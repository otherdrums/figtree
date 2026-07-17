"""Ingestion pipeline: text → atomic figments with boundary + kv_cache capture.

Pipeline:
1. Split text into sentences
2. For each sentence:
   a. Forward through model layers 0..crystal_layer
   b. Capture boundary = hidden state of LAST token at crystal_layer
   c. Compute per-token per-layer K/V (unrotated)
   d. Create Figment
3. Create Image Figment with children = sentence figments
4. Optionally create TrustFigment
"""

from __future__ import annotations

import gc
import re
from pathlib import Path

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from figtree.figment import Figment


def _project_kv(
    hidden: torch.Tensor,
    layer: torch.nn.Module,
    num_kv_heads: int,
    head_dim: int,
    use_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project normed hidden states through a layer's k_proj/v_proj.

    `hidden` is (seq_len, hidden_size). Returns unrotated k, v each
    (seq_len, kv_dim). Prefers the custom CUDA kernel (which now supports
    both dense and bitsandbytes 4-bit weights — the latter are dequantized
    on the fly); falls back to a PyTorch matmul if the kernel cannot be built
    or the dtype is not bf16/fp16.
    """
    h_normed = layer.input_layernorm(hidden)
    if use_kernel and h_normed.dtype in (torch.bfloat16, torch.float16):
        try:
            from figtree.kernel.boundary_project import project_boundaries_to_kv

            k_fig, v_fig = project_boundaries_to_kv(
                h_normed.view(-1, h_normed.shape[-1]).contiguous(), layer
            )
            k = k_fig.reshape(-1, num_kv_heads * head_dim)
            v = v_fig.reshape(-1, num_kv_heads * head_dim)
        except Exception:
            # Kernel unavailable (e.g. no CUDA compiler) — PyTorch fallback.
            k_unrot = layer.self_attn.k_proj(h_normed)
            v = layer.self_attn.v_proj(h_normed)
            k = k_unrot
    else:
        k_unrot = layer.self_attn.k_proj(h_normed)
        v = layer.self_attn.v_proj(h_normed)
        k = k_unrot
    if hasattr(layer.self_attn, "k_norm"):
        k = k.view(-1, num_kv_heads, head_dim)
        k = layer.self_attn.k_norm(k)
        k = k.reshape(-1, num_kv_heads * head_dim)
    return k, v


def split_into_sentences(text: str, min_chars: int = 20) -> list[str]:
    """Split text into sentences, merging very short fragments."""
    raw = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    merged = []
    buf = ""
    for s in raw:
        if len(s) < min_chars and not buf:
            buf = s
        elif buf:
            buf += " " + s
            if len(buf) >= min_chars:
                merged.append(buf)
                buf = ""
        else:
            merged.append(s)
    if buf:
        if merged:
            merged[-1] += " " + buf
        else:
            merged.append(buf)
    return merged


def detect_crystal_layer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    calibration_text: str = "The quick brown fox jumps over the lazy dog.",
) -> int:
    """Auto-detect crystal layer where residual stream stabilises.

    Simple heuristic: find layer where L2 change between consecutive
    layers drops below 5% of the max change.
    """
    device = model.device
    ids = tokenizer.encode(calibration_text, return_tensors="pt").to(device)

    residuals = []
    handles = []

    def make_hook(idx):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            residuals.append((idx, o.detach().cpu().float()))
        return hook

    for li, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make_hook(li)))

    try:
        with torch.no_grad():
            model(ids)
    finally:
        for h in handles:
            h.remove()

    diffs = []
    for i in range(1, len(residuals)):
        prev = residuals[i - 1][1]
        curr = residuals[i][1]
        diff = (curr - prev).norm(dim=-1).mean().item()
        diffs.append(diff)

    max_diff = max(diffs) if diffs else 1.0
    threshold = max_diff * 0.05

    for i, d in enumerate(diffs):
        if d < threshold:
            return i + 1

    return max(1, len(model.model.layers) // 2)


def ingest_text_to_figments(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    output_dir: Path,
    source_id: str = "",
    trust: float = 0.5,
    crystal_layer: int | None = None,
    min_chars: int = 20,
) -> list[Figment]:
    """Ingest text into atomic figments with boundary + kv_cache capture.

    Returns list of Figments: [image, sentence_1, sentence_2, ..., trust_assertion]
    """
    device = model.device
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if crystal_layer is None:
        crystal_layer = detect_crystal_layer(model, tokenizer)

    sentences = split_into_sentences(text, min_chars=min_chars)
    if not sentences:
        raise ValueError("Text produced zero sentences")

    sentence_figments: list[Figment] = []
    num_layers = len(model.model.layers)
    config = model.config
    num_kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
    kv_dim = num_kv_heads * head_dim
    # Try the CUDA kernel for both dense and 4-bit models; _project_kv falls
    # back to a PyTorch matmul if the kernel cannot be built or dtype is unsupported.
    use_kernel = True

    def make_hook(layer_idx, storage):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            storage[layer_idx] = o.detach()
        return hook

    layer_outputs: dict[int, torch.Tensor] = {}
    handles = []
    for li in range(num_layers):
        handles.append(model.model.layers[li].register_forward_hook(make_hook(li, layer_outputs)))

    try:
        with torch.no_grad():
            for sent in sentences:
                ids = tokenizer.encode(sent, return_tensors="pt").to(device)
                if ids.shape[1] == 0:
                    continue

                emb_out = model.get_input_embeddings()(ids)
                boundary_emb = emb_out[0, -1, :].float().cpu()

                model(ids)

                if len(layer_outputs) != num_layers:
                    raise RuntimeError(f"Expected {num_layers} layer outputs, got {len(layer_outputs)}")

                boundaries_list = [
                    layer_outputs[li][0, -1, :].float().cpu()
                    for li in range(num_layers)
                ]
                boundaries_arr = torch.stack(boundaries_list).numpy()
                boundary_crystal = boundaries_arr[crystal_layer]

                seq_len = ids.shape[1]
                kv_cache_list = []
                for li in range(num_layers):
                    if li == 0:
                        h_in = emb_out[0]
                    else:
                        h_in = layer_outputs[li - 1][0]
                    layer = model.model.layers[li]
                    k, v = _project_kv(h_in, layer, num_kv_heads, head_dim, use_kernel)
                    k_cache = k.reshape(seq_len, kv_dim)
                    v_cache = v.reshape(seq_len, kv_dim)
                    kv_cache_list.append(torch.stack([k_cache, v_cache], dim=1))

                kv_cache_t = torch.stack(kv_cache_list).float().cpu().numpy()

                figment = Figment.create(
                    text=sent,
                    boundary=boundary_crystal,
                    boundaries=boundaries_arr,
                    boundary_emb=boundary_emb.numpy(),
                    meta={"source_id": source_id, "crystal_layer": crystal_layer},
                    trust=trust,
                )
                sentence_figments.append(figment)

                figment_dir = figment.save(output_dir)
                np.save(figment_dir / "kv_cache.npy", kv_cache_t)

                layer_outputs.clear()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    finally:
        for h in handles:
            h.remove()

    image = Figment.create(
        text=text,
        boundary=sentence_figments[0].boundary.copy() if sentence_figments else np.zeros(1),
        meta={"source_id": source_id, "crystal_layer": crystal_layer, "is_image": True},
        children=[f.figment_id for f in sentence_figments],
        trust=trust,
    )

    trust_figment = Figment.create(
        text=f"Source {source_id} has trust {trust:.2f}",
        boundary=sentence_figments[0].boundary.copy() if sentence_figments else np.zeros(1),
        meta={"edge_type": "trust", "about_figment": image.figment_id, "score": trust},
        sources=[image.figment_id],
    )

    image.save(output_dir)
    trust_figment.save(output_dir)

    all_figments = [image] + sentence_figments + [trust_figment]
    return all_figments
