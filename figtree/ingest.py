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


def boundary_for_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    crystal_layer: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (boundary, boundaries, boundary_emb) for the last token of ``text``.

    ``boundary`` is the hidden state at ``crystal_layer``; ``boundaries`` is the
    per-layer hidden state of the last token; ``boundary_emb`` is the last-token
    input embedding. Reused by hierarchical summarization to give a higher-level
    Image figment its own genuine boundary instead of copying a child's.
    """
    if crystal_layer is None:
        crystal_layer = detect_crystal_layer(model, tokenizer)
    device = model.device
    num_layers = len(model.model.layers)

    ids = tokenizer.encode(text, return_tensors="pt").to(device)
    layer_outputs: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(idx):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            layer_outputs[idx] = o.detach()
        return hook

    for li in range(num_layers):
        handles.append(model.model.layers[li].register_forward_hook(make_hook(li)))
    try:
        with torch.no_grad():
            emb_out = model.get_input_embeddings()(ids)
            model(ids)
            last = ids.shape[1] - 1
            boundaries_arr = torch.stack(
                [layer_outputs[li][0, last, :].float().cpu() for li in range(num_layers)]
            ).numpy()
            boundary_emb = emb_out[0, last, :].float().cpu().numpy()
    finally:
        for h in handles:
            h.remove()

    return boundaries_arr[crystal_layer], boundaries_arr, boundary_emb



def ingest_text_to_figments(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    source_id: str = "",
    trust: float = 0.5,
    crystal_layer: int | None = None,
    min_chars: int = 20,
    store=None,
    kv_manager=None,
    compute_kv: bool = False,
    summarize_images: bool = False,
) -> list[Figment]:
    """Ingest text into atomic figments with boundary + (optional) K/V capture.

    Storage is LanceDB-backed: ``store`` is required. Figments are upserted as
    compressed rows; the lightweight LanceDB row holds the boundary + text +
    metadata, while K/V caches live outside the row as external quantized blobs.

    K/V caching: by default (``compute_kv=False``) no K/V blob is persisted
    (lazy mode — recomputed on demand by ``KVCacheManager``). Set
    ``compute_kv=True`` (and pass a ``kv_manager``) to eagerly persist quantized
    K/V blobs and record ``kv_uri`` on each figment.
    """
    if store is None:
        raise ValueError(
            "store is required: figments are persisted to a LanceDB store. "
            "Pass a FigmentStore from figtree.lancedb_store.connect()."
        )
    device = model.device

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

    # Build one concatenated token stream with separators between sentences,
    # exactly mirroring FigmentGenerator.generate(). A single forward through the
    # model captures cross-figment attention, so the cached K/V we slice per
    # figment match what generate() would produce -> boundary replay == text forward.
    sep_ids = tokenizer.encode("\n\n", add_special_tokens=False)
    stream: list[int] = []
    starts: list[int] = []
    kept_sentences: list[str] = []
    for sent in sentences:
        ids = tokenizer.encode(sent, add_special_tokens=False)
        if not ids:
            continue
        if stream:
            stream.extend(sep_ids)
        starts.append(len(stream))
        stream.extend(ids)
        kept_sentences.append(sent)

    if not stream:
        raise ValueError("Text produced zero tokens")

    # Each figment's cached slice spans its tokens AND the separator that follows
    # it, so concatenated slices reproduce the full stream (separators included)
    # and generate_from_boundaries can replay without any extra forward pass.
    spans: list[tuple[int, int]] = []  # (start, end_exclusive) per sentence
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(stream)
        spans.append((start, end))

    all_ids = torch.tensor([stream], dtype=torch.long, device=device)
    seq_len_total = all_ids.shape[1]

    layer_outputs: dict[int, torch.Tensor] = {}
    handles = []
    for li in range(num_layers):
        handles.append(model.model.layers[li].register_forward_hook(make_hook(li, layer_outputs)))

    try:
        with torch.no_grad():
            emb_out = model.get_input_embeddings()(all_ids)
            model(all_ids)

            if len(layer_outputs) != num_layers:
                raise RuntimeError(f"Expected {num_layers} layer outputs, got {len(layer_outputs)}")

            # Full-stream per-layer hidden states (the input to each layer).
            # layer li input hidden = emb_out for li==0 else layer_outputs[li-1][0].
            layer_inputs: list[torch.Tensor] = [emb_out[0]]
            for li in range(1, num_layers):
                layer_inputs.append(layer_outputs[li - 1][0])

            # Precompute K/V for every token at every layer across the whole
            # stream ONLY when eagerly persisting K/V caches. In lazy mode the
            # KVCacheManager recomputes on demand, so we skip this cost here.
            full_kv: list[torch.Tensor] | None = None
            if compute_kv:
                full_kv = []  # list over layers of (seq_len_total, 2, kv_dim)
                for li in range(num_layers):
                    k, v = _project_kv(layer_inputs[li], model.model.layers[li], num_kv_heads, head_dim, use_kernel)
                    k = k.reshape(seq_len_total, kv_dim)
                    v = v.reshape(seq_len_total, kv_dim)
                    full_kv.append(torch.stack([k, v], dim=1).float().cpu())

            for si, (sent, (start, end)) in enumerate(zip(kept_sentences, spans)):
                if end <= start:
                    continue
                # Last real token of this sentence (exclude trailing separator if any).
                last_tok = end - 1
                if si + 1 < len(starts):
                    last_tok -= len(sep_ids)
                # boundaries: hidden state of the last real token, per layer.
                boundaries_list = [
                    layer_outputs[li][0, last_tok, :].float().cpu() for li in range(num_layers)
                ]
                boundaries_arr = torch.stack(boundaries_list).numpy()
                boundary_crystal = boundaries_arr[crystal_layer]
                boundary_emb = emb_out[0, last_tok, :].float().cpu().numpy()

                figment = Figment.create(
                    text=sent,
                    boundary=boundary_crystal,
                    boundaries=boundaries_arr,
                    boundary_emb=boundary_emb,
                    meta={"source_id": source_id, "crystal_layer": crystal_layer},
                    trust=trust,
                )
                sentence_figments.append(figment)

                # Persist per-figment K/V slice when eagerly caching.
                if compute_kv and full_kv is not None:
                    kv_cache_list = [full_kv[li][start:end] for li in range(num_layers)]
                    kv_cache_t = torch.stack(kv_cache_list).numpy()
                    if kv_manager is not None:
                        uri = kv_manager._blob_uri(figment.figment_id)  # noqa: SLF001
                        qkv, qmeta = kv_manager._quantize(kv_cache_t)  # noqa: SLF001
                        kv_manager._write_blob(uri, qkv, qmeta)  # noqa: SLF001
                        figment.meta["kv_uri"] = uri
                        figment.meta["has_kv_cache"] = True
                    else:
                        raise ValueError(
                            "compute_kv=True requires a kv_manager to persist K/V blobs."
                        )

            layer_outputs.clear()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    finally:
        for h in handles:
            h.remove()

    image_boundary = sentence_figments[0].boundary.copy() if sentence_figments else np.zeros(1)
    image_boundaries = sentence_figments[0].boundaries.copy() if sentence_figments else None
    image_emb = sentence_figments[0].boundary_emb.copy() if sentence_figments else None
    image_text = text
    if summarize_images and sentence_figments:
        from figtree.summarize import summarize_image

        summary, image_boundary, image_boundaries, image_emb = summarize_image(
            model, tokenizer, sentence_figments, crystal_layer=crystal_layer
        )
        image_text = summary

    image = Figment.create(
        text=image_text,
        boundary=image_boundary,
        boundaries=image_boundaries,
        boundary_emb=image_emb,
        meta={"source_id": source_id, "crystal_layer": crystal_layer, "is_image": True, "base_trust": trust},
        children=[f.figment_id for f in sentence_figments],
        trust=trust,
    )

    trust_figment = Figment.create(
        text=f"Source {source_id} has trust {trust:.2f}",
        boundary=sentence_figments[0].boundary.copy() if sentence_figments else np.zeros(1),
        meta={"edge_type": "trust", "about_figment": image.figment_id, "score": trust, "base_trust": trust},
        sources=[image.figment_id],
    )

    all_figments = [image] + sentence_figments + [trust_figment]

    hidden = image.boundary.shape[0]
    # Upsert in deterministic id order so trust/image overwrite cleanly.
    store.upsert(all_figments, hidden_size=hidden)

    return all_figments
