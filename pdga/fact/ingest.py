"""Ingestion pipeline v2: text → atomic facts with boundary capture.

No pre-computed KV cache. Only boundaries are stored (~10 KB each).

Pipeline:
1. Split text into sentences
2. For each sentence:
   a. Forward through model layers 0..crystal_layer
   b. Capture boundary = hidden state of LAST token at crystal_layer
   c. Create Fact
3. Create narrative Fact with children = sentence facts
4. Optionally create TrustFact
"""

from __future__ import annotations

import gc
import re
from pathlib import Path

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from pdga.fact.primitive import Fact


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

    # Compute L2 differences between consecutive layers
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
            return i + 1  # layer index where stabilization begins

    return max(1, len(model.model.layers) // 2)


def ingest_text_to_facts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    output_dir: Path,
    source_id: str = "",
    trust: float = 0.5,
    crystal_layer: int | None = None,
    min_chars: int = 20,
) -> list[Fact]:
    """Ingest text into atomic facts with boundary capture.

    Returns list of Facts: [narrative, sentence_1, sentence_2, ...]
    """
    device = model.device
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if crystal_layer is None:
        crystal_layer = detect_crystal_layer(model, tokenizer)

    sentences = split_into_sentences(text, min_chars=min_chars)
    if not sentences:
        raise ValueError("Text produced zero sentences")

    # Ingest each sentence as a fact
    sentence_facts: list[Fact] = []
    num_layers = len(model.model.layers)
    config = model.config
    num_kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
    kv_dim = num_kv_heads * head_dim

    def make_hook(layer_idx, storage):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            storage[layer_idx] = o.detach()
        return hook

    # Register hooks on all layers
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

                # Capture embedding output for all tokens
                emb_out = model.get_input_embeddings()(ids)  # (1, seq_len, hidden_size)
                boundary_emb = emb_out[0, -1, :].float().cpu()

                model(ids)  # forward through all layers, hooks capture all outputs

                if len(layer_outputs) != num_layers:
                    raise RuntimeError(f"Expected {num_layers} layer outputs, got {len(layer_outputs)}")

                # Build per-layer boundaries: (num_layers, hidden_size)
                boundaries_list = [
                    layer_outputs[li][0, -1, :].float().cpu()
                    for li in range(num_layers)
                ]
                boundaries_arr = torch.stack(boundaries_list).numpy()

                boundary_crystal = boundaries_arr[crystal_layer]  # for backward compat

                # ── Compute per-token per-layer K/V cache (unrotated) ──
                seq_len = ids.shape[1]
                kv_cache_list = []
                for li in range(num_layers):
                    if li == 0:
                        h_in = emb_out[0]  # (seq_len, hidden_size)
                    else:
                        h_in = layer_outputs[li - 1][0]  # (seq_len, hidden_size)
                    layer = model.model.layers[li]
                    # Apply input_layernorm (pre-norm) before projection,
                    # matching what the layer's forward pass does internally
                    h_normed = layer.input_layernorm(h_in)
                    k_unrot = layer.self_attn.k_proj(h_normed)  # (seq_len, kv_dim)
                    v = layer.self_attn.v_proj(h_normed)  # (seq_len, kv_dim)
                    # Handle k_norm (QK-norm after projection)
                    k_normed = k_unrot.view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)
                    if hasattr(layer.self_attn, "k_norm"):
                        k_normed = layer.self_attn.k_norm(k_normed)
                    k_unrot = k_normed.transpose(1, 2).reshape(seq_len, kv_dim)
                    v = v.reshape(seq_len, kv_dim)
                    kv_cache_list.append(torch.stack([k_unrot, v], dim=1))  # (seq_len, 2, kv_dim)

                kv_cache_t = torch.stack(kv_cache_list).float().cpu().numpy()  # (num_layers, seq_len, 2, kv_dim)

                fact = Fact.create(
                    text=sent,
                    boundary=boundary_crystal,
                    boundaries=boundaries_arr,
                    boundary_emb=boundary_emb.numpy(),
                    meta={"source_id": source_id, "crystal_layer": crystal_layer},
                    trust=trust,
                )
                sentence_facts.append(fact)

                # Save fact + kv_cache immediately
                fact_dir = fact.save(output_dir)
                np.save(fact_dir / "kv_cache.npy", kv_cache_t)

                layer_outputs.clear()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    finally:
        for h in handles:
            h.remove()

    # Create narrative fact
    narrative = Fact.create(
        text=text,
        boundary=sentence_facts[0].boundary.copy() if sentence_facts else np.zeros(1),
        meta={"source_id": source_id, "crystal_layer": crystal_layer, "is_narrative": True},
        children=[f.fact_id for f in sentence_facts],
        trust=trust,
    )

    # Create trust assertion fact
    trust_fact = Fact.create(
        text=f"Source {source_id} has trust {trust:.2f}",
        boundary=sentence_facts[0].boundary.copy() if sentence_facts else np.zeros(1),
        meta={"edge_type": "trust", "about_fact": narrative.fact_id, "score": trust},
        sources=[narrative.fact_id],
    )

    # Save narrative and trust (sentence facts already saved above with kv_cache)
    narrative.save(output_dir)
    trust_fact.save(output_dir)

    all_facts = [narrative] + sentence_facts + [trust_fact]
    return all_facts
