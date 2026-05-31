"""Text ingestion pipeline — convert arbitrary text into ContextDelta.

The pipeline:
1. Auto-detect crystal layer using the first portion of the text
2. Auto-detect injection layer via residual compression analysis
3. Split text into token windows
4. For each window, capture boundary residual at crystal layer (last token)
5. For each window, capture full KV cache for boundary-kv generation
6. GLiNER extracts entities; entity token IDs stored as injection entries
7. Store as ContextDelta: boundaries + injection token IDs + KV cache + fact chunks
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.cache_utils import DynamicCache

from pdga.delta.base import DeltaManifest, Delta
from pdga.delta.context import ContextDelta
from pdga.ingest.extractor import detect_crystal_layer
from pdga.kernel.corrected import detect_injection_layer
from pdga.ingest.gliner_extractor import (
    GlinerExtractor,
    extract_fact_entities_for_window,
)
from pdga.delta.cache_io import save_window_cache

_gliner: Optional[GlinerExtractor] = None


def _get_gliner() -> GlinerExtractor:
    global _gliner
    if _gliner is None:
        _gliner = GlinerExtractor()
    return _gliner


def ingest_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    output_dir: Path,
    window_size: int = 200,
    novel_fraction: float = 0.05,
    trust: float = 0.5,
    source_url: str = "",
    tags: list[str] | None = None,
    crystal_layer: int | None = None,
) -> ContextDelta:
    """Ingest text into a ContextDelta with GLiNER-based injection entries.

    Captures one boundary residual per window, extracts GLiNER entities,
    and stores entity token IDs as injection entries for LARQL-style
    token injection during generation.
    """
    device = model.device
    output_dir = Path(output_dir)
    num_layers = model.config.num_hidden_layers

    if crystal_layer is None:
        crystal_layer = detect_crystal_layer(
            model, tokenizer,
            calibration_text=text[:2000],
        )

    token_ids_all = tokenizer.encode(text)
    if not token_ids_all:
        raise ValueError("Text produced zero tokens after encoding")

    windows = [
        token_ids_all[i : i + window_size]
        for i in range(0, len(token_ids_all), window_size)
    ]

    injection_layer = detect_injection_layer(
        model, tokenizer, windows, calibration_text=text[:1000],
    )

    all_boundaries = []
    all_injection_token_ids: list[list[int]] = []
    all_injection_coeffs: list[list[float]] = []
    fact_chunks_all: list[list[int]] = []
    dynamic_labels: set[str] = set()
    window_token_lists = []

    # Generate delta_id and create .pdga directory early for KV cache storage
    delta_id = Delta.generate_id(text[:10000])
    delta_dir = output_dir / f"{delta_id}.pdga"
    delta_dir.mkdir(parents=True, exist_ok=True)

    per_token_residuals = None

    def capture_hook(module, input, output):
        nonlocal per_token_residuals
        out = output[0] if isinstance(output, tuple) else output
        per_token_residuals = out.detach().float().cpu().numpy()

    target_layer = model.model.layers[crystal_layer]
    handle = target_layer.register_forward_hook(capture_hook)
    gliner = _get_gliner()

    try:
        with torch.inference_mode():
            for i, window_tokens in enumerate(windows):
                if not window_tokens:
                    continue

                input_ids = torch.tensor([window_tokens], dtype=torch.long, device=device)
                L = len(window_tokens)
                pos_ids = torch.arange(0, L, device=device, dtype=torch.long).unsqueeze(0)

                # Manual forward with DynamicCache to capture KV cache
                h = model.get_input_embeddings()(input_ids)
                cache = DynamicCache()

                for li in range(num_layers):
                    layer = model.model.layers[li]
                    pe = model.model.rotary_emb(h, pos_ids)
                    h = layer(
                        h, attention_mask=None, position_ids=pos_ids,
                        position_embeddings=pe, use_cache=True,
                        past_key_values=cache,
                    )

                if per_token_residuals is None:
                    raise RuntimeError(f"Layer {crystal_layer} hook did not fire")

                residuals = per_token_residuals[0]
                seq_len = residuals.shape[0]

                boundary = residuals[seq_len - 1].copy()
                all_boundaries.append(boundary)

                # Save KV cache for boundary-kv generation
                save_window_cache(cache, delta_dir, i, num_layers)
                del cache  # free GPU memory — cache already saved to disk

                window_fact_chunks, entity_positions, w_labels = extract_fact_entities_for_window(
                    window_tokens, residuals, gliner, tokenizer,
                    score_threshold=0.4, span_stability=0.35,
                )

                window_inj_ids: list[int] = []
                window_inj_coeffs: list[float] = []
                for pos in sorted(entity_positions):
                    if pos < len(window_tokens):
                        window_inj_ids.append(window_tokens[pos])
                        window_inj_coeffs.append(1.0)

                all_injection_token_ids.append(window_inj_ids)
                all_injection_coeffs.append(window_inj_coeffs)

                fact_chunks_all.extend(window_fact_chunks)
                dynamic_labels.update(w_labels)
                window_token_lists.append(window_tokens)
                per_token_residuals = None
                del window_tokens  # free variable

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    finally:
        handle.remove()

    if not all_boundaries and not fact_chunks_all:
        raise RuntimeError("No boundaries or facts captured — all windows were empty")

    boundaries_arr = np.stack(all_boundaries).astype(np.float32)

    max_entries = max((len(ids) for ids in all_injection_token_ids), default=0)
    token_ids_arr = np.zeros((len(all_injection_token_ids), max_entries), dtype=np.int32)
    coeffs_arr = np.zeros((len(all_injection_coeffs), max_entries), dtype=np.float32)
    for i, (ids, coeffs) in enumerate(zip(all_injection_token_ids, all_injection_coeffs)):
        token_ids_arr[i, :len(ids)] = ids
        coeffs_arr[i, :len(coeffs)] = coeffs

    hidden_size = model.config.hidden_size

    manifest = DeltaManifest(
        version="0.1.0",
        delta_id=delta_id,
        delta_type="context",
        base_model_id=model.config._name_or_path or "unknown",
        hidden_size=hidden_size,
        num_layers=num_layers,
        crystal_layer=crystal_layer,
        injection_layer=injection_layer,
        window_size=window_size,
        num_windows=len(window_token_lists),
    )

    delta = ContextDelta(
        manifest=manifest,
        boundaries=boundaries_arr,
        window_tokens=window_token_lists,
        fact_tokens=fact_chunks_all if fact_chunks_all else None,
        dynamic_labels=sorted(dynamic_labels) if dynamic_labels else None,
        source_text=text,
        trust=trust,
        source_url=source_url,
        tags=tags or [],
        injection_deltas=boundaries_arr,
        injection_token_ids=token_ids_arr,
        injection_coefficients=coeffs_arr,
    )

    delta.save(output_dir)
    delta.path = output_dir / f"{delta_id}.pdga"

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return delta
