"""Text ingestion pipeline — convert arbitrary text into ContextDelta.

The pipeline:
1. Auto-detect crystal layer using the first portion of the text
2. Split text into token windows
3. For each window, capture boundary residual at crystal layer (last token)
4. For each window, capture full KV cache for boundary-kv generation
5. Store as ContextDelta: boundaries + KV cache + window tokens
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.cache_utils import DynamicCache

from pdga.delta.base import DeltaManifest, Delta
from pdga.delta.context import ContextDelta
from pdga.ingest.extractor import detect_crystal_layer
from pdga.delta.cache_io import save_window_cache


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
    """Ingest text into a ContextDelta with boundary residuals and KV cache.

    Captures one boundary residual per window and the full KV cache for
    each window, enabling instant prefill during generation.
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

    all_boundaries = []
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

                window_token_lists.append(window_tokens)
                per_token_residuals = None
                del window_tokens  # free variable

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    finally:
        handle.remove()

    if not all_boundaries:
        raise RuntimeError("No boundaries captured — all windows were empty")

    boundaries_arr = np.stack(all_boundaries).astype(np.float32)

    hidden_size = model.config.hidden_size

    manifest = DeltaManifest(
        version="0.1.0",
        delta_id=delta_id,
        delta_type="context",
        base_model_id=model.config._name_or_path or "unknown",
        hidden_size=hidden_size,
        num_layers=num_layers,
        crystal_layer=crystal_layer,
        window_size=window_size,
        num_windows=len(window_token_lists),
    )

    delta = ContextDelta(
        manifest=manifest,
        boundaries=boundaries_arr,
        window_tokens=window_token_lists,
        source_text=text,
        trust=trust,
        source_url=source_url,
        tags=tags or [],
    )

    delta.save(output_dir)
    delta.path = output_dir / f"{delta_id}.pdga"

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return delta
