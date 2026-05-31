"""Boundary-KV generation — instant prefill from pre-computed KV caches.

Loads full-article KV caches stored during ingestion directly into DynamicCache,
eliminating the 36-layer prefill for context tokens.  Deltas are processed
sequentially to minimise GPU memory.

All forward passes are wrapped in ``torch.no_grad()`` to avoid keeping 4-bit
dequantized weight buffers in the autograd graph (saves ~450 MB on GPU).

Ingestion should use a single large window (window_size ≈ article length)
to produce one KV cache per article — no RoPE offset issues.
"""

from __future__ import annotations

import gc
import time
import torch
from pathlib import Path
from typing import Optional
from transformers.cache_utils import DynamicCache

from pdga.delta.cache_io import load_window_cache
from pdga.kernel.prompt import build_prompt_ids


def generate_boundary_kv(
    model,
    tokenizer,
    prompt: str,
    delta_paths: list[Path],
    window_indices: list[list[int]] | None = None,
    injection_layer: int = 30,
    injection_deltas: Optional[torch.Tensor] = None,
    max_new_tokens: int = 200,
    sample_temp: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    eos_token_id: Optional[int] = None,
) -> list[dict]:
    """Generate from pre-computed KV caches — deltas processed sequentially.

    Each delta's KV cache is loaded from disk, seeded into a DynamicCache,
    the prompt is prefilled through all layers attending to cached KVs,
    and tokens are decoded autoregressively.  After each delta finishes,
    its GPU memory is freed before moving to the next delta.

    Args:
        delta_paths: List of paths to .pdga directories.
        window_indices: Per-delta list of window indices to load.
            If None, loads all available windows for each delta.
            Use [[0]] to load only the first window (for RoPE safety).

    Returns:
        List of result dicts in the same order as delta_paths.
    """
    device = model.device
    dtype = model.dtype
    embed = model.get_input_embeddings()
    lm_head = model.lm_head
    rotary = model.model.rotary_emb
    final_norm = model.model.norm
    num_layers = model.config.num_hidden_layers
    eos = eos_token_id or tokenizer.eos_token_id

    N = len(delta_paths)
    if N == 0:
        return []

    prompt_ids = build_prompt_ids(tokenizer, prompt, enable_thinking=False)
    P = len(prompt_ids)
    results = []

    # Process deltas sequentially to keep GPU memory low.
    # Each delta loads its KV cache, generates, then frees memory.
    for d in range(N):
        delta_dir = Path(delta_paths[d])

        # Determine which windows to load
        if window_indices is not None and d < len(window_indices):
            wins = window_indices[d]
        else:
            # Load all available windows
            from pdga.delta.cache_io import list_window_caches
            wins = list_window_caches(delta_dir)

        # ── Load KV cache for this delta (layer-by-layer to save memory) ────
        cache = DynamicCache()
        offset = 0

        for w in wins:
            cache_path = delta_dir / f"kv_cache_w{w}.pt"
            if not cache_path.exists():
                continue

            # Load from disk to CPU first, then move to GPU per-layer
            state = torch.load(str(cache_path), map_location="cpu", weights_only=True)
            w_len = None

            if offset == 0:
                for li in range(num_layers):
                    k_key = f"layer_{li}_keys"
                    if k_key not in state:
                        break
                    k = state[k_key].to(device=device, dtype=dtype)
                    v = state[f"layer_{li}_values"].to(device=device, dtype=dtype)
                    cache.update(k, v, li)
                    if w_len is None:
                        w_len = k.shape[2]

                # Free the state dict
                del state
                offset = w_len if w_len else 0

        gc.collect()
        torch.cuda.empty_cache()

        if offset == 0:
            results.append({
                "generated_text": "[no KV cache found]",
                "num_tokens": 0,
                "tokens_per_second": 0,
                "elapsed": 0,
                "path": "boundary-kv",
                "windows_loaded": 0,
            })
            continue

        # ── Prefill prompt tokens ────────────────────────────────────────
        prompt_emb = embed(torch.tensor([prompt_ids], dtype=torch.long, device=device))
        h = prompt_emb
        pos_ids = torch.arange(offset, offset + P, device=device, dtype=torch.long).unsqueeze(0)

        # Explicit causal mask: prompt positions must see all cached KVs
        attn_mask = torch.full(
            (1, 1, P, offset + P), -(2**15), device=device, dtype=dtype,
        )
        for i in range(P):
            attn_mask[:, :, i, :offset + i + 1] = 0.0

        inj = injection_deltas[d:d+1] if injection_deltas is not None else None
        with torch.no_grad():
            for li in range(num_layers):
                layer = model.model.layers[li]
                pe = rotary(h, pos_ids)
                h = layer(
                    h, attention_mask=attn_mask, position_ids=pos_ids,
                    position_embeddings=pe, use_cache=True,
                    past_key_values=cache,
                )
                if li == injection_layer and inj is not None:
                    h[:, -1:, :] = h[:, -1:, :] + inj.to(device=device, dtype=dtype).unsqueeze(0)

            h = final_norm(h)
            logits = lm_head(h[:, -1:, :])

        # Release fragmented memory from prefill before decode
        del attn_mask
        gc.collect()
        torch.cuda.empty_cache()

        # ── Decode ───────────────────────────────────────────────────────
        gen_ids = list(prompt_ids)
        t0 = time.perf_counter()

        for step in range(max_new_tokens):
            with torch.no_grad():
                nxt = _sample(logits, sample_temp, top_k, top_p)
            tid = int(nxt.item()) if hasattr(nxt, "item") else int(nxt)
            if tid == eos:
                break
            gen_ids.append(tid)

            with torch.no_grad():
                tok_emb = embed(torch.tensor([[tid]], dtype=torch.long, device=device))
                cur_pos = offset + P + step
                pos_one = torch.tensor([[cur_pos]], device=device, dtype=torch.long)
                h = tok_emb

                for li in range(num_layers):
                    layer = model.model.layers[li]
                    pe = rotary(h, pos_one)
                    h = layer(
                        h, attention_mask=None, position_ids=pos_one,
                        position_embeddings=pe, use_cache=True,
                        past_key_values=cache,
                    )
                    if li == injection_layer and inj is not None:
                        h[:, 0:1, :] = h[:, 0:1, :] + inj.to(device=device, dtype=dtype).unsqueeze(0)

                h = final_norm(h)
                logits = lm_head(h[:, -1:, :])

        elapsed = time.perf_counter() - t0
        ntok = len(gen_ids) - P
        tps = ntok / elapsed if elapsed > 0 else 0.0
        text = tokenizer.decode(gen_ids[P:], skip_special_tokens=True)
        results.append({
            "generated_text": text,
            "num_tokens": ntok,
            "tokens_per_second": tps,
            "elapsed": elapsed,
            "path": "boundary-kv",
            "windows_loaded": len(wins),
        })

        # Free GPU memory before processing next delta
        del cache, gen_ids
        gc.collect()
        torch.cuda.empty_cache()

    return results


def _sample(logits, temperature, top_k, top_p):
    logits_f = logits.squeeze().float()
    if temperature <= 0:
        return torch.tensor(int(logits_f.argmax(dim=-1).item()))
    probs = torch.softmax(logits_f / temperature, dim=-1)
    if top_k > 0:
        vals, idx = torch.topk(probs, min(top_k, probs.size(-1)))
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask[idx] = True
        probs = probs * mask
    if top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cs = torch.cumsum(sp, dim=-1)
        cutoff = (cs > top_p).nonzero(as_tuple=True)
        if len(cutoff[0]) > 0:
            sp[cutoff[0][0] + 1:] = 0.0
            probs = torch.zeros_like(probs).scatter_(0, si, sp)
    probs = probs / probs.sum()
    return torch.multinomial(probs, 1).item()
