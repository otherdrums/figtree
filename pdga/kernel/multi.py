"""Multi-delta generation with shared-layer KV caching.

Architecture:
- Layers 0..crystal-1: shared DynamicCache (identical for all deltas — same
  dummy at position 0, same prompt tokens at positions 1+)
- Layers crystal..N-1: per-delta DynamicCache (diverges at crystal layer
  where each delta's unique boundary residual replaces position 0)

VRAM overhead is ~4 KB per layer per position per delta.  For 2 deltas at
65 tokens this adds ~13 MB to the ~2.6 GB model footprint (0.5%).

Speed: prefill runs L0-22 once (shared), then L23-35 per delta.  Decode
steps run new-token-only forward through all layers (~5× faster than full
forward re-run).
"""

from __future__ import annotations

import time
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.cache_utils import DynamicCache

from pdga.delta.context import ContextDelta
from pdga.kernel.prompt import build_prompt_ids


class MultiDeltaResidualCache:
    """Routes KV cache writes based on layer index.

    Layers 0..crystal-1 go to a single *shared* DynamicCache (same dummy+prompt
    forward for all deltas).  Layers crystal..N-1 are routed to per-delta
    DynamicCaches via a small forwarding shim.
    """

    def __init__(self, crystal: int, num_deltas: int):
        self.crystal = crystal
        self.shared = DynamicCache()
        self.per_delta = [DynamicCache() for _ in range(num_deltas)]
        # The HuggingFace layer.forward() calls past_key_values.update() with
        # (key_states, value_states, layer_idx, cache_kwargs).  We override
        # DynamicCache's methods via a proxy pattern below.
        #
        # Instead, we build a list of (cache_for_layer_fn) that the generation
        # loop calls explicitly.  This is cleaner than subclassing DynamicCache.

    def get_cache(self, layer_idx: int, delta_idx: int) -> DynamicCache:
        """Return the right DynamicCache for a given layer and delta."""
        if layer_idx < self.crystal:
            return self.shared
        return self.per_delta[delta_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Total sequence length (same for all caches at a given layer)."""
        if layer_idx < self.crystal:
            return self.shared.get_seq_length(layer_idx)
        return self.per_delta[0].get_seq_length(layer_idx)

    def update(self, key: torch.Tensor, value: torch.Tensor,
               layer_idx: int, delta_idx: int, cache_kwargs: dict | None = None):
        """Store K/V for a specific layer and delta."""
        if cache_kwargs is None:
            cache_kwargs = {}
        cache = self.get_cache(layer_idx, delta_idx)
        cache.update(key, value, layer_idx, cache_kwargs)


def generate_multi(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    deltas: list[ContextDelta],
    max_new_tokens: int = 60,
    sample_temp: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    injection_coefficient: float = 1.5,
    eos_token_id: int | None = None,
) -> list[dict]:
    """Generate from multiple deltas in parallel with shared-layer KV caching.

    Prefill:
      1. Run [dummy, prompt] through L0..crystal-1 ONCE → shared DynamicCache.
      2. For each delta: swap boundary at L{crystal}, inject delta, run L{crystal}..L{N-1},
         store per-delta KV cache.

    Decode:
      For each step, for each delta: run the new token through ALL layers
      using the combined shared + per-delta KV caches.  At L0..crystal-1
      the shared cache is used; at L{crystal}..L{N-1} the delta-specific
      cache is used.

    Returns a list of result dicts, one per delta.
    """
    device = model.device
    dtype = model.dtype
    embed = model.get_input_embeddings()
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    eos = eos_token_id or tokenizer.eos_token_id

    prompt_ids = build_prompt_ids(tokenizer, prompt, enable_thinking=False)
    P = len(prompt_ids)
    L = 1 + P

    injection_vecs = []
    crystal_layers = []
    boundaries = []
    delta_metas = []

    for delta in deltas:
        if delta.boundaries is None or delta.boundaries.shape[0] == 0:
            continue
        crystal = delta.manifest.crystal_layer
        b = torch.from_numpy(
            delta.boundaries.mean(axis=0).astype("float32")
        ).to(device=device, dtype=dtype)
        inj = _build_injection_vec(
            delta.injection_token_ids, delta.injection_coefficients,
            set(prompt_ids), embed, device, dtype, injection_coefficient,
        )
        boundaries.append(b)
        crystal_layers.append(crystal)
        injection_vecs.append(inj)
        delta_metas.append(delta)

    if not delta_metas:
        return []

    num_deltas = len(delta_metas)
    # All deltas should have the same crystal layer (from the same model)
    crystal = max(crystal_layers)

    # --- Prefill: shared layers 0..crystal-1 ---
    dummy = torch.zeros(1, 1, hidden_size, device=device, dtype=dtype)
    emb = embed(torch.tensor([prompt_ids], dtype=torch.long, device=device))
    h_shared = torch.cat([dummy, emb], dim=1)
    pos_ids = torch.arange(0, L, device=device, dtype=torch.long).unsqueeze(0)
    rotary = model.model.rotary_emb
    norm = model.model.norm
    lm_head = model.lm_head

    shared_cache = DynamicCache()

    with torch.inference_mode():
        for layer_idx in range(crystal):
            layer = model.model.layers[layer_idx]
            pe = rotary(h_shared, pos_ids)
            h_shared = layer(
                h_shared,
                attention_mask=None,
                position_ids=pos_ids,
                position_embeddings=pe,
                use_cache=True,
                past_key_values=shared_cache,
            )

    # --- Prefill: per-delta layers crystal..N-1 ---
    per_delta_caches = []
    per_delta_logits = []

    with torch.inference_mode():
        for d in range(num_deltas):
            # Start from the shared hidden state
            h = h_shared.clone()

            # Swap boundary at position 0
            h[:, 0:1, :] = boundaries[d].unsqueeze(0)

            delta_cache = DynamicCache()

            for layer_idx in range(crystal, num_layers):
                layer = model.model.layers[layer_idx]
                pe = rotary(h, pos_ids)
                h = layer(
                    h,
                    attention_mask=None,
                    position_ids=pos_ids,
                    position_embeddings=pe,
                    use_cache=True,
                    past_key_values=delta_cache,
                )

                # Injection at crystal layer
                if layer_idx == crystal and injection_vecs[d] is not None:
                    h[:, -1, :] = h[:, -1, :] + injection_vecs[d].unsqueeze(0)

            h = norm(h)
            logits = lm_head(h[:, -1:, :])
            per_delta_logits.append(logits)
            per_delta_caches.append(delta_cache)

    # --- Decode loop ---
    generated_ids = [list(prompt_ids) for _ in range(num_deltas)]
    active = [True] * num_deltas
    P_init = len(prompt_ids)

    t0 = time.perf_counter()

    with torch.inference_mode():
        for _step in range(max_new_tokens):
            # Check if all deltas are done
            if not any(active):
                break

            new_tokens = []
            for d in range(num_deltas):
                if not active[d]:
                    new_tokens.append(None)
                    continue
                next_tok = _sample_token(per_delta_logits[d], sample_temp, top_k, top_p)
                if next_tok == eos:
                    active[d] = False
                    new_tokens.append(None)
                else:
                    new_tokens.append(next_tok)
                    generated_ids[d].append(next_tok)

            if not any(active):
                break

            # Run decode for all active deltas
            for d in range(num_deltas):
                if not active[d]:
                    continue
                tok = new_tokens[d]
                pos = L + _step  # next position

                tok_emb = embed(torch.tensor([[tok]], device=device))
                h = tok_emb
                pos_ids_tok = torch.tensor([[pos]], device=device, dtype=torch.long)

                # Layers 0..crystal-1: use shared cache
                for layer_idx in range(crystal):
                    layer = model.model.layers[layer_idx]
                    pe = rotary(h, pos_ids_tok)
                    h = layer(
                        h,
                        attention_mask=None,
                        position_ids=pos_ids_tok,
                        position_embeddings=pe,
                        use_cache=True,
                        past_key_values=shared_cache,
                    )

                # Layers crystal..N-1: use per-delta cache
                for layer_idx in range(crystal, num_layers):
                    layer = model.model.layers[layer_idx]
                    pe = rotary(h, pos_ids_tok)
                    h = layer(
                        h,
                        attention_mask=None,
                        position_ids=pos_ids_tok,
                        position_embeddings=pe,
                        use_cache=True,
                        past_key_values=per_delta_caches[d],
                    )

                h = norm(h)
                per_delta_logits[d] = lm_head(h[:, -1:, :])

    elapsed = time.perf_counter() - t0

    # --- Build results ---
    results = []
    for d, dm in enumerate(delta_metas):
        new_tokens = len(generated_ids[d]) - P_init
        tps = new_tokens / elapsed if elapsed > 0 else 0.0
        completion = tokenizer.decode(generated_ids[d][P_init:]).lstrip()

        results.append({
            "delta_id": dm.delta_id,
            "trust": dm.trust,
            "generated_text": completion,
            "source_url": dm.source_url,
            "tags": dm.tags,
            "mode": "residual-cached",
            "tokens_per_second": tps,
            "num_tokens": new_tokens,
            "elapsed": elapsed / max(1, num_deltas),
            "crystal_layer": crystal,
            "injection_coefficient": injection_coefficient,
        })

    return results


def _build_injection_vec(tid_arr, coeff_arr, query_set, embed,
                          device, dtype, inject_coeff):
    if tid_arr is None or coeff_arr is None or tid_arr.size == 0:
        return None
    flat_ids = tid_arr.reshape(-1)
    flat_coeffs = coeff_arr.reshape(-1)
    filtered_ids, filtered_coeffs = [], []
    for i in range(len(flat_ids)):
        tid = int(flat_ids[i])
        if tid == 0 or tid in query_set:
            continue
        filtered_ids.append(tid)
        filtered_coeffs.append(float(flat_coeffs[i]))
    if not filtered_ids:
        return None
    ids_t = torch.tensor(filtered_ids, dtype=torch.long, device=device)
    coeffs_t = torch.tensor(filtered_coeffs, dtype=torch.float32).to(
        device=device, dtype=dtype
    )
    embs = embed(ids_t)
    return (embs * coeffs_t.unsqueeze(-1) * inject_coeff).sum(dim=0)


def _sample_token(logits, temperature, top_k, top_p):
    logits_s = logits.squeeze(0).squeeze(0)
    if temperature <= 0:
        return int(logits_s.argmax(dim=-1).item())
    probs = torch.softmax(logits_s / temperature, dim=-1)
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
    return int(torch.multinomial(probs, 1).item())
