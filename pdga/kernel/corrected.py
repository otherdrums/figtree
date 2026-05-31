"""Corrected residual injection engine.

Three key fixes from LARQL Apollo's actual semantics:

1. Injection at L(num_layers - 6), NOT at crystal_layer.
   LARQL research: L23 features are noisy "population code." L30+ features
   are context-selective — injecting there adds signal the MLP can use.

2. Top-K=8 entries, routed by token-overlap between query and stored
   window tokens.  (Previously: 260+ entries summed blindly.)

3. Context = matched window tokens + prompt tokens.
   The original text tokens give the model the background it needs to
   resolve injected entity embeddings into specific facts.

The boundary residual is NOT used for position-0 substitution on
Qwen3/HuggingFace — HF attention cannot mix L23-level boundary with
L0-level token embeddings at the same layer.
"""

from __future__ import annotations

import time
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.cache_utils import DynamicCache

from pdga.delta.context import ContextDelta
from pdga.kernel.prompt import build_prompt_ids


def generate_multi_corrected(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    deltas: list[ContextDelta],
    max_new_tokens: int = 60,
    sample_temp: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    injection_coefficient: float = 10.0,
    injection_topk: int = 8,
    injection_layer: int | None = None,
    eos_token_id: int | None = None,
) -> list[dict]:
    """Generate from multiple deltas in parallel with corrected injection.

    Each delta gets its own window-token context + routed injection entries.
    KV caching speeds up the full forward pass.
    """
    device = model.device
    dtype = model.dtype
    embed = model.get_input_embeddings()
    lm_head = model.lm_head
    rotary = model.model.rotary_emb
    norm = model.model.norm
    num_layers = model.config.num_hidden_layers
    eos = eos_token_id or tokenizer.eos_token_id

    if injection_layer is None:
        injection_layer = _detect_injection_layer(num_layers)

    prompt_ids = build_prompt_ids(tokenizer, prompt, enable_thinking=False)
    P = len(prompt_ids)

    # Pre-compute per-delta data
    delta_data = []
    for delta in deltas:
        entries = _route_and_retrieve(delta, prompt_ids, injection_topk)
        if not entries:
            continue

        # Injection delta
        inj_ids = torch.tensor([e[0] for e in entries], device=device)
        inj_coeffs = torch.tensor([e[1] for e in entries], dtype=torch.float32, device=device).to(dtype)
        inj_embs = embed(inj_ids)
        injection_delta = (inj_embs * inj_coeffs.unsqueeze(-1) * injection_coefficient).sum(dim=0)

        matched_window = entries[0][2]
        window_tokens = list(delta.get_window_tokens(matched_window))

        delta_data.append({
            "delta": delta,
            "injection_delta": injection_delta,
            "window_tokens": window_tokens,
            "matched_window": matched_window,
            "num_entries": len(entries),
        })

    if not delta_data:
        return []

    num_deltas = len(delta_data)
    gen_ids = [list(prompt_ids) for _ in range(num_deltas)]
    active = [True] * num_deltas

    # Prefill: run [window_tokens + prompt] through all layers with KV cache
    per_delta_caches = []
    per_delta_logits = []

    with torch.inference_mode():
        for d in range(num_deltas):
            dd = delta_data[d]
            w_tokens = dd["window_tokens"]
            all_ids = w_tokens + prompt_ids
            L = len(all_ids)
            tok_emb = embed(torch.tensor([all_ids], dtype=torch.long, device=device))
            h = tok_emb
            pos_ids = torch.arange(0, L, device=device).unsqueeze(0)
            cache = DynamicCache()

            for layer_idx in range(num_layers):
                lay = model.model.layers[layer_idx]
                pe = rotary(h, pos_ids)
                h = lay(
                    h,
                    attention_mask=None,
                    position_ids=pos_ids,
                    position_embeddings=pe,
                    use_cache=True,
                    past_key_values=cache,
                )

                if layer_idx == injection_layer and dd["injection_delta"] is not None:
                    h[:, -1, :] = h[:, -1, :] + dd["injection_delta"].unsqueeze(0)

            h = norm(h)
            logits = lm_head(h[:, -1:, :])
            per_delta_logits.append(logits)
            per_delta_caches.append(cache)

    t0 = time.perf_counter()

    # Decode loop
    with torch.inference_mode():
        for step in range(max_new_tokens):
            if not any(active):
                break

            new_tokens = []
            for d in range(num_deltas):
                if not active[d]:
                    new_tokens.append(None)
                    continue
                nxt = _sample(per_delta_logits[d], sample_temp, top_k, top_p)
                if nxt == eos:
                    active[d] = False
                    new_tokens.append(None)
                else:
                    new_tokens.append(nxt)
                    gen_ids[d].append(nxt)

            if not any(active):
                break

            for d in range(num_deltas):
                if not active[d]:
                    continue
                tok = new_tokens[d]
                pos = len(delta_data[d]["window_tokens"]) + len(gen_ids[d])
                tok_emb = embed(torch.tensor([[tok]], device=device))
                h = tok_emb
                pos_id_tok = torch.tensor([[pos]], device=device, dtype=torch.long)
                cache = per_delta_caches[d]

                for layer_idx in range(num_layers):
                    lay = model.model.layers[layer_idx]
                    pe = rotary(h, pos_id_tok)
                    h = lay(
                        h,
                        attention_mask=None,
                        position_ids=pos_id_tok,
                        position_embeddings=pe,
                        use_cache=True,
                        past_key_values=cache,
                    )

                    if layer_idx == injection_layer and delta_data[d]["injection_delta"] is not None:
                        h[:, 0, :] = h[:, 0, :] + delta_data[d]["injection_delta"].unsqueeze(0)

                h = norm(h)
                per_delta_logits[d] = lm_head(h[:, -1:, :])

    elapsed = time.perf_counter() - t0

    results = []
    for d in range(num_deltas):
        dd = delta_data[d]
        delta = dd["delta"]
        new_tokens = len(gen_ids[d]) - P
        tps = new_tokens / elapsed if elapsed > 0 else 0.0
        completion = tokenizer.decode(gen_ids[d][P:]).lstrip()

        results.append({
            "delta_id": delta.delta_id,
            "trust": delta.trust,
            "generated_text": completion,
            "source_url": delta.source_url,
            "tags": delta.tags,
            "mode": "corrected",
            "tokens_per_second": tps,
            "num_tokens": new_tokens,
            "elapsed": elapsed,
            "injection_layer": injection_layer,
            "matched_window": dd["matched_window"],
            "entries_injected": dd["num_entries"],
        })

    return results


def _route_and_retrieve(
    delta: ContextDelta,
    query_ids: list[int],
    topk: int,
) -> list[tuple[int, float, int]]:
    """Route query tokens to windows, retrieve top-K injection entries."""
    query_set = set(query_ids)

    window_scores = []
    for w in range(delta.num_windows):
        overlap = len(set(delta.get_window_tokens(w)) & query_set)
        if overlap > 0:
            window_scores.append((w, overlap))
    window_scores.sort(key=lambda x: -x[1])
    if not window_scores:
        return []

    entries = []
    for w, _score in window_scores[:3]:
        for i in range(delta.injection_token_ids.shape[1]):
            tid = int(delta.injection_token_ids[w, i])
            coeff = float(delta.injection_coefficients[w, i])
            if tid == 0 or tid in query_set:
                continue
            entries.append((tid, coeff, w))

    entries.sort(key=lambda x: -x[1])
    return entries[:topk]


def _detect_injection_layer(num_layers: int) -> int:
    """Auto-detect the injection layer.

    LARQL's research shows the optimal injection point is near the end of
    the retrieval/resolve zone, where MLP features become context-selective
    rather than indiscriminate "population code."  Empirically:
    - Gemma 3 4B (34L): L30  (LARQL hardcoded default)
    - Qwen3 4B  (36L): L30  (validated)

    The rule: num_layers - 6, clamped to [20, num_layers-4].
    """
    return max(20, min(num_layers - 4, num_layers - 6))


def _sample(logits, temperature, top_k, top_p):
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
