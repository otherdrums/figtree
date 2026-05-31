"""Corrected residual injection engine — full forward, no boundary skip.

Based on LARQL Apollo's actual implementation keys:
1. Injection at L30 (injection_layer), NOT L23 (crystal).
2. Top-K=8 entries, routed by token-overlap with query.
3. Context = matched window tokens + query tokens.
4. inject_coefficient = 10.0.

The boundary residual is NOT used for position-0 substitution on Qwen3 —
HuggingFace attention cannot mix L23-level boundary with L0-level token
embeddings at the crystal layer.  Instead we use the full (uncompressed)
forward pass over all context tokens, with injection at L30."""

from __future__ import annotations

import time
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from pdga.delta.context import ContextDelta
from pdga.kernel.prompt import build_prompt_ids


def generate_corrected(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    deltas: list[ContextDelta],
    max_new_tokens: int = 60,
    sample_temp: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    injection_layer: int = 30,
    injection_coefficient: float = 10.0,
    injection_topk: int = 8,
    eos_token_id: int | None = None,
) -> list[dict]:
    """Generate with LARQL-corrected injection semantics.

    Full forward pass (uncompressed) over [window_tokens + generated_tokens]
    with injection delta at `injection_layer`.
    """
    device = model.device
    dtype = model.dtype
    embed = model.get_input_embeddings()
    lm_head = model.lm_head
    rotary = model.model.rotary_emb
    norm = model.model.norm
    num_layers = model.config.num_hidden_layers
    eos = eos_token_id or tokenizer.eos_token_id

    prompt_ids = build_prompt_ids(tokenizer, prompt, enable_thinking=False)

    results = []

    for delta in deltas:
        boundaries = delta.boundaries
        if boundaries is None or boundaries.shape[0] == 0:
            results.append({
                "delta_id": delta.delta_id, "trust": delta.trust,
                "generated_text": "[no boundaries]", "mode": "corrected",
            })
            continue

        # Route entries by token overlap with query
        entries = _route_and_retrieve(
            delta, prompt_ids, injection_topk, device, dtype,
        )
        if not entries:
            results.append({
                "delta_id": delta.delta_id, "trust": delta.trust,
                "generated_text": "[no matching windows]", "mode": "corrected",
            })
            continue

        # Injection delta from top entries
        inj_ids = torch.tensor([e[0] for e in entries], device=device)
        inj_coeffs_t = torch.tensor(
            [e[1] for e in entries], device=device, dtype=torch.float32
        ).to(dtype=dtype)
        inj_embs = embed(inj_ids)
        injection_delta = (
            inj_embs * inj_coeffs_t.unsqueeze(-1) * injection_coefficient
        ).sum(dim=0)

        # Build context: matched window tokens + prompt tokens
        matched_window = entries[0][2]
        window_tokens = list(delta.get_window_tokens(matched_window))

        gen_ids = list(prompt_ids)
        P = len(prompt_ids)

        t0 = time.perf_counter()

        with torch.inference_mode():
            for step in range(max_new_tokens):
                all_ids = window_tokens + gen_ids
                tok_emb = embed(
                    torch.tensor([all_ids], dtype=torch.long, device=device)
                )
                h = tok_emb
                L = h.shape[1]
                pos_ids = torch.arange(0, L, device=device).unsqueeze(0)

                # Full forward through all layers
                for layer_idx in range(num_layers):
                    lay = model.model.layers[layer_idx]
                    pe = rotary(h, pos_ids)
                    h = lay(
                        h,
                        attention_mask=None,
                        position_ids=pos_ids,
                        position_embeddings=pe,
                    )

                    # Inject at injection_layer (L30)
                    if layer_idx == injection_layer:
                        last = L - 1
                        h[:, last:last+1, :] = (
                            h[:, last:last+1, :]
                            + injection_delta.unsqueeze(0)
                        )

                h = norm(h)
                logits = lm_head(h[:, -1:, :])

                next_tok = _sample(logits, sample_temp, top_k, top_p)
                if next_tok == eos:
                    break
                gen_ids.append(next_tok)

        elapsed = time.perf_counter() - t0
        new_tokens = len(gen_ids) - P
        tps = new_tokens / elapsed if elapsed > 0 else 0.0
        completion = tokenizer.decode(gen_ids[P:]).lstrip()

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
            "matched_window": matched_window,
            "entries_injected": len(entries),
        })

    return results


def _route_and_retrieve(
    delta: ContextDelta,
    query_ids: list[int],
    topk: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[tuple[int, float, int]]:
    """Route query tokens to windows, retrieve top-K injection entries."""
    query_set = set(query_ids)

    window_scores = []
    for w in range(delta.num_windows):
        w_tokens = delta.get_window_tokens(w)
        overlap = len(set(w_tokens) & query_set)
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
