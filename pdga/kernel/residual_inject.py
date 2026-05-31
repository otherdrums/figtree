"""Direct residual injection via custom forward pass (no KV cache).

Uses pdga.kernel.forward for LARQL-compatible forward computation:
1. Boundary residual at position 0 (replaces the dummy at crystal layer).
2. Token embeddings at positions 1..P.
3. Causal attention through all layers 0..N-1.
4. Injection delta applied at last position at crystal layer.
5. Full forward re-run at each decode step — no KV cache.
"""

from __future__ import annotations

import time
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from pdga.delta.context import ContextDelta
from pdga.kernel.forward import custom_forward
from pdga.kernel.prompt import build_prompt_ids


def generate_from_residuals(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    deltas: list[ContextDelta],
    max_new_tokens: int = 80,
    sample_temp: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    eos_token_id: int | None = None,
    injection_coefficient: float = 0.75,
    use_chat_template: bool = True,
) -> list[dict]:
    """Generate via custom forward with boundary swap (no KV cache).

    Each delta is sovereign. The boundary residual carries the full context;
    no per-token KV cache is stored.

    Args:
        injection_coefficient: Global scale for Σ(coeff × embed(token_id)).
            0.5–0.75 recommended for full-forward re-run.  Higher values
            (1.0+) cause output degeneration.
        use_chat_template: If True, formats the prompt via Qwen3 ChatML
            with enable_thinking=False (suppresses chain-of-thought mode).
    """
    device = model.device
    embed = model.get_input_embeddings()
    dtype = model.dtype

    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    results = []

    for delta in deltas:
        crystal = delta.manifest.crystal_layer
        boundaries = delta.boundaries
        tid_arr = delta.injection_token_ids
        coeff_arr = delta.injection_coefficients

        if boundaries is None or boundaries.shape[0] == 0:
            results.append({
                "delta_id": delta.delta_id, "trust": delta.trust,
                "generated_text": "[no boundaries]",
                "source_url": delta.source_url, "tags": delta.tags,
                "mode": "residuals", "tokens_per_second": 0.0,
            })
            continue

        # Build prompt tokens (with chat template if requested)
        if use_chat_template:
            prompt_ids = build_prompt_ids(tokenizer, prompt, enable_thinking=False)
        else:
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)

        query_set = set(prompt_ids)
        generated_ids = list(prompt_ids)
        P_init = len(prompt_ids)

        boundary_mean = boundaries.mean(axis=0)
        boundary_t = torch.from_numpy(
            boundary_mean.astype("float32")
        ).to(device=device, dtype=dtype)

        injection_delta = _build_injection_vec(
            tid_arr, coeff_arr, query_set, embed, device, dtype,
            injection_coefficient,
        )

        t0 = time.perf_counter()

        for _step in range(max_new_tokens):
            with torch.inference_mode():
                tok_emb = embed(
                    torch.tensor([generated_ids], dtype=torch.long, device=device)
                )

                h = custom_forward(
                    model=model,
                    boundary_residual=boundary_t,
                    token_embeddings=tok_emb,
                    crystal_layer=crystal,
                    injection_delta=injection_delta,
                )

                logits = model.lm_head(h[:, -1:, :])

            next_token = _sample_token(logits, sample_temp, top_k, top_p)
            if next_token == eos_token_id:
                break
            generated_ids.append(next_token)

        elapsed = time.perf_counter() - t0
        new_tokens = len(generated_ids) - P_init
        tps = new_tokens / elapsed if elapsed > 0 else 0.0

        completion = tokenizer.decode(generated_ids[P_init:]).lstrip()

        results.append({
            "delta_id": delta.delta_id, "trust": delta.trust,
            "generated_text": completion,
            "source_url": delta.source_url, "tags": delta.tags,
            "mode": "residuals", "tokens_per_second": tps,
            "num_tokens": new_tokens, "elapsed": elapsed,
            "crystal_layer": crystal,
        })

    return results


def _build_injection_vec(tid_arr, coeff_arr, query_set, embed,
                          device, dtype, inject_coeff):
    """Build Σ(coeff × embed(token_id)) × global_coeff.

    Filters out: token_id=0, tokens that appear in the query.
    Returns (hidden_size,) tensor, or None.
    """
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
    embs = embed(ids_t)  # (N, hidden_size)
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
