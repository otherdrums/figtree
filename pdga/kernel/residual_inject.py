"""Direct residual injection via Apollo-style layer-skipping forward pass.

Uses pdga.kernel.apollo_engine for LARQL-compatible forward computation:
1. Boundary residual at position 0 (replaces layers 0..crystal-1)
2. Token embeddings at positions 1..P (raw embedding level)
3. Full bidirectional attention at layers crystal..N-1
4. Injection delta applied at last position at crystal layer
5. Full forward re-run at each decode step
"""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from pdga.delta.context import ContextDelta
from pdga.kernel.apollo_engine import apollo_forward


def generate_from_residuals(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    deltas: list[ContextDelta],
    max_new_tokens: int = 256,
    sample_temp: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    eos_token_id: int | None = None,
    injection_coefficient: float = 10.0,
) -> list[dict]:
    """Generate via Apollo-style forward with boundary-as-position-0.

    Each delta is sovereign. The boundary residual serves as virtual position-0,
    skipping layers 0..crystal-1. Token-embedding delta injected at crystal
    layer. Full forward re-run at each decode step.
    """
    device = model.device
    embed = model.get_input_embeddings()

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
                "mode": "residuals",
            })
            continue

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        query_set = set(prompt_ids)
        generated_ids = list(prompt_ids)
        P_init = len(prompt_ids)

        boundary_mean = boundaries.mean(axis=0)
        boundary_t = torch.from_numpy(boundary_mean.astype("float32")).to(device=device, dtype=model.dtype)

        injection_delta = _build_apollo_delta(
            tid_arr, coeff_arr, query_set, embed, device, model.dtype,
            injection_coefficient,
        )

        for _step in range(max_new_tokens):
            with torch.inference_mode():
                tok_emb = embed(torch.tensor([generated_ids], dtype=torch.long, device=device))

                h = apollo_forward(
                    model=model,
                    boundary_residual=boundary_t,
                    token_embeddings=tok_emb,
                    crystal_layer=crystal,
                    injection_delta=injection_delta,
                )

                logits = model.lm_head(h[:, -1:, :])

            if sample_temp > 0:
                probs = torch.softmax(logits.squeeze(0).squeeze(0) / sample_temp, dim=-1)
                if top_k > 0:
                    vals, idx = torch.topk(probs, min(top_k, probs.size(-1)))
                    mask = torch.zeros_like(probs, dtype=torch.bool)
                    mask[idx] = True
                    probs[~mask] = 0.0
                if top_p < 1.0:
                    sp, si = torch.sort(probs, descending=True)
                    cs = torch.cumsum(sp, dim=-1)
                    ci = (cs > top_p).nonzero(as_tuple=True)
                    if len(ci[0]) > 0:
                        sp[ci[0][0] + 1:] = 0.0
                        probs = torch.zeros_like(probs).scatter_(0, si, sp)
                probs = probs / probs.sum()
                next_token = torch.multinomial(probs, 1).item()
            else:
                next_token = logits.squeeze().argmax(dim=-1).item()

            if next_token == eos_token_id:
                break
            generated_ids.append(next_token)

        completion = tokenizer.decode(generated_ids[P_init:]).lstrip()

        results.append({
            "delta_id": delta.delta_id, "trust": delta.trust,
            "generated_text": completion,
            "source_url": delta.source_url, "tags": delta.tags,
            "mode": "residuals",
        })

    return results


def _build_apollo_delta(tid_arr, coeff_arr, query_set, embed, device, dtype, inject_coeff):
    """Build Apollo-style injection delta: Sigma(coeff * embed(token_id))."""
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
    coeffs_t = torch.tensor(filtered_coeffs, dtype=torch.float32).to(device=device, dtype=dtype)
    embs = embed(ids_t)
    return (embs * coeffs_t.unsqueeze(-1) * inject_coeff).sum(dim=0)
