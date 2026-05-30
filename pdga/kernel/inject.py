"""Residual stream injection generation — token-driven LARQL-style injection.

Each delta stores token_ids and coefficients derived from GLiNER entity
positions within windows. During generation, a forward pre_hook at the
injection layer (num_layers - 4) adds the combined token embedding injection
to hidden states during prefill.

This is the proven LARQL approach: important tokens' embeddings injected
at a late layer, perturbing the model toward recalling stored context.
"""

from __future__ import annotations

import torch
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer

from pdga.delta.context import ContextDelta


def generate_from_injection(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    deltas: list[ContextDelta],
    max_new_tokens: int = 256,
    sample_temp: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    eos_token_id: int | None = None,
    injection_coefficient: float = 1.0,
) -> list[dict]:
    """Generate from token injection entries via forward pre_hook.

    Each delta is sovereign. Token IDs from GLiNER entities (or boundary-
    based extraction) are injected at the injection layer during prefill.
    Standard model.generate() handles autoregressive continuation.
    """
    device = model.device
    num_layers = model.config.num_hidden_layers
    embed = model.get_input_embeddings()

    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    results = []

    for delta in deltas:
        injection_vec = _get_injection_entries(delta, embed, device)
        if injection_vec is None:
            results.append({
                "delta_id": delta.delta_id,
                "trust": delta.trust,
                "generated_text": "[no injection entries available]",
                "source_url": delta.source_url,
                "tags": delta.tags,
                "mode": "inject",
            })
            continue

        injection_layer = delta.manifest.injection_layer
        if injection_layer < 0 or injection_layer >= num_layers:
            results.append({
                "delta_id": delta.delta_id,
                "trust": delta.trust,
                "generated_text": "[invalid injection layer]",
                "source_url": delta.source_url,
                "tags": delta.tags,
                "mode": "inject",
            })
            continue

        injection_vec = injection_vec * injection_coefficient

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        injected = {"done": False}

        def inject_hook(module, input):
            if not injected["done"]:
                hidden = input[0]
                hidden = hidden + injection_vec.to(
                    device=hidden.device, dtype=hidden.dtype
                ).unsqueeze(0).unsqueeze(0)
                injected["done"] = True
                return (hidden,) + input[1:]
            return input

        target_layer = model.model.layers[injection_layer]
        handle = target_layer.register_forward_pre_hook(inject_hook)

        try:
            with torch.inference_mode():
                output = model.generate(
                    input_ids=input_tensor,
                    max_new_tokens=max_new_tokens,
                    temperature=sample_temp if sample_temp > 0 else 1.0,
                    do_sample=sample_temp > 0,
                    top_k=top_k if sample_temp > 0 else None,
                    top_p=top_p if sample_temp > 0 else None,
                    eos_token_id=eos_token_id,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
        finally:
            handle.remove()

        completion = tokenizer.decode(output[0][len(prompt_ids):]).lstrip()

        results.append({
            "delta_id": delta.delta_id,
            "trust": delta.trust,
            "generated_text": completion,
            "source_url": delta.source_url,
            "tags": delta.tags,
            "mode": "inject",
        })

    return results


def _get_injection_entries(delta, embed, device):
    """Build injection vector from delta's stored entries.

    Priority: injection_token_ids/coefficients > fact_tokens > None
    """
    tid_arr = delta.injection_token_ids
    coeff_arr = delta.injection_coefficients

    if tid_arr is not None and coeff_arr is not None and tid_arr.shape[0] > 0:
        flat_ids_list = [int(t) for t in tid_arr.reshape(-1) if int(t) != 0]
        if not flat_ids_list:
            return None
        flat_ids = torch.tensor(flat_ids_list, dtype=torch.long, device=device)
        flat_coeffs = torch.from_numpy(
            coeff_arr.reshape(-1)[:len(flat_ids)].astype(np.float32)
        ).to(device=device, dtype=embed.weight.dtype)
        embs = embed(flat_ids)
        return (embs * flat_coeffs.unsqueeze(-1)).sum(dim=0)

    if delta.fact_tokens:
        flat = []
        for chunk in delta.fact_tokens:
            flat.extend(chunk)
        if flat:
            ids = torch.tensor(flat, dtype=torch.long, device=device)
            embs = embed(ids)
            return embs.mean(dim=0)

    return None
