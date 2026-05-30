"""Direct residual injection generation — LARQL Apollo-style crystal-layer injection.

During generation, boundary residuals (captured at the crystal layer output
during ingestion) are injected directly into the residual stream at the
crystal layer output via a forward hook. The injected perturbations are
then processed by all remaining layers.

This is the "real PDGA" path: no token replay, no KV cache seeding.
The boundary residual IS the perturbation.
"""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.cache_utils import DynamicCache

from pdga.delta.context import ContextDelta
from pdga.kernel.cuda_inject import inject_residuals, inject_mean


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
    injection_coefficient: float = 1.0,
    injection_mode: str = "per_position",
) -> list[dict]:
    """Generate via direct residual injection at the crystal layer.

    Each delta is sovereign. Boundary residuals are injected into the
    residual stream at the crystal layer output using a forward hook.
    The CUDA/Triton kernel performs fused per-position injection.

    Args:
        injection_mode: "per_position" — each window's delta at a distinct
            position (best for mult-window deltas). "mean" — average delta
            added to all positions.
    """
    device = model.device
    num_layers = model.config.num_hidden_layers
    embed = model.get_input_embeddings()
    rotary_emb = model.model.rotary_emb

    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    results = []

    for delta in deltas:
        crystal = delta.manifest.crystal_layer
        boundaries = delta.boundaries

        if boundaries is None or boundaries.shape[0] == 0:
            results.append({
                "delta_id": delta.delta_id,
                "trust": delta.trust,
                "generated_text": "[no boundaries available]",
                "source_url": delta.source_url,
                "tags": delta.tags,
                "mode": "residuals",
            })
            continue

        W = boundaries.shape[0]

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        generated_ids = list(prompt_ids)

        deltas_tensor = torch.from_numpy(boundaries.astype("float32")).to(
            device=device, dtype=model.dtype
        )
        deltas_captured = deltas_tensor.clone()

        injected_flag = {"done": False}

        def inject_hook(module, input, output):
            if injected_flag["done"]:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            P = hidden.shape[1]

            if injection_mode == "per_position" and W > 0 and W <= P:
                positions = torch.arange(P - W, P, dtype=torch.long, device=device)
                inject_residuals(
                    hidden, deltas_captured, positions, injection_coefficient,
                )
            else:
                inject_mean(hidden, deltas_captured, injection_coefficient)

            injected_flag["done"] = True
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

        target_layer = model.model.layers[crystal]
        handle = target_layer.register_forward_hook(inject_hook)

        try:
            with torch.inference_mode():
                cache = DynamicCache()
                prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                hidden_states = embed(prompt_tensor)
                P = hidden_states.shape[1]

                for layer_idx in range(num_layers):
                    layer = model.model.layers[layer_idx]
                    pos_ids = torch.arange(0, P, dtype=torch.long, device=device).unsqueeze(0)
                    pe = rotary_emb(hidden_states, pos_ids)
                    layer_out = layer(
                        hidden_states, attention_mask=None,
                        position_ids=pos_ids, position_embeddings=pe,
                        use_cache=True, past_key_values=cache,
                    )
                    hidden_states = layer_out if not isinstance(layer_out, tuple) else layer_out[0]

                past_pos = P
                hidden_states = model.model.norm(hidden_states)
                logits = model.lm_head(hidden_states[:, -1:, :])
        finally:
            handle.remove()

        for _step in range(max_new_tokens):
            with torch.inference_mode():
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

            with torch.inference_mode():
                token_emb = embed(torch.tensor([[next_token]], dtype=torch.long, device=device))
                h = token_emb
                pos_ids = torch.tensor([[past_pos]], dtype=torch.long, device=device)

                for layer_idx in range(num_layers):
                    layer = model.model.layers[layer_idx]
                    pe = rotary_emb(h, pos_ids)
                    layer_out = layer(
                        h, attention_mask=None,
                        position_ids=pos_ids, position_embeddings=pe,
                        use_cache=True, past_key_values=cache,
                    )
                    h = layer_out if not isinstance(layer_out, tuple) else layer_out[0]

                past_pos += 1
                h = model.model.norm(h)
                logits = model.lm_head(h[:, -1:, :])

        completion = tokenizer.decode(generated_ids[len(prompt_ids):]).lstrip()

        results.append({
            "delta_id": delta.delta_id,
            "trust": delta.trust,
            "generated_text": completion,
            "source_url": delta.source_url,
            "tags": delta.tags,
            "mode": "residuals",
        })

    return results
