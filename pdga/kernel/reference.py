"""Reference generation kernel — per-delta independent generation.

Three modes:
1. generate — token replay (reference path)
2. generate_from_residuals — boundary residuals only (real PDGA)
3. generate_hybrid — fact tokens + boundary residuals (best of both)
"""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.cache_utils import DynamicCache

from pdga.delta.context import ContextDelta
from pdga.kernel.prompt import build_prompt_ids


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    deltas: list[ContextDelta],
    max_new_tokens: int = 256,
    sample_temp: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    eos_token_id: int | None = None,
) -> list[dict]:
    """Token-replay generation — used as reference for comparison."""
    device = model.device

    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    results = []

    for delta in deltas:
        context_ids = []
        for window_idx in range(delta.num_windows):
            context_ids.extend(delta.get_window_tokens(window_idx))

        prompt_ids = build_prompt_ids(tokenizer, prompt, enable_thinking=False)
        full_ids = context_ids + prompt_ids
        full_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)

        output = model.generate(
            input_ids=full_tensor, max_new_tokens=max_new_tokens,
            temperature=sample_temp if sample_temp > 0 else 1.0,
            do_sample=sample_temp > 0,
            top_k=top_k if sample_temp > 0 else None,
            top_p=top_p if sample_temp > 0 else None,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        full_output = tokenizer.decode(output[0])
        prompt_text = tokenizer.decode(full_ids)
        completion = full_output[len(prompt_text):].lstrip()

        results.append({
            "delta_id": delta.delta_id,
            "trust": delta.trust,
            "generated_text": completion,
            "source_url": delta.source_url,
            "tags": delta.tags,
            "mode": "replay",
        })

    return results


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
) -> list[dict]:
    """Generate from boundary residuals only — no token replay.

    Each delta's boundary residuals are forwarded through layers crystal+1..N
    to build KV caches. Those KVs are seeded into a DynamicCache before the
    prompt runs. The model attends to residual-derived "virtual tokens" —
    the pure PDGA representation.
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
        delta_kvs = _build_delta_kv(model, delta, crystal, num_layers, device, rotary_emb)

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        generated_ids = list(prompt_ids)

        with torch.inference_mode():
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            prompt_hidden = embed(prompt_tensor)
            prompt_len = prompt_hidden.shape[1]

            cache = DynamicCache()
            for layer_idx in range(num_layers):
                dkv = delta_kvs.get(layer_idx)
                if dkv is not None:
                    dk, dv = dkv
                    cache.update(dk.to(device=device), dv.to(device=device), layer_idx=layer_idx)

            past_pos = max(
                (dkv[0].shape[2] for dkv in delta_kvs.values() if dkv is not None),
                default=0,
            )

            for layer_idx in range(num_layers):
                layer = model.model.layers[layer_idx]
                pos_ids = torch.arange(past_pos, past_pos + prompt_len,
                                       dtype=torch.long, device=device).unsqueeze(0)
                pe = rotary_emb(prompt_hidden, pos_ids)
                layer_out = layer(
                    prompt_hidden, attention_mask=None,
                    position_ids=pos_ids, position_embeddings=pe,
                    use_cache=True, past_key_values=cache,
                )
                prompt_hidden = layer_out if not isinstance(layer_out, tuple) else layer_out[0]

            past_pos += prompt_len
            hidden_states = model.model.norm(prompt_hidden)
            logits = model.lm_head(hidden_states[:, -1:, :])

        for step in range(max_new_tokens):
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
                hidden_states = token_emb
                pos_ids = torch.tensor([[past_pos]], dtype=torch.long, device=device)

                for layer_idx in range(num_layers):
                    layer = model.model.layers[layer_idx]
                    pe = rotary_emb(hidden_states, pos_ids)
                    layer_out = layer(
                        hidden_states, attention_mask=None,
                        position_ids=pos_ids, position_embeddings=pe,
                        use_cache=True, past_key_values=cache,
                    )
                    hidden_states = layer_out if not isinstance(layer_out, tuple) else layer_out[0]

                past_pos += 1
                hidden_states = model.model.norm(hidden_states)
                logits = model.lm_head(hidden_states[:, -1:, :])

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


def _build_delta_kv(
    model: PreTrainedModel,
    delta: ContextDelta,
    crystal: int,
    num_layers: int,
    device: torch.device,
    rotary_emb,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Build KV caches from boundary residuals.

    Each residual vector represents the hidden state at the crystal layer
    for one novel token position. Forward each through layers crystal+1..N
    to produce K,V per layer. Concatenate all residual-derived KVs.
    """
    combined: dict[int, list[torch.Tensor]] = {}
    for layer_idx in range(crystal + 1, num_layers):
        combined[layer_idx] = []

    for vec_idx in range(delta.boundaries.shape[0]):
        residual = delta.boundaries[vec_idx]
        residual_t = torch.from_numpy(residual.astype("float32")).to(
            device=device, dtype=model.dtype
        ).unsqueeze(0).unsqueeze(0)

        hidden_states = residual_t
        pos_ids = torch.tensor([[0]], dtype=torch.long, device=device)
        w_cache = DynamicCache()

        for layer_idx in range(crystal + 1, num_layers):
            layer = model.model.layers[layer_idx]
            pe = rotary_emb(hidden_states, pos_ids)
            layer_out = layer(
                hidden_states, attention_mask=None,
                position_ids=pos_ids, position_embeddings=pe,
                use_cache=True, past_key_values=w_cache,
            )
            hidden_states = layer_out if not isinstance(layer_out, tuple) else layer_out[0]

            if layer_idx in w_cache.layers:
                combined[layer_idx].append(w_cache.layers[layer_idx].keys.cpu())

    result: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_idx, k_list in combined.items():
        if k_list:
            all_k = torch.cat(k_list, dim=2)
            all_v = torch.cat(k_list, dim=2)
            result[layer_idx] = (all_k, all_v)

    return result


def generate_hybrid(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    deltas: list[ContextDelta],
    max_new_tokens: int = 256,
    sample_temp: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    eos_token_id: int | None = None,
) -> list[dict]:
    """Generate using fact token chunks only.

    Fact token chunks contain only the unknowable entities/numbers — the
    minimal information the model can't derive from pretraining. This is
    the compressed "what the model can't know" replay.
    """
    device = model.device

    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    results = []

    for delta in deltas:
        fact_ids = []
        if delta.fact_tokens:
            for chunk in delta.fact_tokens:
                fact_ids.extend(chunk)

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        full_ids = fact_ids + prompt_ids
        full_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)

        output = model.generate(
            input_ids=full_tensor, max_new_tokens=max_new_tokens,
            temperature=sample_temp if sample_temp > 0 else 1.0,
            do_sample=sample_temp > 0,
            top_k=top_k if sample_temp > 0 else None,
            top_p=top_p if sample_temp > 0 else None,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        full_output = tokenizer.decode(output[0])
        prompt_text = tokenizer.decode(full_ids)
        completion = full_output[len(prompt_text):].lstrip()

        results.append({
            "delta_id": delta.delta_id,
            "trust": delta.trust,
            "generated_text": completion,
            "source_url": delta.source_url,
            "tags": delta.tags,
            "mode": "hybrid",
        })

    return results
