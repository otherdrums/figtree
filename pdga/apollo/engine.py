"""Apollo generation engine with KV caching and CUDA injection.

Key mechanics:
1. Pre-fill: [dummy, prompt] through all layers with causal attention.
   Boundary swap at crystal layer, injection at last position.
   KV cache stored for all positions.
2. Decode: new token only, attends to all cached K/V via DynamicCache.
   No boundary swap or injection during decode.
3. CUDA kernels accelerate boundary swap and fused injection.
"""

from __future__ import annotations

import time
import torch
from torch import inference_mode
from transformers import PreTrainedModel, PreTrainedTokenizer, DynamicCache

from pdga.delta.context import ContextDelta
from pdga.apollo.kernels import get_kernels


class ApolloModel:
    """Apollo-style generator with KV-cached generation."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._embed = model.get_input_embeddings()
        self._lm_head = model.lm_head
        self._rotary = model.model.rotary_emb
        self._norm = model.model.norm
        self._layers = model.model.layers
        self.num_layers = model.config.num_hidden_layers
        self.hidden_size = model.config.hidden_size
        self.device = next(model.parameters()).device
        self.dtype = model.dtype

        try:
            get_kernels()
            self._has_kernels = True
        except Exception:
            self._has_kernels = False

    def generate(
        self,
        prompt: str,
        delta: ContextDelta,
        max_new_tokens: int = 60,
        temperature: float = 0.0,
        top_k: int = 50,
        top_p: float = 0.95,
        injection_coefficient: float = 0.75,
        eos_token_id: int | None = None,
    ) -> dict:
        """Generate text using Apollo residual injection.

        Args:
            prompt: Query text.
            delta: Ingested ContextDelta with boundaries + injection tokens.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_k: Top-k sampling parameter.
            top_p: Top-p (nucleus) sampling parameter.
            injection_coefficient: Global scale for token injection delta.
            eos_token_id: End-of-sequence token ID.

        Returns:
            dict with keys: generated_text, delta_id, crystal_layer, tokens_per_second.
        """
        crystal = delta.manifest.crystal_layer
        boundaries = delta.boundaries
        eos_token_id = eos_token_id or self.tokenizer.eos_token_id

        if boundaries is None or boundaries.shape[0] == 0:
            return {"generated_text": "", "delta_id": delta.delta_id,
                    "crystal_layer": crystal, "tokens_per_second": 0.0}

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        query_set = set(prompt_ids)

        # Build boundary + injection delta
        boundary_t = torch.from_numpy(
            boundaries.mean(axis=0).astype("float32")
        ).to(device=self.device, dtype=self.dtype)

        injection_t = _build_injection_delta(
            delta.injection_token_ids, delta.injection_coefficients,
            query_set, self._embed, self.device, self.dtype,
            injection_coefficient,
        )

        t0 = time.perf_counter()

        # Pre-fill: [dummy, prompt] through all layers with boundary swap
        with inference_mode():
            logits, cache = self._prefill(
                boundary_t, crystal, prompt_ids, injection_t,
            )

        generated_ids = list(prompt_ids)
        P_init = len(prompt_ids)
        joint_cache = cache  # DynamicCache, already updated in-place

        for step in range(max_new_tokens):
            with inference_mode():
                next_token = _sample_token(logits, temperature, top_k, top_p)
            if next_token == eos_token_id:
                break
            generated_ids.append(next_token)

            with inference_mode():
                # Decode: new token only, uses joint_cache
                logits = self._decode(next_token, joint_cache)

        elapsed = time.perf_counter() - t0
        new_tokens = len(generated_ids) - P_init
        tps = new_tokens / elapsed if elapsed > 0 else 0.0

        completion = self.tokenizer.decode(generated_ids[P_init:]).lstrip()

        return {
            "generated_text": completion,
            "delta_id": delta.delta_id,
            "crystal_layer": crystal,
            "tokens_per_second": tps,
            "num_tokens": new_tokens,
        }

    def _prefill(
        self,
        boundary: torch.Tensor,
        crystal: int,
        prompt_ids: list[int],
        injection_delta: torch.Tensor | None,
    ) -> tuple[torch.Tensor, DynamicCache]:
        """Pre-fill KV cache: boundary-swapped forward over [dummy, prompt].

        Returns (logits_for_last_position, DynamicCache).
        """
        P = len(prompt_ids)
        L = 1 + P
        dummy = torch.zeros(1, 1, self.hidden_size, device=self.device, dtype=self.dtype)
        emb = self._embed(torch.tensor([prompt_ids], dtype=torch.long, device=self.device))
        h = torch.cat([dummy, emb], dim=1)

        position_ids = torch.arange(0, L, device=self.device, dtype=torch.long).unsqueeze(0)
        cache = DynamicCache()
        has_kernels = self._has_kernels

        for layer_idx in range(self.num_layers):
            layer = self._layers[layer_idx]

            # Boundary swap at crystal layer
            if layer_idx == crystal:
                if has_kernels:
                    # Fused CUDA boundary swap (in-place)
                    get_kernels().apollo_boundary_swap(
                        h, boundary,
                    )
                else:
                    h[:, 0:1, :] = boundary.unsqueeze(0).unsqueeze(0)

            position_embeddings = self._rotary(h, position_ids)

            # Forward layer with KV caching
            out = layer(
                h,
                attention_mask=None,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=True,
                past_key_values=cache,
            )
            h = out[0] if isinstance(out, tuple) else out

            # Fused CUDA injection at crystal layer
            if layer_idx == crystal and injection_delta is not None:
                if has_kernels:
                    n_emb = injection_delta.size(0)
                    ones = torch.ones(n_emb, device=self.device, dtype=torch.float32)
                    get_kernels().apollo_fused_inject(
                        h, injection_delta.contiguous(), ones,
                        L - 1, 1.0,
                    )
                else:
                    combined = injection_delta.sum(dim=0)  # (hidden_size,)
                    h[:, -1, :] = h[:, -1, :] + combined.unsqueeze(0)

        h = self._norm(h)
        logits = self._lm_head(h[:, -1:, :])
        return logits, cache

    def _decode(
        self,
        token_id: int,
        cache: DynamicCache,
    ) -> torch.Tensor:
        """Decode one token using cached KV.

        Returns logits for the newly generated position.
        """
        position = cache.get_seq_length()
        emb = self._embed(torch.tensor([[token_id]], dtype=torch.long, device=self.device))
        h = emb

        position_ids = torch.tensor([[position]], device=self.device, dtype=torch.long)

        for layer_idx in range(self.num_layers):
            layer = self._layers[layer_idx]
            position_embeddings = self._rotary(h, position_ids)
            out = layer(
                h,
                attention_mask=None,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=True,
                past_key_values=cache,
            )
            h = out[0] if isinstance(out, tuple) else out

        h = self._norm(h)
        logits = self._lm_head(h[:, -1:, :])
        return logits


def _build_injection_delta(
    token_ids, coefficients, query_set, embed, device, dtype, global_coeff,
):
    """Build Apollo injection delta: Σ(coeff × embed(token_id)) × global_coeff.

    Filters out: token_id=0, tokens that appear in the query.
    Returns (num_filtered, hidden_size) tensor, or None.
    """
    if token_ids is None or coefficients is None or token_ids.size == 0:
        return None
    flat_ids = token_ids.reshape(-1)
    flat_coeffs = coefficients.reshape(-1)
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
    embs = embed(ids_t)  # (N, hidden_size)
    return embs * coeffs_t.unsqueeze(-1) * global_coeff


def _sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    """Sample next token from logits."""
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


def apollo_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    delta: ContextDelta,
    **kwargs,
) -> dict:
    """Convenience wrapper for single-shot Apollo generation.

    Args match ApolloModel.generate().
    """
    engine = ApolloModel(model, tokenizer)
    return engine.generate(prompt, delta, **kwargs)
