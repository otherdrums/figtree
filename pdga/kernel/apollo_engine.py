"""Apollo-style forward pass — LARQL-compatible residual stream engine.

Corrected algorithm (v2):
1. Dummy token at position 0, prompt embeddings at positions 1+.
2. Forward ALL tokens through ALL layers 0..N-1 with **causal** attention.
3. At crystal layer, BEFORE forward: replace position-0 residual with boundary.
4. At crystal layer, AFTER forward: add injection delta to last position.
5. Full forward re-run at each decode step (no KV cache — the boundary
   residual IS the context carrier; per-token caching violates the architecture).

The full forward re-run at each step has O(N × seq_len²) complexity.  On a
Quadro T1000 with Qwen3-4B (bnb-4bit, 36 layers, h=2560) this is ~450 ms/step
for short sequences.  The C++/CUDA Apollo engine (pdga/apollo/) provides a
path to production-speed inference while preserving the no-cache design.
"""

from __future__ import annotations

import torch
from transformers import PreTrainedModel


def apollo_forward(
    model: PreTrainedModel,
    boundary_residual: torch.Tensor,
    token_embeddings: torch.Tensor,
    crystal_layer: int,
    injection_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apollo forward: boundary swap at crystal layer, full-layer causal forward.

    Args:
        model: HuggingFace CausalLM (Qwen3 family).
        boundary_residual: (hidden_size,) crystal-layer output encoding the
            entire past context.  Replaces position 0 at `crystal_layer`.
        token_embeddings: (1, P, hidden_size) prompt token embeddings.
        crystal_layer: 0-indexed layer where residuals stabilise (e.g. 23
            for Qwen3-4B).
        injection_delta: (hidden_size,) Σ(coeff · embed(token_id)) added to
            the LAST position at `crystal_layer`.  May be ``None``.
            Recommended coefficient: 0.5–0.75 for full forward re-run
            (higher values cause output degeneration).

    Returns:
        Hidden states (1, 1+P, hidden_size) after all layers + final norm.
    """
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    device = token_embeddings.device
    dtype = token_embeddings.dtype
    P = token_embeddings.shape[1]
    L = 1 + P  # 1 boundary position + P prompt positions

    # Position 0: dummy (zero) — minimal perturbation during layers 0..crystal-1,
    # replaced with real boundary at crystal layer.
    dummy = torch.zeros(1, 1, hidden_size, device=device, dtype=dtype)
    h = torch.cat([dummy, token_embeddings], dim=1)  # (1, L, hidden_size)

    rotary_emb = model.model.rotary_emb
    position_ids = torch.arange(0, L, device=device, dtype=torch.long).unsqueeze(0)

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]

        # --- Boundary swap: replace position-0 with real context ---
        if layer_idx == crystal_layer:
            h[:, 0:1, :] = boundary_residual.unsqueeze(0)

        # Compute RoPE cos/sin fresh (h may have changed via boundary swap).
        position_embeddings = rotary_emb(h, position_ids)

        # causal attention is automatic when attention_mask=None (SDPA is_causal=True)
        h = layer(
            h,
            attention_mask=None,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )

        # --- Token-embedding injection at crystal layer ---
        if layer_idx == crystal_layer and injection_delta is not None:
            h[:, -1, :] = h[:, -1, :] + injection_delta.unsqueeze(0)

    h = model.model.norm(h)
    return h
