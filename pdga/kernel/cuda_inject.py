"""CUDA / Triton kernel for direct residual stream injection.

Provides fused GPU kernels that add boundary residual deltas into
hidden state tensors at specified positions. The injection perturbs
the model's residual stream directly — no KV cache seeding needed.

Two backends:
  - Triton (preferred): Pythonic GPU kernel, JIT-compiled
  - PyTorch (fallback): native tensor ops, already runs on CUDA
"""

from __future__ import annotations

import torch

TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    pass


# ── Triton kernel ───────────────────────────────────────────────────────────

if TRITON_AVAILABLE:
    @triton.jit
    def _inject_residuals_kernel(
        hidden_ptr,
        delta_ptr,
        pos_ptr,
        coeff_ptr,
        num_deltas,
        hidden_size,
        stride_seq,
        BLOCK_HIDDEN: tl.constexpr,
    ):
        """Triton kernel: inject deltas into hidden states at given positions.

        Each program block handles one delta at one hidden dimension block.
        """
        pid = tl.program_id(0)
        delta_idx = pid // ((hidden_size + BLOCK_HIDDEN - 1) // BLOCK_HIDDEN)
        block_idx = pid % ((hidden_size + BLOCK_HIDDEN - 1) // BLOCK_HIDDEN)

        if delta_idx >= num_deltas:
            return

        h_offs = block_idx * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
        h_mask = h_offs < hidden_size

        pos = tl.load(pos_ptr + delta_idx)
        coeff = tl.load(coeff_ptr + delta_idx)

        d_offs = delta_idx * hidden_size + h_offs
        delta_vals = tl.load(delta_ptr + d_offs, mask=h_mask)

        store_offs = pos * stride_seq + h_offs
        existing = tl.load(hidden_ptr + store_offs, mask=h_mask)
        tl.store(hidden_ptr + store_offs, existing + coeff * delta_vals, mask=h_mask)


def inject_residuals_triton(
    hidden_states: torch.Tensor,
    deltas: torch.Tensor,
    positions: torch.Tensor | None = None,
    coefficient: float = 1.0,
) -> torch.Tensor:
    """Inject deltas into hidden states using Triton kernel.

    Args:
        hidden_states: (batch, seq_len, hidden_size) on GPU
        deltas: (num_deltas, hidden_size) on GPU
        positions: (num_deltas,) int64 tensor — where to inject.
                   If None, injects at last position.
        coefficient: global scale factor

    Returns:
        Modified hidden_states (in-place operation)
    """
    if not TRITON_AVAILABLE:
        return inject_residuals_pytorch(hidden_states, deltas, positions, coefficient)

    B, S, H = hidden_states.shape
    W = deltas.shape[0]
    assert deltas.shape[1] == H, f"deltas hidden dim {deltas.shape[1]} != {H}"

    if positions is None:
        positions = torch.full((W,), S - 1, dtype=torch.long, device=hidden_states.device)

    coeffs = torch.full((W,), coefficient, dtype=hidden_states.dtype, device=hidden_states.device)

    BLOCK_HIDDEN = min(256, triton.next_power_of_2(H))
    num_blocks = W * ((H + BLOCK_HIDDEN - 1) // BLOCK_HIDDEN)

    if num_blocks > 0:
        _inject_residuals_kernel[(num_blocks,)](
            hidden_states,
            deltas,
            positions,
            coeffs,
            W,
            H,
            S,
            BLOCK_HIDDEN=BLOCK_HIDDEN,
        )

    return hidden_states


# ── PyTorch fallback ────────────────────────────────────────────────────────

def inject_residuals_pytorch(
    hidden_states: torch.Tensor,
    deltas: torch.Tensor,
    positions: torch.Tensor | None = None,
    coefficient: float = 1.0,
) -> torch.Tensor:
    """Inject deltas into hidden states using native PyTorch ops.

    Still runs on CUDA — PyTorch's backend dispatches to cuBLAS kernels.
    This is a drop-in replacement for the Triton kernel.
    """
    B, S, H = hidden_states.shape
    W = deltas.shape[0]
    assert deltas.shape[1] == H

    if positions is None:
        positions = torch.arange(S - W, S, dtype=torch.long, device=hidden_states.device)

    batch_idx = torch.zeros(W, dtype=torch.long, device=hidden_states.device)
    hidden_states[batch_idx, positions, :] += deltas * coefficient
    return hidden_states


# ── Public API ──────────────────────────────────────────────────────────────

def inject_residuals(
    hidden_states: torch.Tensor,
    deltas: torch.Tensor,
    positions: torch.Tensor | None = None,
    coefficient: float = 1.0,
) -> torch.Tensor:
    """Inject residual deltas into hidden states (GPU-accelerated).

    Prefers Triton kernel, falls back to PyTorch native ops.
    Both run on CUDA.

    Args:
        hidden_states: (batch, seq_len, hidden_size) — residual stream tensor
        deltas: (num_deltas, hidden_size) — boundary residual vectors to inject
        positions: (num_deltas,) — sequence positions to target.
                   Default: last num_deltas positions of the sequence.
        coefficient: global scale applied to all deltas

    Returns:
        hidden_states modified in-place
    """
    if TRITON_AVAILABLE:
        return inject_residuals_triton(hidden_states, deltas, positions, coefficient)
    return inject_residuals_pytorch(hidden_states, deltas, positions, coefficient)


def inject_mean(
    hidden_states: torch.Tensor,
    deltas: torch.Tensor,
    coefficient: float = 1.0,
) -> torch.Tensor:
    """Inject the mean delta to every sequence position.

    This is a cruder form of injection that adds the same perturbation
    to all positions. Useful as a baseline comparison.
    """
    mean_delta = deltas.mean(dim=0) * coefficient
    hidden_states = hidden_states + mean_delta.unsqueeze(0).unsqueeze(0)
    return hidden_states
