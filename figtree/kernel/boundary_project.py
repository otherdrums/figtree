"""Python wrapper for the boundary_project CUDA kernel.

Provides:
    project_boundaries_to_kv(boundaries, layer) -> (K_figments, V_figments)

Usage:
    from figtree.kernel.boundary_project import project_boundaries_to_kv
    k_figments, v_figments = project_boundaries_to_kv(boundaries, layer)
"""

from __future__ import annotations

import torch
from torch import nn


_dtype_to_enum = {
    torch.bfloat16: 0,
    torch.float16: 1,
}


def project_boundaries_to_kv(
    boundaries: torch.Tensor,
    layer: nn.Module,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project figment boundaries through a layer's W_k and W_v.

    Args:
        boundaries: (num_figments, hidden_size) float16 or bfloat16
        layer: A transformer self-attention layer with .self_attn.k_proj and .self_attn.v_proj
        device: Target CUDA device (inferred from boundaries if None)

    Returns:
        K_figments: (num_figments, num_kv_heads, head_dim)
        V_figments: (num_figments, num_kv_heads, head_dim)
    """
    if device is None:
        device = boundaries.device

    if boundaries.device != device:
        boundaries = boundaries.to(device)

    dtype = boundaries.dtype
    if dtype not in _dtype_to_enum:
        raise ValueError(f"Unsupported dtype {dtype}; expected bf16 or fp16")

    k_flat = layer.self_attn.k_proj(boundaries)
    v_flat = layer.self_attn.v_proj(boundaries)

    num_figments = boundaries.size(0)
    kv_dim = k_flat.size(1)

    head_dim = layer.self_attn.head_dim
    num_kv_heads = kv_dim // head_dim

    k_figments = k_flat.view(num_figments, num_kv_heads, head_dim)
    v_figments = v_flat.view(num_figments, num_kv_heads, head_dim)

    return k_figments, v_figments
