"""Python wrapper for the boundary_project CUDA kernel.

Provides:
    project_boundaries_to_kv(boundaries, layer) -> (K_facts, V_facts)

Usage:
    from pdga.kernel.boundary_project import project_boundaries_to_kv
    k_facts, v_facts = project_boundaries_to_kv(boundaries, layer)
"""

from __future__ import annotations

import torch
from torch import nn

from pdga.kernel.build import get_extension


_dtype_to_enum = {
    torch.bfloat16: 0,
    torch.float16: 1,
}


def project_boundaries_to_kv(
    boundaries: torch.Tensor,
    layer: nn.Module,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project fact boundaries through a layer's W_k and W_v to produce K/V entries.

    Args:
        boundaries: (num_facts, hidden_size) float16 or bfloat16
        layer: A transformer self-attention layer with .self_attn.k_proj and .self_attn.v_proj
        device: Target CUDA device (inferred from boundaries if None)

    Returns:
        K_facts: (num_facts, num_kv_heads, head_dim)
        V_facts: (num_facts, num_kv_heads, head_dim)
    """
    ext = get_extension()

    if device is None:
        device = boundaries.device

    if boundaries.device != device:
        boundaries = boundaries.to(device)

    dtype = boundaries.dtype
    if dtype not in _dtype_to_enum:
        raise ValueError(f"Unsupported dtype {dtype}; expected bf16 or fp16")

    # For 4-bit quantized models, we can't directly extract the weight matrix.
    # Instead, use the linear layer's forward pass, which handles dequantization internally.
    k_flat = layer.self_attn.k_proj(boundaries)
    v_flat = layer.self_attn.v_proj(boundaries)

    # Reshape: (num_facts, num_kv_heads * head_dim) -> (num_facts, num_kv_heads, head_dim)
    num_facts = boundaries.size(0)
    kv_dim = k_flat.size(1)

    # Infer num_kv_heads and head_dim
    head_dim = layer.self_attn.head_dim
    num_kv_heads = kv_dim // head_dim

    k_facts = k_flat.view(num_facts, num_kv_heads, head_dim)
    v_facts = v_flat.view(num_facts, num_kv_heads, head_dim)

    return k_facts, v_facts
