"""Python wrapper for the boundary_project CUDA kernel.

Provides:
    project_boundaries_to_kv(boundaries, layer) -> (K_figments, V_figments)

Usage:
    from figtree.kernel.boundary_project import project_boundaries_to_kv
    k_figments, v_figments = project_boundaries_to_kv(boundaries, layer)

The kernel multiplies ``boundaries @ W`` in CUDA. Quantized (bitsandbytes
4-bit) projections store their weights in a packed uint8 format the raw
kernel cannot read, so we dequantize the weight to a normal bf16/fp16 matrix
on the fly before handing it to the kernel. This keeps the kernel usable for
both quantized and non-quantized models.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from figtree.kernel.build import get_extension


_dtype_to_enum = {
    torch.bfloat16: 0,
    torch.float16: 1,
}


def _dequant_weight(proj: nn.Module, dtype: torch.dtype) -> torch.Tensor:
    """Return a plain (out_features, in_features) weight matrix for `proj`.

    For bitsandbytes 4-bit linear layers the weight is packed uint8; we
    dequantize it via `bitsandbytes.functional.dequantize_4bit`. For ordinary
    linear layers we simply transpose `.weight`.
    """
    if proj.__class__.__name__ == "Linear4bit" or hasattr(proj, "quant_state"):
        from bitsandbytes.functional import dequantize_4bit

        # dequantize_4bit yields (out_features, in_features) = (kv_dim, hidden_size);
        # the kernel expects a contiguous W of shape (hidden_size, kv_dim), so
        # transpose AND make contiguous (a raw .T view has swapped strides the
        # kernel's row-major indexing would misread).
        return dequantize_4bit(proj.weight.data, quant_state=proj.quant_state).to(dtype).T.contiguous()
    # Dense Linear: weight is (out_features, in_features); transpose to a
    # contiguous (hidden_size, kv_dim) matrix for the kernel.
    return proj.weight.data.to(dtype).T.contiguous()


def project_boundaries_to_kv(
    boundaries: torch.Tensor,
    layer: nn.Module,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project figment boundaries through a layer's W_k and W_v via CUDA kernel.

    Args:
        boundaries: (num_figments, hidden_size) float16 or bfloat16
        layer: A transformer self-attention layer with .self_attn.k_proj and
            .self_attn.v_proj (supports both dense and bitsandbytes 4-bit).
        device: Target CUDA device (inferred from boundaries if None)

    Returns:
        K_figments: (num_figments, num_kv_heads, head_dim)
        V_figments: (num_figments, num_kv_heads, head_dim)
    """
    if device is None:
        device = boundaries.device

    boundaries = boundaries.to(device=device)
    dtype = boundaries.dtype
    if dtype not in _dtype_to_enum:
        raise ValueError(f"Unsupported dtype {dtype}; expected bf16 or fp16")

    k_proj = layer.self_attn.k_proj
    v_proj = layer.self_attn.v_proj
    Wk = _dequant_weight(k_proj, dtype).to(device)
    Wv = _dequant_weight(v_proj, dtype).to(device)

    ext = get_extension()
    k_flat = ext.boundary_project(boundaries, Wk, _dtype_to_enum[dtype])
    v_flat = ext.boundary_project(boundaries, Wv, _dtype_to_enum[dtype])

    num_figments = boundaries.size(0)
    kv_dim = k_flat.size(1)

    head_dim = layer.self_attn.head_dim
    num_kv_heads = kv_dim // head_dim

    k_figments = k_flat.view(num_figments, num_kv_heads, head_dim)
    v_figments = v_flat.view(num_figments, num_kv_heads, head_dim)

    return k_figments, v_figments
