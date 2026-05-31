"""Apollo-style CUDA kernel for residual stream injection.

Provides a fused CUDA C++ kernel that adds a delta vector to a specific
position in the hidden states tensor. Used during the Apollo-style forward
pass where boundary residuals serve as virtual position-0 and token-embedding
deltas are injected at the crystal layer.

The kernel is compiled with torch.utils.cpp_extension.load_inline() and
works with bfloat16/float16 inputs.
"""

from __future__ import annotations

import torch
from torch.utils.cpp_extension import load_inline

_APOLLO_KERNEL = None


def _get_kernel():
    """Lazy-compile the Apollo CUDA kernel."""
    global _APOLLO_KERNEL
    if _APOLLO_KERNEL is not None:
        return _APOLLO_KERNEL

    cuda_source = """
    #include <cuda_fp16.h>
    #include <cuda_bf16.h>
    #include <torch/extension.h>

    template <typename scalar_t>
    __global__ void apollo_inject_kernel(
        scalar_t* __restrict__ hidden,
        const scalar_t* __restrict__ delta,
        int hidden_size,
        int target_position,
        int stride
    ) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= hidden_size) return;

        int store_idx = target_position * stride + tid;
        hidden[store_idx] = __hadd(hidden[store_idx], delta[tid]);
    }

    // Specialization for fp32
    template <>
    __global__ void apollo_inject_kernel<float>(
        float* __restrict__ hidden,
        const float* __restrict__ delta,
        int hidden_size,
        int target_position,
        int stride
    ) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= hidden_size) return;
        hidden[target_position * stride + tid] += delta[tid];
    }

    torch::Tensor apollo_inject(
        torch::Tensor hidden,
        torch::Tensor delta,
        int target_position
    ) {
        int batch_size = hidden.size(0);
        int seq_len = hidden.size(1);
        int hidden_size = hidden.size(2);
        int stride = seq_len;

        int threads = 256;
        int blocks = (hidden_size + threads - 1) / threads;

        for (int b = 0; b < batch_size; b++) {
            scalar_t* h_ptr = hidden[b].data_ptr<scalar_t>();
            const scalar_t* d_ptr = delta.data_ptr<const scalar_t>();

            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half, at::ScalarType::BFloat16,
                hidden.scalar_type(), "apollo_inject_cuda", ([&] {
                    apollo_inject_kernel<scalar_t><<<blocks, threads>>>(
                        h_ptr, d_ptr, hidden_size, target_position, stride
                    );
                })
            );
        }

        return hidden;
    }
    """

    cpp_source = """
    torch::Tensor apollo_inject(
        torch::Tensor hidden,
        torch::Tensor delta,
        int target_position
    );
    """

    _APOLLO_KERNEL = load_inline(
        name="apollo_cuda",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["apollo_inject"],
        verbose=False,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-use_fast_math"],
    )

    return _APOLLO_KERNEL


def inject_at_position(
    hidden: torch.Tensor,
    delta: torch.Tensor,
    position: int,
) -> torch.Tensor:
    """Inject delta vector at a specific position using CUDA kernel.

    Args:
        hidden: (batch, seq_len, hidden_size) — modified in-place
        delta: (hidden_size,) — the vector to add
        position: which sequence position to modify (0-indexed)

    Returns:
        hidden (same tensor, modified in-place)
    """
    kernel = _get_kernel()
    return kernel.apollo_inject(hidden, delta, position)


def inject_at_last(hidden: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Inject delta at the last sequence position."""
    return inject_at_position(hidden, delta, hidden.shape[1] - 1)
