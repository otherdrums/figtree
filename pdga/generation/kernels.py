"""Fused CUDA kernels for custom residual injection.

Compiles at runtime via torch.utils.cpp_extension.load_inline().
Provides:
- boundary_swap: Copy boundary residual to position 0 (bf16/fp16)
- fused_inject: Add weighted sum of token embeddings to target pos
"""

from __future__ import annotations

from torch.utils.cpp_extension import load_inline

_cuda_source = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// ---- BF16 kernels ----
__global__ void boundary_swap_bf16_kernel(
    const __nv_bfloat16* __restrict__ boundary,
    __nv_bfloat16* __restrict__ hidden,
    int hidden_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= hidden_size) return;
    hidden[tid] = boundary[tid];
}

__global__ void fused_inject_bf16_kernel(
    const __nv_bfloat16* __restrict__ delta,
    const float* __restrict__ coeffs,
    __nv_bfloat16* __restrict__ hidden,
    int target_position, int num_coeffs, int hidden_size,
    int stride, float global_scale
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= hidden_size) return;
    float sum = 0.0f;
    for (int c = 0; c < num_coeffs; ++c) {
        float d = __bfloat162float(delta[c * hidden_size + tid]);
        sum += d * coeffs[c];
    }
    sum *= global_scale;
    int store_idx = target_position * stride + tid;
    hidden[store_idx] = __float2bfloat16_rn(
        __bfloat162float(hidden[store_idx]) + sum
    );
}

// ---- FP16 kernels ----
__global__ void boundary_swap_fp16_kernel(
    const __half* __restrict__ boundary,
    __half* __restrict__ hidden,
    int hidden_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= hidden_size) return;
    hidden[tid] = boundary[tid];
}

__global__ void fused_inject_fp16_kernel(
    const __half* __restrict__ delta,
    const float* __restrict__ coeffs,
    __half* __restrict__ hidden,
    int target_position, int num_coeffs, int hidden_size,
    int stride, float global_scale
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= hidden_size) return;
    float sum = 0.0f;
    for (int c = 0; c < num_coeffs; ++c) {
        float d = __half2float(delta[c * hidden_size + tid]);
        sum += d * coeffs[c];
    }
    sum *= global_scale;
    int store_idx = target_position * stride + tid;
    hidden[store_idx] = __float2half_rn(
        __half2float(hidden[store_idx]) + sum
    );
}

// ---- Host-callable wrappers ----
torch::Tensor boundary_swap(
    torch::Tensor hidden_in,
    torch::Tensor boundary_in
) {
    auto hidden = hidden_in.contiguous();
    auto boundary = boundary_in.contiguous();
    int hidden_size = boundary.size(0);
    int threads = 256;
    int blocks = (hidden_size + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();

    if (hidden.dtype() == torch::kBFloat16) {
        boundary_swap_bf16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(boundary.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(hidden.data_ptr()),
            hidden_size
        );
    } else if (hidden.dtype() == torch::kFloat16) {
        boundary_swap_fp16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __half*>(boundary.data_ptr()),
            reinterpret_cast<__half*>(hidden.data_ptr()),
            hidden_size
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", hidden.dtype());
    }
    return hidden;
}

torch::Tensor fused_inject(
    torch::Tensor hidden_in,
    torch::Tensor delta_in,
    torch::Tensor coeffs_in,
    int64_t target_position,
    float global_scale
) {
    auto hidden = hidden_in.contiguous();
    auto delta = delta_in.contiguous();
    auto coeffs = coeffs_in.contiguous();
    int hidden_size = delta.size(1);
    int num_coeffs = delta.size(0);
    int stride = hidden.stride(1);
    int threads = 256;
    int blocks = (hidden_size + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();

    if (hidden.dtype() == torch::kBFloat16) {
        fused_inject_bf16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(delta.data_ptr()),
            coeffs.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(hidden.data_ptr()),
            (int)target_position, num_coeffs, hidden_size, stride, global_scale
        );
    } else if (hidden.dtype() == torch::kFloat16) {
        fused_inject_fp16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __half*>(delta.data_ptr()),
            coeffs.data_ptr<float>(),
            reinterpret_cast<__half*>(hidden.data_ptr()),
            (int)target_position, num_coeffs, hidden_size, stride, global_scale
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", hidden.dtype());
    }
    return hidden;
}
"""

_cpp_source = r"""
// Empty — all code is in cuda_sources (it includes torch/extension.h for PYBIND11)
torch::Tensor boundary_swap(torch::Tensor, torch::Tensor);
torch::Tensor fused_inject(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, float);
"""

_cuda_kernels: dict = {}

def get_kernels():
    """Compile (once) and return the CUDA extension module."""
    global _cuda_kernels
    if _cuda_kernels:
        return _cuda_kernels

    _cuda_kernels = load_inline(
        name="pdga_generation_kernels",
        cpp_sources=_cpp_source,
        cuda_sources=_cuda_source,
        functions=["boundary_swap", "fused_inject"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        verbose=False,
    )
    return _cuda_kernels
