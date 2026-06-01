/*
 * boundary_project_kernel.cu
 *
 * Project fact-boundary vectors through a layer's W_k and W_v weight matrices
 * to produce per-fact K/V entries that are injected directly into the KV cache.
 *
 *   boundaries: (num_facts, hidden_size)
 *   W:          (hidden_size, kv_dim)   — i.e. W_k.weight.T or W_v.weight.T
 *   out:        (num_facts, kv_dim)     — reshaped to (num_facts, num_kv_heads, head_dim) by caller
 *
 * Each thread computes one element of out[fact, kv_idx].
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>

/* ── bf16 ── */
__global__ void boundary_project_bf16_kernel(
    const __nv_bfloat16* __restrict__ boundaries,
    const __nv_bfloat16* __restrict__ W,
    __nv_bfloat16* __restrict__ out,
    int num_facts,
    int hidden_size,
    int kv_dim)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_facts * kv_dim;
    if (tid >= total) return;

    int fact_idx = tid / kv_dim;
    int kv_idx   = tid % kv_dim;

    float acc = 0.0f;
    const __nv_bfloat16* b_row = boundaries + fact_idx * hidden_size;

    /* simple dot-product along hidden_size */
    #pragma unroll 4
    for (int h = 0; h < hidden_size; ++h) {
        float b = __bfloat162float(b_row[h]);
        float w = __bfloat162float(W[h * kv_dim + kv_idx]);
        acc += b * w;
    }
    out[tid] = __float2bfloat16_rn(acc);
}

/* ── fp16 ── */
__global__ void boundary_project_fp16_kernel(
    const __half* __restrict__ boundaries,
    const __half* __restrict__ W,
    __half* __restrict__ out,
    int num_facts,
    int hidden_size,
    int kv_dim)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_facts * kv_dim;
    if (tid >= total) return;

    int fact_idx = tid / kv_dim;
    int kv_idx   = tid % kv_dim;

    float acc = 0.0f;
    const __half* b_row = boundaries + fact_idx * hidden_size;

    #pragma unroll 4
    for (int h = 0; h < hidden_size; ++h) {
        float b = __half2float(b_row[h]);
        float w = __half2float(W[h * kv_dim + kv_idx]);
        acc += b * w;
    }
    out[tid] = __float2half_rn(acc);
}

/* ── C++ host wrappers (called from PyTorch C++ extension) ── */
#include <torch/extension.h>

torch::Tensor boundary_project(
    torch::Tensor boundaries,   // (num_facts, hidden_size)
    torch::Tensor W,          // (hidden_size, kv_dim)
    int dtype_enum)            // 0=bf16, 1=fp16
{
    TORCH_CHECK(boundaries.is_cuda(), "boundaries must be CUDA");
    TORCH_CHECK(W.is_cuda(), "W must be CUDA");
    TORCH_CHECK(boundaries.dim() == 2, "boundaries must be 2D");
    TORCH_CHECK(W.dim() == 2, "W must be 2D");

    int num_facts   = boundaries.size(0);
    int hidden_size = boundaries.size(1);
    int kv_dim      = W.size(1);

    TORCH_CHECK(W.size(0) == hidden_size,
                "W rows (", W.size(0), ") != hidden_size (", hidden_size, ")");

    auto out = torch::empty({num_facts, kv_dim},
                            torch::TensorOptions()
                                .dtype(boundaries.dtype())
                                .device(boundaries.device()));

    int total   = num_facts * kv_dim;
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream();

    if (boundaries.dtype() == torch::kBFloat16) {
        boundary_project_bf16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(boundaries.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            num_facts, hidden_size, kv_dim);
    } else if (boundaries.dtype() == torch::kFloat16) {
        boundary_project_fp16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __half*>(boundaries.data_ptr()),
            reinterpret_cast<const __half*>(W.data_ptr()),
            reinterpret_cast<__half*>(out.data_ptr()),
            num_facts, hidden_size, kv_dim);
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", boundaries.dtype());
    }

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed");
    return out;
}

/* ── pybind11 exports ── */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("boundary_project", &boundary_project,
          "Project boundary vectors through a weight matrix (K or V). "
          "Args: boundaries(N,H), W(H,K), dtype_enum. Returns (N,K).");
}
