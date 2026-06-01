# CUDA Attention Kernel for Boundary-KV Injection

## Problem

Standard HF attention (`Q @ K^T` softmax) treats all K/V entries identically — they come from normally processed token sequences. Boundary-projected K/V entries come from a different distribution (pre-computed layer residuals), and the model hasn't learned to interpret them. The result: boundary-based generation produces noise, not factual recall.

## Current Status: Solved

This plan is now **superseded** by a simpler approach that works with standard HF attention.

The key insight was that boundary-projected K/V needs to match the model's internal computation exactly. Instead of a custom kernel with learned gates, we:

1. During ingestion, compute per-token K/V by applying `input_layernorm` before projecting through `k_proj`/`v_proj` with `k_norm`
2. Store the unrotated K/V as `kv_cache.npy`
3. During generation, apply RoPE based on global position IDs and insert into standard `DynamicCache`

This produces factual recall with standard HF attention because the K/V entries are **identical** to what the model would compute during a forward pass. No custom kernel or learned gates needed.

## Original Approach (Historical)

The original plan used a **separate boundary-attention path** with learned gating:

```
Standard path:            Q_std @ K_std^T → softmax → V_std
Boundary path (new):      Q_bd @ K_bd^T → softmax → V_bd ← learned gate
Output: gate * V_bd + (1 - gate) * V_std
```

Each layer's attention gets an additional **boundary K/V cache** containing the projected boundary entries. A learned **boundary gate** parameter (one scalar per layer, initialized to 0) controls how much the layer attends to boundary information.

### Kernel Design (Original)

```cuda
__global__ void fused_boundary_attention_kernel(
    const __nv_bfloat16* Q,           // (batch, heads, seq_len, head_dim)
    const __nv_bfloat16* K_std,       // (batch, heads, kv_seq_len, head_dim)
    const __nv_bfloat16* V_std,       // (batch, heads, kv_seq_len, head_dim)
    const __nv_bfloat16* K_bd,        // (batch, heads, num_figments, head_dim)
    const __nv_bfloat16* V_bd,        // (batch, heads, num_figments, head_dim)
    const float* gate,                 // per-layer learned scalar
    __nv_bfloat16* output,            // (batch, heads, seq_len, head_dim)
    ...
)
```

### Training (Original)

The boundary gate would need to be trained with 36 parameters (one per layer), freezing all model weights.

### Dependencies

- CUDA Toolkit (already required)
- No additional Python packages (uses existing torch + transformers)
- Training data: can be generated from the Davos narratives or any text corpus
