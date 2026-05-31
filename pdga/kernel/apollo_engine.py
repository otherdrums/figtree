"""Apollo-style forward pass — LARQL-compatible residual stream engine.

Implements the full Apollo injection pattern using PyTorch operations
(already on CUDA) with an optional fused CUDA kernel for injection speed.

Key mechanics (matching LARQL Apollo):
1. Boundary residual at position 0 (crystal-layer level, replaces layers 0..crystal-1)
2. Prompt embeddings at positions 1..P (embedding level)
3. Forward through layers crystal..N-1 with full bidirectional attention
4. Injection delta added at last position at crystal layer
5. Re-run full forward at each decode step (no KV cache mismatch)
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

# Try optional CUDA kernel
try:
    from pdga.kernel.cuda_inject import inject_at_position
    _cuda_inject = inject_at_position
except Exception:
    _cuda_inject = None


def apollo_forward(
    model: PreTrainedModel,
    boundary_residual: torch.Tensor,
    token_embeddings: torch.Tensor,
    crystal_layer: int,
    injection_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apollo-style forward: boundary at pos-0, embeddings at pos-1+, skip 0..crystal-1."""
    num_layers = model.config.num_hidden_layers
    device = token_embeddings.device
    dtype = token_embeddings.dtype
    config = model.config

    B = boundary_residual.unsqueeze(0).unsqueeze(0)
    P = token_embeddings.shape[1]
    h = torch.cat([B, token_embeddings], dim=1)
    L = 1 + P

    is_qwen3 = hasattr(config, 'model_type') and config.model_type == 'qwen3'

    for layer_idx in range(crystal_layer, num_layers):
        layer = model.model.layers[layer_idx]

        residual = h
        h_norm = layer.input_layernorm(h)

        attn_out = _compute_attention(
            model, layer.self_attn, h_norm, L, config, device, dtype, is_qwen3,
        )
        h = residual + attn_out

        residual = h
        h_norm = layer.post_attention_layernorm(h)

        gate = layer.mlp.gate_proj(h_norm)
        up = layer.mlp.up_proj(h_norm)
        act = _get_activation_fn(model)
        down = layer.mlp.down_proj(act(gate) * up)
        h = residual + down

        if layer_idx == crystal_layer and injection_delta is not None:
            if _cuda_inject is not None:
                h = _cuda_inject(h, injection_delta, L - 1)
            else:
                h[:, -1, :] = h[:, -1, :] + injection_delta.unsqueeze(0)

    h = model.model.norm(h)
    return h


def _compute_attention(model, attn, hidden, seq_len, config, device, dtype, is_qwen3):
    """Attention with full bidirectional masking, Qwen2/3 compatible."""
    bsz = hidden.shape[0]
    q_len = hidden.shape[1]
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = attn.head_dim
    num_kv_groups = num_heads // num_kv_heads

    query = attn.q_proj(hidden)
    key = attn.k_proj(hidden)
    value = attn.v_proj(hidden)

    query = query.view(bsz, q_len, num_heads, head_dim)
    key = key.view(bsz, q_len, num_kv_heads, head_dim)
    value = value.view(bsz, q_len, num_kv_heads, head_dim)

    if is_qwen3 and hasattr(attn, 'q_norm'):
        query = attn.q_norm(query)
    if is_qwen3 and hasattr(attn, 'k_norm'):
        key = attn.k_norm(key)

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    pos_ids = torch.arange(0, seq_len, device=device, dtype=torch.long).unsqueeze(0)
    cos, sin = model.model.rotary_emb(value, pos_ids)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    key = repeat_kv(key, num_kv_groups)
    value = repeat_kv(value, num_kv_groups)

    attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, num_heads * head_dim)
    attn_output = attn.o_proj(attn_output)
    return attn_output


def _get_activation_fn(model):
    """Get the activation function from the model config."""
    config = model.config
    if hasattr(config, 'hidden_act'):
        act_name = config.hidden_act
        if act_name == 'silu':
            return F.silu
        elif act_name == 'gelu':
            return F.gelu
        elif act_name == 'gelu_pytorch_tanh':
            return lambda x: F.gelu(x, approximate='tanh')
    return F.silu
