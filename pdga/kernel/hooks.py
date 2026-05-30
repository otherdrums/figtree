"""Model forward hook utilities for PDGA multi-delta injection."""

from __future__ import annotations

import torch
from transformers import PreTrainedModel


def get_model_device(model: PreTrainedModel) -> torch.device:
    """Get the device a model is on."""
    return next(model.parameters()).device


def forward_from_residual(
    model: PreTrainedModel,
    residual: torch.Tensor,
    start_layer: int,
    num_layers: int,
    use_cache: bool = True,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
    """Forward a residual vector through layers start_layer..N.

    Args:
        model: HuggingFace model
        residual: (batch, num_tokens, hidden_size) residual at start_layer-1
        start_layer: First layer to run (0-indexed)
        num_layers: Total number of layers
        use_cache: Whether to return KV cache
        attention_mask: Optional attention mask

    Returns:
        (hidden_states, past_key_values)
        hidden_states: (batch, num_tokens, hidden_size) after all layers
        past_key_values: list of (k, v) per layer starting from start_layer
    """
    hidden_states = residual
    past_key_values = [None] * start_layer
    batch_size, seq_len, _ = hidden_states.shape
    device = hidden_states.device

    if attention_mask is None:
        attention_mask = torch.ones(batch_size, seq_len, device=device)

    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    rotary_emb = model.model.rotary_emb

    for layer_idx in range(start_layer, num_layers):
        layer = model.model.layers[layer_idx]
        position_embeddings = rotary_emb(hidden_states, position_ids)
        layer_out = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
        )
        hidden_states = layer_out if not isinstance(layer_out, tuple) else layer_out[0]

        if use_cache and isinstance(layer_out, tuple) and len(layer_out) > 1 and layer_out[1] is not None:
            past_key_values.append(layer_out[1])
        elif use_cache:
            past_key_values.append(None)

    hidden_states = model.model.norm(hidden_states)
    return hidden_states, past_key_values if use_cache else None


def build_delta_kv_cache(
    model: PreTrainedModel,
    residual: torch.Tensor,
    crystal_layer: int,
    num_layers: int,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Build KV cache for a single delta boundary residual.

    Forwards the residual through layers crystal_layer+1..N, capturing
    K and V at each layer for compact context injection.

    Args:
        model: HuggingFace model
        residual: (1, 1, hidden_size) boundary residual vector
        crystal_layer: Layer where residual was captured
        num_layers: Total layers in model

    Returns:
        dict mapping layer_idx -> (K, V) in past_key_values format:
        K: (1, num_kv_heads, 1, head_dim)
        V: (1, num_kv_heads, 1, head_dim)
        Layers 0..crystal_layer have None (will be zero-filled during generation)
    """
    hidden_states = residual.clone().to(model.device)
    kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    batch_size, seq_len, _ = hidden_states.shape
    device = hidden_states.device
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    rotary_emb = model.model.rotary_emb

    for layer_idx in range(crystal_layer + 1, num_layers):
        layer = model.model.layers[layer_idx]
        position_embeddings = rotary_emb(hidden_states, position_ids)
        layer_out = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=True,
        )
        hidden_states = layer_out if not isinstance(layer_out, tuple) else layer_out[0]
        present_kv = layer_out[1] if isinstance(layer_out, tuple) and len(layer_out) > 1 else None

        if present_kv is not None:
            key, value = present_kv
            kv_cache[layer_idx] = (key.cpu(), value.cpu())

    return kv_cache


def seed_past_key_values(
    model: PreTrainedModel,
    delta_kv_caches: list[dict[int, tuple[torch.Tensor, torch.Tensor]]],
    crystal_layer: int,
) -> list[tuple[torch.Tensor, torch.Tensor] | None]:
    """Build seeded past_key_values for multiple deltas.

    Concatenates KV from all deltas along the sequence dimension.
    Layers 0..crystal_layer get None (standard generation-specific self-attention).
    Layers crystal_layer+1..N get concatenated delta KV.

    Returns:
        List of (K, V) tuples per layer, suitable for model.forward(past_key_values=...)
    """
    num_layers = model.config.num_hidden_layers
    device = next(model.parameters()).device
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    past_key_values = []
    for layer_idx in range(num_layers):
        if layer_idx <= crystal_layer:
            past_key_values.append(None)
            continue

        all_k = []
        all_v = []
        for delta_kv in delta_kv_caches:
            if layer_idx in delta_kv:
                k, v = delta_kv[layer_idx]
                all_k.append(k.to(device))
                all_v.append(v.to(device))

        if not all_k:
            k_empty = torch.zeros(1, num_kv_heads, 0, head_dim, device=device)
            v_empty = torch.zeros(1, num_kv_heads, 0, head_dim, device=device)
            past_key_values.append((k_empty, v_empty))
        else:
            combined_k = torch.cat(all_k, dim=2)
            combined_v = torch.cat(all_v, dim=2)
            past_key_values.append((combined_k, combined_v))

    return past_key_values
