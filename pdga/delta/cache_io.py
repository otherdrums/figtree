"""KV cache serialization — save and load DynamicCache for boundary-kv generation.

Stores pre-computed K/V tensors per layer per window, enabling instant prefill
during generation (load from disk instead of re-running window tokens through
the model).

Format (per window):
    kv_cache_W{window_idx}.pt — PyTorch archive containing:
        layer_{li}_keys:  (1, num_kv_heads, seq_len, head_dim) bf16/fp16
        layer_{li}_values: (1, num_kv_heads, seq_len, head_dim) bf16/fp16
"""

from __future__ import annotations

import torch
from pathlib import Path
from transformers.cache_utils import DynamicCache


def save_window_cache(
    cache: DynamicCache,
    delta_dir: Path,
    window_idx: int,
    num_layers: int,
) -> Path:
    """Save a window's KV cache to the delta directory.

    Args:
        cache: DynamicCache populated with K/V for this window's tokens.
        delta_dir: Path to the .pdga directory.
        window_idx: Zero-based window index within the delta.
        num_layers: Total number of layers (for validation).

    Returns:
        Path to the saved cache file.
    """
    delta_dir = Path(delta_dir)
    delta_dir.mkdir(parents=True, exist_ok=True)

    state = {}
    for li in range(num_layers):
        if li >= len(cache.layers):
            break
        layer = cache.layers[li]
        state[f"layer_{li}_keys"] = layer.keys.cpu()
        state[f"layer_{li}_values"] = layer.values.cpu()

    out_path = delta_dir / f"kv_cache_w{window_idx}.pt"
    torch.save(state, out_path, _use_new_zipfile_serialization=True)
    return out_path


def load_window_cache(
    cache_path: Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    num_layers: int = 36,
) -> DynamicCache:
    """Load a window's KV cache from disk and reconstruct a DynamicCache.

    Args:
        cache_path: Path to kv_cache_w{N}.pt file.
        device: Target device (cuda:0, cpu).
        dtype: Data type for the loaded tensors.
        num_layers: Expected number of layers.

    Returns:
        DynamicCache seeded with the pre-computed K/V tensors.
    """
    state = torch.load(str(cache_path), map_location="cpu", weights_only=True)
    cache = DynamicCache()

    for li in range(num_layers):
        k_key = f"layer_{li}_keys"
        v_key = f"layer_{li}_values"
        if k_key not in state:
            break
        keys = state[k_key].to(device=device, dtype=dtype)
        values = state[v_key].to(device=device, dtype=dtype)
        cache.update(keys, values, li)

    return cache


def get_cache_size_per_window(
    window_size: int,
    num_layers: int = 36,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    bytes_per_element: int = 2,  # bf16/fp16
) -> int:
    """Estimate KV cache size in bytes for one window."""
    # (K + V) × heads × dim × tokens × layers × bytes
    return 2 * num_kv_heads * head_dim * window_size * num_layers * bytes_per_element


def list_window_caches(delta_dir: Path) -> list[int]:
    """List available window cache indices in a delta directory."""
    delta_dir = Path(delta_dir)
    paths = sorted(delta_dir.glob("kv_cache_w*.pt"))
    indices = []
    for p in paths:
        try:
            idx = int(p.stem.split("kv_cache_w")[1])
            indices.append(idx)
        except (ValueError, IndexError):
            pass
    return indices
