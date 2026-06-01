"""Retrieval-assisted generation — score windows and load only top-k.

Uses boundary residuals (captured at crystal layer during ingestion) to score
relevance of each window to a query. Only the top-k most relevant windows'
KV caches are loaded, reducing GPU memory and speeding up prefill.
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_norm, b_norm))


def retrieve_top_windows(
    delta_dir: Path,
    boundaries: np.ndarray,
    query: str,
    tokenizer,
    model,
    top_k: int = 2,
) -> list[Path]:
    """Score windows by relevance to query, return paths for top-k.

    Args:
        delta_dir: Path to .pdga directory containing kv_cache_w{N}.pt files
        boundaries: (num_windows, hidden_size) boundary residuals
        query: User query text
        tokenizer: Model tokenizer
        model: Model (used for embedding query)
        top_k: Number of top windows to return

    Returns:
        List of kv_cache_w{N}.pt paths for top-k windows
    """
    num_windows = boundaries.shape[0]
    if num_windows == 0:
        return []

    # Encode query as token IDs, get average embedding
    query_ids = tokenizer.encode(query, add_special_tokens=False)
    if not query_ids:
        # Fallback: return all windows if query is empty
        return [delta_dir / f"kv_cache_w{i}.pt" for i in range(num_windows)]

    # Get query embedding by averaging token embeddings
    device = model.device
    embed = model.get_input_embeddings()
    query_tensor = torch.tensor([query_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        query_emb = embed(query_tensor).mean(dim=1).squeeze().float().cpu().numpy()

    # Score each window
    scores = []
    for i in range(num_windows):
        boundary = boundaries[i]
        score = cosine_similarity(query_emb, boundary)
        scores.append((score, i))

    # Sort by score descending, take top_k
    scores.sort(key=lambda x: -x[0])
    top_indices = [idx for _, idx in scores[:top_k]]

    # Return paths in original window order (preserves position continuity)
    top_indices_sorted = sorted(top_indices)
    return [delta_dir / f"kv_cache_w{i}.pt" for i in top_indices_sorted]
