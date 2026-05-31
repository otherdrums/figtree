"""Sparse novelty detection for boundary residuals.

During ingestion, capture the residual at the crystal layer for EVERY token
position. Compute pairwise cosine distance between consecutive positions.
Positions where the residual shifted significantly represent novel
information the model didn't already possess.

This module handles novelty scoring and semantic span computation.
"""

from __future__ import annotations

import numpy as np


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine_similarity: 0 = identical, 2 = opposite."""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(1.0 - np.dot(a_norm, b_norm))


def detect_novel_positions_adaptive(
    residuals: np.ndarray,
    target_fraction: float = 0.05,
    min_positions: int = 4,
    max_positions: int = 40,
) -> tuple[list[int], list[float]]:
    """Adaptive novelty detection — retain the top `target_fraction` of positions.

    Always includes the last position. Picks the most novel positions based
    on pairwise cosine distance, up to the target fraction of total positions.
    """
    seq_len = residuals.shape[0]
    if seq_len <= 1:
        return [0], [0.0]

    target_count = max(min_positions, min(max_positions, int(seq_len * target_fraction)))

    scores: list[tuple[int, float]] = []
    for i in range(1, seq_len):
        dist = cosine_distance(residuals[i - 1], residuals[i])
        scores.append((i, dist))

    scores.sort(key=lambda x: x[1], reverse=True)

    novel = {seq_len - 1}

    for pos, _dist in scores:
        if len(novel) >= target_count:
            break
        novel.add(pos)

    result_positions = sorted(novel)
    result_scores = [
        cosine_distance(residuals[0], residuals[p]) if p > 0 else 0.0
        for p in result_positions
    ]

    return result_positions, result_scores
