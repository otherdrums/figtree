"""Locality-Sensitive Hashing (LSH) for boundary residual retrieval.

Cosine-preserving LSH using random hyperplane projection (SimHash style).
Each boundary residual vector is hashed into L hash tables with k-bit keys.
Bucket collisions = semantic similarity candidates.

This serves dual purpose:
1. Retrieval: O(1) per-table lookup for relevant context deltas
2. Cogitation: Bucket groups are natural topic clusters for generalization
"""

from __future__ import annotations

import json
import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from pdga.db.schema import get_db_path


@dataclass
class LSHConfig:
    num_tables: int = 4
    bits_per_hash: int = 16
    hidden_size: int = 0


class BoundaryLSH:
    """Cosine-LSH over boundary residual vectors.

    Each hash table: k random hyperplane normals → k-bit hash via sign projection.
    L tables → L independent hashings → higher recall through union of results.
    """

    def __init__(
        self,
        config: LSHConfig,
        db_path: Path | None = None,
    ):
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_tables = config.num_tables
        self.bits_per_hash = config.bits_per_hash

        if db_path is None:
            db_path = get_db_path()
        self.db_path = db_path

        self._rng = np.random.RandomState(42)
        self._planes: list[np.ndarray] = []
        self._init_planes()
        self._init_db()

    def _init_planes(self) -> None:
        if self.hidden_size == 0:
            return
        for _ in range(self.num_tables):
            planes = self._rng.randn(self.bits_per_hash, self.hidden_size).astype(np.float32)
            norms = np.linalg.norm(planes, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            planes = planes / norms
            self._planes.append(planes)

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS lsh_tables (
                table_idx   INTEGER NOT NULL,
                bucket_key  TEXT NOT NULL,
                delta_id    TEXT NOT NULL,
                window_idx  INTEGER NOT NULL,
                boundary_hash TEXT,
                PRIMARY KEY (table_idx, bucket_key, delta_id, window_idx)
            )"""
        )
        conn.commit()
        conn.close()

    def _compute_hash(self, table_idx: int, vector: np.ndarray) -> str:
        """Compute k-bit hash for a single table."""
        projections = self._planes[table_idx] @ vector
        bits = (projections > 0).astype(np.uint8)
        packed = np.packbits(bits)
        return packed.tobytes().hex()

    def _compute_boundary_hash(self, vector: np.ndarray) -> str:
        """Short hash of the boundary itself for quick dedup."""
        return hashlib.sha256(vector.tobytes()).hexdigest()[:16]

    def hash_all(self, vector: np.ndarray) -> list[str]:
        """Return bucket keys across all L tables for a single vector."""
        return [self._compute_hash(i, vector) for i in range(self.num_tables)]

    def insert(
        self,
        delta_id: str,
        boundaries: np.ndarray,
    ) -> list[str]:
        """Insert all windows of a delta into all L hash tables.

        Args:
            delta_id: Delta identifier
            boundaries: (num_windows, hidden_size) f32 array

        Returns:
            List of all bucket keys this delta occupies
        """
        conn = sqlite3.connect(str(self.db_path))
        all_buckets = []

        try:
            for window_idx in range(boundaries.shape[0]):
                boundary = boundaries[window_idx].astype(np.float32)
                bhash = self._compute_boundary_hash(boundary)

                for table_idx in range(self.num_tables):
                    bucket_key = self._compute_hash(table_idx, boundary)
                    conn.execute(
                        """INSERT OR IGNORE INTO lsh_tables
                           (table_idx, bucket_key, delta_id, window_idx, boundary_hash)
                           VALUES (?, ?, ?, ?, ?)""",
                        (table_idx, bucket_key, delta_id, window_idx, bhash),
                    )
                    all_buckets.append(bucket_key)

            conn.commit()
        finally:
            conn.close()

        return all_buckets

    def query(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
    ) -> list[tuple[str, int]]:
        """Find candidate (delta_id, window_idx) pairs matching query boundary.

        Queries all L tables, returns unique candidates by union of bucket hits.
        """
        conn = sqlite3.connect(str(self.db_path))
        candidates: dict[tuple[str, int], int] = {}

        try:
            query_vec = query_vector.astype(np.float32).squeeze()

            for table_idx in range(self.num_tables):
                bucket_key = self._compute_hash(table_idx, query_vec)
                rows = conn.execute(
                    """SELECT delta_id, window_idx FROM lsh_tables
                       WHERE table_idx = ? AND bucket_key = ?""",
                    (table_idx, bucket_key),
                ).fetchall()

                for delta_id, window_idx in rows:
                    key = (delta_id, window_idx)
                    candidates[key] = candidates.get(key, 0) + 1

        finally:
            conn.close()

        sorted_candidates = sorted(
            candidates.items(), key=lambda x: x[1], reverse=True
        )
        return [k for k, _ in sorted_candidates[:top_k]]

    def get_bucket_contents(
        self,
        table_idx: int,
        bucket_key: str,
    ) -> list[tuple[str, int]]:
        """Get all (delta_id, window_idx) in a specific bucket.

        Used by cogitation to gather all deltas in a topic cluster.
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute(
                """SELECT delta_id, window_idx FROM lsh_tables
                   WHERE table_idx = ? AND bucket_key = ?""",
                (table_idx, bucket_key),
            ).fetchall()
            return [(r[0], r[1]) for r in rows]
        finally:
            conn.close()

    def get_all_buckets(
        self,
        table_idx: int,
        min_size: int = 2,
    ) -> list[tuple[str, int]]:
        """Get all buckets with at least min_size entries.

        Used to find topic clusters ready for cogitation.
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute(
                """SELECT bucket_key, COUNT(*) as cnt FROM lsh_tables
                   WHERE table_idx = ?
                   GROUP BY bucket_key
                   HAVING cnt >= ?
                   ORDER BY cnt DESC""",
                (table_idx, min_size),
            ).fetchall()
            return [(r[0], r[1]) for r in rows]
        finally:
            conn.close()

    def remove_delta(self, delta_id: str) -> None:
        """Remove all entries for a delta from LSH tables."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("DELETE FROM lsh_tables WHERE delta_id = ?", (delta_id,))
        conn.commit()
        conn.close()

    def stats(self) -> dict:
        """Return LSH index statistics."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            total = conn.execute("SELECT COUNT(*) FROM lsh_tables").fetchone()[0]
            buckets = conn.execute(
                "SELECT COUNT(DISTINCT bucket_key || ':' || table_idx) FROM lsh_tables"
            ).fetchone()[0]
            deltas = conn.execute(
                "SELECT COUNT(DISTINCT delta_id) FROM lsh_tables"
            ).fetchone()[0]
            return {
                "total_entries": total,
                "unique_buckets": buckets,
                "indexed_deltas": deltas,
                "num_tables": self.num_tables,
                "bits_per_hash": self.bits_per_hash,
            }
        finally:
            conn.close()


def create_lsh_for_model(
    hidden_size: int,
    num_tables: int = 4,
    bits_per_hash: int = 16,
    db_path: Path | None = None,
) -> BoundaryLSH:
    """Factory for creating an LSH index matching a model's hidden size."""
    config = LSHConfig(
        num_tables=num_tables,
        bits_per_hash=bits_per_hash,
        hidden_size=hidden_size,
    )
    return BoundaryLSH(config, db_path=db_path)
