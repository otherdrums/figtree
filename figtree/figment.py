"""Figment — the universal unit of knowledge in Figtree.

Everything is a Figment:
- A sentence from a news article
- An Image (Figment with children=[...])
- An edge (Figment with meta["edge_type"] = "supports")
- A trust assertion (Figment with meta["edge_type"] = "trust")
- Even the system itself (meta-figments about Figtree)

Figments are persisted as rows in a LanceDB table (see ``figtree/lancedb_store.py``);
K/V caches live outside the row as external quantized blobs managed by
``figtree/kv_cache_manager.py``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Figment:
    """A single atomic unit of knowledge."""

    figment_id: str             # SHA-256(text)[:16]
    text: str                   # Natural language statement
    boundary: np.ndarray        # (hidden_size,) float32 — crystal layer
    meta: dict[str, Any]        # edge_type, about_figment, etc.
    children: list[str]         # Child figment IDs (Images = figments with children)
    sources: list[str]          # Parent figment IDs
    trust: float                # Cached trust score
    boundaries: np.ndarray | None = None  # (num_layers, hidden_size) float32 — all layers
    boundary_emb: np.ndarray | None = None  # (hidden_size,) float32 — last-token embedding

    @property
    def hidden_size(self) -> int:
        return self.boundaries.shape[1] if self.boundaries is not None else self.boundary.shape[0]

    @classmethod
    def create(
        cls,
        text: str,
        boundary: np.ndarray,
        meta: dict[str, Any] | None = None,
        children: list[str] | None = None,
        sources: list[str] | None = None,
        trust: float = 0.5,
        boundaries: np.ndarray | None = None,
        boundary_emb: np.ndarray | None = None,
        figment_id: str | None = None,
    ) -> "Figment":
        """Factory: auto-generate figment_id from text (or use a provided id).

        A provided ``figment_id`` enables idempotent, re-runnable figments
        (e.g. one canonical trust Figment per source that can be overwritten).
        """
        figment_id = figment_id or hashlib.sha256(text.encode()).hexdigest()[:16]
        return cls(
            figment_id=figment_id,
            text=text,
            boundary=boundary.astype(np.float32),
            boundaries=boundaries.astype(np.float32) if boundaries is not None else None,
            boundary_emb=boundary_emb.astype(np.float32) if boundary_emb is not None else None,
            meta=meta or {},
            children=children or [],
            sources=sources or [],
            trust=trust,
        )

    def is_image(self) -> bool:
        """True if this figment contains other figments (i.e., has children)."""
        return len(self.children) > 0

    def is_edge(self) -> bool:
        """True if this figment represents a graph edge."""
        return self.meta.get("edge_type") is not None

    def is_trust_assertion(self) -> bool:
        """True if this figment represents a trust score."""
        return self.meta.get("edge_type") == "trust"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-friendly dict (independent of the store).

        Arrays become nested lists; use :meth:`from_dict` to reconstruct.
        """
        return {
            "figment_id": self.figment_id,
            "text": self.text,
            "boundary": self.boundary.astype(np.float32).tolist(),
            "boundaries": (
                self.boundaries.astype(np.float32).tolist() if self.boundaries is not None else None
            ),
            "boundary_emb": (
                self.boundary_emb.astype(np.float32).tolist() if self.boundary_emb is not None else None
            ),
            "meta": dict(self.meta),
            "children": list(self.children),
            "sources": list(self.sources),
            "trust": float(self.trust),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Figment":
        """Reconstruct a Figment from :meth:`to_dict` output."""
        boundary = np.asarray(d["boundary"], dtype=np.float32)
        boundaries = d.get("boundaries")
        boundary_emb = d.get("boundary_emb")
        return cls(
            figment_id=d["figment_id"],
            text=d["text"],
            boundary=boundary,
            boundaries=(
                np.asarray(boundaries, dtype=np.float32) if boundaries is not None else None
            ),
            boundary_emb=(
                np.asarray(boundary_emb, dtype=np.float32) if boundary_emb is not None else None
            ),
            meta=dict(d.get("meta", {})),
            children=list(d.get("children", [])),
            sources=list(d.get("sources", [])),
            trust=float(d.get("trust", 0.5)),
        )

    def __repr__(self) -> str:
        kind = "image" if self.is_image() else "edge" if self.is_edge() else "atomic"
        return f"Figment({kind}, id={self.figment_id[:8]}..., trust={self.trust:.2f}, text={self.text[:40]!r})"
