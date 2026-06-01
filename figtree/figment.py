"""Figment — the universal unit of knowledge in Figtree.

Everything is a Figment:
- A sentence from a news article
- An Image (Figment with children=[...])
- An edge (Figment with meta["edge_type"] = "supports")
- A trust assertion (Figment with meta["edge_type"] = "trust")
- Even the system itself (meta-figments about Figtree)

Storage (.figment format):
    figment.figment/
    ├── manifest.json     # figment_id, children, meta, sources, trust
    ├── boundary.npy      # (hidden_size,) float32 — crystal layer (backward compat)
    ├── boundaries.npy    # (num_layers, hidden_size) float32 — per-layer boundaries
    ├── boundary_emb.npy  # (hidden_size,) float32 — last-token embedding
    ├── kv_cache.npy      # (num_layers, seq_len, 2, kv_dim) — unrotated K/V
    └── text.txt          # Natural language statement
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
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
    ) -> "Figment":
        """Factory: auto-generate figment_id from text."""
        figment_id = hashlib.sha256(text.encode()).hexdigest()[:16]
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

    def save(self, output_dir: Path) -> Path:
        """Write to .figment directory. Returns path."""
        output_dir = Path(output_dir)
        figment_dir = output_dir / f"{self.figment_id}.figment"
        figment_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "figment_id": self.figment_id,
            "text": self.text,
            "meta": self.meta,
            "children": self.children,
            "sources": self.sources,
            "trust": self.trust,
            "hidden_size": int(self.hidden_size),
        }
        (figment_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        np.save(figment_dir / "boundary.npy", self.boundary)
        if self.boundaries is not None:
            np.save(figment_dir / "boundaries.npy", self.boundaries)
        if self.boundary_emb is not None:
            np.save(figment_dir / "boundary_emb.npy", self.boundary_emb)
        (figment_dir / "text.txt").write_text(self.text)

        return figment_dir

    @classmethod
    def load(cls, figment_dir: Path) -> "Figment":
        """Load from .figment directory."""
        figment_dir = Path(figment_dir)
        manifest = json.loads((figment_dir / "manifest.json").read_text())
        boundary = np.load(figment_dir / "boundary.npy")
        boundaries = boundary_like = None
        bd_path = figment_dir / "boundaries.npy"
        if bd_path.exists():
            boundaries = np.load(bd_path)
        emb_path = figment_dir / "boundary_emb.npy"
        if emb_path.exists():
            boundary_like = np.load(emb_path)
        figment_id = manifest["figment_id"]
        return cls(
            figment_id=figment_id,
            text=manifest["text"],
            boundary=boundary,
            boundaries=boundaries,
            boundary_emb=boundary_like,
            meta=manifest.get("meta", {}),
            children=manifest.get("children", []),
            sources=manifest.get("sources", []),
            trust=manifest.get("trust", 0.5),
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

    def __repr__(self) -> str:
        kind = "image" if self.is_image() else "edge" if self.is_edge() else "atomic"
        return f"Figment({kind}, id={self.figment_id[:8]}..., trust={self.trust:.2f}, text={self.text[:40]!r})"
