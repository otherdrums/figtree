"""Fact primitive — the universal unit of knowledge in PDGA v2.

Everything is a Fact:
- A sentence from a news article
- A narrative (Fact with children=[...])
- An edge (Fact with meta["edge_type"] = "supports")
- A trust assertion (Fact with meta["edge_type"] = "trust")
- Even the system itself (meta-facts about PDGA)

Storage (v2 .pdga format):
    fact.pdga/
    ├── manifest.json    # fact_id, children, meta, sources, trust
    ├── boundary.npy     # (hidden_size,) float32 — ONLY stored representation
    └── text.txt         # Natural language statement

No pre-computed KV cache. No window_tokens. Just boundary + text.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Fact:
    """A single atomic unit of knowledge."""

    fact_id: str          # SHA-256(text)[:16]
    text: str             # Natural language statement
    boundary: np.ndarray  # (hidden_size,) float32 — the ONLY stored tensor
    meta: dict[str, Any]  # edge_type, about_fact, etc.
    children: list[str]   # Child fact IDs (narratives = facts with children)
    sources: list[str]    # Parent fact IDs
    trust: float          # Cached trust score

    @property
    def hidden_size(self) -> int:
        return self.boundary.shape[0]

    @classmethod
    def create(
        cls,
        text: str,
        boundary: np.ndarray,
        meta: dict[str, Any] | None = None,
        children: list[str] | None = None,
        sources: list[str] | None = None,
        trust: float = 0.5,
    ) -> "Fact":
        """Factory: auto-generate fact_id from text."""
        fact_id = hashlib.sha256(text.encode()).hexdigest()[:16]
        return cls(
            fact_id=fact_id,
            text=text,
            boundary=boundary.astype(np.float32),
            meta=meta or {},
            children=children or [],
            sources=sources or [],
            trust=trust,
        )

    def save(self, output_dir: Path) -> Path:
        """Write to .pdga directory. Returns path."""
        output_dir = Path(output_dir)
        fact_dir = output_dir / f"{self.fact_id}.pdga"
        fact_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "fact_id": self.fact_id,
            "text": self.text,
            "meta": self.meta,
            "children": self.children,
            "sources": self.sources,
            "trust": self.trust,
            "hidden_size": int(self.hidden_size),
        }
        (fact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        np.save(fact_dir / "boundary.npy", self.boundary)
        (fact_dir / "text.txt").write_text(self.text)

        return fact_dir

    @classmethod
    def load(cls, fact_dir: Path) -> "Fact":
        """Load from .pdga directory."""
        fact_dir = Path(fact_dir)
        manifest = json.loads((fact_dir / "manifest.json").read_text())
        boundary = np.load(fact_dir / "boundary.npy")
        return cls(
            fact_id=manifest["fact_id"],
            text=manifest["text"],
            boundary=boundary,
            meta=manifest.get("meta", {}),
            children=manifest.get("children", []),
            sources=manifest.get("sources", []),
            trust=manifest.get("trust", 0.5),
        )

    def is_narrative(self) -> bool:
        """True if this fact contains other facts (i.e., has children)."""
        return len(self.children) > 0

    def is_edge(self) -> bool:
        """True if this fact represents a graph edge."""
        return self.meta.get("edge_type") is not None

    def is_trust_assertion(self) -> bool:
        """True if this fact represents a trust score."""
        return self.meta.get("edge_type") == "trust"

    def __repr__(self) -> str:
        kind = "narrative" if self.is_narrative() else "edge" if self.is_edge() else "atomic"
        return f"Fact({kind}, id={self.fact_id[:8]}..., trust={self.trust:.2f}, text={self.text[:40]!r})"
