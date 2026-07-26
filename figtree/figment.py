"""Figment — the universal unit of knowledge in Figtree.

Everything is a Figment. Each figment carries a free-form ``kind`` that lets
applications build their own hierarchies (article, paragraph, sentence, role,
edge, trust, etc.). A few kinds are conventional:

- ``atomic`` — default leaf figment
- ``image`` / ``article`` — container figments
- ``edge`` — relationship figments
- ``trust`` — trust assertion figments

Figments are persisted as rows in a LanceDB table (see ``figtree/lancedb_store.py``);
K/V caches live outside the row as external quantized blobs managed by
``figtree/kv_cache_manager.py``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

# Conventional built-in kinds. Applications are free to invent more.
KIND_ATOMIC = "atomic"
KIND_IMAGE = "image"
KIND_ARTICLE = "article"
KIND_PARAGRAPH = "paragraph"
KIND_SENTENCE = "sentence"
KIND_ROLE = "role"
KIND_EDGE = "edge"
KIND_TRUST = "trust"
KIND_CONTAINER = "container"

CONTAINER_KINDS = {
    KIND_IMAGE, KIND_ARTICLE, KIND_PARAGRAPH, KIND_SENTENCE, KIND_CONTAINER,
}


@dataclass
class Figment:
    """A single unit of knowledge. Kinds are application-defined; the library
    only treats ``kind`` as a filterable label, plus a few conventional helpers.
    """

    figment_id: str             # SHA-256(text)[:16]
    text: str                   # Natural language statement
    boundary: np.ndarray        # (hidden_size,) float32 — crystal layer
    meta: dict[str, Any]        # edge_type, about_figment, etc.
    children: list[str]         # Child figment IDs
    sources: list[str]          # Parent figment IDs
    trust: float                # Cached trust score
    kind: str = KIND_ATOMIC     # Free-form type label
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
        kind: str | None = None,
        boundaries: np.ndarray | None = None,
        boundary_emb: np.ndarray | None = None,
        figment_id: str | None = None,
    ) -> "Figment":
        """Factory: auto-generate figment_id from text (or use a provided id).

        A provided ``figment_id`` enables idempotent, re-runnable figments
        (e.g. one canonical trust Figment per source that can be overwritten).
        """
        figment_id = figment_id or hashlib.sha256(text.encode()).hexdigest()[:16]
        # Backward compatibility: legacy images set ``is_image`` in meta.
        if kind is None:
            meta = meta or {}
            if meta.get("edge_type"):
                if meta.get("edge_type") == "trust":
                    kind = KIND_TRUST
                else:
                    kind = KIND_EDGE
            elif meta.get("is_image") or len(children or []) > 0:
                kind = KIND_IMAGE
            else:
                kind = KIND_ATOMIC
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
            kind=kind,
        )

    def is_container(self) -> bool:
        """True if this figment has children or a container-like kind."""
        return self.kind in CONTAINER_KINDS or len(self.children) > 0

    def is_image(self) -> bool:
        """Legacy helper: True if this figment is an image/article container.

        Historically this was any figment with children. With ``kind`` it is now
        explicit: article, paragraph, image, etc. are containers, but a generic
        edge with children is not an image.
        """
        return self.kind in {KIND_IMAGE, KIND_ARTICLE, KIND_PARAGRAPH, KIND_CONTAINER} or (
            self.meta.get("is_image") is True
        )

    def is_edge(self) -> bool:
        """True if this figment represents a graph edge."""
        return self.kind == KIND_EDGE or self.meta.get("edge_type") is not None

    def is_trust_assertion(self) -> bool:
        """True if this figment represents a trust score."""
        return self.kind == KIND_TRUST or self.meta.get("edge_type") == "trust"

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
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Figment":
        """Reconstruct a Figment from :meth:`to_dict` output."""
        boundary = np.asarray(d["boundary"], dtype=np.float32)
        boundaries = d.get("boundaries")
        boundary_emb = d.get("boundary_emb")
        meta = dict(d.get("meta", {}))
        children = list(d.get("children", []))
        kind = d.get("kind", "")
        if not kind:
            # Backward compatibility: infer kind from old metadata.
            if meta.get("edge_type") == "trust":
                kind = KIND_TRUST
            elif meta.get("edge_type"):
                kind = KIND_EDGE
            elif meta.get("is_image") or len(children) > 0:
                kind = KIND_IMAGE
            else:
                kind = KIND_ATOMIC
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
            meta=meta,
            children=children,
            sources=list(d.get("sources", [])),
            trust=float(d.get("trust", 0.5)),
            kind=kind,
        )

    def __repr__(self) -> str:
        return f"Figment({self.kind}, id={self.figment_id[:8]}..., trust={self.trust:.2f}, text={self.text[:40]!r})"
