"""Graph module: all edges, trust, and relationships are first-class Figments.

Instead of a separate graph database, every relationship is a Figment:
- "Figment A supports Figment B" -> Figment with meta["edge_type"] = "supports"
- "Figment C has trust 0.95" -> Figment with meta["edge_type"] = "trust"
- "Figment D contradicts Figment E" -> Figment with meta["edge_type"] = "contradicts"
"""

from __future__ import annotations


import numpy as np

from figtree.figment import Figment


class Figtree:
    """Graph of figments — manages relationships, dedup, and trust propagation."""

    def __init__(self, figments: list[Figment] | None = None):
        self.figments: dict[str, Figment] = {}
        self.canonical: dict[str, str] = {}
        if figments:
            for f in figments:
                self.add_figment(f)

    def add_figment(self, figment: Figment) -> None:
        self.figments[figment.figment_id] = figment

    def deduplicate(self) -> list[Figment]:
        """Deduplicate figments by exact text match + semantic boundary similarity."""
        edges = []
        fid_list = list(self.figments.keys())
        for i in range(len(fid_list)):
            for j in range(i + 1, len(fid_list)):
                a = self.figments[fid_list[i]]
                b = self.figments[fid_list[j]]

                if a.text.strip().lower() == b.text.strip().lower():
                    canonical = a.figment_id
                    duplicate = b.figment_id
                    self.canonical[duplicate] = canonical

                    edge = Figment.create(
                        text=f"Figment {canonical[:8]} supports figment {fid_list[j][:8]}",
                        boundary=a.boundary.copy(),
                        meta={"edge_type": "supports", "dedup": "exact"},
                        sources=[canonical],
                        children=[duplicate],
                    )
                    edges.append(edge)
                    continue

                if a.boundaries is not None and b.boundaries is not None:
                    sim = _boundary_similarity(a.boundaries, b.boundaries)
                    if sim > 0.95:
                        canonical = a.figment_id
                        duplicate = b.figment_id
                        self.canonical[duplicate] = canonical

                        edge = Figment.create(
                            text=f"Figment {canonical[:8]} supports figment {duplicate[:8]}",
                            boundary=a.boundary.copy(),
                            meta={"edge_type": "supports", "dedup": "semantic", "similarity": float(sim)},
                            sources=[canonical],
                            children=[duplicate],
                        )
                        edges.append(edge)

        return edges

    def create_edges(self) -> list[Figment]:
        """Auto-create SUPPORTS, SAME_ENTITY edges based on entity overlap."""
        edges = []

        entities: dict[str, list[str]] = {}
        for fid, fig in self.figments.items():
            if fig.is_edge() or fig.is_trust_assertion():
                continue
            tokens = fig.text.split()
            caps = {t.strip(".,!?;:") for t in tokens if t and t[0].isupper()}
            entities[fid] = list(caps)

        fid_list = list(entities.keys())
        for i in range(len(fid_list)):
            for j in range(i + 1, len(fid_list)):
                shared = set(entities[fid_list[i]]) & set(entities[fid_list[j]])
                if shared:
                    edge = Figment.create(
                        text=f"Figments share entities: {', '.join(list(shared)[:3])}",
                        boundary=self.figments[fid_list[i]].boundary.copy(),
                        meta={"edge_type": "same_entity", "entities": list(shared)},
                        sources=[fid_list[i]],
                        children=[fid_list[j]],
                    )
                    edges.append(edge)

        return edges

    def propagate_trust(self) -> list[dict]:
        """Propagate trust through the figment graph."""
        updates = []

        for fid, fig in self.figments.items():
            if not fig.is_trust_assertion():
                continue
            about = fig.meta.get("about_figment")
            score = fig.meta.get("score", 0.5)
            if about and about in self.figments:
                self.figments[about].trust = score
                updates.append({"figment_id": about, "trust": score, "source": "base"})

        for fid, fig in self.figments.items():
            if not fig.is_trust_assertion() and not fig.is_edge():
                self.figments[fid].trust = self.figments[fid].trust or 0.3

        children_map: dict[str, list[str]] = {}
        for fid, fig in self.figments.items():
            if fig.is_trust_assertion():
                about = fig.meta.get("about_figment")
                if about:
                    children_map.setdefault(fig.meta.get("about_figment"), []).append(fid)

        for fid, fig in self.figments.items():
            if not fig.is_edge() or fig.meta.get("edge_type") != "contradicts":
                continue
            for child in fig.children:
                if child in self.figments:
                    self.figments[child].trust *= 0.5
                    updates.append({"figment_id": child, "trust": self.figments[child].trust, "source": "contradiction"})

        return updates

    def get_top_figments(self, limit: int = 10) -> list[Figment]:
        """Return figments sorted by trust (highest first)."""
        candidates = [f for f in self.figments.values() if not f.is_edge() and not f.is_trust_assertion()]
        return sorted(candidates, key=lambda x: x.trust, reverse=True)[:limit]

    def get_edges(self, edge_type: str | None = None) -> list[Figment]:
        """Return all edge figments, optionally filtered by type."""
        result = [f for f in self.figments.values() if f.is_edge()]
        if edge_type:
            result = [f for f in result if f.meta.get("edge_type") == edge_type]
        return result


def _boundary_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.ravel().astype(np.float64)
    b_f = b.ravel().astype(np.float64)
    dot = np.dot(a_f, b_f)
    norm = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    return float(dot / norm) if norm > 0 else 0.0
