"""Graph module: all edges, trust, and relationships are first-class Figments.

Instead of a separate graph database, every relationship is a Figment:
- "Figment A supports Figment B" -> Figment with meta["edge_type"] = "supports"
- "Figment C has trust 0.95" -> Figment with meta["edge_type"] = "trust"
- "Figment D contradicts Figment E" -> Figment with meta["edge_type"] = "contradicts"

Trust design (recall-aware + mutable):
- Trust is SOURCE-based. Each source carries a base trust score; the system
  recalls every perspective and explains each one's credibility from (a) the
  source's base trust and (b) how many *other* sources corroborate or contradict
  its claims.
- Trust Figments are canonical per source_id and are persisted to the LanceDB
  store by `propagate_trust(store=...)` so the score is re-runnable and editable:
  a future "accuracy proven -> trust up" step only edits `meta["score"]` and
  re-runs `propagate_trust`, overwriting the same row. No schema change needed.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from figtree.figment import Figment

# Lexicons used for lightweight, explainable contradiction detection.
_POSITIVE_CUES = {
    "unanimous", "binding", "endorsed", "adopted", "agreement", "landmark",
    "committed", "guarantees", "legally", "comprehensive", "historic",
}
_NEGATIVE_CUES = {
    "not binding", "no agreement", "failed", "vague", "contradict", "walked out",
    "unenforceable", "aspirational", "loopholes", "detained", "hidden", "refused",
    "critics", "skeptic", "doubts", "disputed",
}


class Figtree:
    """Graph of figments — manages relationships, trust, and credibility."""

    def __init__(self, figments: list[Figment] | None = None, store=None):
        self.figments: dict[str, Figment] = {}
        self.canonical: dict[str, str] = {}
        self.store = store
        if figments:
            for f in figments:
                self.add_figment(f)
        # source_id -> list of atomic figment ids
        self.by_source: dict[str, list[str]] = defaultdict(list)
        self._reindex_sources()

    def add_figment(self, figment: Figment) -> None:
        """Register a figment in the in-memory index (keyed by figment_id)."""
        self.figments[figment.figment_id] = figment

    def _reindex_sources(self) -> None:
        self.by_source = defaultdict(list)
        for fid, fig in self.figments.items():
            if fig.is_edge() or fig.is_trust_assertion():
                continue
            src = fig.meta.get("source_id", "")
            if src:
                self.by_source[src].append(fid)

    # ------------------------------------------------------------------ #
    # Credibility model
    # ------------------------------------------------------------------ #
    def _source_base_trust(self) -> dict[str, float]:
        """Map source_id -> immutable base trust (from the image figment).

        Base trust is fixed at ingest time (stored on the image figment's
        `meta["base_trust"]`). Persisted `trust:{src}` figments carry the
        *adjusted* score and must NOT be read back as base, otherwise trust
        would drift on every reload.
        """
        base: dict[str, float] = {}
        for src in self.by_source:
            src_base = 0.5
            for fid in self.by_source[src]:
                fig = self.figments[fid]
                # the image figment (parent) carries base_trust
                if fig.meta.get("is_image"):
                    src_base = float(fig.meta.get("base_trust", fig.trust))
                    break
            base[src] = src_base
        return base

    def _entities(self, fig: Figment) -> set[str]:
        tokens = fig.text.split()
        return {t.strip(".,!?;:").lower() for t in tokens if t and t[0].isupper()}

    def _cue(self, fig: Figment) -> str:
        t = fig.text.lower()
        if any(c in t for c in _NEGATIVE_CUES):
            return "negative"
        if any(c in t for c in _POSITIVE_CUES):
            return "positive"
        return "neutral"

    def analyze_sources(self) -> dict[str, dict]:
        """Compute per-source corroboration/contradiction.

        Distinguishes three kinds of cross-source relationship:
        - ``related``: sources discuss the same entities (topic overlap) — neutral;
          this is NOT a claim of agreement.
        - ``agreeing``: same entities AND both sources carry the *same explicit
          non-neutral* sentiment cue (both positive => shared endorsement; both
          negative => shared criticism). Genuine alignment only.
        - ``contradicting``: same entities AND opposite sentiment cues.

        Returns source_id -> {
            "base_trust", "related": [sources], "agreeing": [sources],
            "contradicting": [sources], "corroborated_frac",
            "adjusted_trust", "rationale": str
        }
        """
        base = self._source_base_trust()
        sources = list(self.by_source.keys())

        related: dict[str, set] = defaultdict(set)
        agreeing: dict[str, set] = defaultdict(set)
        contradicting: dict[str, set] = defaultdict(set)
        fig_corroborated: dict[str, bool] = {}

        src_figs = {s: [self.figments[i] for i in self.by_source[s]] for s in sources}
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                a, b = sources[i], sources[j]
                for fa in src_figs[a]:
                    for fb in src_figs[b]:
                        shared = self._entities(fa) & self._entities(fb)
                        if not shared:
                            continue
                        related[a].add(b)
                        related[b].add(a)
                        cue_a, cue_b = self._cue(fa), self._cue(fb)
                        if cue_a != cue_b and "neutral" not in (cue_a, cue_b):
                            # Opposite explicit sentiment => contradiction.
                            contradicting[a].add(b)
                            contradicting[b].add(a)
                        elif cue_a == cue_b and cue_a != "neutral":
                            # Both share the same explicit (pos/neg) stance =>
                            # genuine alignment on that point.
                            agreeing[a].add(b)
                            agreeing[b].add(a)
                            fig_corroborated[fa.figment_id] = True
                            fig_corroborated[fb.figment_id] = True

        result: dict[str, dict] = {}
        for s in sources:
            figs = src_figs[s]
            corr = round(sum(1 for f in figs if fig_corroborated.get(f.figment_id, False)) / max(1, len(figs)), 2)
            b_trust = base.get(s, 0.5)
            adj = 0.6 * b_trust + 0.4 * corr
            if contradicting[s]:
                adj *= 0.85
            adj = float(min(1.0, max(0.0, adj)))
            rationale = (
                f"base_trust={b_trust:.2f}, related to {sorted(related[s]) or 'none'} "
                f"(topic overlap), agrees with {sorted(agreeing[s]) or 'none'}, "
                f"contradicted by {sorted(contradicting[s]) or 'none'} "
                f"({corr * 100:.0f}% of its claims corroborated)"
            )
            result[s] = {
                "base_trust": b_trust,
                "related": sorted(related[s]),
                "agreeing": sorted(agreeing[s]),
                "contradicting": sorted(contradicting[s]),
                "corroborated_frac": corr,
                "adjusted_trust": adj,
                "rationale": rationale,
            }
        return result

    # ------------------------------------------------------------------ #
    # Edges
    # ------------------------------------------------------------------ #
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
            caps = {t.strip(".,!?;:") for t in fig.text.split() if t and t[0].isupper()}
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

    def propagate_trust(self, store=None) -> list[dict]:
        """Recompute + persist canonical trust Figments per source.

        Idempotent: a single trust Figment per source_id is (re)created with a
        deterministic id and overwrites the previous row in the store, so future
        trust adjustments only need to edit `meta["score"]` and re-run. Requires a
        ``store`` (LanceDB-backed persistence).
        """
        if store is None:
            raise ValueError(
                "store is required: trust Figments are persisted to a LanceDB store. "
                "Pass a FigmentStore from figtree.lancedb_store.connect()."
            )
        analysis = self.analyze_sources()
        updates = []
        for src, info in analysis.items():
            fid = f"trust:{src}"
            trust_fig = Figment.create(
                text=f"Source {src} has adjusted trust {info['adjusted_trust']:.2f} "
                     f"(base {info['base_trust']:.2f})",
                boundary=np.zeros(1, dtype=np.float32),
                meta={
                    "edge_type": "trust",
                    "source_id": src,
                    "score": info["adjusted_trust"],
                    "base_trust": info["base_trust"],
                    "related": info["related"],
                    "agreeing": info["agreeing"],
                    "contradicting": info["contradicting"],
                    "corroborated_frac": info["corroborated_frac"],
                    "rationale": info["rationale"],
                },
                figment_id=fid,
                trust=info["adjusted_trust"],
            )
            for f in self.by_source.get(src, []):
                self.figments[f].trust = info["adjusted_trust"]
            self.figments[fid] = trust_fig
            updates.append({"source_id": src, "trust": info["adjusted_trust"], **info})

            store.upsert_one(trust_fig)

        return updates

    # ------------------------------------------------------------------ #
    # Trust-aware retrieval / explanation
    # ------------------------------------------------------------------ #
    def build_trust_aware_context(self, query: str, limit: int = 6) -> dict:
        """Recall all perspectives relevant to `query`, with credibility notes.

        Returns {
            "query",
            "by_source": {src: {
                "trust", "rationale", "figments",
                "related", "agreeing", "contradicting"  # source-id lists
            }},
            "rationale": str,   # synthesized credibility explanation
        }
        """
        analysis = self.analyze_sources()
        qwords = set(query.lower().split())
        scored = []
        for src, fids in self.by_source.items():
            for f in fids:
                fig = self.figments[f]
                if fig.is_edge() or fig.is_trust_assertion():
                    continue
                overlap = len(qwords & set(fig.text.lower().split()))
                if overlap > 0:
                    scored.append((overlap, src, fig.text))
        scored.sort(key=lambda x: x[0], reverse=True)

        by_source: dict[str, dict] = {}
        for _, src, text in scored[:limit]:
            info = analysis.get(src, {})
            by_source.setdefault(src, {
                "trust": info.get("adjusted_trust", 0.5),
                "rationale": info.get("rationale", ""),
                "figments": [],
                "related": info.get("related", []),
                "agreeing": info.get("agreeing", []),
                "contradicting": info.get("contradicting", []),
            })
            by_source[src]["figments"].append(text)

        parts = []
        for src, d in by_source.items():
            parts.append(f"{src} (trust {d['trust']:.2f}): {d['rationale']}")
        rationale = "\n".join(parts) if parts else "No relevant figments recalled."

        return {"query": query, "by_source": by_source, "rationale": rationale}

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
