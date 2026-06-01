"""Graph module v2: all edges, trust, and relationships are first-class Facts.

Instead of a separate graph database, every relationship is a Fact:
- "Fact A supports Fact B" → Fact with meta["edge_type"] = "supports"
- "Fact C has trust 0.95" → Fact with meta["edge_type"] = "trust", meta["score"] = 0.95
- Deduplication: boundaries with cosine similarity > 0.92 → same canonical fact

Operations:
- deduplicate(facts): merge equivalent facts, return canonical list + edge facts
- create_edges(facts): generate SUPPORTS, CONTRADICTS, SAME_ENTITY edges
- propagate_trust(facts): compute trust scores via graph traversal
"""

from __future__ import annotations


import numpy as np

from pdga.fact.primitive import Fact


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D arrays."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class FactGraph:
    """Graph of facts where edges and trust are themselves facts."""

    DEDUP_THRESHOLD = 0.92
    ALIGNMENT_BOOST = 0.05
    CONTRADICTION_PENALTY = 0.15

    def __init__(self, facts: list[Fact] | None = None):
        self.facts: dict[str, Fact] = {}  # fact_id -> Fact
        self.canonical: dict[str, str] = {}  # fact_id -> canonical_id
        if facts:
            for f in facts:
                self.add_fact(f)

    def add_fact(self, fact: Fact) -> None:
        self.facts[fact.fact_id] = fact

    def deduplicate(self) -> list[Fact]:
        """Two-step dedup: exact text match + semantic boundary similarity.

        Returns new edge facts for SUPPORTS relationships.
        """
        # Step 1: exact text match
        text_to_canonical: dict[str, str] = {}
        for fid, f in self.facts.items():
            if f.is_edge() or f.is_trust_assertion():
                continue
            norm = " ".join(f.text.lower().split())
            if norm in text_to_canonical:
                self.canonical[fid] = text_to_canonical[norm]
            else:
                text_to_canonical[norm] = fid
                self.canonical[fid] = fid

        # Step 2: semantic similarity for unmatched facts
        unmatched = [fid for fid, cid in self.canonical.items() if fid == cid]
        merged = set()

        for i in range(len(unmatched)):
            fid1 = unmatched[i]
            if fid1 in merged:
                continue
            f1 = self.facts[fid1]
            for j in range(i + 1, len(unmatched)):
                fid2 = unmatched[j]
                if fid2 in merged:
                    continue
                f2 = self.facts[fid2]
                sim = cosine_similarity(f1.boundary, f2.boundary)
                if sim >= self.DEDUP_THRESHOLD:
                    self.canonical[fid2] = fid1
                    merged.add(fid2)

        # Create SUPPORTS edge facts for merged pairs
        edge_facts = []
        for fid, cid in self.canonical.items():
            if fid != cid:
                edge = Fact.create(
                    text=f"Fact {cid[:8]} supports fact {fid[:8]}",
                    boundary=self.facts[fid].boundary.copy(),
                    meta={"edge_type": "supports", "source": cid, "target": fid},
                    sources=[cid, fid],
                )
                edge_facts.append(edge)

        for ef in edge_facts:
            self.add_fact(ef)

        return edge_facts

    def create_edges(self) -> list[Fact]:
        """Auto-generate SAME_ENTITY and CONTRADICTS edges."""
        # Extract entities from each fact
        canonical_ids = set(self.canonical.values())
        edge_facts = []

        # SAME_ENTITY: shared noun phrases
        cids = list(canonical_ids)
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                f1 = self.facts[cids[i]]
                f2 = self.facts[cids[j]]
                shared = _extract_entities(f1.text) & _extract_entities(f2.text)
                if shared:
                    edge = Fact.create(
                        text=f"{f1.text[:40]} and {f2.text[:40]} share entities",
                        boundary=(f1.boundary + f2.boundary) / 2,
                        meta={"edge_type": "same_entity", "shared": list(shared),
                              "source": cids[i], "target": cids[j]},
                        sources=[cids[i], cids[j]],
                    )
                    edge_facts.append(edge)

        # CONTRADICTS: negation detection
        negation_words = {"not", "no", "never", "none", "neither", "hidden",
                          "concealed", "false", "fake", "fabricated", "covert"}
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                f1 = self.facts[cids[i]]
                f2 = self.facts[cids[j]]
                shared = _extract_entities(f1.text) & _extract_entities(f2.text)
                if not shared:
                    continue
                n1 = any(w in f1.text.lower() for w in negation_words)
                n2 = any(w in f2.text.lower() for w in negation_words)
                if n1 != n2:
                    edge = Fact.create(
                        text=f"{f1.text[:40]} contradicts {f2.text[:40]}",
                        boundary=(f1.boundary + f2.boundary) / 2,
                        meta={"edge_type": "contradicts", "shared": list(shared),
                              "source": cids[i], "target": cids[j]},
                        sources=[cids[i], cids[j]],
                    )
                    edge_facts.append(edge)

        for ef in edge_facts:
            self.add_fact(ef)

        return edge_facts

    def propagate_trust(self) -> dict[str, float]:
        """Compute trust scores for all facts.

        Algorithm:
        1. Base trust = source trust (from TrustFact about this fact)
        2. Alignment boost: +0.05 per additional corroborating source
        3. Contradiction penalty: -0.15 × delta if high-trust fact contradicts
        """
        scores: dict[str, float] = {}

        # Step 1: base trust from TrustFacts
        for fid, f in self.facts.items():
            if f.is_trust_assertion():
                about = f.meta.get("about_fact")
                score = f.meta.get("score", 0.5)
                if about:
                    scores[about] = score

        # Default trust for facts without explicit TrustFact
        for fid, f in self.facts.items():
            if fid not in scores and not f.is_edge() and not f.is_trust_assertion():
                scores[fid] = f.trust

        # Step 2: alignment boost from SUPPORTS edges
        for fid, f in self.facts.items():
            if f.meta.get("edge_type") == "supports":
                target = f.meta.get("target")
                source = f.meta.get("source")
                if target and source:
                    if target in scores:
                        scores[target] = min(1.0, scores[target] + self.ALIGNMENT_BOOST)

        # Step 3: contradiction penalty
        for fid, f in self.facts.items():
            if f.meta.get("edge_type") == "contradicts":
                src = f.meta.get("source")
                tgt = f.meta.get("target")
                if src and tgt and src in scores and tgt in scores:
                    if scores[src] > scores[tgt]:
                        delta = scores[src] - scores[tgt]
                        scores[tgt] = max(0.0, scores[tgt] - self.CONTRADICTION_PENALTY * delta)
                    elif scores[tgt] > scores[src]:
                        delta = scores[tgt] - scores[src]
                        scores[src] = max(0.0, scores[src] - self.CONTRADICTION_PENALTY * delta)

        # Update fact objects
        for fid, score in scores.items():
            if fid in self.facts:
                self.facts[fid].trust = score

        return scores

    def get_top_facts(self, limit: int = 10) -> list[Fact]:
        """Return facts sorted by trust (highest first)."""
        atomic = [f for f in self.facts.values() if not f.is_edge() and not f.is_trust_assertion()]
        atomic.sort(key=lambda f: f.trust, reverse=True)
        return atomic[:limit]

    def get_edges(self, edge_type: str | None = None) -> list[Fact]:
        """Return edge facts, optionally filtered by type."""
        edges = [f for f in self.facts.values() if f.is_edge()]
        if edge_type:
            edges = [e for e in edges if e.meta.get("edge_type") == edge_type]
        return edges


def _extract_entities(text: str) -> set[str]:
    """Extract potential named entities (capitalized phrases)."""
    import re
    entities = set()
    words = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){0,2}\b', text)
    for w in words:
        w_clean = w.strip().lower()
        if len(w_clean) > 2 and w_clean not in {
            "the", "a", "an", "this", "that", "these", "those",
            "he", "she", "it", "they", "we", "you", "i",
            "was", "were", "is", "are", "been", "being",
            "have", "has", "had", "will", "would", "could", "should",
        }:
            entities.add(w_clean)
    return entities
