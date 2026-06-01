"""Trust propagation across the fact graph.

Source trust flows down to facts. Multi-source corroboration boosts trust.
Contradictions with high-trust facts penalize trust.
"""

from __future__ import annotations

from pdga.db.store import DeltaDB
from pdga.graph.edges import EdgeOps, EdgeType


class TrustPropagator:
    """Propagate trust scores through the fact graph.

    Algorithm (single pass):
    1. Base trust = mean of source narrative trusts
    2. Alignment boost: +0.05 per additional independent source
    3. Contradiction penalty: -0.1 * delta if high-trust fact contradicts
    4. Clamp to [0, 1]
    """

    ALIGNMENT_BOOST = 0.05
    CONTRADICTION_PENALTY = 0.15

    def __init__(self, db: DeltaDB):
        self.db = db
        self.edge_ops = EdgeOps(db)

    def propagate(self) -> dict[str, float]:
        """Compute propagated trust for all fact deltas.

        Returns: {delta_id -> trust_score}
        """
        # Load all facts and their source trusts
        facts = self.db.list_all(delta_type="fact")
        if not facts:
            return {}

        # Map fact_id -> list of source trusts
        fact_sources: dict[str, list[float]] = {}
        for fact in facts:
            fid = fact["delta_id"]
            trust = fact.get("trust", 0.5)
            fact_sources.setdefault(fid, []).append(trust)

        # Compute base trust (mean of source trusts)
        trust_scores: dict[str, float] = {}
        for fid, trusts in fact_sources.items():
            trust_scores[fid] = sum(trusts) / len(trusts)

        # Apply alignment boost for multi-source facts
        # Find SUPPORTS edges to count corroborating sources
        for fid in list(trust_scores.keys()):
            supports = self.edge_ops.get(fid, EdgeType.SUPPORTS)
            num_extra = len(supports)
            if num_extra > 0:
                boost = self.ALIGNMENT_BOOST * num_extra
                trust_scores[fid] = min(1.0, trust_scores[fid] + boost)

        # Apply contradiction penalty
        for fid in list(trust_scores.keys()):
            contradictions = self.edge_ops.find_contradictions(fid)
            for edge in contradictions:
                other_id = edge["target_id"] if edge["source_id"] == fid else edge["source_id"]
                other_trust = trust_scores.get(other_id, 0.5)
                my_trust = trust_scores[fid]
                if other_trust > my_trust:
                    penalty = self.CONTRADICTION_PENALTY * (other_trust - my_trust)
                    trust_scores[fid] = max(0.0, trust_scores[fid] - penalty)

        # Update database
        for fid, score in trust_scores.items():
            self.db.update_trust(fid, score)

        return trust_scores

    def get_narrative_trust(self, narrative_id: str) -> float:
        """Get the propagated trust for a narrative (mean of its facts)."""
        # Find all part_of edges from facts to this narrative
        rows = self.db.conn.execute(
            "SELECT source_id FROM edges WHERE target_id=? AND edge_type=?",
            (narrative_id, EdgeType.PART_OF.value),
        ).fetchall()
        if not rows:
            return 0.5
        fact_ids = [r[0] for r in rows]
        trusts = []
        for fid in fact_ids:
            fact = self.db.get(fid)
            if fact:
                trusts.append(fact.get("trust", 0.5))
        if not trusts:
            return 0.5
        return sum(trusts) / len(trusts)

    def rank_facts(self, limit: int = 20) -> list[dict]:
        """Return facts ranked by propagated trust (highest first)."""
        facts = self.db.list_all(delta_type="fact")
        facts.sort(key=lambda f: f.get("trust", 0.5), reverse=True)
        return facts[:limit]
