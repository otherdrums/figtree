"""Auto-generate graph edges between facts and narratives."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pdga.graph.edges import EdgeOps, EdgeType
from pdga.db.store import DeltaDB


class AutoEdgeGenerator:
    """Generate edges automatically from fact relationships."""

    EDGE_TYPES = {
        "part_of": "part_of",
        "supports": "supports",
        "contradicts": "contradicts",
        "same_entity": "same_entity",
    }

    def __init__(self, db: DeltaDB):
        self.db = db
        self.edge_ops = EdgeOps(db)

    def generate_all(
        self,
        narratives: list[Path],
        canonical_map: dict[str, dict],
        fact_to_canonical: dict[str, str],
    ):
        """Generate all edges for a set of narratives and deduplicated facts."""
        # PART_OF: each fact -> its narrative
        self._generate_part_of(narratives, fact_to_canonical)

        # SUPPORTS: canonical facts with multiple sources
        self._generate_supports(canonical_map)

        # SAME_ENTITY: facts sharing entities
        self._generate_same_entity(canonical_map)

        # CONTRADICTS: detect contradictions via entity-value pairs
        self._generate_contradictions(canonical_map)

    def _generate_part_of(self, narratives: list[Path], fact_to_canonical: dict):
        """Link each fact to its parent narrative."""
        for narr_dir in narratives:
            narrative_json = narr_dir / "narrative.json"
            if not narrative_json.exists():
                continue
            meta = json.loads(narrative_json.read_text())
            narr_id = meta["narrative_id"]

            facts_dir = narr_dir / "facts"
            if not facts_dir.exists():
                continue

            for fact_dir in facts_dir.iterdir():
                if not fact_dir.is_dir():
                    continue
                manifest = fact_dir / "manifest.json"
                if not manifest.exists():
                    continue
                fact_meta = json.loads(manifest.read_text())
                fact_id = fact_meta["fact_id"]
                canonical_id = fact_to_canonical.get(fact_id, fact_id)

                self.edge_ops.add(
                    canonical_id,
                    EdgeType.PART_OF,
                    narr_id,
                    weight=1.0,
                    metadata={"fact_text": fact_meta.get("text", "")[:200]},
                )

    def _generate_supports(self, canonical_map: dict[str, dict]):
        """Link canonical facts to their supporting source facts."""
        for cid, canon in canonical_map.items():
            sources = canon.get("sources", [])
            if len(sources) <= 1:
                continue
            # Create edges between all pairs of source facts
            for i in range(len(sources)):
                for j in range(i + 1, len(sources)):
                    src1 = sources[i]["fact_id"]
                    src2 = sources[j]["fact_id"]
                    self.edge_ops.add(
                        src1,
                        EdgeType.SUPPORTS,
                        src2,
                        weight=1.0,
                        metadata={"canonical_id": cid},
                    )

    def _generate_same_entity(self, canonical_map: dict[str, dict]):
        """Link facts that share entities."""
        # Extract entities from each canonical fact
        canonical_entities = {}
        for cid, canon in canonical_map.items():
            entities = self._extract_entities(canon["text"])
            canonical_entities[cid] = entities

        # Find shared entities
        cids = list(canonical_map.keys())
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                cid1, cid2 = cids[i], cids[j]
                shared = canonical_entities[cid1] & canonical_entities[cid2]
                if shared:
                    weight = len(shared) / max(
                        len(canonical_entities[cid1]),
                        len(canonical_entities[cid2]),
                    )
                    self.edge_ops.add(
                        cid1,
                        EdgeType.SAME_ENTITY,
                        cid2,
                        weight=weight,
                        metadata={"shared_entities": list(shared)},
                    )

    def _generate_contradictions(self, canonical_map: dict[str, dict]):
        """Detect contradictions using entity + negation patterns."""
        # Simple heuristic: same entities, but one contains negation words
        negation_words = {"not", "no", "never", "nothing", "none", "neither",
                          "nowhere", "hardly", "scarcely", "barely", "deny",
                          "refute", "reject", "dismiss", "false", "fake",
                          "fabricated", "concealed", "hidden", "covert"}

        cids = list(canonical_map.keys())
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                cid1, cid2 = cids[i], cids[j]
                text1 = canonical_map[cid1]["text"].lower()
                text2 = canonical_map[cid2]["text"].lower()

                # Check for shared significant entities
                entities1 = self._extract_entities(text1)
                entities2 = self._extract_entities(text2)
                shared = entities1 & entities2

                if not shared:
                    continue

                # Check if one has negation and the other doesn't
                has_neg1 = any(w in text1 for w in negation_words)
                has_neg2 = any(w in text2 for w in negation_words)

                if has_neg1 != has_neg2:
                    # Potential contradiction
                    self.edge_ops.add(
                        cid1,
                        EdgeType.CONTRADICTS,
                        cid2,
                        weight=0.7,
                        metadata={
                            "shared_entities": list(shared),
                            "negation_1": has_neg1,
                            "negation_2": has_neg2,
                        },
                    )

    @staticmethod
    def _extract_entities(text: str) -> set[str]:
        """Extract potential named entities using simple heuristics."""
        # Look for capitalized phrases of 1-3 words
        entities = set()
        words = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){0,2}\b', text)
        for w in words:
            w_clean = w.strip()
            if len(w_clean) > 2 and w_clean.lower() not in {
                "the", "a", "an", "this", "that", "these", "those",
                "he", "she", "it", "they", "we", "you", "i",
                "was", "were", "is", "are", "been", "being",
                "have", "has", "had", "do", "does", "did",
                "will", "would", "could", "should", "may", "might",
                "can", "must", "shall", "said", "says", "say",
                "told", "tells", "tell", "according", "also",
                "however", "therefore", "furthermore", "moreover",
                "nevertheless", "meanwhile", "additionally",
                "consequently", "nonetheless", "notwithstanding",
            }:
                entities.add(w_clean.lower())
        return entities
