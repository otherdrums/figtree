"""Deduplicate atomic facts across narratives using exact + semantic matching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch


def _normalize_text(text: str) -> str:
    """Normalize text for exact matching."""
    return " ".join(text.lower().strip().split())


class FactDeduper:
    """Deduplicate facts across multiple narratives.

    Two-step process:
    1. Exact text match (normalized) → same canonical fact
    2. Semantic similarity via embeddings → same canonical fact if above threshold
    """

    def __init__(
        self,
        embed_fn: Callable[[str], torch.Tensor] | None = None,
        semantic_threshold: float = 0.92,
    ):
        self.embed_fn = embed_fn
        self.semantic_threshold = semantic_threshold
        # canonical_id -> {text, sources: [(narrative_id, fact_id)], embedding}
        self.canonical: dict[str, dict] = {}
        # fact_id -> canonical_id
        self.fact_to_canonical: dict[str, str] = {}

    def load_narrative_facts(self, narrative_dir: Path) -> list[dict]:
        """Load all facts from a narrative directory."""
        facts = []
        facts_dir = narrative_dir / "facts"
        if not facts_dir.exists():
            return facts

        for fact_dir in sorted(facts_dir.iterdir()):
            if not fact_dir.is_dir():
                continue
            manifest = fact_dir / "manifest.json"
            if not manifest.exists():
                continue
            meta = json.loads(manifest.read_text())
            facts.append({
                "fact_id": meta["fact_id"],
                "narrative_id": meta["narrative_id"],
                "text": meta["text"],
                "start_pos": meta["start_pos"],
                "end_pos": meta["end_pos"],
                "token_count": meta["token_count"],
                "path": str(fact_dir),
            })
        return facts

    def deduplicate(self, all_facts: list[dict]) -> dict[str, dict]:
        """Deduplicate a list of facts. Returns canonical map."""
        # Step 1: Exact text match
        for fact in all_facts:
            norm = _normalize_text(fact["text"])
            matched = False
            for cid, canon in self.canonical.items():
                if _normalize_text(canon["text"]) == norm:
                    canon["sources"].append({
                        "narrative_id": fact["narrative_id"],
                        "fact_id": fact["fact_id"],
                        "path": fact["path"],
                    })
                    self.fact_to_canonical[fact["fact_id"]] = cid
                    matched = True
                    break
            if not matched:
                cid = fact["fact_id"]
                self.canonical[cid] = {
                    "text": fact["text"],
                    "sources": [{
                        "narrative_id": fact["narrative_id"],
                        "fact_id": fact["fact_id"],
                        "path": fact["path"],
                    }],
                    "embedding": None,
                }
                self.fact_to_canonical[fact["fact_id"]] = cid

        # Step 2: Semantic similarity (if embed_fn provided)
        if self.embed_fn:
            self._semantic_dedup()

        return self.canonical

    def _semantic_dedup(self):
        """Merge canonical facts that are semantically similar."""
        # Compute embeddings for canonical facts without them
        for cid, canon in self.canonical.items():
            if canon["embedding"] is None:
                canon["embedding"] = self.embed_fn(canon["text"])

        # Find similar pairs and merge
        cids = list(self.canonical.keys())
        merged = set()

        for i in range(len(cids)):
            cid1 = cids[i]
            if cid1 in merged:
                continue
            emb1 = self.canonical[cid1]["embedding"]
            for j in range(i + 1, len(cids)):
                cid2 = cids[j]
                if cid2 in merged:
                    continue
                emb2 = self.canonical[cid2]["embedding"]
                sim = _cosine_sim(emb1, emb2)
                if sim >= self.semantic_threshold:
                    # Merge cid2 into cid1
                    self.canonical[cid1]["sources"].extend(
                        self.canonical[cid2]["sources"]
                    )
                    # Update mappings
                    for src in self.canonical[cid2]["sources"]:
                        self.fact_to_canonical[src["fact_id"]] = cid1
                    merged.add(cid2)

        # Remove merged canonicals
        for cid in merged:
            del self.canonical[cid]

    def get_canonical_id(self, fact_id: str) -> str | None:
        return self.fact_to_canonical.get(fact_id)

    def get_shared_facts(self) -> list[dict]:
        """Return canonical facts that appear in multiple sources."""
        return [
            {"canonical_id": cid, **canon}
            for cid, canon in self.canonical.items()
            if len(canon["sources"]) > 1
        ]

    def get_unique_facts(self, narrative_id: str) -> list[dict]:
        """Return canonical facts unique to a given narrative."""
        return [
            {"canonical_id": cid, **canon}
            for cid, canon in self.canonical.items()
            if len(canon["sources"]) == 1
            and canon["sources"][0]["narrative_id"] == narrative_id
        ]


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two 1D tensors."""
    a = a.flatten()
    b = b.flatten()
    dot = torch.dot(a, b).item()
    norm_a = torch.norm(a).item()
    norm_b = torch.norm(b).item()
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def make_embed_fn(model, tokenizer) -> Callable[[str], torch.Tensor]:
    """Create an embedding function using the model's token embeddings."""
    device = next(model.parameters()).device

    def embed(text: str) -> torch.Tensor:
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        with torch.no_grad():
            emb = model.model.embed_tokens(input_ids)
        # Mean pooling
        return emb[0].mean(dim=0).float().cpu()

    return embed
