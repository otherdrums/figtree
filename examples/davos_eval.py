"""Shared evaluation harness for the Davos multi-narrative task.

Used by both the Figtree demo and the conventional RAG baseline so the two are
compared on identical queries and metrics:

- factual fidelity: do the generated answers reproduce each source's key figures?
- contradiction awareness: does a cross-source answer note disagreement?
- VRAM peak and wall-clock latency (measured by the caller).

Metrics are intentionally simple and honest (mirroring AGENTS.md's candor about
the small target GPU) — this is measurement, not a leaderboard.
"""

from __future__ import annotations

import re

# Three conflicting narratives about one event (Davos). Each carries a few
# distinctive, checkable figures so fidelity is measurable.
SOURCES = {
    "pro_globalist": {"name": "Reuters-style", "trust": 0.95},
    "anti_globalist": {"name": "Guardian-style", "trust": 0.60},
    "conspiracy": {"name": "Fringe Blog", "trust": 0.15},
}

# Queries grouped by what they exercise.
QUERIES = {
    # Per-source factual recall — the model should reproduce each narrative's
    # own figures verbatim.
    "q_pro": "According to the pro-globalist account, how many delegates attended and what was committed?",
    "q_anti": "According to the anti-globalist account, what was the real cost and who benefited?",
    "q_conspiracy": "According to the fringe blog, what was secretly decided at Davos?",
    # Cross-source synthesis + contradiction awareness.
    "q_conflict": "Compare the three accounts of Davos. Where do they disagree and what hard numbers differ?",
    "q_synth": "Summarize what happened at Davos using only facts stated by at least two sources.",
}

# Source-specific checkable figures for fidelity scoring. A query is "faithful"
# to a source if the generated text contains that source's distinctive figures.
SOURCE_FIGURES = {
    "pro_globalist": ["3,000", "3000", "2 billion", "2bn", "pledge", "pledged"],
    "anti_globalist": ["50 million", "50m", "elite", "luxury", "caviar"],
    "conspiracy": ["secret", "treaty", "15 families", "climate lock"],
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def fidelity_score(output: str, source_id: str) -> float:
    """Fraction of a source's distinctive figures present in the output (0..1)."""
    figs = SOURCE_FIGURES.get(source_id)
    if not figs:
        return 0.0
    out = normalize(output)
    hits = sum(1 for f in figs if f in out)
    return hits / len(figs)


def contradiction_aware(output: str) -> bool:
    """Heuristic: does the text acknowledge disagreement between sources?"""
    out = normalize(output)
    cues = [
        "disagree", "conflict", "contrast", "differ", "contradict", "versus",
        "while the", "whereas", "on the other hand", "in contrast", "differ",
    ]
    return any(c in out for c in cues)


def vram_peak_mb() -> float | None:
    try:
        import torch
    except Exception:
        return None
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return None
