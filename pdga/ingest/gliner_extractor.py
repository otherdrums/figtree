"""GLiNER-based entity extraction with model-driven chunking.

Replaces regex-based fact detection with zero-shot NER via GLiNER.
Finds "who what where when how much" entities, then uses the model's
own residual stream to determine semantic spans around each entity.

Dynamic labels: a universal label set is used on each document.
Labels that produce zero entities are pruned. The surviving labels
become the document's self-documenting fact categories.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from pdga.ingest.sparse import cosine_distance

UNIVERSAL_LABELS = [
    "person", "organization", "location", "date", "time",
    "money", "percent", "quantity", "event", "law",
    "treaty", "agreement", "policy", "regulation", "provision",
    "tax", "tariff", "levy", "revenue", "fund", "investment", "index",
    "carbon", "emission", "climate", "energy",
    "pharmaceutical", "patent", "technology", "product", "industry", "company",
    "country", "nation", "agency", "commission", "ambassador",
    "market", "currency", "trade", "import", "export", "gdp", "sanction",
    "summit", "conference", "negotiation", "resolution", "amendment", "clause",
    "bill", "framework", "mechanism", "standard",
]


class GlinerExtractor:
    """Zero-shot entity extraction using GLiNER.

    Loaded once per session. Uses a universal label set and prunes
    unused labels per document to create a dynamic label profile.

    Default: gliner_small-v2.1 (~100MB, <0.1s per document).
    Switch to medium for higher recall in production.
    """

    def __init__(self, model_name: str = "urchade/gliner_small-v2.1"):
        from gliner import GLiNER
        self.model = GLiNER.from_pretrained(model_name)
        self.model_name = model_name

    def extract(
        self,
        text: str,
        labels: list[str] | None = None,
        score_threshold: float = 0.3,
    ) -> tuple[list[dict], list[str]]:
        """Extract entities from text, returning filtered entities and active labels.

        Args:
            text: Document text
            labels: Label set to use (defaults to UNIVERSAL_LABELS)
            score_threshold: Minimum confidence score to keep an entity

        Returns:
            (entities, active_labels)
            entities: list of {start, end, label, text, score} dicts
            active_labels: labels that produced at least one entity
        """
        if labels is None:
            labels = UNIVERSAL_LABELS

        raw = self.model.predict_entities(text, labels, threshold=score_threshold)
        raw = sorted(raw, key=lambda e: e["start"])
        entities = self._deduplicate(raw)
        active_labels = sorted(set(e["label"] for e in entities))
        return entities, active_labels

    def _deduplicate(self, entities: list[dict]) -> list[dict]:
        if not entities:
            return []
        result = [entities[0]]
        for e in entities[1:]:
            prev = result[-1]
            if e["start"] <= prev["end"] and e["label"] == prev["label"]:
                if e["score"] > prev["score"] or e["end"] > prev["end"]:
                    result[-1] = e
            else:
                result.append(e)
        return result


def semantic_span(
    entity_token_pos: int,
    residuals: np.ndarray,
    stability_threshold: float = 0.25,
    max_span: int = 12,
    min_span: int = 3,
) -> tuple[int, int]:
    """Find the semantic span around an entity position using residual stability.

    Walks backward and forward from the entity position in the residual
    stream. Stops when the residual shifts significantly (cosine distance
    exceeds threshold) — indicating a semantic boundary. This uses the
    model's own internal representation instead of sentence splitting.

    Args:
        entity_token_pos: Token position of the entity within the window
        residuals: (seq_len, hidden_size) per-token residuals at crystal layer
        stability_threshold: Cosine distance above which to stop expanding
        max_span: Maximum tokens to expand in either direction
        min_span: Minimum span size in each direction

    Returns:
        (start_pos, end_pos) — start is inclusive, end is exclusive
    """
    seq_len = residuals.shape[0]
    pos = min(max(entity_token_pos, 0), seq_len - 1)

    start = pos
    for i in range(max(0, pos - min_span), max(0, pos - max_span), -1):
        if i <= 0:
            start = 0
            break
        dist = cosine_distance(residuals[i], residuals[i - 1])
        if dist > stability_threshold and pos - i >= min_span:
            start = i
            break
        start = i

    end = pos + 1
    for i in range(pos + 1, min(seq_len, pos + max_span)):
        if i >= seq_len - 1:
            end = seq_len
            break
        dist = cosine_distance(residuals[i], residuals[i - 1])
        if dist > stability_threshold and i - pos >= min_span:
            end = i + 1
            break
        end = i + 1

    return start, end


def map_char_spans_to_token_positions(
    entity: dict,
    window_tokens: list[int],
    tokenizer,
) -> Optional[tuple[int, int]]:
    """Map a character-level entity span to token positions within a window.

    Returns (token_start, token_end) or None if mapping fails.
    """
    char_start = entity["start"]
    char_end = entity["end"]

    prefix_text = tokenizer.decode(window_tokens[:1])

    cumulative = 0
    token_start = None
    token_end = None

    for i, tok in enumerate(window_tokens):
        tok_text = tokenizer.decode([tok])
        if token_start is None and cumulative + len(prefix_text) if i == 0 else cumulative >= char_start:
            pass

        tok_len = len(tok_text)

        if token_start is None and char_start < cumulative + tok_len:
            token_start = i
        if char_end <= cumulative + tok_len:
            token_end = i + 1
            break
        cumulative += tok_len

    if token_start is not None and token_end is not None:
        return (token_start, token_end)
    return None


def map_entity_to_token_position(
    entity: dict,
    window_tokens: list[int],
    tokenizer,
) -> Optional[int]:
    """Find the token position within a window that best contains this entity.

    Uses approximate character-position-to-token mapping.
    Returns a single best token index, or None.
    """
    result = map_char_spans_to_token_positions(entity, window_tokens, tokenizer)
    if result is not None:
        start, end = result
        return (start + end) // 2
    return None


def extract_fact_entities_for_window(
    window_tokens: list[int],
    residuals: np.ndarray,
    gliner_extractor: GlinerExtractor,
    tokenizer,
    score_threshold: float = 0.3,
    span_stability: float = 0.15,
) -> tuple[list[list[int]], set[int], list[str]]:
    """Extract entities from a window and compute their semantic spans.

    Returns:
        (fact_chunks, entity_token_positions, dynamic_labels)
    """
    window_text = tokenizer.decode(window_tokens)
    entities, dynamic_labels = gliner_extractor.extract(window_text, score_threshold=score_threshold)

    entity_positions: set[int] = set()
    fact_chunks: list[list[int]] = []
    chunk_ranges_covered: list[tuple[int, int]] = []

    for entity in entities:
        token_pos = map_entity_to_token_position(entity, window_tokens, tokenizer)
        if token_pos is None:
            continue

        span_start, span_end = semantic_span(
            token_pos, residuals,
            stability_threshold=span_stability,
        )

        span_start = max(0, span_start)
        span_end = min(len(window_tokens), span_end)

        chunk_ranges_covered.append((span_start, span_end))
        for p in range(span_start, span_end):
            entity_positions.add(p)

    if chunk_ranges_covered:
        chunk_ranges_covered.sort()
        merged = [chunk_ranges_covered[0]]
        for rng in chunk_ranges_covered[1:]:
            prev = merged[-1]
            if rng[0] <= prev[1]:
                merged[-1] = (prev[0], max(prev[1], rng[1]))
            else:
                merged.append(rng)
        for start, end in merged:
            if start < end:
                fact_chunks.append(window_tokens[start:end])

    return fact_chunks, entity_positions, dynamic_labels
