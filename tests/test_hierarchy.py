"""Tests for hierarchical Figments (summarized image boundaries).

Runs without a GPU by mocking the model forward pass, so it is CI-friendly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from figtree.figment import Figment
from figtree.summarize import summarize_image


def _fake_children(n=3, hidden=16):
    return [
        Figment.create(
            text=f"Statement number {i} about Davos.",
            boundary=np.random.randn(hidden).astype(np.float32),
            boundaries=np.random.randn(4, hidden).astype(np.float32),
            boundary_emb=np.random.randn(hidden).astype(np.float32),
        )
        for i in range(n)
    ]


def test_summarize_image_returns_summary_and_boundary():
    children = _fake_children()
    model = MagicMock()
    model.device = "cpu"
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "A concise summary of the statements."

    with patch("figtree.summarize.boundary_for_text") as bft:
        bft.return_value = (
            np.zeros(16, dtype=np.float32),
            np.zeros((4, 16), dtype=np.float32),
            np.zeros(16, dtype=np.float32),
        )
        summary, boundary, boundaries, emb = summarize_image(
            model, tokenizer, children
        )

    assert summary == "A concise summary of the statements."
    assert boundary.shape == (16,)
    assert boundaries.shape == (4, 16)
    assert emb.shape == (16,)
    # tokenizer.build_prompt_ids / generate were exercised
    assert tokenizer.decode.called


def test_summarize_image_fallback_on_empty_generation():
    children = _fake_children()
    model = MagicMock()
    model.device = "cpu"
    tokenizer = MagicMock()
    tokenizer.decode.return_value = ""  # empty generation -> fallback

    with patch("figtree.summarize.boundary_for_text") as bft:
        bft.return_value = (
            np.zeros(16, dtype=np.float32),
            np.zeros((4, 16), dtype=np.float32),
            np.zeros(16, dtype=np.float32),
        )
        summary, _, _, _ = summarize_image(model, tokenizer, children)

    # Falls back to the joined child texts rather than empty string.
    assert "Statement number 0" in summary


def test_image_figment_links_children():
    children = _fake_children()
    image = Figment.create(
        text="Image of the Davos narrative",
        boundary=children[0].boundary.copy(),
        meta={"is_image": True, "source_id": "s"},
        children=[c.figment_id for c in children],
        trust=0.5,
    )
    assert image.is_image()
    assert image.children == [c.figment_id for c in children]
