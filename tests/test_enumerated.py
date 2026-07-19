"""Tests for long-source enumerated chunking (no GPU/model required)."""

from __future__ import annotations

from figtree.generate import _chunk_text


class _WordTokenizer:
    """Minimal word tokenizer standing in for a real HF tokenizer."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(ids)


def test_short_text_is_single_chunk():
    tok = _WordTokenizer()
    chunks = _chunk_text(tok, "alpha beta gamma", chunk_tokens=10, overlap_tokens=2)
    assert chunks == ["alpha beta gamma"]


def test_long_text_splits_into_multiple_chunks():
    tok = _WordTokenizer()
    text = " ".join(f"w{i}" for i in range(20))
    chunks = _chunk_text(tok, text, chunk_tokens=5, overlap_tokens=1)
    assert len(chunks) > 1
    # Every original word must appear in at least one chunk.
    seen = " ".join(chunks).split()
    for i in range(20):
        assert f"w{i}" in seen


def test_empty_text_returns_empty():
    tok = _WordTokenizer()
    assert _chunk_text(tok, "   ", chunk_tokens=5, overlap_tokens=1) == []
