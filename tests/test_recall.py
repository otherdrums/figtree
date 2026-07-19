"""Tests for recall verification (no GPU required)."""

from __future__ import annotations

from figtree.recall import (
    build_recall_prompt,
    extract_atoms,
    missing_atoms,
    recall_score,
)


def test_extract_atoms_finds_numbers_and_entities():
    text = (
        "2,700 delegates from 130 countries met. A $2 trillion fund and "
        "150 billion dollars were pledged. The WTO and MSCI World Index rose 2.1%."
    )
    atoms = extract_atoms(text)
    joined = " ".join(atoms)
    for expected in ["2700", "130", "2 trillion", "150 billion", "2.1%", "wto",
                     "msci"]:
        assert expected in joined, f"expected atom {expected!r} in {atoms}"


def test_missing_atoms_detects_dropped_figure():
    source = "Davos had 2,700 delegates and a 2 trillion dollar fund was pledged."
    generated = "Davos had many delegates and a fund was pledged."
    miss = missing_atoms(source, generated)
    assert "2700" in " ".join(miss)
    assert "2 trillion" in " ".join(miss)


def test_recall_score_perfect_when_all_present():
    source = "2,700 delegates met. 150 billion was pledged."
    assert recall_score(source, source) == 1.0


def test_recall_score_zero_when_all_missing():
    source = "2,700 delegates met."
    assert recall_score(source, "nothing relevant") == 0.0


def test_build_recall_prompt_lists_missing():
    p = build_recall_prompt(["2700", "2 trillion"])
    assert "2700" in p and "2 trillion" in p


def test_numeric_core_match_counts_as_recalled():
    # "2,700" recalled as "2700" (comma stripped) should not be flagged missing.
    miss = missing_atoms("2,700 delegates", "there were 2700 delegates")
    assert miss == []
