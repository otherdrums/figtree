"""CPU-only tests for graph operations: trust propagation, dedup, edges.

Uses synthetic Figments (random boundaries, crafted text) so no language model
is required. Trust math is asserted against the documented formula
``0.6*base + 0.4*corroborated``, then ``*0.85`` if the source is contradicted.
"""

from __future__ import annotations

import numpy as np
import pytest

from figtree import Figment, Figtree, connect

HIDDEN = 16


def _atomic(text: str, source_id: str, seed: int) -> Figment:
    rng = np.random.default_rng(seed)
    return Figment.create(
        text, rng.standard_normal(HIDDEN).astype("float32"),
        meta={"source_id": source_id},
    )


def _image(source_id: str, base_trust: float) -> Figment:
    rng = np.random.default_rng(abs(hash(source_id)) % (2**31))
    return Figment.create(
        f"Article from {source_id} source.",
        rng.standard_normal(HIDDEN).astype("float32"),
        meta={"source_id": source_id, "is_image": True, "base_trust": base_trust},
    )


def test_deduplicate_exact_match():
    # Distinct casing -> distinct ids, but equal after lowercasing -> exact dedup.
    a = _atomic("The summit reached a deal.", "s1", 1)
    b = _atomic("the summit reached a deal.", "s1", 2)
    g = Figtree([a, b])
    edges = g.deduplicate()
    assert len(edges) == 1
    assert edges[0].meta["edge_type"] == "supports"
    assert edges[0].meta.get("dedup") == "exact"


def test_deduplicate_semantic_match():
    shared = np.random.default_rng(3).standard_normal((4, HIDDEN)).astype("float32")
    a = Figment.create(
        "Alpha note.", np.random.default_rng(10).standard_normal(HIDDEN).astype("float32"),
        boundaries=shared.copy(), meta={"source_id": "s1"},
    )
    b = Figment.create(
        "Beta note.", np.random.default_rng(11).standard_normal(HIDDEN).astype("float32"),
        boundaries=shared.copy(), meta={"source_id": "s1"},
    )
    g = Figtree([a, b])
    edges = g.deduplicate()
    assert any(e.meta.get("dedup") == "semantic" for e in edges)


def test_create_edges_same_entity():
    a = _atomic("The Paris summit concluded.", "s1", 4)
    b = _atomic("Paris announced a plan.", "s2", 5)
    g = Figtree([a, b])
    edges = g.create_edges()
    assert any(
        e.meta.get("edge_type") == "same_entity"
        and "paris" in [x.lower() for x in e.meta.get("entities", [])]
        for e in edges
    )


def test_single_source_trust_is_base_weighted():
    store = _tmp_store()
    src = "alpha"
    figs = [_image(src, 0.9), _atomic("The council endorsed the plan.", src, 1)]
    g = Figtree(figs, store=store)
    updates = g.propagate_trust(store=store)
    info = {u["source_id"]: u for u in updates}["alpha"]
    # No cross-source corroboration -> corroborated_frac = 0.
    assert info["adjusted_trust"] == pytest.approx(0.6 * 0.9, abs=1e-6)


def test_trust_idempotent_and_persisted():
    store = _tmp_store()
    src = "alpha"
    figs = [_image(src, 0.9), _atomic("The council endorsed the plan.", src, 1)]
    g = Figtree(figs, store=store)
    g.propagate_trust(store=store)
    first = store.get("trust:alpha")
    assert first is not None
    expected = first.meta["score"]
    # Re-run: overwrites the same canonical row, score unchanged.
    g2 = Figtree(figs, store=store)
    g2.propagate_trust(store=store)
    second = store.get("trust:alpha")
    assert second.meta["score"] == pytest.approx(expected, abs=1e-6)
    assert store.count() == 1  # only the one canonical trust figment


def test_two_sources_agreement_boosts_trust():
    store = _tmp_store()
    figs = [
        _image("alpha", 0.9),
        _atomic("The Treaty was endorsed by leaders.", "alpha", 1),
        _image("beta", 0.6),
        _atomic("The Treaty was endorsed by the council.", "beta", 2),
    ]
    g = Figtree(figs, store=store)
    updates = {u["source_id"]: u for u in g.propagate_trust(store=store)}
    # Each source: 1 atomic fig corroborated of 2 figs (image + atomic) -> 0.5.
    assert updates["alpha"]["adjusted_trust"] == pytest.approx(0.6 * 0.9 + 0.4 * 0.5, abs=1e-6)
    assert updates["beta"]["adjusted_trust"] == pytest.approx(0.6 * 0.6 + 0.4 * 0.5, abs=1e-6)
    assert updates["alpha"]["agreeing"] == ["beta"]


def test_two_sources_contradiction_penalizes_trust():
    store = _tmp_store()
    figs = [
        _image("alpha", 0.9),
        _atomic("The Treaty was endorsed by leaders.", "alpha", 1),
        _image("beta", 0.6),
        _atomic("The Treaty failed to gain support.", "beta", 2),
    ]
    g = Figtree(figs, store=store)
    updates = {u["source_id"]: u for u in g.propagate_trust(store=store)}
    # corroborated_frac = 0; contradicted -> *0.85.
    assert updates["alpha"]["adjusted_trust"] == pytest.approx(0.6 * 0.9 * 0.85, abs=1e-6)
    assert updates["beta"]["adjusted_trust"] == pytest.approx(0.6 * 0.6 * 0.85, abs=1e-6)
    assert updates["alpha"]["contradicting"] == ["beta"]


def _tmp_store():
    import tempfile
    from pathlib import Path
    return connect(Path(tempfile.mkdtemp(prefix="ft_trust_")) / "store.lance")
