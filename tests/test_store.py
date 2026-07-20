"""CPU-only tests for the LanceDB FigmentStore (no model / GPU required).

Figments are constructed with random boundaries so the store round-trip,
idempotent upsert, source indexing, and similarity search can be exercised
without loading a language model.
"""

from __future__ import annotations

import numpy as np
import pytest

from figtree import Figment, connect
from figtree.lancedb_store import FigmentStore

HIDDEN = 16


def _mk(text: str, seed: int, source_id: str = "") -> Figment:
    rng = np.random.default_rng(seed)
    return Figment.create(
        text, rng.standard_normal(HIDDEN).astype("float32"),
        meta={"source_id": source_id},
    )


@pytest.fixture
def store(tmp_path) -> FigmentStore:
    return connect(tmp_path / "store.lance")


def test_roundtrip_by_id(store):
    f = _mk("The summit concluded with a compact.", seed=1, source_id="s1")
    store.upsert([f])
    got = store.get(f.figment_id)
    assert got is not None
    assert got.text == f.text
    assert got.figment_id == f.figment_id
    assert np.allclose(got.boundary, f.boundary)
    assert got.meta.get("source_id") == "s1"


def test_all_and_by_source(store):
    a = _mk("Alpha reached a deal.", seed=2, source_id="srcA")
    b = _mk("Beta rejected the deal.", seed=3, source_id="srcB")
    c = _mk("Alpha published a statement.", seed=4, source_id="srcA")
    store.upsert([a, b, c])
    assert store.count() == 3
    assert {x.figment_id for x in store.all()} == {a.figment_id, b.figment_id, c.figment_id}
    src_a = store.by_source("srcA")
    assert {x.figment_id for x in src_a} == {a.figment_id, c.figment_id}


def test_upsert_is_idempotent(store):
    f = _mk("Original boundary.", seed=5)
    store.upsert([f])
    assert store.count() == 1
    # Re-upsert the same id with a new boundary -> still one row, updated value.
    f2 = Figment.create(f.text, np.full(HIDDEN, 9.0, dtype="float32"),
                        figment_id=f.figment_id)
    store.upsert([f2])
    assert store.count() == 1
    got = store.get(f.figment_id)
    assert np.allclose(got.boundary, 9.0)


def test_search_ranks_nearest_first(store):
    near = _mk("Anchor point.", seed=6)
    close = Figment.create("Close point.", (near.boundary + 0.01).astype("float32"))
    far = Figment.create("Far point.", -near.boundary)
    store.upsert([near, close, far])
    results = store.search(near.boundary, k=3)
    ids = [r[0].figment_id for r in results]
    assert ids[0] == near.figment_id
    # The near/close pair should both outrank the opposite-sign vector.
    assert far.figment_id not in ids[:2]


def test_missing_get_returns_none(store):
    assert store.get("does-not-exist") is None


def test_figment_dict_roundtrip():
    f = _mk("Serialize me.", seed=7, source_id="s9")
    f.children = ["c1", "c2"]
    f.sources = ["p1"]
    f.trust = 0.83
    d = f.to_dict()
    assert isinstance(d["boundary"], list)
    f2 = Figment.from_dict(d)
    assert f2.figment_id == f.figment_id
    assert f2.text == f.text
    assert f2.children == ["c1", "c2"]
    assert f2.sources == ["p1"]
    assert f2.trust == 0.83
    assert np.allclose(f2.boundary, f.boundary)
