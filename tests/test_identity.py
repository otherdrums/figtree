"""CPU tests for the identity merge engine (no model)."""

from __future__ import annotations

import numpy as np
from figtree import (
    Figment,
    assert_identity,
    connect,
    expand_identities,
    identity_groups,
    merge_role_figments,
    propose_identity_merges,
)
from figtree.learn import _normalize, role_figment_id


def _role_fig(role: str, text: str, refs: list[str] | None = None) -> Figment:
    return Figment.create(
        text=text,
        boundary=np.zeros(8, dtype="float32"),
        meta={
            "role": role,
            "normalized": _normalize(text),
            "references": list(refs or []),
            "reference_count": len(refs or []),
        },
        figment_id=role_figment_id(role, text),
        kind="role",
    )


def test_propose_identity_merges_finds_variants(tmp_path):
    store = connect(str(tmp_path / "id.lance"))
    variants = [
        _role_fig("who", "Donald Trump"),
        _role_fig("who", "Trump"),
        _role_fig("who", "President Trump"),
    ]
    store.upsert(variants, hidden_size=8)

    props = propose_identity_merges(store, string_overlap_threshold=0.4)
    # Every pair shares a token overlap, so all 3 pairs are proposed.
    assert len(props) == 3
    assert all(p["role"] == "who" for p in props)
    assert all("string_overlap" in p["reasons"] for p in props)
    assert props[0]["confidence"] >= props[-1]["confidence"]


def test_assert_and_expand_identities(tmp_path):
    store = connect(str(tmp_path / "id.lance"))
    a = _role_fig("who", "Donald Trump")
    b = _role_fig("who", "DJT")
    store.upsert([a, b], hidden_size=8)

    edge = assert_identity(store, a.figment_id, b.figment_id)
    assert edge is not None
    assert edge.meta["edge_type"] == "association"

    # Idempotent: asserting the same pair returns the existing edge.
    again = assert_identity(store, a.figment_id, b.figment_id)
    assert again.meta["edge_type"] == "association"

    reachable = expand_identities(store, a.figment_id)
    assert reachable == {a.figment_id, b.figment_id}

    groups = identity_groups(store)
    assert any(len(members) == 2 for members in groups.values())


def test_merge_role_figments_rewrites_references(tmp_path):
    store = connect(str(tmp_path / "id.lance"))
    keep = _role_fig("who", "Donald Trump", refs=["s1"])
    variant = _role_fig("who", "DJT", refs=["s2"])
    store.upsert([keep, variant], hidden_size=8)

    article = Figment.create(
        text="Article mentioning both forms.",
        boundary=np.zeros(8, dtype="float32"),
        meta={"role_figments": [keep.figment_id, variant.figment_id], "decomposed": True},
        figment_id="a1",
        kind="article",
    )
    store.upsert([article], hidden_size=8)

    mutations = merge_role_figments(store, keep.figment_id, [variant.figment_id])
    assert mutations > 0

    assert store.get(variant.figment_id) is None  # variant row deleted
    kept = store.get(keep.figment_id)
    assert kept.meta["is_association"] is True
    assert set(kept.meta["references"]) == {"s1", "s2"}  # references unioned
    # Article reference rewritten: variant id replaced by canonical id.
    stored_article = store.get("a1")
    assert stored_article.meta["role_figments"] == [keep.figment_id]


def test_retrieve_news_role_figments_resolve_to_articles(tmp_path):
    """role_lookup matches news-domain role figments and resolves via article_id."""
    from figtree.retrieve import role_lookup

    store = connect(str(tmp_path / "id.lance"))
    news_role = _role_fig("who", "Donald Trump")
    news_role.meta["article_id"] = "art1"
    article = Figment.create(
        text="Trump visits Davos.",
        boundary=np.zeros(8, dtype="float32"),
        meta={"source_id": "reuters"},
        figment_id="art1",
        kind="article",
    )
    store.upsert([news_role, article], hidden_size=8)

    hits = role_lookup(store, "who", "Donald Trump")
    assert len(hits) == 1
    assert hits[0][0].figment_id == news_role.figment_id
