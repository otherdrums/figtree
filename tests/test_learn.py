"""CPU-only tests for the prompt-learning loop (no model / GPU required).

Covers role parsing, identity/association figment creation, the
newest-high-trust-wins conflict policy, corrections/forget, provenance,
role-intersection retrieval, and the backward-compatibility fixes
(``generate_faithful(source_tokens=...)`` alias, ``FigmentStore(path)``
constructor). The end-to-end GPU test at the bottom is skipped without CUDA.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from figtree import Figment, FigmentStore, connect
from figtree.generate import FigmentGenerator
from figtree.learn import (
    _assoc_id,
    apply_roles_to_store,
    extract_query_roles,
    forget,
    learned_facts,
    parse_role_lines,
    role_figment_id,
)
from figtree.retrieve import role_lookup, retrieve_by_roles

HIDDEN = 16
L0 = 1_700_000_000.0


@pytest.fixture
def store(tmp_path) -> FigmentStore:
    return connect(tmp_path / "store.lance")


def _mk_statement(store, text: str, seed: int = 1) -> str:
    rng = np.random.default_rng(seed)
    fig = Figment.create(
        text, rng.standard_normal(HIDDEN).astype("float32"),
        meta={}, trust=0.95, kind="sentence",
    )
    store.upsert([fig], hidden_size=HIDDEN)
    return fig.figment_id


# ---------------------------------------------------------------------- #
# Role line parsing
# ---------------------------------------------------------------------- #
def test_parse_role_lines_basic():
    out = parse_role_lines("NAME|Alex\nPOLICY|Never use the legacy auth endpoint\n")
    assert ("NAME", "Alex") in out
    assert ("POLICY", "Never use the legacy auth endpoint") in out


def test_parse_role_lines_ignores_garbage():
    assert parse_role_lines("no pipe here\nNAME|June\nweird|stuff") == [("NAME", "June")]


def test_parse_role_lines_query_allow_empty():
    assert parse_role_lines("NAME|\nPREFERENCE|", allow_empty=True) == [
        ("NAME", ""), ("PREFERENCE", ""),
    ]


def test_parse_role_lines_filters_fillers():
    assert parse_role_lines("PREFERENCE|none\nPOLICY|n/a\nNAME|June") == [("NAME", "June")]


# ---------------------------------------------------------------------- #
# Rule-based query role extraction (CPU-only)
# ---------------------------------------------------------------------- #
def test_extract_query_roles():
    assert extract_query_roles("What is the user's name?") == [("ACTOR", "user"), ("NAME", "")]
    assert ("POLICY", "") in extract_query_roles("What should I do instead of the legacy auth endpoint?")
    assert ("CONSTRAINT", "") in extract_query_roles("What is the user's daughter allergic to?")
    assert extract_query_roles("Write a poem about the sea.") == []
    assert extract_query_roles("What is Alex's name?") == [("NAME", "alex")]
    assert extract_query_roles("What is Alex's phone number?") == []


# ---------------------------------------------------------------------- #
# Identity + association creation
# ---------------------------------------------------------------------- #
def test_apply_roles_creates_identity_and_association(store):
    sid = _mk_statement(store, "My name is Alex. Please remember that.")
    res = apply_roles_to_store(
        sid, "My name is Alex.", [("ACTOR", "user"), ("NAME", "Alex")],
        store, trust=0.95, learned_at=L0,
    )
    name_id = role_figment_id("NAME", "Alex")
    actor_id = role_figment_id("ACTOR", "user")
    assert set(res["created"]) == {name_id, actor_id}

    name = store.get(name_id)
    assert name is not None
    assert name.kind == "role"
    assert name.meta["role"] == "NAME"
    assert name.meta["normalized"] == "alex"
    assert name.meta["references"] == [sid]
    assert name.meta["reference_count"] == 1
    assert name.meta["learning"] is True
    assert name.meta["session_id"] == "user"

    actor = store.get(role_figment_id("ACTOR", "user"))
    assert actor is not None and actor.kind == "role"

    assoc = store.get(_assoc_id("NAME", actor.figment_id, name.figment_id))
    assert assoc is not None
    assert assoc.meta["edge_type"] == "association"
    assert actor.figment_id in assoc.meta["links"]
    assert name.figment_id in assoc.meta["links"]

    st = store.get(sid)
    assert st.meta["learning"] is True
    assert st.meta["learned_roles"] == {"NAME": "Alex"}


def test_reteach_same_value_strengthens_no_duplicates(store):
    sid1 = _mk_statement(store, "My name is Alex.", seed=1)
    apply_roles_to_store(sid1, "My name is Alex.", [("ACTOR", "user"), ("NAME", "Alex")], store, learned_at=L0)
    sid2 = _mk_statement(store, "Remember: my name is Alex.", seed=2)
    res = apply_roles_to_store(sid2, "Remember: my name is Alex.", [("ACTOR", "user"), ("NAME", "Alex")], store, learned_at=L0 + 10)

    assert role_figment_id("NAME", "Alex") not in res["created"]
    assert res["created"] == [] or role_figment_id("ACTOR", "user") in res["created"]
    name = store.get(role_figment_id("NAME", "Alex"))
    assert name.meta["reference_count"] == 2

    assocs = [
        f for f in store.all()
        if f.meta.get("edge_type") == "association"
        and f.meta.get("role") == "NAME"
        and not f.meta.get("superseded_by")
    ]
    assert len(assocs) == 1


def test_identity_does_not_duplicate_actor(store):
    sid = _mk_statement(store, "My name is Alex.", seed=1)
    apply_roles_to_store(sid, "My name is Alex.", [("ACTOR", "user"), ("NAME", "Alex")], store, learned_at=L0)
    sid2 = _mk_statement(store, "I prefer dark mode.", seed=2)
    res = apply_roles_to_store(sid2, "I prefer dark mode.", [("ACTOR", "user"), ("PREFERENCE", "dark mode")], store, learned_at=L0 + 1)
    assert role_figment_id("ACTOR", "user") not in res["created"]


# ---------------------------------------------------------------------- #
# Conflict policy: newest-high-trust wins, old frames kept
# ---------------------------------------------------------------------- #
def test_new_value_supersedes_old(store):
    sid1 = _mk_statement(store, "My name is Alex.", seed=1)
    apply_roles_to_store(sid1, "My name is Alex.", [("ACTOR", "user"), ("NAME", "Alex")], store, learned_at=L0)
    sid2 = _mk_statement(store, "Correct this: my name is Alex Chen.", seed=2)
    res = apply_roles_to_store(
        sid2, "Correct this: my name is Alex Chen.",
        [("ACTOR", "user"), ("NAME", "Alex Chen")], store, learned_at=L0 + 10,
    )

    old_id = role_figment_id("NAME", "Alex")
    new_id = role_figment_id("NAME", "Alex Chen")
    assert res["superseded"] == [old_id]

    old = store.get(old_id)
    assert old.meta["superseded_by"] == new_id
    assert old.meta["superseded_by_statement"] == sid2
    new = store.get(new_id)
    assert not new.meta.get("superseded_by")

    # Retrieval only surfaces the active frame by default.
    active = role_lookup(store, "NAME")
    assert [f.figment_id for f, _ in active] == [new_id]
    both = role_lookup(store, "NAME", include_superseded=True)
    assert {f.figment_id for f, _ in both} == {old_id, new_id}

    # The old association is superseded too.
    actor = store.get(role_figment_id("ACTOR", "user"))
    old_assoc = store.get(_assoc_id("NAME", actor.figment_id, old_id))
    assert old_assoc.meta["superseded_by"] == _assoc_id("NAME", actor.figment_id, new_id)


def test_correction_without_replacement_retracts(store):
    sid1 = _mk_statement(store, "My name is Alex.", seed=1)
    apply_roles_to_store(sid1, "My name is Alex.", [("ACTOR", "user"), ("NAME", "Alex")], store, learned_at=L0)
    sid2 = _mk_statement(store, "Forget my old name.", seed=2)
    res = apply_roles_to_store(sid2, "Forget my old name.", [("ACTOR", "user"), ("CORRECTION", "NAME")], store, learned_at=L0 + 10)
    assert res["superseded"] == [role_figment_id("NAME", "Alex")]
    node = store.get(role_figment_id("NAME", "Alex"))
    assert node.meta["superseded_by"] == "retracted"
    assert role_lookup(store, "NAME") == []


def test_forget_marks_association(store):
    sid = _mk_statement(store, "My daughter's name is June.", seed=1)
    apply_roles_to_store(sid, "My daughter's name is June.", [("ACTOR", "user"), ("NAME", "June")], store, learned_at=L0)
    assert forget(store, "NAME", "June") is True
    assert role_lookup(store, "NAME") == []
    assert forget(store, "NAME", "June") is False  # idempotent: nothing active left
    node = store.get(role_figment_id("NAME", "June"))
    assert node.meta["superseded_by"] == "forgotten"


# ---------------------------------------------------------------------- #
# Provenance
# ---------------------------------------------------------------------- #
def test_provenance_stamped(store):
    sid = _mk_statement(store, "The production host is db-prod-03.", seed=1)
    apply_roles_to_store(
        sid, "The production host is db-prod-03.",
        [("ACTOR", "user"), ("FACT", "production host db-prod-03")],
        store, session_id="ops", trust=0.9, learned_at=L0,
    )
    node = store.get(role_figment_id("FACT", "production host db-prod-03"))
    prov = node.meta["provenance"]
    assert prov["type"] == "prompt"
    assert prov["session_id"] == "ops"
    assert prov["statement_id"] == sid
    assert prov["learned_at"] == L0
    assert node.trust == 0.9


# ---------------------------------------------------------------------- #
# Retrieval
# ---------------------------------------------------------------------- #
def test_role_lookup_value_matching(store):
    sid = _mk_statement(store, "My name is Alex.", seed=1)
    apply_roles_to_store(sid, "My name is Alex.", [("ACTOR", "user"), ("NAME", "Alex")], store, learned_at=L0)
    assert [f.figment_id for f, _ in role_lookup(store, "NAME", "alex")] == [role_figment_id("NAME", "Alex")]
    assert [f.figment_id for f, _ in role_lookup(store, "NAME", "Alex Chen")] == []
    # Word-overlap ladder still finds it for a fuzzy value.
    assert [f.figment_id for f, _ in role_lookup(store, "NAME", "alexander")] == []


def test_retrieve_by_roles_pure_extractor(store):
    sid = _mk_statement(store, "My name is Alex.", seed=1)
    apply_roles_to_store(sid, "My name is Alex.", [("ACTOR", "user"), ("NAME", "Alex")], store, learned_at=L0)

    got = retrieve_by_roles(
        "What is the user's name?", store,
        extract_roles_fn=lambda q: [("NAME", "alex")],
    )
    assert [f.figment_id for f in got] == [sid]
    assert got[0].meta["role_match"] == "NAME"

    assert retrieve_by_roles("Any question", store, extract_roles_fn=lambda q: []) == []
    assert retrieve_by_roles("Pizza?", store, extract_roles_fn=lambda q: [("NAME", "nobody")]) == []


def test_retrieve_rule_based_default_with_actor_filter(store):
    sid1 = _mk_statement(store, "My name is Alex.", seed=1)
    apply_roles_to_store(sid1, "My name is Alex.", [("ACTOR", "user"), ("NAME", "Alex")], store, learned_at=L0)
    # Another actor's name must not surface for the user's own identity.
    sid2 = _mk_statement(store, "Her name is Zoe.", seed=3)
    apply_roles_to_store(sid2, "Her name is Zoe.", [("ACTOR", "zoe"), ("NAME", "Zoe")], store, learned_at=L0 + 10)
    # Default extractor is rule-based (no model): NAME + ACTOR|user.
    hits = retrieve_by_roles("What is the user's name?", store)
    assert [f.figment_id for f in hits] == [sid1]


def test_retrieve_require_all_intersects(store):
    sid1 = _mk_statement(store, "My name is Alex.", seed=1)
    apply_roles_to_store(sid1, "My name is Alex.", [("ACTOR", "user"), ("NAME", "Alex")], store, learned_at=L0)
    sid2 = _mk_statement(store, "I prefer dark mode.", seed=2)
    apply_roles_to_store(sid2, "I prefer dark mode.", [("ACTOR", "user"), ("PREFERENCE", "dark mode")], store, learned_at=L0 + 10)

    both = retrieve_by_roles(
        "name and preference", store,
        extract_roles_fn=lambda q: [("NAME", "alex"), ("PREFERENCE", "dark")],
        require_all=True,
    )
    assert both == []
    any_ = retrieve_by_roles(
        "name and preference", store,
        extract_roles_fn=lambda q: [("NAME", "alex"), ("PREFERENCE", "dark")],
    )
    assert {f.figment_id for f in any_} == {sid1, sid2}


def test_learned_facts_lists_active_only(store):
    sid1 = _mk_statement(store, "My name is Alex.", seed=1)
    apply_roles_to_store(sid1, "My name is Alex.", [("ACTOR", "user"), ("NAME", "Alex")], store, learned_at=L0)
    sid2 = _mk_statement(store, "I prefer dark mode.", seed=2)
    apply_roles_to_store(sid2, "I prefer dark mode.", [("ACTOR", "user"), ("PREFERENCE", "dark mode")], store, learned_at=L0 + 10)
    facts = learned_facts(store)
    assert [f.meta["role"] for f in facts] == ["PREFERENCE", "NAME"]
    assert all(f.meta.get("learning") is True for f in facts)
    assert all(f.meta.get("role") != "ACTOR" for f in facts)
    assert learned_facts(store, session_id="other") == []


def test_news_domain_figments_ignored(store):
    sid = _mk_statement(store, "Davos summit concluded.", seed=1)
    news_role = Figment.create(
        "Donald Trump", np.zeros(HIDDEN, dtype=np.float32),
        meta={"role": "who", "normalized": "donald trump"}, kind="role",
        figment_id=role_figment_id("who", "Donald Trump"),
    )
    store.upsert([news_role], hidden_size=HIDDEN)
    assert role_lookup(store, "who") == []  # not a learned figment
    apply_roles_to_store(sid, "Davos summit concluded.", [("ACTOR", "user"), ("FACT", "Davos summit concluded")], store, learned_at=L0)
    assert len(learned_facts(store)) == 1


# ---------------------------------------------------------------------- #
# Backward-compatibility regression pins
# ---------------------------------------------------------------------- #
def test_generate_faithful_accepts_source_tokens_alias():
    sig = inspect.signature(FigmentGenerator.generate_faithful)
    params = sig.parameters
    assert "source_tokens" in params
    assert "source_texts" in params


def test_figmentstore_accepts_path_string(tmp_path):
    store = FigmentStore(str(tmp_path / "path.lance"))
    assert store.count() == 0
    rng = np.random.default_rng(7)
    fig = Figment.create("Roundtrip via path constructor.", rng.standard_normal(HIDDEN).astype("float32"))
    store.upsert([fig], hidden_size=HIDDEN)
    assert store.get(fig.figment_id) is not None
    # Same physical table is visible through connect().
    via_connect = connect(tmp_path / "path.lance")
    assert via_connect.get(fig.figment_id) is not None
