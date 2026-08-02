"""Role-intersection retrieval for learned facts.

Queries surface learned facts by extracting the roles the question asks about
(NAME, PREFERENCE, POLICY, ...) and resolving them against the identity +
association figments created by :mod:`figtree.learn`. Statement figments whose
text carries the fact are returned, ready to feed ``FigmentGenerator``.

Only figments stamped ``meta["learning"]=True`` are considered, so news-domain
role figments (figtree-news) in the same store are never returned.
"""

from __future__ import annotations

from typing import Any

from figtree.figment import Figment
from figtree.learn import _normalize, extract_query_roles, extract_roles, role_figment_id


def role_lookup(
    store,
    role: str,
    value: str | None = None,
    include_superseded: bool = False,
) -> list[tuple[Figment, float]]:
    """Find learned role figments for ``role``, optionally matching a value.

    Value matching (stricter than string containment, since identity between
    surface forms is the job of association figments, not string matching):
    exact normalized match (score 1.0), query-token-set subset of the node's
    tokens ("dark" -> "dark mode", score 0.95), then token jaccard >= 0.6
    (0.85). Superseded/forgotten frames are excluded unless
    ``include_superseded``.
    """
    matches: list[tuple[Figment, float]] = []
    want = _normalize(value) if value else None
    for f in store.all():
        if f.kind != "role" or f.meta.get("learning") is not True:
            continue
        if f.meta.get("role") != role:
            continue
        if not include_superseded and (
            f.meta.get("superseded_by") or f.meta.get("forgotten")
        ):
            continue
        if want is None:
            matches.append((f, 1.0))
            continue
        norm = f.meta.get("normalized", _normalize(f.text))
        if norm == want:
            score = 1.0
        else:
            q_tokens = set(want.split())
            n_tokens = set(norm.split())
            if q_tokens and q_tokens <= n_tokens:
                score = 0.95
            else:
                jac = len(q_tokens & n_tokens) / len(q_tokens | n_tokens)
                if jac < 0.6:
                    continue
                score = 0.85
        matches.append((f, score))
    matches.sort(key=lambda m: m[1], reverse=True)
    return matches


def retrieve_by_roles(
    query: str,
    store,
    model=None,
    tokenizer=None,
    extract_roles_fn=None,
    limit: int = 10,
    require_all: bool = False,
) -> list[Figment]:
    """Retrieve learned statements that answer ``query`` by role intersection.

    Roles are extracted from the query with the deterministic rule-based
    :func:`figtree.learn.extract_query_roles` by default (CPU-only); pass
    ``extract_roles_fn`` to supply a custom callable, or ``model`` +
    ``tokenizer`` for LLM-based extraction (:func:`extract_roles` with
    ``query_mode=True``). When the query refers to the user (ACTOR role), only
    statements whose role figments are associated with that actor are kept.

    For each role, matching role figments are resolved; the statement figments
    they reference are intersected (``require_all``) or unioned, then ranked by
    trust then recency. The returned figments carry ``meta["role_match"]`` so
    callers can see why each was selected.

    Returns an empty list when the query asks nothing about learned facts.
    """
    if extract_roles_fn is None:
        if model is not None and tokenizer is not None:
            extract_roles_fn = lambda q: extract_roles(  # noqa: E731
                model, tokenizer, q, query_mode=True
            )
        else:
            extract_roles_fn = extract_query_roles

    roles = extract_roles_fn(query) or []
    role_figs: dict[str, list[Figment]] = {}

    actor_id = None
    for role, value in roles:
        if role == "ACTOR" and value:
            candidate = role_figment_id("ACTOR", value)
            if store.get(candidate) is not None:
                actor_id = candidate
    linked = None
    if actor_id is not None:
        linked = {
            other for f in store.all()
            if f.meta.get("edge_type") == "association"
            and actor_id in (f.meta.get("links") or [])
            for other in f.meta.get("links", [])
            if other != actor_id
        }

    for role, value in roles:
        if role in ("ACTOR", "CORRECTION"):
            continue  # ACTOR only restricts; it is not a retrieval role
        matches = role_lookup(store, role, value or None)
        if linked is not None:
            matches = [(f, s) for f, s in matches if f.figment_id in linked]
        if matches:
            role_figs.setdefault(role, [f for f, _ in matches])

    if not role_figs:
        return []

    by_statement: dict[str, dict[str, Any]] = {}
    for role, figs in role_figs.items():
        for f in figs:
            refs = [r for r in (f.meta.get("references") or []) if store.get(r) is not None]
            for sid in refs:
                entry = by_statement.setdefault(
                    sid, {"statement_id": sid, "roles": set(), "trust": 0.0}
                )
                entry["roles"].add(role)
                entry["trust"] = max(entry["trust"], float(f.trust))

    if require_all and len(role_figs) > 1:
        needed = set(role_figs.keys())
        by_statement = {
            sid: e for sid, e in by_statement.items() if needed <= e["roles"]
        }

    ranked = sorted(
        by_statement.values(),
        key=lambda e: (-e["trust"], -_learned_at(store.get(e["statement_id"]))),
    )

    out: list[Figment] = []
    for entry in ranked[:limit]:
        st = store.get(entry["statement_id"])
        if st is None:
            continue
        st.meta["role_match"] = ",".join(sorted(entry["roles"]))
        out.append(st)
    return out


def _learned_at(fig: Figment | None) -> float:
    if fig is None:
        return 0.0
    return float(fig.meta.get("learned_at", 0.0))
