"""Identity merging: surface-form variants of an entity -> canonical role figment.

Role figments are canonicalized by exact normalized text match, so surface
variants such as ``"Donald Trump"``, ``"Trump"``, and ``"DJT"`` become separate
rows. The identity layer links them so downstream retrieval can expand through
aliases and fetch every role instance of the same entity.

Merges are proposed with cheap heuristics (boundary similarity, token overlap,
edit similarity) — no LLM required — and applied by :func:`merge_role_figments`,
which promotes one row to the canonical node (``is_association=True``) and
rewrites *every* reference to the removed variants across all storage
locations, then deletes the variant rows.

All identity links are first-class figments (``edge_type="association"``)
stored in the same LanceDB table, so they are traversable and queryable like
everything else in the system. figtree-news thin-wraps this module
(``figtree_news/associations.py``) with its historical API names.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

import numpy as np

from figtree.figment import Figment
from figtree.learn import _normalize


def _boundary_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    dot = float(np.dot(a_f, b_f))
    n = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    return dot / n if n > 0 else 0.0


def _string_overlap(a: str, b: str) -> float:
    """Fraction of the shorter string's tokens that appear in the other."""
    ta = set(_normalize(a).split())
    tb = set(_normalize(b).split())
    if not ta or not tb:
        return 0.0
    smaller = ta if len(ta) <= len(tb) else tb
    larger = tb if smaller is ta else ta
    return len(smaller & larger) / len(smaller)


def _editsim(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def propose_identity_merges(
    store,
    role_figments: list[Figment] | None = None,
    boundary_threshold: float = 0.90,
    string_overlap_threshold: float = 0.50,
    editsim_threshold: float = 0.85,
    min_co_occurrence: int = 3,
) -> list[dict[str, Any]]:
    """Propose identity merges between role figments of the same role.

    Scans the store for role figments and proposes a merge where two variants
    share the same ``meta["role"]`` and one or more of: boundary cosine
    similarity >= ``boundary_threshold``, string overlap >=
    ``string_overlap_threshold``, edit-distance ratio >= ``editsim_threshold``,
    or co-occurrence weight in an existing relationship edge >=
    ``min_co_occurrence``.

    Returns a list of dicts ready to be reviewed or auto-applied:
    ``figment_a_id``/``figment_b_id``/``role``/``text_a``/``text_b``/
    ``confidence``/``reasons``, sorted by confidence descending.
    """
    if role_figments is None:
        all_figs = store.all()
        role_figments = [f for f in all_figs if f.meta.get("role")]

    by_role: dict[str, list[Figment]] = defaultdict(list)
    for f in role_figments:
        by_role[f.meta.get("role", "")].append(f)

    proposals: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for role, figs in by_role.items():
        if len(figs) < 2:
            continue
        for i in range(len(figs)):
            for j in range(i + 1, len(figs)):
                a, b = figs[i], figs[j]
                pair = tuple(sorted([a.figment_id, b.figment_id]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                if _already_associated(store, a.figment_id, b.figment_id):
                    continue

                reasons: list[str] = []
                scores: list[float] = []

                sim = _boundary_sim(a.boundary, b.boundary)
                if sim >= boundary_threshold:
                    reasons.append("boundary_similarity")
                    scores.append(sim)

                olap = _string_overlap(a.text, b.text)
                if olap >= string_overlap_threshold:
                    reasons.append("string_overlap")
                    scores.append(olap)

                ed = _editsim(a.text, b.text)
                if ed >= editsim_threshold:
                    reasons.append("edit_similarity")
                    scores.append(ed)

                if not reasons:
                    continue

                cooccur = _co_occurrence_weight(store, a.figment_id, b.figment_id)
                if cooccur >= min_co_occurrence:
                    reasons.append(f"co_occurrence:{cooccur}")
                    scores.append(min(cooccur / 10.0, 1.0))

                confidence = float(max(scores)) if scores else 0.0

                proposals.append(
                    {
                        "figment_a_id": a.figment_id,
                        "figment_b_id": b.figment_id,
                        "role": role,
                        "text_a": a.text,
                        "text_b": b.text,
                        "confidence": confidence,
                        "reasons": reasons,
                    }
                )

    proposals.sort(key=lambda p: p["confidence"], reverse=True)
    return proposals


def _already_associated(store, id_a: str, id_b: str) -> bool:
    """Quick check: are these two figments already linked by an association?"""
    for f in store.all():
        if f.meta.get("edge_type") != "association":
            continue
        links = f.meta.get("links", [])
        if id_a in links and id_b in links:
            return True
    return False


def _co_occurrence_weight(store, id_a: str, id_b: str) -> int:
    """Return the weight of the relationship edge between two role figments."""
    for f in store.all():
        if f.meta.get("edge_type") != "relationship":
            continue
        fa = f.meta.get("figment_a")
        fb = f.meta.get("figment_b")
        if (fa == id_a and fb == id_b) or (fa == id_b and fb == id_a):
            return int(f.meta.get("weight", 0))
    return 0


def assert_identity(
    store,
    figment_a_id: str,
    figment_b_id: str,
    confidence: float = 1.0,
    evidence: str = "manual",
    hidden_size: int | None = None,
) -> Figment | None:
    """Create a bidirectional association edge between two role figments.

    Returns the association Figment that was upserted, or None if both
    figments are already linked. The id is deterministic:
    ``sha256(f"assoc:{role}:{a}:{b}")[:16]``.
    """
    a = store.get(figment_a_id)
    b = store.get(figment_b_id)
    if a is None or b is None:
        return None
    role = a.meta.get("role") or b.meta.get("role", "")

    for f in store.all():
        if f.meta.get("edge_type") != "association":
            continue
        links = f.meta.get("links", [])
        if figment_a_id in links and figment_b_id in links:
            return f

    figment_id = hashlib.sha256(
        f"assoc:{role}:{figment_a_id}:{figment_b_id}".encode()
    ).hexdigest()[:16]

    association = Figment.create(
        text=f"Association: {a.text[:40]} <-> {b.text[:40]} ({role})",
        boundary=(a.boundary if a.boundary.shape[0] > 0 else np.zeros(1, dtype=np.float32)),
        meta={
            "edge_type": "association",
            "role": role,
            "links": [figment_a_id, figment_b_id],
            "confidence": confidence,
            "evidence": evidence,
        },
        figment_id=figment_id,
        kind="edge",
    )

    hs = hidden_size or association.boundary.shape[0]
    store.upsert([association], hidden_size=hs)
    return association


def expand_identities(store, role_figment_id: str, max_hops: int = 2) -> set[str]:
    """Return the full set of variant figment IDs reachable from *role_figment_id*.

    Walks association edges up to ``max_hops`` (default 2). The starting
    figment ID is always included in the result.
    """
    result: set[str] = {role_figment_id}
    frontier: set[str] = {role_figment_id}

    for _ in range(max_hops):
        next_frontier: set[str] = set()
        for fid in frontier:
            for f in store.all():
                if f.meta.get("edge_type") != "association":
                    continue
                links = f.meta.get("links", [])
                if fid in links:
                    for other in links:
                        if other not in result:
                            result.add(other)
                            next_frontier.add(other)
        frontier = next_frontier
        if not frontier:
            break

    return result


def identity_groups(store) -> dict[str, list[str]]:
    """Return all identity clusters as ``{canonical_id: [variant_ids]}``.

    Uses union-find over association edges; every linked id appears under its
    cluster root (which is not necessarily the promoted canonical node).
    """
    associations: list[Figment] = [
        f for f in store.all() if f.meta.get("edge_type") == "association"
    ]
    if not associations:
        return {}

    all_ids: set[str] = set()
    for a in associations:
        for lid in a.meta.get("links", []):
            all_ids.add(lid)

    parent = {fid: fid for fid in all_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a_id: str, b_id: str) -> None:
        ra, rb = find(a_id), find(b_id)
        if ra != rb:
            parent[rb] = ra

    for a in associations:
        links = a.meta.get("links", [])
        for i in range(len(links)):
            for j in range(i + 1, len(links)):
                union(links[i], links[j])

    groups: dict[str, list[str]] = defaultdict(list)
    for fid in all_ids:
        groups[find(fid)].append(fid)

    return dict(groups)


def merge_role_figments(
    store,
    keep_id: str,
    remove_ids: list[str],
    all_figs: list[Figment] | None = None,
) -> int:
    """Merge confirmed-equivalent role figments into one canonical node.

    ``keep_id`` is promoted to the canonical node (``is_association=True``).
    Every reference to any ID in ``remove_ids`` is rewritten to point at
    ``keep_id`` across all figment storage locations (article/paragraph
    ``role_figments``, ``sentence.children``, relationship edges, association
    edges, and dedup_obs). The removed rows are then deleted.

    Returns the number of store mutation operations performed.
    """
    all_f = all_figs if all_figs is not None else store.all()
    by_id = {f.figment_id: f for f in all_f}

    keep = by_id.get(keep_id)
    if not keep or keep.kind != "role":
        return 0

    removes = [by_id.get(rid) for rid in remove_ids if by_id.get(rid)]
    if not removes:
        return 0

    # ── Union references ────────────────────────────────────────────────
    all_refs: set[str] = set(keep.meta.get("references", []))
    for r in removes:
        all_refs.update(r.meta.get("references", []))
    keep.meta["references"] = list(all_refs)
    keep.meta["reference_count"] = len(all_refs)

    merged_from: list[str] = keep.meta.get("merged_from", [])
    for r in removes:
        if r.figment_id not in merged_from:
            merged_from.append(r.figment_id)
    keep.meta["merged_from"] = merged_from
    keep.meta["is_association"] = True

    remove_set: set[str] = set(remove_ids)
    mutations = 0

    to_upsert: dict[str, Figment] = {keep_id: keep}
    to_delete: set[str] = set(remove_ids)

    for fig in all_f:
        if fig.figment_id in remove_set:
            continue

        # 1. article / image meta["role_figments"]
        # 2. paragraph meta["role_figments"]
        if fig.kind in ("article", "image", "paragraph"):
            rfs = fig.meta.get("role_figments", [])
            new_rfs = _rewrite_list(rfs, remove_set, keep_id)
            if new_rfs is not rfs:
                fig.meta["role_figments"] = new_rfs
                to_upsert[fig.figment_id] = fig
                mutations += 1

        # 3. sentence.children
        if fig.kind == "sentence":
            children = list(fig.children)
            new_children = _rewrite_list(children, remove_set, keep_id)
            if new_children is not children:
                fig.children = new_children
                to_upsert[fig.figment_id] = fig
                mutations += 1

        # 4. relationship edges (edge_type="relationship")
        if fig.meta.get("edge_type") == "relationship":
            fa = fig.meta.get("figment_a")
            fb = fig.meta.get("figment_b")
            if fa in remove_set or fb in remove_set:
                old_rel_id = fig.figment_id
                new_fa = keep_id if fa in remove_set else fa
                new_fb = keep_id if fb in remove_set else fb
                pair = tuple(sorted([new_fa, new_fb]))
                new_id = hashlib.sha256(f"rel:{pair[0]}:{pair[1]}".encode()).hexdigest()[:16]
                weight = fig.meta.get("weight", 1)
                if new_id == old_rel_id:
                    fig.meta["figment_a"] = new_fa
                    fig.meta["figment_b"] = new_fb
                    to_upsert[fig.figment_id] = fig
                    mutations += 1
                else:
                    existing = by_id.get(new_id) or store.get(new_id)
                    if existing and existing.figment_id != old_rel_id:
                        existing.meta["weight"] = existing.meta.get("weight", 0) + weight
                        to_upsert[existing.figment_id] = existing
                        mutations += 1
                    else:
                        fig.figment_id = new_id
                        fig.meta["figment_a"] = new_fa
                        fig.meta["figment_b"] = new_fb
                        fig.text = f"Relationship: {new_fa[:8]} <-> {new_fb[:8]}"
                        to_upsert[fig.figment_id] = fig
                        mutations += 1
                    to_delete.add(old_rel_id)
                    mutations += 1  # deletion

        # 5. association edges (edge_type="association")
        if fig.meta.get("edge_type") == "association":
            links = fig.meta.get("links", [])
            new_links = _rewrite_list(links, remove_set, keep_id)
            if new_links is not links:
                old_assoc_id = fig.figment_id
                sorted_links = sorted(new_links)
                role = keep.meta.get("role", "") or fig.meta.get("role", "")
                new_id = hashlib.sha256(
                    f"assoc:{role}:{sorted_links[0]}:{sorted_links[1]}".encode()
                ).hexdigest()[:16]
                if new_id == old_assoc_id:
                    fig.meta["links"] = new_links
                    to_upsert[fig.figment_id] = fig
                    mutations += 1
                else:
                    existing = by_id.get(new_id) or store.get(new_id)
                    if existing and existing.figment_id != old_assoc_id:
                        pass  # edge already exists for new pair; skip
                    else:
                        fig.figment_id = new_id
                        fig.meta["links"] = new_links
                        fig.text = f"Association: {new_links[0][:8]} <-> {new_links[1][:8]} ({role})"
                        to_upsert[fig.figment_id] = fig
                        mutations += 1
                    to_delete.add(old_assoc_id)
                    mutations += 1  # deletion

        # 6. dedup_obs (kind="dedup_obs")
        if fig.kind == "dedup_obs":
            role_fig_a = fig.meta.get("role_figment_a")
            role_fig_b = fig.meta.get("role_figment_b")
            if role_fig_a in remove_set or role_fig_b in remove_set:
                fig.meta["role_figment_a"] = keep_id if role_fig_a in remove_set else role_fig_a
                fig.meta["role_figment_b"] = keep_id if role_fig_b in remove_set else role_fig_b
                to_upsert[fig.figment_id] = fig
                mutations += 1

    # ── Persist ──────────────────────────────────────────────────────────
    hidden = keep.boundary.shape[0]

    if to_upsert:
        store.upsert(list(to_upsert.values()), hidden_size=hidden)

    if to_delete:
        for fid in to_delete:
            try:
                store.delete(fid)
                mutations += 1
            except Exception:
                pass

    return mutations


def integrate_identity(store, role: str, normalized_text: str) -> list[str]:
    """Given a role + normalized text, return all variant figment IDs.

    Looks up an existing role figment by exact match, then expands through
    association edges. Returns the full set of variant IDs (including the
    original), or [] when no role figment matches.
    """
    for f in store.all():
        if f.meta.get("role") == role and f.meta.get("normalized") == normalized_text:
            return sorted(expand_identities(store, f.figment_id))
    return []


def _rewrite_list(
    items: list[str],
    remove_set: set[str],
    keep_id: str,
) -> list[str]:
    """Replace any ID in *remove_set* with *keep_id*, preserving order and deduplicating."""
    result: list[str] = []
    for item in items:
        if item in remove_set:
            if keep_id not in result:
                result.append(keep_id)
        else:
            result.append(item)
    if result == items:
        return items
    return result
