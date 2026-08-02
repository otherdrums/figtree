"""Prompt learning: single statements become permanent figments (one-shot memory).

A user statement (a name, a policy, a preference, a correction) is ingested
exactly like any other source, and durable facts are extracted from it and
linked into the figment graph:

- The statement itself becomes a small hierarchy of figments (article /
  paragraph / sentence) via :func:`figtree.ingest.ingest_text_to_figments`.
  Role extraction runs in the SAME forward pass through ``decode_prompt_fn``.
- Each extracted fact becomes a ``kind="role"`` figment with the deterministic
  id ``sha256(f"role:{role}:{normalized}")[:16]`` (the same id scheme used by
  figtree-news), so re-teaching the same fact updates in place and different
  surface forms hash to different ids — identity is then established by
  association edges, mirroring the app-layer association model.
- An ``edge_type="association"`` edge links the speaker (ACTOR node) to each
  fact node. Re-teaching a fact strengthens it; teaching a NEW value for the
  same role+actor supersedes the old value (old frames stay retrievable with
  ``meta["superseded_by"]``) — newest-high-trust-wins with provenance.

Learned figments carry ``meta["learning"]=True`` and provenance
(``learned_at``, ``session_id``) and deliberately do NOT set ``source_id``,
so they never leak into source-based trust propagation
(:meth:`figtree.graph.Figtree.analyze_sources`).
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Any

import numpy as np

from figtree.figment import Figment
from figtree.ingest import ingest_text_to_figments

# Roles the extractor may emit for a teaching statement.
LEARN_ROLES = {
    "ACTOR", "NAME", "PREFERENCE", "RELATIONSHIP", "POLICY", "CONSTRAINT",
    "FACT", "CORRECTION",
}

_EXTRACT_INSTRUCTION = """Extract durable facts from the statement below. Output one line per fact, exactly this format: ROLE|value

Roles:
- ACTOR   | who the fact is about. If the statement is first-person ("I", "me", "my"), value is "user"
- NAME    | an identity fact: a name to remember (person, database, host, function, client, ...)
- PREFERENCE   | something the user prefers or wants
- POLICY   | a standing rule or preferred method ("never X", "always Y", "use Z instead")
- CONSTRAINT   | a restriction or prohibition
- RELATIONSHIP | how two things relate
- FACT     | any other durable fact

If the statement corrects or replaces an earlier fact, first output: CORRECTION|ROLE
If the statement has no durable facts, output nothing.
Extract ONLY from the statement below — do not invent or repeat other facts.

Statement:
{statement}"""

_QUERY_INSTRUCTION = """The question below asks about remembered facts. Emit one line per fact category the question asks about, exactly this format: ROLE|value

Categories:
- ACTOR   | if the question refers to the user ("user", "my", "me", "I"), value is "user"
- NAME    | a name the question asks about
- PREFERENCE   | a preference the question asks about
- POLICY   | a rule or method the question asks about
- CONSTRAINT   | a restriction the question asks about
- RELATIONSHIP | a relationship the question asks about
- FACT     | any other learned fact

value = the entity word(s) from the question when present, otherwise empty.
Do NOT answer the question. If it asks about none of these categories, output nothing.

Question:
{question}"""


def _normalize(text: str) -> str:
    """Surface-form normalizer for identity (mirrors figtree-news normalize)."""
    t = text.lower()
    for prefix in (
        "mr ", "mrs ", "ms ", "dr ", "prof ", "sen ", "rep ", "gov ", "gen ",
        "col ", "lt ", "cpt ", "maj ", "capt ", "sgt ", "ambassador ",
        "judge ", "attorney ", "sheriff ", "officer ", "detective ",
    ):
        if t.startswith(prefix):
            t = t[len(prefix):]
            break
    t = t.replace("'s", "")
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def role_figment_id(role: str, value: str) -> str:
    """Deterministic id for a role figment: ``sha256(f"role:{role}:{normalized}")``.

    Shared with figtree-news's role scheme, so both domains can coexist in one
    store and exact duplicates auto-dedupe.
    """
    return hashlib.sha256(f"role:{role}:{_normalize(value)}".encode()).hexdigest()[:16]


def _assoc_id(role: str, a_id: str, b_id: str) -> str:
    return hashlib.sha256(f"assoc:{role}:{a_id}:{b_id}".encode()).hexdigest()[:16]


_FILLER_VALUES = {"none", "n/a", "na", "unknown", "no fact", "no", "same", "-", "none.", "n/a."}

# Rule-based query role extraction (small models fail at abstract extraction;
# statement extraction stays LLM-based, but questions are matched lexically).
_QUERY_LEXICON: dict[str, tuple[str, ...]] = {
    "ACTOR": ("user", "my", "me", "mine", "i'm", " i "),
    "NAME": ("name", "named", "called", "who is", "who's"),
    "PREFERENCE": ("prefer", "prefers", "favorite", "favourite", "want", "likes"),
    "POLICY": ("should", "use", "using", "instead", "method", "policy", "rule", "wrapper"),
    "CONSTRAINT": ("never", "forbidden", "prohibited", "not allowed", "allergic",
                   "must not", "don't", "restriction"),
    "RELATIONSHIP": ("related", "sister", "brother", "mother", "father", "wife",
                     "husband", "daughter", "son", "friend", "boss", "client"),
    "FACT": (),
}
_QUERY_STOPWORDS = {
    "what", "whats", "who", "whos", "how", "when", "where", "why", "the", "user",
    "users", "my", "your", "name", "is", "are", "do", "does", "did", "should",
    "of", "in", "to", "a", "an", "for", "with", "i", "me", "mine", "that", "this",
}


def extract_query_roles(query: str) -> list[tuple[str, str]]:
    """Rule-based extraction of learned-fact categories a question asks about.

    Deterministic and CPU-only (small models extract poorly from abstract
    questions). Returns ``[(ROLE, value)]`` with an empty value meaning "any";
    value is the first capitalized entity word in the question, if any.
    """
    q = query.lower()
    roles = [
        role for role, cues in _QUERY_LEXICON.items()
        if role != "FACT"
        and any(re.search(rf"(?<![a-z]){re.escape(cue)}(?![a-z])", q) for cue in cues)
    ]
    value = ""
    for tok in re.findall(r"[A-Z][A-Za-z']+", query):
        if tok.lower() not in _QUERY_STOPWORDS:
            value = tok.lower().strip("'s")
            break
    out = []
    for role in roles:
        out.append((role, "user" if role == "ACTOR" else value))
    return out


def parse_role_lines(text: str, allow_empty: bool = False) -> list[tuple[str, str]]:
    """Parse ``ROLE|value`` lines from LLM output into (role, value) pairs."""
    out: list[tuple[str, str]] = []
    for line in (text or "").splitlines():
        line = line.strip().strip('"').strip("'").rstrip(",")
        if not line or "|" not in line:
            continue
        role, _, value = line.partition("|")
        role = role.strip().upper()
        value = value.strip()
        if role == "CORRECTION":
            value = value.upper().strip()
        if role not in LEARN_ROLES:
            continue
        if role != "CORRECTION" and not value and not allow_empty:
            continue
        if role != "CORRECTION" and _normalize(value) in _FILLER_VALUES:
            continue
        out.append((role, value))
    return out


def _llm_decode(model, tokenizer, content: str, max_tokens: int = 96) -> str:
    """Greedily decode ``content`` with the model (ChatML, thinking disabled)."""
    import torch
    from transformers.cache_utils import DynamicCache

    from figtree.kernel.prompt import build_prompt_ids

    prompt_ids = build_prompt_ids(tokenizer, content, enable_thinking=False)
    if not prompt_ids:
        return ""
    device = model.device
    embed = model.get_input_embeddings()
    lm_head = model.lm_head
    final_norm = model.model.norm
    rotary = model.model.rotary_emb
    num_layers = model.config.num_hidden_layers
    eos = tokenizer.eos_token_id
    P = len(prompt_ids)

    attn_mask = torch.full((1, 1, P, P), float("-inf"), device=device, dtype=torch.float32)
    for i in range(P):
        attn_mask[:, :, i, : i + 1] = 0.0

    gen_ids: list[int] = []
    with torch.no_grad():
        h = embed(torch.tensor([prompt_ids], dtype=torch.long, device=device))
        pos = torch.arange(P, device=device, dtype=torch.long).unsqueeze(0)
        pe = rotary(h, pos)
        cache = DynamicCache()
        for li in range(num_layers):
            layer = model.model.layers[li]
            h = layer(
                h, attention_mask=attn_mask, position_ids=pos,
                position_embeddings=pe, use_cache=True, past_key_values=cache,
            )
        h = final_norm(h)
        logits = lm_head(h[:, -1:, :])

        for _ in range(max_tokens):
            nxt = int(logits[0, -1, :].argmax(dim=-1).item())
            if nxt == eos:
                break
            gen_ids.append(nxt)
            # Repetition guard: stop when the last 4 tokens repeat the 4 before.
            if len(gen_ids) >= 8 and gen_ids[-4:] == gen_ids[-8:-4]:
                break
            tok_emb = embed(torch.tensor([[nxt]], dtype=torch.long, device=device))
            cur_pos = torch.tensor([[P + len(gen_ids) - 1]], device=device, dtype=torch.long)
            pe_d = rotary(tok_emb, cur_pos)
            for li in range(num_layers):
                layer = model.model.layers[li]
                h = layer(
                    h, attention_mask=None, position_ids=cur_pos,
                    position_embeddings=pe_d, use_cache=True, past_key_values=cache,
                )
            h = final_norm(h)
            logits = lm_head(h[:, -1:, :])

    del cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def extract_roles(
    model,
    tokenizer,
    text: str,
    query_mode: bool = False,
    max_tokens: int = 48,
) -> list[tuple[str, str]]:
    """Extract (role, value) facts from a statement or query with the model.

    ``query_mode=True`` answers which learned-fact roles a question asks about
    (values may be empty, meaning "any").
    """
    instruction = _QUERY_INSTRUCTION if query_mode else _EXTRACT_INSTRUCTION
    prompt = instruction.format(
        question=text if query_mode else "",
        statement=text if not query_mode else "",
    )
    out = _llm_decode(model, tokenizer, prompt, max_tokens=max_tokens)
    return parse_role_lines(out, allow_empty=query_mode)


def apply_roles_to_store(
    statement_id: str,
    statement_text: str,
    roles: list[tuple[str, str]],
    store,
    session_id: str = "user",
    trust: float = 0.95,
    learned_at: float | None = None,
    hidden_size: int | None = None,
) -> dict[str, Any]:
    """Create/update identity + association figments for extracted roles.

    Pure store logic (no model needed) so the identity/conflict machinery is
    unit-testable. Roles are ``(ROLE, value)`` pairs from :func:`extract_roles`
    or :func:`parse_role_lines`. Returns a summary dict with ``roles``,
    ``created``, ``updated``, ``superseded``, ``corrections``.
    """
    result: dict[str, Any] = {
        "roles": list(roles),
        "created": [],
        "updated": [],
        "superseded": [],
        "corrections": [],
    }
    if not roles:
        return result

    statement = store.get(statement_id)
    if statement is None:
        raise ValueError(f"Statement figment {statement_id!r} not in store")
    learned_at = learned_at if learned_at is not None else time.time()
    hs = hidden_size or statement.boundary.shape[0]
    boundary = statement.boundary.astype(np.float32).copy()

    corrections = {v for r, v in roles if r == "CORRECTION"}
    fact_roles = [(r, v) for r, v in roles if r not in ("CORRECTION", "ACTOR")]
    result["corrections"] = sorted(corrections)

    actor_value = next((v for r, v in roles if r == "ACTOR"), session_id)
    actor_id = role_figment_id("ACTOR", actor_value)

    def _stamp(meta: dict, kind: str) -> dict:
        meta["learning"] = True
        meta["session_id"] = session_id
        meta["learned_at"] = learned_at
        meta["provenance"] = {
            "type": "prompt",
            "statement_id": statement_id,
            "session_id": session_id,
            "learned_at": learned_at,
        }
        meta["kind_note"] = kind
        return meta

    def _get_or_create_role(role: str, value: str) -> tuple[Figment, bool]:
        normalized = _normalize(value)
        fid = role_figment_id(role, value)
        existing = store.get(fid)
        if existing is not None and existing.kind == "role":
            refs = list(existing.meta.get("references", []))
            if statement_id not in refs:
                refs.append(statement_id)
            existing.meta["references"] = refs
            existing.meta["reference_count"] = len(refs)
            existing.meta["learned_at"] = learned_at
            result["updated"].append(fid)
            return existing, False
        fig = Figment.create(
            text=value,
            boundary=boundary,
            meta=_stamp(
                {
                    "role": role,
                    "normalized": normalized,
                    "references": [statement_id],
                    "reference_count": 1,
                },
                f"learned:{role}",
            ),
            figment_id=fid,
            trust=min(1.0, max(0.0, trust)),
            kind="role",
        )
        result["created"].append(fid)
        return fig, True

    # ACTOR node: the speaker/session identity every learned fact attaches to.
    actor, actor_new = _get_or_create_role("ACTOR", actor_value)
    figments: list[Figment] = []
    if actor_new:
        figments.append(actor)

    all_figs = store.all()
    active_assocs = [
        f for f in all_figs
        if f.meta.get("edge_type") == "association"
        and f.meta.get("role") is not None
        and actor_id in (f.meta.get("links") or [])
        and not f.meta.get("superseded_by")
        and not f.meta.get("forgotten")
    ]

    for role, value in fact_roles:
        if not value.strip():
            continue
        node, is_new = _get_or_create_role(role, value)
        if is_new:
            figments.append(node)
        else:
            figments.append(node)
        node_id = node.figment_id

        # Find the currently-active value this actor holds for `role`.
        current = None
        for assoc in active_assocs:
            if assoc.meta.get("role") == role:
                other = [x for x in assoc.meta.get("links", []) if x != actor_id]
                current = (assoc, other[0] if other else None)
                break

        if current is not None:
            cur_assoc, cur_node_id = current
            if cur_node_id == node_id:
                # Same value re-taught: strengthen, never duplicate.
                cur_assoc.meta["strengthened_at"] = learned_at
                figments.append(cur_assoc)
                continue
            # New value: newest-high-trust wins; old frames stay retrievable.
            old_node = store.get(cur_node_id) if cur_node_id else None
            if old_node is not None and not old_node.meta.get("superseded_by"):
                old_node.meta["superseded_by"] = node_id
                old_node.meta["superseded_at"] = learned_at
                old_node.meta["superseded_by_statement"] = statement_id
                result["superseded"].append(cur_node_id)
                figments.append(old_node)
            cur_assoc.meta["superseded_by"] = _assoc_id(role, actor_id, node_id)
            cur_assoc.meta["superseded_at"] = learned_at
            figments.append(cur_assoc)

        assoc = Figment.create(
            text=f"Association: {actor.text[:40]} <-> {node.text[:40]} ({role})",
            boundary=boundary,
            meta=_stamp(
                {
                    "edge_type": "association",
                    "role": role,
                    "links": [actor_id, node_id],
                    "confidence": min(1.0, max(0.0, trust)),
                    "evidence": "prompt",
                    "statement_id": statement_id,
                },
                f"learned:assoc:{role}",
            ),
            figment_id=_assoc_id(role, actor_id, node_id),
            trust=min(1.0, max(0.0, trust)),
            kind="edge",
        )
        figments.append(assoc)

    # Corrections with no replacement: retract the currently-active value.
    for role in sorted(corrections - {r for r, _ in fact_roles}):
        for assoc in active_assocs:
            if assoc.meta.get("role") != role:
                continue
            other = [x for x in assoc.meta.get("links", []) if x != actor_id]
            node = store.get(other[0]) if other else None
            if node is not None and not node.meta.get("superseded_by"):
                node.meta["superseded_by"] = "retracted"
                node.meta["superseded_at"] = learned_at
                node.meta["superseded_by_statement"] = statement_id
                result["superseded"].append(node.figment_id)
                figments.append(node)
            assoc.meta["superseded_by"] = "retracted"
            assoc.meta["superseded_at"] = learned_at
            figments.append(assoc)

    # Tag the statement itself with what it taught.
    statement.meta["learning"] = True
    statement.meta["learned_roles"] = {r: v for r, v in fact_roles}
    statement.meta["learned_at"] = learned_at
    statement.meta["provenance"] = {
        "type": "prompt",
        "statement_id": statement_id,
        "session_id": session_id,
        "learned_at": learned_at,
    }
    figments.append(statement)

    if figments:
        store.upsert(figments, hidden_size=hs)
    result["learned_at"] = learned_at
    result["statement_id"] = statement_id
    result["actor_id"] = actor_id
    return result


def teach(
    model,
    tokenizer,
    statement: str,
    store,
    session_id: str = "user",
    trust: float = 0.95,
    min_chars: int = 10,
    decode_max_tokens: int = 96,
    **ingest_kwargs: Any,
) -> dict[str, Any]:
    """Teach the system a single statement; returns the learning summary.

    The statement is ingested like any source (single forward pass) with role
    extraction appended to the same cache, then identity/association figments
    are created by :func:`apply_roles_to_store`. The result contains
    ``statement_id``, ``roles``, ``created``, ``updated``, ``superseded``,
    ``learned_at`` and the raw ``decode_output``.
    """
    if store is None:
        raise ValueError(
            "store is required: learned figments are persisted to a LanceDB store. "
            "Pass a FigmentStore from figtree.lancedb_store.connect()."
        )

    def _decode_prompt_fn(paragraphs, kept_sentences, sentence_to_paragraph):
        return _EXTRACT_INSTRUCTION.format(statement=statement)

    figments = ingest_text_to_figments(
        model=model,
        tokenizer=tokenizer,
        text=statement,
        source_id="",
        trust=trust,
        store=store,
        min_chars=min_chars,
        decode_prompt_fn=_decode_prompt_fn,
        decode_max_tokens=decode_max_tokens,
        decode_temperature=0.0,
        **ingest_kwargs,
    )

    image = figments[0]
    sentence = next((f for f in figments if f.kind == "sentence"), None)
    if sentence is None:
        raise ValueError("Statement produced no sentence figment")
    decode_output = image.meta.get("decode_output", "") or ""
    roles = parse_role_lines(decode_output)
    if not roles:
        roles = [("FACT", statement.strip())]

    result = apply_roles_to_store(
        sentence.figment_id, sentence.text, roles, store,
        session_id=session_id, trust=trust,
    )
    result["decode_output"] = decode_output
    result["statement_id"] = sentence.figment_id

    # Stamp the statement hierarchy root (image) with provenance.
    image.meta["learning"] = True
    image.meta["provenance"] = {
        "type": "prompt",
        "statement_id": sentence.figment_id,
        "session_id": session_id,
        "learned_at": result["learned_at"],
    }
    store.upsert_one(image)
    return result


def forget(store, role: str, value: str, session_id: str = "user") -> bool:
    """Forget a learned fact: mark its association forgotten (frames kept).

    Returns True if an active learned association for (role, value) was found
    and retired.
    """
    actor_id = role_figment_id("ACTOR", session_id)
    target = _normalize(value)
    for f in store.all():
        if f.meta.get("edge_type") != "association":
            continue
        if f.meta.get("role") != role or f.meta.get("forgotten") or f.meta.get("superseded_by"):
            continue
        links = f.meta.get("links") or []
        if actor_id not in links:
            continue
        other = [x for x in links if x != actor_id]
        if not other:
            continue
        node = store.get(other[0])
        if node is not None and node.meta.get("normalized") == target:
            f.meta["forgotten"] = True
            f.meta["forgotten_at"] = time.time()
            node.meta["superseded_by"] = "forgotten"
            node.meta["superseded_at"] = time.time()
            store.upsert([f, node], hidden_size=node.boundary.shape[0])
            return True
    return False


def learned_facts(store, session_id: str | None = None) -> list[Figment]:
    """Return learned role figments (optionally for one session), newest first."""
    out = []
    for f in store.all():
        if f.kind != "role" or f.meta.get("learning") is not True:
            continue
        if f.meta.get("role") == "ACTOR":
            continue
        if session_id is not None and f.meta.get("session_id") != session_id:
            continue
        out.append(f)
    out.sort(key=lambda f: float(f.meta.get("learned_at", 0.0)), reverse=True)
    return out
