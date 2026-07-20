# Plan: Flawless Recall — Root-Cause Fix (not patches)

## Context

We added a verify-and-patch loop (`figtree/recall.py` +
`FigmentGenerator.generate_with_recall`) to force per-source recall to 1.0. The
user correctly asks: *why do we need all these checks at all?* The diagnosis
(`explore` agent, grounded in `figtree/generate.py`) is clear:

**Retrieval is perfect. The figures are fully in the K/V cache and the prompt
attends to the entire source** (causal mask, prompt placed after all figments,
`generate.py:104-107`). Recall failure is a **generation-behavior** problem with
three root causes:

1. **Token-budget truncation (dominant).** A Davos narrative is ~440 tokens
   (`examples/davos_narratives/*.txt`, ~chars/4). A faithful "recount every
   figure" answer needs ~the same. QUERY 1 caps output at `max_new_tokens=400`
   (`run_davos_v2.py:222`) — *at or below* the minimum, so the model runs out of
   room before re-verbalizing early figures. The dropped "2,700 delegates" was
   the FIRST sentence: a truncation-order artifact, NOT attention blindness.
2. **Summarization prior.** "Recount the key details" makes the model *summarize*
   rather than *enumerate*; it never gets forced to surface each atom.
3. **Decoding fights rare tokens.** `temperature=0.7` lets it stochastically skip
   figures; `repetition_penalty=1.15` (`generate.py:404-413`) *suppresses*
   already-seen digit tokens, making the model less likely to emit a new but
   similar-looking number.

The verify loop exists only because open-ended generation under a tight budget
and summarization instinct won't enumerate every figure without per-atom
pressure. The real fixes attack roots 1–3 directly.

## Decisions (from user)
- **Primary mechanism: phased** — E1+E2 first (cheap, high-impact), then E3
  (chunked generation) as a follow-up.
- **Remove the verify-and-patch loop** once E1–E3 make single-pass recall reliable
  (guarded: only delete after measurement shows the loop is dead code).
- **Scope: all queries** — per-source recount AND trust-aware cross-source
  synthesis (QUERY 2/3).

## Phase 1 — Decode + budget + enumeration (cheap, high impact)

**E1 — Right-size budget + force enumeration.**
- In `figtree/generate.py` add a recall-oriented path (or params): for recall
  queries, set `max_new_tokens` from source length, e.g.
  `max_new_tokens = max(requested, int(source_token_estimate * 1.2) + 64)`.
  Source token estimate can come from `len(tokenizer.encode(text))` summed over
  fed figments, passed by the caller.
- Change QUERY 1/4 prompt in `examples/run_davos_v2.py` (and QUERY 2/3 grounded
  prompt) from "recount the key details, figures, and claims" to an explicit
  **enumeration template**, e.g.:
  "List EVERY figure from the source verbatim as a bullet list: each number,
  percent, year, and named entity. Do not summarize. Include all of them."
- Applies to all four queries (per-source + trust-aware), per user scope.

**E2 — Decode for fidelity on recall paths.**
- For recall-mode generation, use **greedy** (`temperature=0`, `top_k=1`,
  `top_p=1.0`) and **lower `repetition_penalty`** (1.0–1.05) so numbers aren't
  suppressed. Implement via a `faithful: bool` flag on `generate()` /
  `generate_with_recall()` that selects decode params, rather than hard-coding.

**E4 — Keep verifier as a temporary safety net + measurement.**
- Keep `generate_with_recall` for now but measure: after E1+E2, run the three
  Davos narratives and assert `recall_score == 1.0` *without* the patch ever
  triggering (log how many patch attempts occur). Add a counter so we can see
  the loop is dead.

**Tests / verification (Phase 1):**
- `tests/test_v2_pipeline.py`: keep the flawless-recall assertion; confirm patch
  attempts == 0 (loop is inert).
- Extend `davos_eval.py` to score Figtree recall on all queries, and report it
  alongside the RAG baseline (E5 measurement) so "flawless recall" is a comparable
  number, not just a per-source claim.

## Phase 2 — Chunked generation (structural guarantee)

**E3 — Per-figure-span generation.**
- Add `FigmentGenerator.generate_enumerated(figments, source_texts, ...)`:
  split each source into figure-bearing spans (reuse `split_into_sentences` +
  keep any span containing an atom from `figtree/recall.extract_atoms`), generate
  a 1–2 sentence verbatim restatement **per span**, then concatenate. Each figure
  sits in its own generation's focus window, so recall becomes a function of
  coverage, not the model's voluntary enumeration.
- This is the architectural fix; it changes output shape (concatenated spans) and
  costs more forward passes — acceptable on the 3 GB card for the recall task.

**Remove the verify-and-patch loop (final step, guarded):**
- Only after Phase 2 measurement shows single-pass (E1+E2+E3) recall == 1.0 on
  all narratives with zero patch triggers: delete `figtree/recall.py`'s patch
  usage from `generate.py`, remove `generate_with_recall`, and drop the
  `recall_score` plumbing. Keep `figtree/recall.py`'s *atom extraction/scoring*
  functions (they remain useful for the `davos_eval.py` metric, E5).

## Files touched (expected)
- `figtree/generate.py` — `faithful` decode flag; budget-from-source; new
  `generate_enumerated` (Phase 2); eventual removal of `generate_with_recall`.
- `examples/run_davos_v2.py` — enumeration prompts for QUERY 1–4; pass source
  lengths / `faithful=True`.
- `examples/davos_eval.py` — recall metric across all queries (E5).
- `tests/test_v2_pipeline.py` — assert flawless recall + inert verifier in Phase 1.
- `figtree/recall.py` — kept for atom scoring; patch logic removed in Phase 2.

## Quality gates
- ruff clean; `tests/test_recall.py` (CPU) + `tests/test_v2_pipeline.py` pass.
- Measured: all 3 Davos narratives + QUERY 2/3 reach `recall_score == 1.0`;
  verifier patch-trigger count == 0 before its removal.
- `figtree compare` still runs (RAG baseline unchanged).

## Open note
Phase 2 (E3) is the bigger change; shipping Phase 1 first keeps a reviewable,
low-risk PR. The verifier removal is explicitly deferred until measured dead.
