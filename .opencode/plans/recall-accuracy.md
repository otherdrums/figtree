# Plan: Faithful recall + source-based trust awareness (P0–P3)

## Context

Ran `python examples/run_davos_v2.py all`. The pipeline works end-to-end, but
output quality has defects that hurt faithful recall and trust-aware reasoning.

Confirmed facts from the run + code inspection:
- Atomic figments carry `meta["source_id"]` (ingest.py), so grouping by source is
  directly possible.
- Cached K/V are faithful to the model (kernel dequant == Linear4bit forward, diff
  0.0). The boundary path degrades only because of replay defects (below), not bad
  storage.
- Trust Figments already exist (`edge_type="trust"`, `score`, `about_figment`) and
  flow through the graph, but `propagate_trust` mutates trust only in memory and
  never persists it.

### Observed defects
1. **Boundary path run-on (QUERY 4).** `generate_from_boundaries`
   (generate.py:194-233) concatenates figment K/V with **no separators** between
   figments, unlike `generate()` which now inserts `\n\n` separators
   (generate.py:66-71). Short text → coherent; full ~14-figment article → the model
   falls into "complete the document" mode ("Write a 2000-word article…"). This is a
   *replay* defect.
2. **Repetition loops (QUERY 2).** "What are the key differences…" ×4 — no
   repetition penalty in `_sample` (generate.py:317+).
3. **No chat template.** Bare prompt (`add_special_tokens=False`) in both paths;
   `figtree/kernel/prompt.py:build_prompt_ids` (Qwen3 ChatML) is unused.
4. **Trust is not source-aware or explainable.** The demo's "Top Trusted" list is
   just the highest-source-trust narrative's sentences. There is no
   credibility-aware recall of *conflicting* narratives and no explanation of *why*
   a differing narrative is more/less credible.

### Guiding principle (user)
The whole point is to **recall conflicting narratives with trust awareness** and
**mutable** trust (trust can rise as a source's accuracy is proven — a *future*
trigger, but the design must support it). Decisions below keep trust as a
first-class, persisted, re-runnable Figment.

## Goal
Answers that (a) faithfully recall ingested facts, (b) never run on / loop,
(c) recall all source perspectives, and (d) explain each perspective's credibility
from source trust + cross-source corroboration. Trust lives on disk as Figments and
can be recomputed/edited without losing history.

---

## P0 — Boundary replay == text forward (highest recall leverage)

Make `generate_from_boundaries` produce output equivalent to `generate()`.

- **Ingest: one concatenated forward with separators.** In `figtree/ingest.py`,
  build the concatenated token stream with the same `\n\n` separators `generate()`
  uses, forward it once through the model capturing a `DynamicCache`, then slice
  per-figment K/V by each figment's [start,end) token span and save per-figment
  `kv_cache.npy`. Keeps `boundaries`/`boundary_emb` per figment as today. This makes
  cached K/V encode the same cross-figment context the text path has.
- **Replay: add separators** in `generate_from_boundaries` too — forward the few
  separator tokens through the model to obtain their K/V and interleave between
  figment slices, matching `generate()`.

## P1 — Sampling quality
- Add `repetition_penalty` (≈1.15) to `_sample` and thread it through both
  `generate` and `generate_from_boundaries`. Kills QUERY-2-style loops.

## P2 — Chat template grounding
- Use `figtree/kernel/prompt.py:build_prompt_ids` to wrap the prompt in the Qwen3
  ChatML template (assistant turn, thinking disabled) in BOTH paths. Improves
  grounding and stops "complete the document" behavior.

## P3 — Source-based trust, multi-perspective recall, explainability (mutable)

Reframed per user direction: **trust is source-based**, built from each source's
trust score; the system recalls every perspective and explains credibility.

### P3a — Trust as persisted, re-runnable Figments
- `graph.propagate_trust` becomes **idempotent and disk-persistent**: it recomputes
  the trust Figment(s) for each source (score from the source's base trust, adjusted
  by corroboration/contradiction) and **overwrites the existing `*.figment` trust
  Figment on disk** (not just in-memory). Future "accuracy proven → trust up" only
  edits `meta["score"]` on that Figment and re-runs `propagate_trust`.
- Keep a single canonical trust Figment per `source_id` (id derived from
  `source_id` so re-runs overwrite the same file). Store `base_trust`,
  `adjusted_trust`, and the list of corroborating/contradicting source_ids in
  `meta` for explainability.

### P3b — Corroboration & contradiction (source-based, not fuzzy)
- Group atomic figments by `source_id`. For each *claim cluster* (figments across
  sources that share entities — reuse `create_edges`' entity overlap), compute:
  - `corroborating_sources` = distinct sources asserting the cluster.
  - `contradicting_sources` = sources whose figments share entities but carry
    opposing cue words ("not binding", "failed", "no agreement" vs "unanimous",
    "binding", "endorsed").
  - `credibility` = function of (mean trust of corroborating sources) and
    (whether high-trust sources corroborate vs. only a low-trust source asserts it).
- Emit `supports` / `contradicts` edge Figments per cluster (source-keyed, not
  sentence-fuzzy), so the graph is explainable.

### P3c — Trust-aware context builder + explanation generation
- New `Figtree.build_trust_aware_context(query, figments) -> dict` that returns, for
  the query, the grouped narratives by source (text + trust score) and a short
  credibility rationale (which sources agree, which contradict, and the resulting
  confidence). This is the "explain why a differing narrative may/may not be
  credible" capability.
- Wire into `run_davos_v2.py`: replace/augment QUERY 2 ("agree") and QUERY 3
  ("disagree") with trust-aware variants that (1) recall all perspectives and (2)
  surface the credibility rationale alongside the generated answer. QUERY 4
  (boundary) gets the same trust-aware context.

### P3d — Selection respects (but does not hide) conflict
- Generation should recall *all* perspectives (not just the highest-trust one); the
  answer is expected to note disagreements and their credibility. Provide a
  `trust_weighted` flag: when True, lower-trust-only claims are still included but
  labeled low-confidence in the context.

---

## Files touched
- `figtree/ingest.py` — concatenated-with-separators forward; per-figment kv_cache slicing (P0).
- `figtree/generate.py` — separators in boundary replay (P0); repetition_penalty (P1);
  chat template via `prompt.py` (P2).
- `figtree/graph.py` — persistent/idempotent `propagate_trust` (P3a); source-based
  corroboration/contradiction (P3b); `build_trust_aware_context` (P3c).
- `figtree/figment.py` — ensure trust Figment re-save is clean (idempotent id from
  source_id).
- `examples/run_davos_v2.py` — trust-aware QUERY 2/3/4; pass trust-weighted context.
- `tests/test_v2_pipeline.py` — assert boundary output is on-topic AND close to text
  output on a multi-figment input; assert trust Figments are persisted to disk and
  re-propagatable.

## Verification
- `ruff check figtree/ examples/ tests/`.
- `python tests/test_v2_pipeline.py` — boundary≈text equivalence + persisted trust.
- `python examples/run_davos_v2.py all` — QUERY 4 matches QUERY 1 quality, no
  repetition loops, and QUERY 2/3 recall all perspectives with a credibility
  rationale. Inspect `davos_run_*.log` / `davos_graph_*.log`.

## Notes / future hooks
- The "accuracy proven → trust up" trigger is intentionally out of scope but the
  design supports it: edit a source's trust Figment `meta["score"]` and re-run
  `propagate_trust` (idempotent overwrite). No schema change needed later.
- Contradiction cues are a small lexicon now; can later be replaced by a classifier
  without changing the Figment/graph shape.
