# Plan: fix cross-source recall bug + bring docs up to date

## Context

Ran `python examples/run_davos_v2.py all`. Per-source recall (QUERY 1 text + QUERY 4
boundary) is faithful and accurate. The cross-source trust-aware queries have one
real bug and one clarity issue; and the user-facing docs (README.md, AGENTS.md) are
stale/missing relative to the current code (P0–P3 work).

### Recall-accuracy verdict
- QUERY 1 (text) & QUERY 4 (boundary): accurate — all source figures/claims reproduced.
- QUERY 3 (disagree): accurate.
- QUERY 2 (agree): **BUG** — confabulates agreement the low-trust source never states.

### Bugs
1. **QUERY 2 fabricates agreement.** `examples/run_davos_v2.py:_run_trust_aware`
   prints the credibility context but does NOT inject it into the generation prompt;
   it feeds every keyword-overlapping figment from all sources, so the model invents
   cross-source "agreement" (e.g. claims the conspiracy text "implies" 193 UN
   endorsement). Fix in code + prompt.
2. **`shares claims with` == `contradicted by` (clarity).** In `figtree/graph.py:
   analyze_sources`, neutral topic overlap counts as "shares claims", so every source
   is listed as both corroborated and contradicted by the others. Separate
   *topic overlap* from *explicit agreement/disagreement*.
3. **Year drift 2023/2024 (minor).** Source texts have no year; model guesses. Add a
   guard instruction "only use facts present in the provided text; do not invent dates."

### Documentation issues (README.md, AGENTS.md)
- README:154 / AGENTS:184,222,228: "~22% faster" boundary generation — stale (current
  demo shows boundary ≈ text timing).
- README:184 / AGENTS:68: "4-bit models fall back to PyTorch matmul" — WRONG; the CUDA
  kernel now dequantizes 4-bit weights (commit 78125b6) and runs on the default model.
- AGENTS:158: `propagate_trust()` described as in-memory "alignment boost"; real impl is
  source-based, idempotent, disk-persisted.
- Missing: source-based trust, `analyze_sources`, `build_trust_aware_context`, persisted
  `trust:{source}` figments (mutable), `prompt.py` ChatML usage, separators,
  `repetition_penalty`.
- Benchmark tables list specific timings/sizes no longer matching current runs.

## Plan

### A. Fix QUERY 2 confabulation (bug 1)
- In `graph.build_trust_aware_context`, also return the structured relationships from
  `analyze_sources`: for each source, `agreeing_with`, `contradicting_with`,
  `related_with` (source-id lists).
- In `run_davos_v2.py:_run_trust_aware`, build the generation prompt to embed the
  grounded context, e.g.:
  "Sources that AGREE on this topic: {agreeing}. Sources that CONTRADICT: {contradicting}.
   Only state agreement that is explicitly present in the agreed sources' text."
- For the "agree" query, restrict `all_relevant` to figments from sources that actually
  corroborate each other (per `analyze_sources`); for "disagree", include the
  contradicting pairs explicitly. This removes room for fabrication.
- Add the "do not invent facts/dates not in the text" guard to the prompt used by both
  trust-aware queries (bug 3).

### B. Clarify credibility signals (bug 2)
- In `analyze_sources`, compute three disjoint-ish sets per source pair:
  - `related_with`: share entities (topic overlap).
  - `agreeing_with`: share entities AND not opposite cues (genuine alignment).
  - `contradicting_with`: share entities AND opposite cues.
- Update `propagate_trust` meta + rationale wording to use these distinct labels so the
  displayed credibility is not self-contradictory.

### C. Documentation (README.md + AGENTS.md)
- Remove/fix the "~22% faster" claim: either re-run `davos_benchmark_v2.py` and report
  real current numbers, or replace with a qualitative statement ("boundary path skips
  the figment forward pass; wall-clock speedup varies with context length").
- Fix the 4-bit kernel description: state the CUDA kernel dequantizes bitsandbytes 4-bit
  weights on the fly and is used for the default model; PyTorch matmul only as a fallback
  if the kernel cannot be built.
- Update `propagate_trust` description to source-based / idempotent / disk-persisted.
- Document the trust architecture: source-based trust, `analyze_sources`,
  `build_trust_aware_context`, persisted `trust:{source}` Figments, idempotent + mutable
  (future "accuracy proven → trust up" edits `meta["score"]` and re-runs).
- Document `prompt.py` ChatML usage, figment separators, and `repetition_penalty`.
- Refresh benchmark tables with current measured numbers (or mark as illustrative).
- Note known limitations: per-source recall is verbatim-grounded; cross-source synthesis
  is model-generated and can confabulate unless grounded (now mitigated by A); model may
  infer unspecified details (e.g. year).

## Files touched
- `figtree/graph.py` — `analyze_sources` distinct agree/relate/contradict sets; rationale.
- `examples/run_davos_v2.py` — `_run_trust_aware` grounds prompt with relationships +
  scopes retrieved figments; adds no-invent guard.
- `README.md`, `AGENTS.md` — accuracy + new trust architecture docs.

## Verification
- `ruff check figtree/ examples/ tests/`.
- `python tests/test_v2_pipeline.py` (still passes: boundary==text recall, persisted trust).
- `python examples/run_davos_v2.py all` — QUERY 2 must no longer claim the conspiracy
  source agrees on 193 UN / voluntary frameworks; QUERY 3 unchanged accurate; credibility
  context shows distinct agree/relate/contradict sets.
- Re-run benchmark; update doc numbers.
