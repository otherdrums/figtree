# Plan: Long-source chunked enumerated generation (E3) + branch cleanup

## Context

Flawless recall is already achieved for the Davos narratives (short sources, ~440
tokens) via `FigmentGenerator.generate_faithful`: greedy decode + a source-sized
budget (≥1.2× source length) reproduces every atom in a single pass
(`recall_score = 1.0`).

That budget scales linearly. For a long source (e.g. a 5,000-token report) the
single-pass output would need ~6,000 tokens, which (a) risks truncation/drift
before the final figures are restated and (b) costs one very long decode. **E3**
adds an opt-in `generate_enumerated` path that splits a long source into
figure-bearing spans, generates a short faithful restatement per span, and
concatenates — so every figure sits in a focused generation window and total
output stays bounded per span.

### Feedback reconciliation (already true in repo, just verify)

The progress feedback flagged two gaps. Both are already satisfied in the current
`master` working tree; the plan confirms and finishes the small loose ends:

1. **LICENSE file** — already present at repo root (`LICENSE`, MIT, © 2026 Brian
   Mulkern) and tracked. No action needed beyond confirming it stays committed.
2. **`feat/maturation` branch** — already fully merged into `master` (it is an
   ancestor of `master`; the only commits ahead of it on `master` are the merge
   commit `21e9a4f` and the patcher-removal `410e82d`). Cleanup = delete the
   now-merged branch locally and on the remote. No divergence to reconcile.

## Design

### New method: `FigmentGenerator.generate_enumerated`

Signature:
```python
def generate_enumerated(
    self,
    figments: list[Figment],
    prompt: str,
    source_texts: list[str] | None = None,
    max_new_tokens: int = 400,
    chunk_tokens: int = 350,      # ~1 figure-bearing paragraph
    overlap_tokens: int = 40,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.15,
) -> dict:
```

Behavior:
1. If `source_texts` is None or empty, delegate straight to `generate_faithful`
   (single-pass path; no chunking). This keeps short sources unchanged.
2. Concatenate `source_texts` into one blob. Tokenize with the model tokenizer
   (`add_special_tokens=False`).
3. Walk the token list in windows of `chunk_tokens` with `overlap_tokens`
   overlap. Decode each window back to text (lossless-ish; boundaries may cut a
   word, which is fine for a restatement prompt).
4. For each chunk, build a per-chunk prompt:
   ```
   {prompt}

   Restate ONLY the facts in the following source excerpt, verbatim, as a bullet
   list — every number, percent, year, and named entity. Do not add or infer.

   Excerpt:
   "<chunk text>"
   ```
   Call `self.generate(figments=figments, prompt=..., faithful=True,
   max_new_tokens=max(chunk_max, int(len(chunk_tokens)*1.2)+32))`.
5. Concatenate the per-chunk outputs (strip + join with `\n\n`).
6. Attach `recall_score` / `missing_atoms` over the full `source_blob` (same
   measurement as `generate_faithful`) and `num_tokens` summed across chunks.
7. Return the same dict shape as `generate_faithful` so callers/reports are
   unchanged.

`figments` are passed to every chunk call so the model's attention still sees the
ingested K/V context for grounding; the chunk text is the *focus* excerpt in the
prompt.

### Boundary path

`generate_from_boundaries` already supports `faithful=True` + `source_tokens`.
E3 only adds the chunking orchestration in `generate_enumerated`. To keep scope
tight and avoid a second code path, `generate_enumerated` uses the **text**
`generate` path (figments → K/V on the fly) for each chunk. The boundary path is
used elsewhere via `generate_from_boundaries` directly; enumerated chunking over
boundary K/V is a later extension if needed. (Documented as a known limitation.)

### Wiring into the demo (`examples/run_davos_v2.py`)

Add a threshold constant `ENUMERATE_TOKEN_THRESHOLD = 600` (tuned so Davos
narratives ~440 tokens stay on the single-pass path). In the per-source recall
helper (`_run_recall` / the QUERY 1 block) and the trust-aware block
(`_run_trust_aware`), if total source tokens exceed the threshold, call
`gen.generate_enumerated(...)` instead of `gen.generate_faithful(...)`. Both
return the same dict shape, so reporting code is unchanged.

No behavior change for current Davos demo (all sources < threshold).

### Tests

- `tests/test_recall.py` (CPU, no GPU): add `test_split_source_into_chunks` that
  asserts a long synthetic source (concat of several atom-bearing paragraphs) is
  split into ≥2 chunks by a small pure helper. Extract the chunking into a
  module-level helper `_chunk_text(tokenizer, text, chunk_tokens, overlap_tokens)`
  in `figtree/generate.py` so it is unit-testable without a model.
- `tests/test_v2_pipeline.py` (GPU, guarded): keep the existing `recall >= 1.0`
  assertion. Add a second recall block that builds a *long* synthetic source
  (repeat the Davos TEXT several times with distinct atoms) and calls
  `generate_enumerated`, asserting `recall_score >= 1.0`. Only runs when CUDA is
  available (the script already guards on GPU).

### Docs

- `AGENTS.md`: update design decision #7 wording to mention `generate_enumerated`
  as the long-source fallback (opt-in by token threshold); single-pass
  `generate_faithful` for short sources.
- `README.md`: add one line under the "Flawless recall" section noting that
  sources exceeding the threshold auto-switch to chunked enumerated generation.

## Files to change

- `figtree/generate.py` — add `_chunk_text` helper + `generate_enumerated`.
- `examples/run_davos_v2.py` — threshold + branch to `generate_enumerated`.
- `tests/test_recall.py` — chunk helper unit test.
- `tests/test_v2_pipeline.py` — long-source enumerated recall assertion (GPU).
- `AGENTS.md`, `README.md` — doc note.

## Verification

- `ruff check figtree/ examples/ tests/`
- `python -m pytest tests/test_recall.py` (CPU, 6 tests)
- `python tests/test_v2_pipeline.py` (GPU): existing single-pass recall 1.0 AND
  new long-source enumerated recall 1.0.
- Delete `feat/maturation` branch: `git branch -d feat/maturation` and
  `git push origin --delete feat/maturation` (after confirming it is an ancestor
  of master — it is).

## Commits (on master)

1. `feat(generate): enumerated chunked generation for long sources (E3)` —
   helper + method + demo threshold + tests + docs.
2. `chore: delete merged feat/maturation branch` — after merge confirmed.

## Out of scope

- Boundary-K/V enumerated chunking (separate later extension).
- Any change to the faithful single-pass path (verified flawless; untouched).
