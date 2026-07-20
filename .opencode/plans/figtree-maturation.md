# Plan: FigTree Maturation — Close the Assessment Gaps

**Goal:** Address the four highest-leverage next steps from the v0.2.0 assessment so
FigTree reads as a coherent, self-consistent research system rather than a transitional
one. All four workstreams are in scope; license = **MIT**.

**Current state (verified):**
- No LICENSE; `pyproject.toml` has deps + `dev`/`storage` extras but **no entry points**.
- `.figment/` still exists in `figment.py:80-129` (`save`/`load`) and as backward-compat
  branches in `ingest.py:157,165-167,290-293,326-330` (only when `store` omitted) and
  `generate.py:208-218` (only when `kv_manager is None` + `cache_dir`). Main path is LanceDB.
- Tests = one end-to-end smoke script (`tests/test_v2_pipeline.py`); run as a script, no
  pytest unit tests.
- KV is sentence-level atomic, lazy by default, fp16/int8 quantization available.
- `graph.py` multi-source conflict handling is stance-heuristic, no LLM resolution.
- "Image" boundary = copy of first sentence's boundary (`ingest.py:307,315`); no summarizer.
- No conventional-RAG baseline comparison anywhere.

---

## Workstream 1 — LICENSE + Packaging

**1.1 Add `LICENSE` (MIT).** New file at repo root: standard MIT text, copyright
"2026 <your name / Figtree authors>". Keep it permissive and short.

**1.2 Tighten `pyproject.toml`:**
- Add `license = {text = "MIT"}` (or `license-files`), `authors`, `readme = "README.md"`,
  `keywords`, `classifiers` (License :: OSI Approved :: MIT License; Development Status ::
  3 - Alpha; Programming Language :: Python :: 3.10/3.11/3.12; Topic :: Scientific/Engineering :: AI).
- Add `[project.urls]` (Homepage/Repository → GitHub).
- Add `[project.scripts]` console entry points so the package is runnable, e.g.:
  `figtree = "figtree.cli:app"` — **or**, to avoid building a new CLI, point at the existing
  Davos demo: `figtree-davos = "examples.run_davos_v2:main"`. Recommend a thin
  `figtree/cli.py` Typer app wrapping `run_davos_v2` phases (ingest/generate/benchmark) so
  the package has a real entry point. (Confirm with user whether a CLI stub is wanted vs
  just wiring the demo as a script entry.)
- Bump version `0.2.0` → `0.2.1` (post-assessment maturation; keep semver honest as Alpha).

**1.3 Verify:** `pip install -e .` still works; `figtree --help` (or the chosen script)
runs; `python -c "import figtree"` clean.

---

## Workstream 2 — Kill the old `.figment/` path

Make LanceDB + external KV blobs the **sole** persistence path. Removes transitional cruft
the assessment flagged.

**2.1 `figtree/figment.py`:** Delete `save()` (80-103) and `load()` (105-129). `Figment`
stays a pure dataclass + `to_record`/`from_record` (Lance rows). Keep `hidden_size` attr.

**2.2 `figtree/ingest.py`:**
- Remove `output_dir` backward-compat branch (157, 165-167, 290-293, 326-330). Require `store`
  (raise `ValueError` if `store is None` with a clear message). Keep `kv_manager` optional
  (lazy KV is the default; eager only if `compute_kv=True` + `kv_manager` given).
- Update docstring; drop `.figment` mentions.

**2.3 `figtree/generate.py`:**
- Remove `cache_dir` fallback path (208-218); require `kv_manager` for boundary generation.
  Update docstring. Keep `generate` (text-based) unchanged.

**2.4 Cleanup:** Remove `*.figment/` from `.gitignore` (no longer produced). Grep for any
  remaining `*.figment`, `figment_id}.figment`, `Figment.save`/`load` references across repo
  and remove. `lancedb_store.py:3` docstring: reword from "Replaces the raw `.figment/`
  layout" to "Figments are persisted in a LanceDB table."

**2.5 Docs:** `README.md` / `AGENTS.md` already describe LanceDB as current — confirm no
  `.figment/` "deprecated" language remains (done in prior commit). Add a one-line note that
  the directory format is removed.

**2.6 Tests:** `test_v2_pipeline.py` must not rely on `.figment` (it uses the store — verify
  and adjust if any `cache_dir`/`output_dir` usage remains).

---

## Workstream 3 — Conventional RAG baseline on Davos task

Add a fair, low-effort baseline to quantify FigTree's delta (fidelity, contradiction
awareness, VRAM, latency) — the assessment's "measure against a baseline" ask.

**3.1 New `examples/rag_baseline_davos.py` (or `tests/bench_rag_baseline.py`):**
- Same model (`unsloth/Qwen3-4B-bnb-4bit`) + same 3 Davos narratives + same queries.
- Baseline = chunk each narrative by sentence, embed with the **model's last-token hidden
  state at the same crystal layer** (reuse `ingest` boundary extraction so embedding parity
  holds), store in a plain in-memory list / small Lance table, retrieve top-k by cosine,
  stuff into the same ChatML prompt, generate. **No** KV replay, **no** trust graph.
- Keep it dependency-light (numpy + existing model; reuse `lancedb_store.search` or a simple
  cosine top-k to avoid new deps).

**3.2 Shared harness:** Extract the Davos query set + scoring into a small helper
(`examples/davos_eval.py`) reused by both FigTree benchmark and RAG baseline:
- Queries: per-source factual recall (QUERY 1), cross-source contradiction awareness
  (QUERY 2/3), open synthesis (QUERY 4).
- Metrics: (a) factual fidelity — exact-figure match vs source text; (b) contradiction
  awareness — does output note disagreement between sources; (c) VRAM peak (`torch.cuda.max_memory_allocated`); (d) latency (ingest + generate wall-clock).

**3.3 Output:** `examples/davos_benchmark_v2.py` gains a `--baseline` mode (or a new
`davos_compare.py`) printing a side-by-side table: FigTree vs RAG on the 4 metrics. Keep
claims modest (mirror AGENTS.md candor).

**3.4 Note:** This is measurement, not a new architecture. Guard GPU cost; allow `--max-new-tokens` small default so it runs on the 3 GB card.

---

## Workstream 4 — Hierarchical Figments (summarized boundaries)

Give higher-level "image" figments a **real** summarized boundary instead of copying the
first sentence's (current `ingest.py:307,315`).

**4.1 Add `figtree/summarize.py`:**
- `summarize_image(model, tokenizer, child_figments, prompt=...)` → produces a natural-language
  summary text of the image's children (one short generation pass, max_new_tokens small), then
  runs that summary text through the same `ingest_text_to_figments` boundary path to get a
  **summary boundary** + `boundary_emb`. Optionally also store the summary as the image's
  `text` (or keep source text + add `meta["summary"]`).
- Keep it optional (`summarize_images=False` default) to avoid surprising VRAM/latency.

**4.2 `figtree/ingest.py`:** when building the image figment, if `summarize_images`, use the
  summarized boundary; else fall back to current first-sentence copy (documented). Image's
  `children` already lists atomic figment ids — unchanged.

**4.3 `figtree/graph.py` / retrieval:** ensure ANN search can use the image-level boundary for
  coarse retrieval and drill into children (already possible since children are separate
  figments). Document the two-level retrieval pattern.

**4.4 Tests:** add a small unit test (`tests/test_hierarchy.py`) that ingests a 2-sentence text
  with `summarize_images=True`, asserts the image boundary differs from the raw first-sentence
  copy and that `children` are linked. Keep it CPU-friendly (tiny model or boundary-shape mock).

---

## Testing & Quality Gates (all workstreams)
- `ruff check figtree/ examples/ tests/` clean.
- `python3 tests/test_v2_pipeline.py` passes (store round-trip, compression, lazy-KV gen,
  persisted/idempotent trust).
- New: `tests/test_hierarchy.py` (Workstream 4) and a non-GPU unit test for `rag_baseline`
  retrieval parity (cosine top-k) so it runs in CI without a 3 GB GPU.
- Confirm `pip install -e .` + entry point works (Workstream 1).

## Docs
- `README.md`: add License section + badge-less note; mention RAG baseline results table;
  note hierarchical `summarize_images` option; state `.figment/` format removed.
- `AGENTS.md`: update architecture tree if `summarize.py`/`cli.py` added; refresh Known
  Limitations (remove transitional `.figment` item; note baseline exists).

## Commit / PR strategy
- Feature branch `feat/maturation` off `master`; one commit per workstream (or squashed PR).
- PR description summarizes the four changes + benchmark delta.

## Open questions for user — RESOLVED
- Q1: Packaging entry point → **Option A: thin `figtree/cli.py` Typer wrapper** exposing
  `ingest` / `generate` / `benchmark` subcommands that call the existing Davos demo phases.
- Q2: MIT LICENSE copyright holder → **Brian Mulkern**.
- Q3: Hierarchical summarization default → **opt-in `False`** (protects low-VRAM users on the
  3 GB card; pass `summarize_images=True` to enable).
