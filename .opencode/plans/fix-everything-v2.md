# Plan: Fix Figtree v2 — broken example, dead kernel, unexercised boundary path, and git push

## Context

A code review of the current `master` (4e04886) found the core Figtree pipeline is
implemented and coherent, but there are concrete defects and dead code:

1. **Broken example — `get_top_facts`** (`examples/davos_shell_v2.py:101`)
   The method was renamed to `get_top_figments` in commit `29e0e1b`, but the
   interactive shell still calls the old name, so the `/trust` command crashes with
   `AttributeError`. `run_davos_v2.py:216` already uses the correct name.

2. **Dead CUDA kernel.** `figtree/kernel/boundary_project.cu` + `boundary_project.py`
   (`project_boundaries_to_kv`) compile and are exported from `kernel/__init__.py`,
   but **nothing imports them**. Ingestion computes per-token K/V directly in PyTorch
   (`ingest.py:171-180`). For the default 4-bit model the README/AGENTS.md already
   state the kernel falls back to matmul, so it is effectively unreachable dead code.

3. **`generate_from_boundaries` is never exercised.** Demo + test + benchmark only
   call `generate()` (text-based). The pre-computed `kv_cache.npy`, the ~22% faster
   boundary path, and `prompt.py` (Qwen3 ChatML helper) are tested only in isolation,
   so regressions there would go unnoticed.

4. **`generate()` concatenates figment tokens with no delimiter** (`generate.py:66-92`).
   All figment texts are fed as one continuous prefix with a single causal mask and no
   inter-figment separator/special tokens, so figment boundaries blur together. This
   hurts fidelity vs. the boundary path and may degrade recall.

5. **`ruff` is not installed** in the project venv (`.venv_f39`), despite AGENTS.md
   prescribing `ruff check figtree/`. Add it as a dev dependency so the documented lint
   command works.

Environment: torch 2.13.0+cu130 with CUDA available, `nvcc` present, git remote
`origin` -> `https://github.com/otherdrums/figtree.git`, branch `master` tracking
`origin/master`. Changes committed and pushed to `origin/master`.

## Scope (approved: "fix everything")

- Keep the CUDA kernel but actually wire it in (do NOT delete): ingestion uses
  `project_boundaries_to_kv` when the model is non-quantized, falling back to the
  existing PyTorch path for 4-bit bnb models. Makes the kernel reachable + honors README.
- Fix the shell bug (`get_top_facts` -> `get_top_figments`).
- Add figment delimiters in `generate()` to preserve boundaries.
- Exercise the boundary path: add a `generate_from_boundaries` run to the demo and a
  recall assertion to the test.
- Make `ruff` available and lint clean.
- Commit + push to `origin/master`.

## Steps

### 1. Fix broken shell example
- `examples/davos_shell_v2.py:101`: `graph.get_top_facts(10)` -> `graph.get_top_figments(10)`.

### 2. Wire CUDA kernel into ingestion (non-quantized path)
- In `figtree/ingest.py`, detect quantization via `getattr(model, "is_quantized", False)`
  or a `bnb.nn.Linear4bit` check on `self_attn.k_proj`. If non-quantized and dtype is
  bf16/fp16, use `figtree.kernel.boundary_project.project_boundaries_to_kv` per layer,
  feeding `input_layernorm(h_in)` reshaped to `(seq_len, hidden_size)` and applying
  `k_norm` to match current behavior. Keep the existing PyTorch fallback verbatim for
  4-bit models.
- Guard the import so a kernel build failure degrades gracefully to the PyTorch path.

### 3. Add figment delimiters in `generate()`
- In `generate.py`, when concatenating figment token ids, insert a separator between
  figments (e.g. `tokenizer.encode("\n\n")` or a dedicated delimiter) so each figment's
  text is bounded. Keep the single causal mask valid (each position still attends to
  all prior positions).

### 4. Exercise the boundary path
- `examples/run_davos_v2.py` `do_generate()`: add a `generate_from_boundaries` call
  (use `cache_dir=FIGMENTS_DIR/key`) alongside the text-based run, and print both.
- `tests/test_v2_pipeline.py`: after ingest, assert `generate_from_boundaries` runs
  without error and that a key entity from the text appears in the generated output
  (recall check).

### 5. Make `ruff` available + lint
- Add `ruff` to `pyproject.toml` `[project.optional-dependencies].dev`, install into
  `.venv_f39`, run `ruff check figtree/` and fix any findings.

### 6. Verify
- Run `python3 tests/test_v2_pipeline.py` (requires GPU + model download). If GPU/model
  unavailable in this env, at minimum confirm imports + lint pass, and document that the
  full run needs the T1000 + cached Qwen3-4B.

### 7. Commit + push
- `git add` the intended files (figtree/ingest.py, figtree/generate.py,
  examples/davos_shell_v2.py, examples/run_davos_v2.py, tests/test_v2_pipeline.py,
  pyproject.toml). Do NOT stage `.venv_f39/`.
- Commit with a message matching repo style (imperative, concise).
- `git push origin master`.

## Files touched
- `examples/davos_shell_v2.py` — rename fix.
- `figtree/ingest.py` — kernel wiring + quantization guard.
- `figtree/generate.py` — figment delimiters.
- `examples/run_davos_v2.py` — add boundary-path query.
- `tests/test_v2_pipeline.py` — boundary-path recall assertion.
- `pyproject.toml` — add `ruff` dev dep.

## Risks / notes
- The CUDA kernel requires the model in bf16/fp16 and a build step; the graceful
  fallback keeps the default 4-bit demo working unchanged.
- Full functional verification needs the GPU + model; lint + import checks are the
  always-runnable gate.
