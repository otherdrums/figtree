# Plan: Migrate FigTree storage to LanceDB (+ compression, KV cache manager, remote storage)

## Objective
Replace the raw `.figment/` filesystem storage with **LanceDB** as the primary
store. Enable remote/object storage (S3-compatible) backends, compression
(zstd for text/metadata, lz4 for hot columns, dictionary/FSST encodings), and a
dedicated `KVCacheManager` that materializes K/V on demand (lazy default) or
eagerly, with quantization + tiered (LRU + object) caching. Deprecate the
filesystem path. Keep the existing faithful-recall + source-based trust behavior
intact (no recall regressions).

## Context / current state
- `Figment` is a dataclass persisted as `*.figment/` dirs (`figment.py`):
  `manifest.json`, `boundary.npy`, `boundaries.npy`, `boundary_emb.npy`,
  `text.txt`. K/V is a separate `kv_cache.npy` written next to each figment.
- `ingest.py` writes figment dirs + `kv_cache.npy` to disk.
- `generate.py` has two paths: `generate` (re-forward text) and
  `generate_from_boundaries` (loads `kv_cache.npy` from `cache_dir`).
- `graph.py` works in-memory on `Figment` objects loaded from disk.
- Demos (`run_davos_v2.py`, `davos_benchmark_v2.py`, `davos_shell_v2.py`) and
  `tests/test_v2_pipeline.py` all `glob("*.figment")` + `Figment.load`.
- `.gitignore` ignores `*.figment/`; `pyproject.toml` has no `lancedb`.

## Dependencies
- `pyproject.toml`: add `lancedb` (+ `pyarrow`). Add `s3fs` (or rely on Lance's
  built-in S3 via `storage_options`) only if a local MinIO test is wanted; Lance
  talks S3 natively through `storage_options`, so `boto3`/`s3fs` not strictly
  required. Add a `[storage]` optional extra for object-store helpers.

## New module: `figtree/lancedb_store.py`
Connection + table management with compression settings.
- `connect(uri, storage_options=None) -> DBConnection` — accepts local path or
  `s3://`/`gs://`/`az://` URIs. Default local: `./figtree.lance`.
- `FigmentSchema` via `lancedb.pydantic.LanceModel` with fields:
  - `figment_id: str` (primary key), `text: str`, `source_id: str`,
    `edge_type: str | None`, `trust: float`, `is_image: bool`,
    `children: list[str]`, `sources: list[str]`,
    `meta: dict` (JSON/struct column),
  - `boundary: Vector(hidden_size)` — the crystal-layer vector, for ANN search,
  - `boundaries: list[float] | None` — flattened `(num_layers*hidden_size)` for
    reconstruction (nullable; only when ingested with full boundaries),
  - `boundary_emb: list[float] | None`,
  - `has_kv_cache: bool` (False by default — lazy), `kv_uri: str | None`
    (where quantized K/V lives, e.g. `s3://.../<id>.kv` or local path).
  - ANN index on `boundary` (IVF_PQ) created on first query if absent.
- Compression: applied via `storage_options` at connection/`create_table`
  (`new_table_data_storage_version="stable"`). Plus per-write
  `compression_overrides` mapping:
  - `text`, `meta`, `source_id` -> `zstd` (ratio); `kv_uri`/`figment_id` ->
    `lz4`; rely on Lance auto dictionary/FSST for string columns.
- CRUD: `upsert(figments)`, `get(figment_id)`, `delete(figment_id)`,
  `all()`, `by_source(src)`, `search(vector, k)` (ANN), `filter(meta_expr)`.
  All return `Figment` objects (so `graph.py`/`generate.py` keep working).

## `figtree/figment.py`
- Keep the dataclass (used in-memory everywhere). Add:
  - `to_record() -> dict` (numpy arrays -> lists, meta -> dict).
  - `classmethod from_record(rec) -> Figment` (lists -> numpy).
  - Keep `save`/`load` as **deprecated** (emit `DeprecationWarning`), kept only
    for the migration one-off and backward-compat tests.

## `figtree/kv_cache_manager.py` (new)
`KVCacheManager` abstracts K/V materialization, decoupling it from figment rows.
- Store K/V blobs separately from the row (Lance isn't ideal for multi-MB
  variable tensors): `kv_uri` points at an object store / local file.
  - **Eager**: `ingest` computes K/V and writes it (quantized fp16/int8) to
    `kv_uri` (default local `./figtree_kv/<id>.kv.npy` or S3).
  - **Lazy (default)**: K/V is NOT persisted at ingest; `materialize(figments,
    model, tokenizer)` recomputes it on demand (same concat+separator forward
    as current `ingest.py`), caches in an in-memory LRU, and (optionally) writes
    to `kv_uri` for reuse.
- `get_kv(figment, model, tokenizer, device)` -> `(k, v)` tensors ready for
  `generate_from_boundaries`. Tiered: LRU (hot) -> object store (`kv_uri`) ->
  recompute (cold).
- Quantization: `quantize="fp16"|"int8"` when writing to object store to cut
  storage; dequant on load. Default fp16 (lossless-ish for bf16 K/V).
- Honors a `compute_kv` flag threaded from ingest/generate.

## `figtree/ingest.py`
- New signature keeps `output_dir` but it becomes the LanceDB `uri` (path or
  `s3://`). Add `compute_kv: bool = False` (lazy default per the spec) and
  `kv_manager: KVCacheManager | None`.
- Build `Figment`s as today, write rows via `lancedb_store.upsert`. When
  `compute_kv=True`, also persist K/V through the manager (quantized) and set
  `has_kv_cache=True`, `kv_uri=...`.
- Keep deterministic id + `base_trust` on image figment (trust model unchanged).

## `figtree/generate.py`
- Add `kv_manager` to `FigmentGenerator`. `generate_from_boundaries` pulls K/V
  via `kv_manager.get_kv(figment, ...)` instead of `kv_cache.npy` from a
  `cache_dir`. Falls back to lazy recompute if `has_kv_cache=False`.
- `generate` (text) unchanged in behavior; both paths still use ChatML +
  `repetition_penalty`.

## `figtree/graph.py`
- `Figtree.__init__` accepts either a `list[Figment]` or a LanceDB-backed
  `FigmentStore`; loading sources via `store.by_source(src)` / `store.all()`.
  No change to `analyze_sources`/`propagate_trust` math. `propagate_trust`
  writes the `trust:{src}` figment row via `store.upsert` (idempotent).

## Demos / examples / tests
- `run_davos_v2.py`, `davos_benchmark_v2.py`, `davos_shell_v2.py`: replace
  `glob("*.figment")`+`Figment.load` with `store.all()`/`store.by_source`.
  Add a `--store-uri` (default local `./figtree.lance`) and
  `--compute-kv` flag; benchmark reports storage size via Lance (`du` of the
  `.lance` dir) to show compression benefit.
- `tests/test_v2_pipeline.py`: add LanceDB round-trip + compression assertions
  (row count, boundary preserved, `has_kv_cache` lazy default), lazy vs eager
  KV parity (recompute == eager blob), and an optional MinIO/S3 test gated by an
  env var (`FIGTREE_TEST_S3_URI`). Keep the existing recall/trust assertions
  (boundary==text, persisted trust) — regression guard.

## Docs
- `README.md` + `AGENTS.md`: new "Storage: LanceDB" section — architecture
  (Figment records + vector index + external K/V blobs), compression strategy
  (zstd text/metadata, lz4 hot, dictionary/FSST), remote storage (S3 URI +
  `storage_options`), `KVCacheManager` (lazy/eager/quantized/tiered), and
  production considerations. Mark `.figment/` as deprecated. Update benchmark
  table with LanceDB storage size vs the old ~2.8 MB/figment.
- Add `.venv_f39/` to `.gitignore` (the `_f39` suffix denotes the f39 toolbox
  venv; pattern `.venv/` doesn't match it). Also ignore `*.lance/` and
  `figtree_kv/`.

## Commits (feature branch -> PR)
- `feat: add lancedb_store with compression + S3/object storage support`
- `feat: add KVCacheManager (lazy/eager, quantized, tiered)`
- `refactor: migrate ingest/generate/graph to LanceDB store`
- `test: LanceDB round-trip, compression, lazy vs eager KV, optional S3`
- `docs: storage architecture, compression, remote storage in README/AGENTS`
- `chore: deprecate .figment/ fs path; update .gitignore`

## Decisions (confirmed with user)
1. **KV storage**: external quantized blobs (local path or S3), referenced by
   `kv_uri` in the Lance row. K/V is NOT embedded in the row. ✅
2. **Phasing**: one feature branch (`feat/lancedb-storage`), single squashed PR
   covering Phases 0–5. (Agent discretion on commit granularity below.) ✅
3. **S3 test**: gated behind env var `FIGTREE_TEST_S3_URI`; skipped by default.
   No MinIO standup in CI. ✅
4. **Default KV mode**: lazy (no K/V persisted at ingest); eager available via
   `--compute-kv` / `compute_kv=True`. ✅

First implementation step (currently blocked by plan mode): add `.venv_f39/` to
`.gitignore` (plus `*.lance/`, `figtree_kv/`), then proceed per phases.

## Commit granularity (agent discretion)
- `feat: add lancedb_store with compression + object storage support`
- `feat: add KVCacheManager (lazy/eager, quantized, tiered)`
- `refactor: migrate ingest/generate/graph to LanceDB store`
- `test: LanceDB round-trip, compression, lazy vs eager KV, optional S3`
- `docs: storage architecture, compression, remote storage in README/AGENTS`
- `chore: deprecate .figment/ fs path; update .gitignore`

