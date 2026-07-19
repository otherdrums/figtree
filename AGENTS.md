# Figtree v2 — Figment-Centric Architecture

## Philosophy

Everything is a **Figment**. Images are figments containing other figments. Trust scores are figments. Graph edges are figments. Even the system itself is represented as figments.

Figments are stored in a **LanceDB table** (see `figtree/lancedb_store.py`). The lightweight
row carries:
- **boundary** — (hidden_size,) float32, ~10 KB. Compressed vector for ANN retrieval and dedup.
- **boundaries** — (num_layers, hidden_size) float32, ~360 KB. Per-layer hidden states (all layers).
- **boundary_emb** — (hidden_size,) float32. Last-token embedding.
- **text** — Natural language statement. Used for text-based generation and lazy K/V recompute.
- **meta_json** — Metadata (children, sources, trust, edge_type, kv_uri, …), zstd-compressed.

K/V caches (`kv_cache`, ~2.8 MB per 20-token figment before quantization) live **outside** the
row as external quantized blobs addressed by `kv_uri`, managed by `KVCacheManager` (lazy by
default; eager via `compute_kv=True`). The legacy `.figment/` directory format has been removed.

## Build / Test / Lint

```bash
pip install -e .
ruff check figtree/

# Quick pipeline test
python3 tests/test_v2_pipeline.py

# Full Davos demo
python3 examples/run_davos_v2.py all

# Interactive shell
python3 examples/davos_shell_v2.py

# Benchmark
python3 examples/davos_benchmark_v2.py
```

## Architecture

```
figtree/
├── figment.py          # Figment dataclass + to/from records
├── ingest.py           # Text → figments with boundary + optional K/V capture
├── generate.py         # On-the-fly KV + cached boundary KV generation
├── graph.py            # Edges/trust as figments + source-based credibility
├── lancedb_store.py    # LanceDB store (compression + object storage)
├── kv_cache_manager.py # K/V materialization: lazy/eager, quantized, tiered
├── summarize.py        # Hierarchical figments: summarized image boundaries
├── cli.py              # Typer CLI: ingest/generate/graph/benchmark/compare
└── kernel/
    ├── boundary_project.cu   # CUDA kernel: boundaries @ W_k/W_v
    ├── boundary_project.py   # Python wrapper (4-bit dequant)
    └── build.py              # torch.utils.cpp_extension.load
```

### Custom CUDA Kernel

**File:** `figtree/kernel/boundary_project.cu`

Projects boundary vectors through a layer's W_k and W_v weight matrices:
```cuda
__global__ void boundary_project_bf16_kernel(
    const __nv_bfloat16* boundaries,  // (num_figments, hidden_size)
    const __nv_bfloat16* W,            // (hidden_size, kv_dim)
    __nv_bfloat16* out,                // (num_figments, kv_dim)
    ...
)
```

**Usage:**
```python
from figtree.kernel.boundary_project import project_boundaries_to_kv
k_figments, v_figments = project_boundaries_to_kv(boundaries, layer, device)
```

**Note:** For 4-bit quantized models (e.g. `unsloth/Qwen3-4B-bnb-4bit`), the Python wrapper in `boundary_project.py` **dequantizes** the bitsandbytes 4-bit weight to bf16 on the fly (contiguous `(hidden, kv_dim)` layout) and runs the same CUDA kernel — no PyTorch `matmul` fallback needed. The kernel is only bypassed if it fails to build, in which case a PyTorch `matmul` fallback is used.

### Figment Primitive

```python
@dataclass
class Figment:
    figment_id: str          # SHA-256(text)[:16]
    text: str                # Natural language statement
    boundary: np.ndarray     # (hidden_size,) float32
    meta: dict               # edge_type, about_figment, etc.
    children: list[str]      # Child figment IDs
    sources: list[str]       # Parent figment IDs
    trust: float             # Cached trust score
    boundaries: np.ndarray | None = None  # (num_layers, hidden_size)
    boundary_emb: np.ndarray | None = None  # (hidden_size,) last-token embedding
```

**Image = Figment with `children=[figment_1, figment_2, ...]`**
**Edge = Figment with `meta["edge_type"] = "supports"`**
**Trust = Figment with `meta["edge_type"] = "trust", meta["score"]=0.95`**

### Ingestion Pipeline

```python
from figtree.ingest import ingest_text_to_figments
from figtree.lancedb_store import connect

store = connect("./figtree.lance")
figments = ingest_text_to_figments(
    model, tokenizer, text,
    store=store,
    source_id="reuters",
    trust=0.95,
)
# Returns: [image, atomic_1, atomic_2, ..., trust_assertion]
```

For each sentence:
1. Forward through model layers 0..crystal_layer
2. Capture boundary = hidden state of LAST token at crystal_layer
3. Compute per-token per-layer K/V (unrotated) by projecting each layer's
   normed input through W_k/W_v with k_norm
4. Upsert figments into a **LanceDB store** (`lancedb_store.connect(uri)`)
   with compression (zstd text/metadata, lz4 hot cols, auto dictionary/FSST).
   K/V is NOT stored in the row by default (lazy); set `compute_kv=True`
   (or `--compute-kv`) to also persist quantized K/V blobs via the
   `KVCacheManager` (external files / object storage).

### Generation Engine

```python
from figtree.generate import FigmentGenerator

gen = FigmentGenerator(model, tokenizer)

# Text-based generation (forward pass for each figment)
result = gen.generate(
    figments=[fig1, fig2, fig3],
    prompt="What happened at Davos?",
    max_new_tokens=100,
)

# Boundary-based generation (lazy K/V via KVCacheManager; recomputes on demand)
result = gen.generate_from_boundaries(
    figments=[fig1, fig2, fig3],
    prompt="What happened at Davos?",
    max_new_tokens=100,
    kv_manager=kv_manager,
)
```

**Text-based KV generation (`generate`):**
1. For each selected figment, tokenize its text and run through the model
2. Populates a `DynamicCache` with the figment's full KV entries
3. Prompt tokens are forwarded with the pre-populated cache
4. Standard causal attention (explicit mask) ensures prompts see all figment positions
5. Decode autoregressively

**Boundary-based KV generation (`generate_from_boundaries`):**
1. Loads pre-computed per-token unrotated K/V from the `KVCacheManager`
   (lazy: recomputed on demand; eager: loaded from the `kv_uri` blob)
2. Applies RoPE based on global position IDs
3. Inserts into `DynamicCache` directly — no forward pass for figment tokens
4. Prompt prefill + decode proceed as usual

The boundary-based approach avoids re-running the forward pass for figment tokens, trading ~2.8 MB/figment disk storage. On the 3GB test GPU the KV-load + RoPE cost roughly cancels the saved forward pass, so wall-clock is comparable to text-based generation; the benefit is the storage/compute trade-off. Per-token K/V is computed during ingestion by capturing each layer's input hidden state, applying `input_layernorm`, and projecting through `k_proj`/`v_proj` with `k_norm` applied. This matches the model's internal computation exactly, enabling factual recall with standard HF attention.

Both `generate` and `generate_from_boundaries` wrap the prompt in the Qwen3 ChatML template (`figtree/kernel/prompt.py:build_prompt_ids`, thinking disabled) and apply `repetition_penalty=1.15` during decoding. Figment texts are concatenated with `\n\n` separators during ingestion; the boundary path replays the exact same per-figment K/V slices, so boundary-based output matches text-based output in content.

### Graph as Figments

```python
from figtree.graph import Figtree

graph = Figtree(all_figments, store=store)
graph.deduplicate()            # exact + semantic boundary similarity
graph.create_edges()           # SUPPORTS, SAME_ENTITY, CONTRADICTS
graph.propagate_trust(store=store)  # source-based, idempotent, store-persisted
```

All graph operations produce **new Figments**:
- Deduplication creates `Figment(meta={"edge_type": "supports"})`
- Trust propagation creates `Figment(meta={"edge_type": "trust", "score": 0.95})`
- Contradictions create `Figment(meta={"edge_type": "contradicts"})`

### Storage: LanceDB (`figtree/lancedb_store.py`)

Figments are stored in a **LanceDB table**. Benefits:

- **Compression**: `text`/`meta` columns use `zstd`; small/hot string columns
  (`figment_id`, `source_id`, `edge_type`, `kv_uri`) use `lz4`; Lance applies
  automatic dictionary/FSST encodings for strings and bit-packing for numerics.
  A text-heavy 200-row table of ~280 KB raw compresses to ~26 KB on disk.
- **Remote / object storage**: pass an object-store URI to `connect`
  (`s3://bucket/path`, `gs://...`, `az://...`) plus `storage_options`
  (credentials, region, endpoint). Same code path as local.
- **ANN search**: the `boundary` vector column supports similarity retrieval
  (`store.search(vector, k=...)`).
- **Idempotent upsert**: keyed on `figment_id` (delete+add under the hood,
  because this LanceDB version's `merge_insert`/`update` mishandle vector
  columns).

Schema fields: `figment_id` (pk), `text`, `source_id`, `edge_type`, `trust`,
`is_image`, `has_kv_cache`, `kv_uri`, `children`, `sources`, `meta_json`,
`boundary` (vector), `boundaries`, `boundary_emb`.

### KV Cache Manager (`figtree/kv_cache_manager.py`)

K/V caches (`(num_layers, seq_len, 2, kv_dim)`) are large and variable-shape, so
they live **outside** the Lance row, addressed by `kv_uri` on the figment:

- **Lazy (default)**: K/V is not persisted at ingest. `materialize()` recomputes
  it on demand (single concatenated forward pass, same separators as ingest),
  caches it in an in-memory LRU, and (if eager) writes a quantized blob.
- **Eager** (`compute_kv=True`): K/V is computed at ingest and written (fp16/int8
  quantized) to `kv_uri`, recording `has_kv_cache=True` on the figment.
- **Tiered**: LRU (hot) → external blob at `kv_uri` (warm) → recompute (cold).
- Blobs are local files by default; `kv_root` may be an `s3://` URI.

## Model

Default: Qwen3-4B (unsloth bnb-4bit, cached at ~/.cache/huggingface/hub/)
GPU: Quadro T1000 (3GB VRAM)

## Model

Default: Qwen3-4B (unsloth bnb-4bit, cached at ~/.cache/huggingface/hub/)
GPU: Quadro T1000 (3GB VRAM)

## Benchmark Results

```bash
python3 examples/davos_benchmark_v2.py
```

| Phase | Time | Throughput |
|-------|------|------------|
| Ingestion | ~18s | ~38 atomic figments |
| Generation (text) | ~50s | 4 sources × 400 tokens |
| Generation (boundary) | ~58s | 4 sources × 400 tokens |
| Graph | <1s | 3 sources, persisted trust |
| **Total** | **~2 min** | |

Boundary-based generation (`generate_from_boundaries`) skips the per-figment
forward pass by replaying cached K/V, at the cost of ~2.8 MB/figment disk
storage. On the 3GB test GPU the wall-clock is comparable to text-based
generation (KV-load + RoPE roughly cancels the saved forward pass); the benefit
is the storage/compute trade-off, not a fixed speedup.

## Key Commands

```bash
# Davos v2 demo (ingest + generate + graph)
python3 examples/run_davos_v2.py all

# Interactive shell
python3 examples/davos_shell_v2.py

# Benchmark
python3 examples/davos_benchmark_v2.py

# Pipeline test
python3 tests/test_v2_pipeline.py

# Boundary-based generation test
python3 -c "
from figtree.figment import Figment
from figtree.generate import FigmentGenerator
gen = FigmentGenerator(model, tokenizer)
result = gen.generate_from_boundaries(figments, prompt, kv_manager=kv_manager)
"
```

## Design Decisions

1. **Everything is a Figment**: Images, edges, trust assertions, even the system itself. This unifies the data model and enables recursive reasoning (meta-figments about figments).

2. **Boundaries for retrieval, text for generation**: Boundaries (~10 KB) enable fast similarity search and deduplication. Text is used to regenerate full KV caches on-the-fly during generation. This gives compact storage without sacrificing recall.

3. **Custom CUDA kernel for boundary projection**: The kernel compiles and works for non-quantized models. For 4-bit quantized models (bitsandbytes packed weights), the Python wrapper dequantizes the weight to bf16 on the fly and runs the same kernel — no PyTorch `matmul` fallback needed unless the kernel fails to build.

4. **On-the-fly KV generation**: Instead of storing KV caches on disk (~90 MB per image), we regenerate them from text during generation. This trades computation for storage and keeps the system lightweight.

5. **Pre-computed KV cache**: Per-token unrotated K/V for all layers is computed with proper `input_layernorm` and stored as an external quantized blob addressed by `kv_uri` (lazy by default; eager via `compute_kv=True`), managed by `KVCacheManager`. RoPE is applied lazily during generation. This skips the forward pass for figment tokens (boundary path); wall-clock saving depends on GPU/context.

6. **Figment-centric graph**: All relationships (SUPPORTS, CONTRADICTS, TRUST) are first-class Figments with their own boundaries and text. During generation, the model can load meta-figments alongside content figments, enabling trust-aware reasoning through attention.

## Known Limitations

1. **Boundary-based generation is not a fixed speedup**: On the 3GB test GPU, text-based generation is ~50s and boundary-based ~58s for 4 sources × 400 tokens. The skipped forward pass is offset by KV-load + RoPE. The trade-off is disk storage (~2.8 MB/figment) vs recompute, not wall-clock.

2. **GPU memory constrained**: Qwen3-4B (3.4GB) on 3GB GPU leaves ~1.1GB headroom. Works for 300–500 token contexts. For longer contexts, use Qwen3-2B or larger GPU.

3. **External K/V storage**: ~2.8 MB per 20-token figment before quantization,
   but lives as an external quantized blob (default lazy — not persisted at ingest).
   In LanceDB the lightweight row is ~10 KB (boundary + text + metadata, compressed).
   100 figments ≈ 1 MB in the store; K/V only materializes on demand.

6. **`.figment/` format removed**: the legacy directory layout and `Figment.save`/
   `Figment.load` are gone; all code reads/writes the LanceDB store. `store` is
   required at ingest/graph time.

7. **Conventional RAG baseline exists**: `examples/rag_baseline_davos.py` +
   `examples/davos_eval.py` compare Figtree against top-k cosine retrieval +
   generate on the same Davos task (fidelity, contradiction awareness, VRAM,
   latency) via `figtree compare`.

8. **Hierarchical figments are opt-in**: pass `summarize_images=True` to ingestion
   to give an Image its own summarized `boundary` (`figtree/summarize.py`);
   otherwise the Image boundary copies its first child. Opt-in protects low-VRAM users.

4. **Per-source recall is verbatim-grounded; cross-source synthesis is model-generated**. QUERY 1/4 (per-source) reproduce each narrative's figures faithfully. The trust-aware QUERY 2/3 synthesize across sources via the LLM; to avoid fabricated agreement the demo grounds the prompt with `analyze_sources` relationships (`related`/`agreeing`/`contradicting`) and instructs the model to only state facts present in ≥2 source texts. Even so, the small model can under-describe disagreements.

5. **Source texts omit a year** (they say "concluded yesterday"), so the model may infer a date (it guessed 2023/2024 inconsistently). This is model inference, not a recall error.

## Source-Based Trust Model

- Each source has an **immutable** `base_trust` stored on its image figment's `meta["base_trust"]` at ingest. Persisted `trust:{source_id}` figments carry the *adjusted* score and are never read back as base (prevents trust drift across reloads).
- `Figtree.analyze_sources()` returns, per source: `related` (topic overlap), `agreeing` (same explicit non-neutral stance), `contradicting` (opposite stance), `corroborated_frac`, and `adjusted_trust = min(1, max(0, 0.6·base + 0.4·corroborated)) × 0.85 if contradicted`.
- `Figtree.propagate_trust(store=...)` is **idempotent and store-persistent**: it
  (re)creates one trust figment per source with a deterministic id
  (`trust:{source_id}`) in the LanceDB store, overwriting the previous row. A
  future "accuracy proven → trust up" step only edits `meta["score"]` on that
  figment and re-runs `propagate_trust`. No schema change.
- `Figtree.build_trust_aware_context(query)` recalls every perspective relevant to a query with its credibility relationships; the demo injects these into the generation prompt so cross-source claims are grounded in the actual texts.
