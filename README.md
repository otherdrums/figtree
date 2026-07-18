# FigTree — coherent context images from a single primitive

**Everything is a Figment.** There is one type — the **Figment** — and every
operation in the system produces, transforms, or retrieves figments. Figments are
persisted in a LanceDB table (with K/V caches held externally as quantized blobs).

## What is a Figment?

A Figment is a **self-contained unit of knowledge** that carries everything
needed to recall, verify, and relate it within a language model's latent space:

| Component | Size | Purpose |
|-----------|------|---------|
| `text.txt` | ~100 B | Natural language statement — the human-readable payload |
| `boundary.npy` | ~10 KB | Single hidden-state vector at the **crystal layer**, used for similarity search and deduplication |
| `boundaries.npy` | ~360 KB | Per-layer hidden states (all layers), used for boundary projection and cross-layer analysis |
| `kv_cache` | ~2.8 MB | Pre-computed unrotated K/V for every token at every layer (held externally as a quantized blob) — enables generation without a forward pass |
| `manifest` | ~200 B | Metadata: children, sources, trust score, edge type |

A Figment persists as a **row in a LanceDB table** — the table is the unit of
storage, with vector columns for similarity search and compressed columns for
text/metadata. K/V caches live outside the row, addressed by `kv_uri`.

### What can be a Figment?

**An Image** is a Figment with children:
```python
image = Figment(text="Davos Summit article", children=[sent1, sent2, ...])
```

**An Edge** is a Figment with `meta["edge_type"]`:
```python
edge = Figment(text="Figment A supports Figment B", meta={"edge_type": "supports"})
```

**A Trust score** is a Figment with `meta["edge_type"] = "trust"`:
```python
trust = Figment(text="Source Reuters has trust 0.95", meta={"edge_type": "trust", "score": 0.95})
```

**A Graph** is simply a collection of figments — edges, trust assertions, and
content figments living in the same namespace. There is no separate graph
database. Relationships are figments that reference other figments.

**The system itself** is represented as figments. A meta-figment describes the
FigTree installation, its configuration, and its current state.

### The radical idea

Traditional RAG systems have layers of abstraction:
```
Documents → chunks → embeddings → vector DB → relational DB → graph DB → cache
```

FigTree collapses everything into one type:
```
Anything → Figment → more Figments
```

Because figments carry both a **text representation** (for the model to read)
and a **boundary representation** (for the model to recall), the model can
reason about its own knowledge structure. A figment about another figment is
just another figment — recursive, composable, and uniform.

## Quick Start

```bash
pip install -e .

# Full Davos demo (ingest + generate + graph)
python3 examples/run_davos_v2.py all

# Interactive shell
python3 examples/davos_shell_v2.py

# Benchmark
python3 examples/davos_benchmark_v2.py
```

## Demo

```bash
python3 examples/run_davos_v2.py all
```

Three conflicting news narratives are ingested into a **LanceDB store** with
boundary capture (~10 KB boundary + compressed text/metadata per figment).
K/V caches are external quantized blobs (lazy by default, eager with
`--compute-kv`):

```
Text → sentence splitting → boundary capture → LanceDB store (compressed)
  ↓
Query → ANN retrieve by boundary → KVCacheManager (lazy recompute or blob) → generate
```

**Results** (Qwen3-4B on Quadro T1000, 3GB; times vary with context length):

| Phase | Time | Output |
|-------|------|--------|
| Ingestion | ~18s | ~38 atomic figments, ~2.8 MB KV each |
| Generation (text-based) | ~50s | 4 sources × 400 tokens |
| Generation (boundary-based) | ~58s | 4 sources × 400 tokens |
| Graph | <1s | 3 sources, persisted trust figments |
| **Total** | **~2 min** | |

The boundary-based path skips the per-figment forward pass by replaying cached
K/V, but on this small GPU the KV-load + RoPE cost roughly cancels the saving, so
wall-clock is comparable to text-based generation. The benefit is storage/compute
trade-off, not a fixed speedup.

## Architecture

```
figtree/
├── figment.py       # Figment dataclass — universal primitive + to/from records
├── ingest.py        # Text → figments with boundary + optional K/V capture
├── generate.py      # Text-based and boundary-based generation
├── graph.py         # Edges/trust as figments + source-based credibility
├── lancedb_store.py # LanceDB store (compression + object storage)
├── kv_cache_manager.py # K/V materialization: lazy/eager, quantized, tiered
└── kernel/          # CUDA kernels
    ├── boundary_project.cu    # Boundary → K/V projection (CUDA)
    ├── boundary_project.py    # Python wrapper (handles 4-bit models)
    ├── prompt.py              # Qwen3 ChatML prompt builder
    └── build.py               # Compilation script
```

### Figment Storage (LanceDB)

Figments live in a **LanceDB table** (local path or `s3://`/`gs://`/`az://`). The
row is lightweight and compressed; K/V lives outside the row as a quantized
blob addressed by `kv_uri`:

| Field | Type | Notes |
|-------|------|-------|
| `figment_id` | str (pk) | SHA-256(text)[:16] (or deterministic id for trust) |
| `text` | str | zstd-compressed |
| `source_id` / `edge_type` | str | lz4 hot columns |
| `trust` / `is_image` | float / bool | |
| `has_kv_cache` / `kv_uri` | bool / str | K/V blob pointer (external) |
| `children` / `sources` | list[str] | |
| `meta_json` | str (JSON) | zstd-compressed metadata |
| `boundary` | vector(hidden) | for ANN similarity search |
| `boundaries` / `boundary_emb` | list[float] | per-layer states / last-token emb |

## Model

Default: `unsloth/Qwen3-4B-bnb-4bit` (36 layers, hidden_size=2560, 4-bit quantized)
Tested on: Quadro T1000 (3GB VRAM)

## How It Works

### 1. Ingestion (`figtree/ingest.py`)

Text is split into sentences. For each sentence:

1. Forward through model layers 0..crystal_layer
2. Capture boundary = hidden state of last token at crystal layer
3. For each layer, capture input hidden state, apply `input_layernorm`,
   project through `k_proj`/`v_proj` with `k_norm`, store unrotated K/V
4. Upsert into the LanceDB store (`lancedb_store.connect(uri)`) with
   compression. K/V is external: lazy by default, or persisted as a quantized
   blob via `KVCacheManager` when `compute_kv=True`.

### 2. Generation (`figtree/generate.py`)

**Text-based** (forward pass for each figment):

1. For each selected figment, tokenize its text
2. Forward through all 36 layers with DynamicCache
3. Figment KV entries populate the cache

**Boundary-based** (lazy K/V via `KVCacheManager`, skips per-figment forward pass):

1. `kv_manager.materialize(figments)` returns per-figment K/V (recomputed on
   demand, or loaded from the `kv_uri` blob if persisted eagerly)
2. Apply RoPE based on global position IDs
3. Insert into DynamicCache directly — no forward pass

**Prefill:**

1. Prompt tokens embed → forward through layers with cached K/V
2. Explicit causal 4D mask ensures prompt sees all figment positions
3. Final norm → lm_head → logits

**Decode:**

1. Sample next token
2. Forward through all layers with cached K/V
3. KV cache grows by 1 position per layer per step

### 3. Boundary Projection (`figtree/kernel/boundary_project.cu`)

Custom CUDA kernel (`boundary_project_bf16_kernel`) projects boundary vectors
through a layer's `W_k` / `W_v`:

```cuda
boundary_project_bf16_kernel(
    boundaries,  // (num_figments, hidden_size)
    W,           // (hidden_size, kv_dim)
    out,         // (num_figments, kv_dim)
    ...
)
```

For **4-bit quantized models** (e.g. `unsloth/Qwen3-4B-bnb-4bit`), the Python
wrapper in `boundary_project.py` **dequantizes** the `bitsandbytes` 4-bit weight
to bf16 on the fly (contiguous `(hidden, kv_dim)` layout) and runs the same CUDA
kernel — no PyTorch `matmul` fallback needed. The kernel is only bypassed if it
fails to build, in which case a PyTorch `matmul` fallback is used.

### 4. Source-based Trust (`figtree/graph.py`)

Trust is **source-based and mutable**, not a fixed graph attribute:

- Each source carries an immutable `base_trust` (set at ingest on the image
  figment's `meta["base_trust"]`). Persisted `trust:{source_id}` figments carry
  the *adjusted* score and are never read back as base (this prevents trust drift
  across reloads).
- `analyze_sources()` computes, per source: which other sources it is `related`
  to (topic overlap), `agreeing` with (same explicit non-neutral stance), and
  `contradicting` (opposite stance). Adjusted trust = `0.6·base + 0.4·corroborated`
  then ×0.85 if contradicted by others.
- `propagate_trust(output_dir=...)` is **idempotent and disk-persistent**: it (re)
  creates one trust figment per source with a deterministic id, overwriting the
  previous file. A future "accuracy proven → trust up" step only edits
  `meta["score"]` on that figment and re-runs `propagate_trust`. No schema change.
- `build_trust_aware_context(query)` recalls every perspective relevant to a query
  together with its credibility relationships, which the demo injects into the
  generation prompt so cross-source agreement/disagreement is grounded in the
  actual source texts (preventing the model from inventing agreement).

### 5. Prompt & decoding details (`figtree/generate.py`, `figtree/kernel/prompt.py`)

- Both generation paths wrap the prompt in the **Qwen3 ChatML template**
  (`build_prompt_ids`, thinking disabled) so the model produces grounded answers.
- Figment texts are concatenated with `\n\n` separators during ingestion; the
  boundary path replays the exact same per-figment K/V slices, so
  boundary-based output matches text-based output token-for-token in content.
- A `repetition_penalty` (1.15) is applied during sampling to reduce looping.

## Critical Technical Details

### SDPA Causal Mask Fix

When using SDPA with a pre-existing KV cache (figments at positions 0..T-1,
prompt at positions T..T+P-1), `is_causal=True` is **incorrect** — it assumes
Q starts at position 0, blocking attention from prompt to figments.

Fix: explicit 4D mask `(1, 1, P, T+P)` where each prompt token sees all
figment positions plus previous prompt positions.

### Memory Fix for 4-bit Models

`model.eval()` does NOT disable gradient tracking. The `MatMul4Bit` autograd
function keeps dequantized buffers. Wrapping in `torch.no_grad()` saves ~450 MB:

| | Without `no_grad` | With `no_grad` |
|---|---|---|
| Prefill peak | ~3,036 MB | ~2,583 MB |
| Decode | OOMs | Stable at ~2,600 MB |

### Pre-norm for KV Projection

During ingestion, the input to each layer is `input_layernorm`'d before
projecting through `k_proj`/`v_proj`, and `k_norm` is applied. This matches
the model's internal computation exactly. The K/V is stored unrotated — RoPE
is applied lazily during generation based on global position IDs.

## Testing

```bash
# Quick pipeline test
python3 tests/test_v2_pipeline.py

# Full demo: ingest + generate + graph
python3 examples/run_davos_v2.py all

# Interactive shell
python3 examples/davos_shell_v2.py

# Benchmark
python3 examples/davos_benchmark_v2.py
```

## Known Limitations

1. **kv_cache.npy storage**: ~2.8 MB per 20-token figment. 100 figments = ~280 MB.
   Manageable on modern drives but not as compact as boundary-only storage.

2. **GPU memory constrained**: Qwen3-4B (3.4GB) on 3GB GPU leaves ~1.1GB
   headroom. Works for 300–500 token contexts. For longer context or larger
   models, use Qwen3-2B (~1.5GB) or a 6GB+ GPU.

## Inspiration

Directly inspired by [**chrishayuk/larql**](https://github.com/chrishayuk/larql):

- **Boundary residuals** — LARQL stores one residual vector per text window at
  the crystal layer. The Apollo 11 transcript demo (370K tokens) compresses to
  ~2.8MB of boundary residuals — 20,000× over full KV cache.
- **Crystal layer detection** — The model layer where the residual stream
  stabilizes, enabling early-layer skipping.
- **"The model IS the database"** — Model weights reorganized as queryable
  knowledge with patches as lightweight overlays.
- **Boundary-KV engine** — LARQL's boundary retrieval store: boundaries for
  LSH retrieval, KV cache for generation.

Figtree extends these concepts with:

- **One universal primitive** — Everything is a Figment (images, edges, trust,
  metadata). An Image is a Figment with children.
- **Pre-computed per-token KV cache** — During ingestion, each figment's
  unrotated K/V is computed for all layers and stored as `kv_cache.npy`
  (~2.8 MB per 20-token figment). During generation, RoPE is applied and K/V
  is inserted directly into the cache — no forward pass needed.
- **Boundary + KV hybrid storage** — Boundaries (~10 KB) for retrieval and
  similarity search; KV cache (~2.8 MB per figment) for generation.
- **Graph as figments** — All relationships and trust are first-class Figments.

## Credit

**chrishayuk/larql** — The boundary residual concept, crystal layer detection,
boundary-kv engine architecture, and "model IS the database" philosophy.
