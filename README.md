# FigTree — coherent context images from a single primitive

**Everything is a Figment.** There is no database. There is no graph store.
There is no separate KV cache layer. There is one type — the **Figment** — and
every operation in the system produces, transforms, or retrieves figments.

## What is a Figment?

A Figment is a **self-contained unit of knowledge** that carries everything
needed to recall, verify, and relate it within a language model's latent space:

| Component | Size | Purpose |
|-----------|------|---------|
| `text.txt` | ~100 B | Natural language statement — the human-readable payload |
| `boundary.npy` | ~10 KB | Single hidden-state vector at the **crystal layer**, used for similarity search and deduplication |
| `boundaries.npy` | ~360 KB | Per-layer hidden states (all layers), used for boundary projection and cross-layer analysis |
| `kv_cache.npy` | ~2.8 MB | Pre-computed unrotated K/V for every token at every layer — enables generation without a forward pass |
| `manifest.json` | ~200 B | Metadata: children, sources, trust score, edge type |

A Figment is stored as a **`.figment/` directory** on disk — a directory is the
unit of persistence. This is not a file format; it is a filesystem primitive.

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

Three conflicting news narratives are ingested with boundary + KV capture
(~10 KB boundary + ~2.8 MB KV per figment). Generation uses cached K/V:

```
Text → sentence splitting → boundary + KV capture → .figment directory
  ↓
Query → retrieve figments → load KV cache → apply RoPE → generate
```

**Results** (Qwen3-4B on Quadro T1000, 3GB):

| Phase | Time | Output |
|-------|------|--------|
| Ingestion | 23.4s | 41 figments, 445 KB |
| Generation | 33.6s | 100 tokens, 35 figments |
| Graph | 0.0s | 38 figments, 3 edges |
| **Total** | **57.0s** | |

## Architecture

```
figtree/
├── figment.py       # Figment dataclass — universal primitive + save/load
├── ingest.py        # Text → figments with boundary + kv_cache capture
├── generate.py      # Text-based and boundary-based generation
├── graph.py         # Edges/trust as figments + dedup
└── kernel/          # CUDA kernels
    ├── boundary_project.cu    # Boundary → K/V projection (CUDA)
    ├── boundary_project.py    # Python wrapper
    └── build.py               # Compilation script
```

### Figment Format (.figment)

```
figment.figment/
├── manifest.json     # figment_id, children, meta, sources, trust, edge_type
├── boundary.npy      # (hidden_size,) float32 — ~10 KB
├── boundary_emb.npy  # (hidden_size,) float32 — last-token embedding
├── boundaries.npy    # (num_layers, hidden_size) float32 — per-layer states
├── kv_cache.npy      # (num_layers, seq_len, 2, kv_dim) float32 — ~2.8 MB
└── text.txt          # Natural language statement
```

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
4. Save as `.figment`: boundary.npy + text.txt + manifest.json + kv_cache.npy

### 2. Generation (`figtree/generate.py`)

**Text-based** (forward pass for each figment):

1. For each selected figment, tokenize its text
2. Forward through all 36 layers with DynamicCache
3. Figment KV entries populate the cache

**Boundary-based** (load cached K/V from disk, ~22% faster):

1. Load `kv_cache.npy` for each figment
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

Custom CUDA kernel for non-quantized models:
```cuda
boundary_project_bf16_kernel(
    boundaries,  // (num_figments, hidden_size)
    W,           // (hidden_size, kv_dim)
    out,         // (num_figments, kv_dim)
    ...
)
```

For 4-bit quantized models, falls back to PyTorch `matmul`.

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
