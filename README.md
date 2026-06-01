# PDGA — Parallel Delta Graph Architecture

A fact-centric system for ingesting arbitrary text into atomic facts with
compressed boundary representations. During generation, full KV caches are
regenerated on-the-fly from fact text, enabling factual recall on constrained
GPUs with minimal storage.

## Inspiration

Directly inspired by [**chrishayuk/larql**](https://github.com/chrishayuk/larql):

- **Boundary residuals** — LARQL stores one residual vector per text window at the crystal layer. The Apollo 11 transcript demo (370K tokens) compresses to ~2.8MB of boundary residuals — 20,000× over full KV cache.
- **Crystal layer detection** — The model layer where the residual stream stabilizes, enabling early-layer skipping.
- **"The model IS the database"** — Model weights reorganized as queryable knowledge with patches as lightweight overlays.
- **Boundary-KV engine** — LARQL's boundary retrieval store: boundaries for LSH retrieval, KV cache for generation. This is retrieval + generation, not compression.

PDGA extends these concepts with:
- **Fact-centric architecture** — Everything is a Fact (narratives, edges, trust)
- **On-the-fly KV generation** — No pre-computed KV cache storage; regenerate from text
- **Custom CUDA kernel** for boundary projection through W_k/W_v
- **Graph as facts** — All relationships are first-class Facts

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

Three conflicting news narratives are ingested with boundary capture only
(~10 KB per fact). The demo runs on-the-fly KV generation:

```
Text → sentence splitting → boundary capture → .pdga directory (~10 KB/fact)
  ↓
Query → retrieve facts → tokenize text → forward → KV cache → generate
```

**Results** (Qwen3-4B on Quadro T1000, 3GB):

| Phase | Time | Output |
|-------|------|--------|
| Ingestion | 23.4s | 41 facts, 445 KB |
| Generation | 33.6s | 100 tokens, 35 facts |
| Graph | 0.0s | 38 facts, 3 edges |
| **Total** | **57.0s** | |

## Architecture

```
pdga/
├── fact/          # Fact primitive, ingestion, generation, graph
│   ├── primitive.py    # Fact dataclass — universal primitive
│   ├── ingest.py       # Text → facts with boundary capture
│   ├── generate.py     # On-the-fly KV + standard attention
│   └── graph.py        # Edges/trust as facts + dedup
└── kernel/        # CUDA kernels
    ├── boundary_project.cu   # Boundary → K/V projection
    ├── boundary_project.py   # Python wrapper
    └── build.py              # Compilation script
```

### Fact Format (.pdga)

```
fact.pdga/
├── manifest.json    # fact_id, children, meta, sources, trust
├── boundary.npy     # (hidden_size,) float32 — ~10 KB
└── text.txt         # Natural language statement
```

## Model

Default: `unsloth/Qwen3-4B-bnb-4bit` (36 layers, hidden_size=2560, 4-bit quantized)
Tested on: Quadro T1000 (3GB VRAM)

## How It Works

### 1. Ingestion (`pdga/fact/ingest.py`)

Text is split into sentences. For each sentence:
- Forward through model layers 0..crystal_layer
- Capture boundary = hidden state of last token at crystal layer
- Save as `.pdga`: boundary.npy + text.txt + manifest.json

### 2. Generation (`pdga/fact/generate.py`)

**Fact KV generation** (on-the-fly):
1. For each selected fact, tokenize its text
2. Forward through all 36 layers with DynamicCache
3. Fact KV entries populate the cache

**Prefill:**
1. Prompt tokens embed → forward through layers with cached fact K/V
2. Explicit causal 4D mask ensures prompt sees all fact positions
3. Final norm → lm_head → logits

**Decode:**
1. Sample next token
2. Forward through all layers with cached K/V
3. KV cache grows by 1 position per layer per step

### 3. Custom CUDA Kernel (`pdga/kernel/boundary_project.cu`)

**Boundary projection kernel** for non-quantized models:
```cuda
boundary_project_bf16_kernel(
    boundaries,  // (num_facts, hidden_size)
    W,           // (hidden_size, kv_dim)
    out,         // (num_facts, kv_dim)
    ...
)
```

For 4-bit quantized models, the Python wrapper falls back to PyTorch `matmul`.

## Critical Technical Details

### SDPA Causal Mask Fix

When using SDPA with a pre-existing KV cache (facts at positions 0..T-1,
prompt at positions T..T+P-1), `is_causal=True` is **incorrect** — it assumes Q starts
at position 0, blocking attention from prompt to facts.

Fix: explicit 4D mask `(1, 1, P, T+P)` where each prompt token sees all
fact positions plus previous prompt positions.

### Memory Fix for 4-bit Models

`model.eval()` does NOT disable gradient tracking. The `MatMul4Bit` autograd
function keeps dequantized buffers. Wrapping in `torch.no_grad()` saves ~450 MB:

| | Without `no_grad` | With `no_grad` |
|---|---|---|
| Prefill peak | ~3,036 MB | ~2,583 MB |
| Decode | OOMs | Stable at ~2,600 MB |

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

1. **Boundary-only generation non-functional**: Boundary residual at position 0 with zero
   RoPE does not produce factual recall through HF attention. Requires custom
   CUDA attention kernel or fine-tuned model. The projection kernel is built and
   ready for this future path.

2. **GPU memory constrained**: Qwen3-4B (3.4GB) on 3GB GPU leaves ~1.1GB
   headroom. Works for 300–500 token articles. For longer context or larger
   models, use Qwen3-2B (~1.5GB) or a 6GB+ GPU.

## Credit

**chrishayuk/larql** — The boundary residual concept, crystal layer detection,
boundary-kv engine architecture, and "model IS the database" philosophy.
