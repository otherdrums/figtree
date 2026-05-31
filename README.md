# PDGA — Parallel Delta Graph Architecture

A system for ingesting arbitrary text into delta representations that capture
the model's full KV cache state. During generation, pre-computed KV caches are
loaded from disk into GPU memory, enabling instant prefill and factual recall
from article-length context on constrained GPUs.

## Inspiration

Directly inspired by [**chrishayuk/larql**](https://github.com/chrishayuk/larql):

- **Boundary residuals** — LARQL's generation engine stores one residual vector per text window at the crystal layer. The generation 11 transcript demo (370K tokens) compresses to ~2.8MB of boundary residuals — 20,000× over full KV cache.
- **Crystal layer detection** — The model layer where the residual stream stabilizes, enabling early-layer skipping.
- **"The model IS the database"** — Model weights reorganized as queryable knowledge with patches as lightweight overlays.
- **Boundary-KV engine** — generation stores boundary residuals for LSH retrieval, and full KV caches for generation. This is retrieval + generation, not compression.

PDGA extends these concepts with KV cache serialization, progressive GPU loading
from system RAM, SDPA-based streaming attention, and a delta graph architecture.

## Quick Start

```bash
pip install -e .

# Ingest an article (captures full KV cache to disk)
pdga ingest article.txt --trust 0.99 --tags summit_event

# List stored deltas
pdga list-deltas

# Generate with loaded KV cache (instant prefill)
pdga generate "What happened?" --deltas abc123

# Multi-stream parallel generation
pdga think "Analyze" --streams "conscious:d=abc123:st=0.5|explore:d=def456:st=0.9"

# Retrieve relevant deltas
pdga retrieve "trade summit"

# Manage graph edges
pdga graph link --source abc123 --edge-type contradicts --target def456
pdga graph show
```

## Demo

```bash
python3 examples/run_demo.py all
```

Two conflicting news articles are ingested with full KV cache capture.
The demo runs the **streaming generator** with progressive KV loading:

```
Full article (372 tokens) → KV cache on disk → progressive GPU load → SDPA → 350 tokens generated
```

**Results** (Qwen3-4B on Quadro T1000, 3GB):

| Article | Tokens | Facts Found | Recall |
|---------|--------|-------------|--------|
| Article A (pro-deal) | 372 | 19/20 | 95% |
| Article B (skeptical) | 366 | 12/14 | 86% |

**Sovereignty**: Zero cross-contamination. Each article's output contains only
facts from its own context.

## Architecture

```
pdga/
├── delta/         Delta types, ContextDelta, .pdga format I/O
├── db/            SQLite delta registry and edges
├── graph/         Typed graph edge management
├── ingest/        Crystal layer detection, text→ContextDelta pipeline, KV cache capture
├── kernel/        Per-delta generation (corrected engine with injection)
├── retrieval/     LSH-based boundary residual index
├── generation/        Streaming generator + boundary-kv engine
└── cli/           Typer CLI
```

### Delta Format (.pdga)

```
delta.pdga/
├── boundaries.npy          Boundary residuals at crystal layer (for LSH retrieval)
├── injection_deltas.npz    Injection delta vectors per window
├── injection_token_ids.npz Entity token IDs for injection routing
├── window_tokens.npz       Token IDs per window
├── kv_cache_w0.pt          Full KV cache for window 0 (K/V per layer)
├── manifest.json           Model config, crystal layer, provenance
└── metadata.json           Trust, source URL, tags
```

## Model

Default: `unsloth/Qwen3-4B-bnb-4bit` (36 layers, hidden_size=2560, 4-bit quantized)  
Entity extraction: `urchade/gliner_small-v2.1` (dynamic labels, zero-shot NER)  
Tested on: Quadro T1000 (3GB VRAM)

## How It Works

### 1. Ingestion (`pdga/ingest/text.py`)

Text is split into windows (default 500 tokens, or larger than article length).
For each window:
- Manual forward through all 36 layers with `DynamicCache`
- KV cache saved to `kv_cache_w{N}.pt`
- Boundary residual captured at crystal layer
- GLiNER extracts entities for injection entries

### 2. Generation (`pdga/generation/streaming.py`)

**Prefill** (instant — no window token re-processing):
1. Load full KV cache from disk → CPU RAM
2. For each layer: move window K/V from CPU → GPU, update `DynamicCache`
3. Forward prompt tokens through layer attending to cached KVs
4. Explicit causal 4D mask ensures prompt sees all window positions

**Decode** (autoregressive):
1. Sample next token
2. Forward through all layers with cached KVs
3. KV cache grows by 1 position per layer per step

All forward passes wrapped in `torch.no_grad()` — saves ~450 MB GPU by preventing
4-bit dequantized weight buffers from accumulating in the autograd graph.

### 3. Boundary-KV Engine (`pdga/generation/boundary_kv.py`)

Alternative path that loads KV caches directly into `DynamicCache` without
the progressive layer-by-layer loading. Same causal mask fix and `no_grad`
optimization. Used for multi-delta sequential processing.

## Critical Technical Details

### SDPA Causal Mask Fix

When using SDPA with a pre-existing KV cache (window at positions 0..T-1,
prompt at T..T+P-1), `is_causal=True` is **incorrect** — it assumes Q starts
at position 0, blocking attention from prompt to window.

Fix: explicit 4D mask `(1, 1, P, T+P)` where each prompt token sees all
window positions plus previous prompt positions.

### Memory Fix for 4-bit Models

`model.eval()` does NOT disable gradient tracking. The `MatMul4Bit` autograd
function keeps dequantized buffers. Wrapping in `torch.no_grad()` saves ~450 MB:

| | Without `no_grad` | With `no_grad` |
|---|---|---|
| Prefill peak | ~3,036 MB | ~2,583 MB |
| Decode | OOMs | Stable at ~2,600 MB |

## Testing

```bash
# Full demo: ingest + generate with fact verification
python3 examples/run_demo.py all

# Boundary-kv end-to-end test
python3 tests/test_boundary_kv.py

# Comprehensive tests (multi-delta, facts, benchmark)
python3 tests/test_generation_comprehensive.py multi
python3 tests/test_generation_comprehensive.py facts
python3 tests/test_generation_comprehensive.py bench
```

## Known Limitations

1. **Compressed path non-functional**: Boundary residual at position 0 with zero
   RoPE does not produce factual recall through HF attention. Requires custom
   CUDA kernel or fine-tuned model.

2. **GLiNER injection entries are noise**: Subword fragments at uniform 1.0
   coefficient — no discriminative power. Needs better entity extraction.

3. **GPU memory constrained**: Qwen3-4B (3.4GB) on 3GB GPU leaves ~1.1GB
   headroom. Works for 300–500 token articles. For longer context or larger
   models, use Qwen3-2B (~1.5GB) or a 6GB+ GPU.

## Credit

**chrishayuk/larql** — The boundary residual concept, crystal layer detection,
boundary-kv engine architecture, and "model IS the database" philosophy.

**urchade/GLiNER** — Zero-shot NER for dynamic fact extraction.
