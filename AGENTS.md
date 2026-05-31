# PDGA — Parallel Delta Graph Architecture

## Build / Test / Lint

```bash
# Install in dev mode
pip install -e .

# Run CLI
pdga --help

# Run tests
pytest tests/

# Lint
ruff check pdga/
```

## Key Commands

```bash
# Ingest text into context delta
pdga ingest article.txt --trust 0.99 --tags summit_event

# List stored deltas
pdga list-deltas

# Generate with generation engine (multi-delta, KV cached, with injection)
pdga generate "What happened at the summit?" --deltas abc123,def456

# Multi-stream parallel generation
pdga think "Analyze the event" --streams "conscious:d=abc123:dt=0.8:st=0.3|explore:d=def456:dt=0.9:st=1.0"

# Retrieve relevant deltas for a query
pdga retrieve "trade summit agreement"

# Manage graph edges
pdga graph link --source abc123 --edge-type contradicts --target def456
pdga graph show
```

## Architecture

- `pdga/delta/` — Delta types, ContextDelta, .pdga format I/O
- `pdga/db/` — SQLite delta registry and edges
- `pdga/graph/` — Typed graph edge management
- `pdga/ingest/` — Crystal layer detection + text→ContextDelta pipeline
- `pdga/kernel/` — Multi-delta attention, generation, multi-stream orchestrator
- `pdga/retrieval/` — LSH-based boundary residual index
- `pdga/generation/` — generation generation engine with KV caching and injection
- `pdga/cli/` — Typer CLI

## Model

Default: Qwen3-4B (unsloth bnb-4bit, cached at ~/.cache/huggingface/hub/)
GPU: Quadro T1000 (3GB VRAM)

## Current State

### Working

- **Streaming Generator** (`pdga/generation/streaming.py`): Progressive KV loading from CPU RAM to GPU
  - Full article KV cache stored in system RAM during generation (one `.pt` file per article)
  - Progressive layer-by-layer loading: each window K/V moved to GPU just before that layer's forward pass
  - Explicit causal 4D mask handles position offset (window at 0..T-1, prompt at T..T+P-1)
  - SDPA backend (memory-efficient tiled attention)
  - All forward passes wrapped in `torch.no_grad()` to prevent 4-bit dequantized weight buffers from accumulating in autograd graph (saves ~450 MB GPU)
  - Verified numerically identical to standard SDPA (max diff <0.05)
  - Speed: ~3.5–6 t/s on Quadro T1000 with 300–500 token context
  - **Demo factual recall**: 19/20 facts (95%) from 372-token article, zero cross-contamination

- **Boundary-KV Engine** (`pdga/generation/boundary_kv.py`): Instant prefill from pre-computed KV caches
  - Loads KV caches from disk directly into DynamicCache, skips 36-layer window token prefill
  - Sequential delta processing: each delta's KV loaded, generated, then freed before next delta
  - Same `torch.no_grad()` optimization as streaming generator
  - Explicit causal mask for position offset (same fix as streaming)

- **KV Cache Serialization** (`pdga/delta/cache_io.py`): Save/load DynamicCache per window
  - `save_window_cache()`: Extracts K/V tensors from DynamicCache, saves to `.pt` file
  - `load_window_cache()`: Reconstructs DynamicCache from `.pt` file
  - Single-file format: `kv_cache_w{N}.pt` with keys `layer_{L}_keys`, `layer_{L}_values`
  - Stored alongside `.pdga` directory

- **Ingestion** (`pdga/ingest/text.py`): Text → ContextDelta with crystal detection,
  boundary capture, **full KV cache capture**
  - Uses `torch.inference_mode()` (equivalent to `torch.no_grad()`) during forward passes
  - Manual forward through all layers with DynamicCache to capture per-window KV cache
  - `del cache` after each window to free GPU memory before next window

- **Delta format**: `.pdga` directory with `boundaries.npy`, `window_tokens.npz`, **plus `kv_cache_w{N}.pt` files**

- **Retrieval**: LSH over boundary residuals for query-delta matching

- **Graph**: Edge management (contradicts, about_same_event)

### Not Working

- **Compressed forward path**: Boundary residual at position 0 with zero RoPE does NOT
  produce factual recall through HF attention. The single 2560-d vector doesn't decode
  into specific facts during multi-step generation. LARQL's custom Rust attention engine
  likely handles boundary KV differently (no RoPE, direct KV injection, or custom
  attention patterns). A custom CUDA attention kernel is needed for this path.

## generation engine Architecture

### Streaming Generator (`pdga/generation/streaming.py`)

```
Prefill:
  1. Load full KV cache from disk → CPU RAM (torch.load, map_location="cpu")
  2. For each layer L in 0..35:
     a. Move window K/V for layer L from CPU → GPU
     b. Update DynamicCache with window K/V
     c. Forward prompt tokens through layer L with cached KVs
     d. Prompt positions attend to all window positions (explicit causal mask)
  3. Final norm → lm_head → logits for next token

Decode (autoregressive, per-delta):
  1. Sample next token from logits
  2. Forward new token through all layers with cached KVs
  3. KV cache grows by 1 position per layer per decode step
```

### Boundary-KV Engine (`pdga/generation/boundary_kv.py`)

```
Prefill:
  1. Load KV cache from disk → DynamicCache (layer-by-layer to save memory)
  2. Prompt tokens embed → forward through L0..L35 attending to cached KVs
  3. Explicit causal mask ensures prompt sees all cached positions
  4. Final norm → lm_head → logits

Decode:
  Same as standard autoregressive decode with cached KVs
```

### Corrected Engine (`pdga/kernel/corrected.py`)

RAG-style generation with injection at L29:
- Window tokens + prompt as context → full forward with KV caching
- Injection delta added at injection layer (h[:, -1, :] += delta × 10.0)
- Each delta processes independently with its own KV cache
- Requires re-running window tokens through model (no pre-computed KV)

### Compressed path (experimental, non-functional)

- Position 0 = BOS embedding → replaced by boundary residual at crystal layer
- RoPE zeroed (cos=1, sin=0) at position 0 for layers crystal..N
- Boundary goes through Q/K/V projections without rotation
- Produces coherent next-token predictions but collapses during multi-step decode
- **Confirmed**: 11 variants tested, none produce factual recall through HF attention

## Critical Memory Fix

**The `torch.no_grad()` requirement for 4-bit models:**

`model.eval()` does NOT disable PyTorch gradient tracking. The `MatMul4Bit`
autograd function (bitsandbytes) keeps dequantized 4-bit weight buffers in
the computation graph during forward passes. Each MLP layer temporarily
dequantizes ~106 MB of weights — with gradients enabled, these buffers
accumulate instead of being freed.

**Impact on Quadro T1000:**
- Without `torch.no_grad()`: prefill peaks at ~3,036 MB, decode OOMs
- With `torch.no_grad()`: prefill peaks at ~2,583 MB, decode stable at ~2,600 MB
- **Saves ~450 MB** — the difference between OOM and smooth generation

**Both `StreamingGenerator` and `generate_boundary_kv()` wrap all forward
passes in `torch.no_grad()`.** Ingestion uses `torch.inference_mode()`
(equivalent). The demo sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
to further reduce fragmentation.

## Memory Breakdown (Quadro T1000, 3.63 GB)

| Component | Memory |
|---|---|
| Qwen3-4B 4-bit weights | 2,530 MB |
| 300-token KV cache (loaded) | ~46 MB |
| Attention / MLP temp (with `no_grad`) | ~20 MB |
| **Total allocated** | **~2,600 MB** |
| GPU capacity | 3,717 MB |
| **Headroom** | **~1,100 MB** |

## SDPA Causal Mask Fix

When using SDPA with a pre-existing KV cache (window tokens at positions
0..T-1, prompt at positions T..T+P-1), `is_causal=True` is **incorrect**.
SDPA's causal mask assumes Q starts at position 0, which blocks attention
from prompt tokens to earlier window positions.

**Fix**: Explicit 4D mask `(1, 1, P, T+P)` with:
```python
mask = torch.full((1, 1, P, T+P), -(2**15), device=device, dtype=dtype)
for i in range(P):
    mask[:, :, i, :T + i + 1] = 0.0
```

This ensures each prompt token attends to all window positions plus all
previous prompt positions.

## Demo Results

```bash
python3 examples/run_demo.py all
```

**Article A (372 tokens, pro-deal):** 17/20 facts (85% recall)
- Found: 47 nations, Maria Okonkwo, Sarah Chen, $45, $12 billion, $900 billion,
  3%, 1.8%, 20 to 12, Brussels, digital services, carbon tariffs,
  pharmaceutical, climate adaptation, global GDP, landmark, S&P
- Missing: multilateral cooperation, shared prosperity, turning point

**Article B (366 tokens, skeptical):** 14/14 facts (100% recall)
- Found: United States, China, walked out, $900 billion, $45, Geneva,
  Maria Okonkwo, IMF, S&P, 1.7%, Global Trade Summit, pharmaceutical,
  carbon tariffs, digital taxation

**Sovereignty**: Zero cross-contamination. Article A output contains no
B-only facts; Article B output contains no A-only facts.

## Testing

```bash
# Streaming attention correctness test (verifies SDPA output match)
python3 -c "from pdga.generation.streaming import StreamingGenerator; ..."

# Boundary-kv end-to-end test
python3 tests/test_boundary_kv.py

# Full demo with ingestion + generation pipeline
python3 examples/run_demo.py all

# Multi-delta generation test
python3 tests/test_generation_comprehensive.py multi

# Factual recall test (300 tokens)
python3 tests/test_generation_comprehensive.py facts

# Speed benchmark
python3 tests/test_generation_comprehensive.py bench
```

## Design Decisions

1. **Boundary-kv is retrieval + generation, not compression**: LARQL's generation is
   a "boundary retrieval store" — boundaries for LSH retrieval, KV cache for
   generation. The compressed path (boundary at position 0) doesn't decode facts
   through HF; uncompressed path with pre-computed KV caches is the working
   architecture.

2. **System RAM staging, not disk streaming**: Full KV cache loaded to system RAM
   at generator init; tiles moved to GPU one layer at a time. Faster than disk
   I/O during generation.

3. **Single large window per article**: `window_size=500` (or larger than article)
   produces one KV cache per article, avoiding multi-window RoPE offset issues.

4. **Sequential delta processing**: Each delta's KV cache loaded to GPU,
   generation completes, then freed before next delta — keeps peak memory at
   model + 1 delta's KV + prompt.

5. **Explicit causal mask over `is_causal=True`**: Required when Q starts at
   non-zero position offset. Verified numerically identical to standard SDPA.
