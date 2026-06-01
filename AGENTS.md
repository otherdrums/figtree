# PDGA v2 — Fact-Centric Architecture

## Philosophy

Everything is a **Fact**. Narratives are facts containing other facts. Trust scores are facts. Graph edges are facts. Even the system itself is represented as facts.

Facts are stored as:
- **boundary.npy** — (hidden_size,) float32, ~10 KB. Compressed representation for retrieval and deduplication.
- **boundaries.npy** — (num_layers, hidden_size) float32, ~360 KB. Per-layer hidden states for all layers.
- **boundary_emb.npy** — (hidden_size,) float32. Last-token embedding.
- **kv_cache.npy** — (num_layers, seq_len, 2, kv_dim) float32, ~2.8 MB per 20-token fact. Pre-computed unrotated K/V for every token at every layer.
- **text.txt** — Natural language statement. Used for text-based generation fallback.
- **manifest.json** — Metadata (children, sources, trust, edge_type).

## Build / Test / Lint

```bash
pip install -e .
ruff check pdga/

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
pdga/
├── fact/
│   ├── primitive.py    # Fact dataclass + save/load
│   ├── ingest.py       # Text → facts with boundary + kv_cache capture
│   ├── generate.py     # On-the-fly KV + cached boundary KV generation
│   └── graph.py        # Edges/trust as facts + dedup
└── kernel/
    ├── boundary_project.cu   # CUDA kernel: boundaries @ W_k/W_v
    ├── boundary_project.py   # Python wrapper
    └── build.py              # torch.utils.cpp_extension.load
```

### Custom CUDA Kernel

**File:** `pdga/kernel/boundary_project.cu`

Projects fact-boundary vectors through a layer's W_k and W_v weight matrices:
```cuda
__global__ void boundary_project_bf16_kernel(
    const __nv_bfloat16* boundaries,  // (num_facts, hidden_size)
    const __nv_bfloat16* W,            // (hidden_size, kv_dim)
    __nv_bfloat16* out,                // (num_facts, kv_dim)
    ...
)
```

**Usage:**
```python
from pdga.kernel.boundary_project import project_boundaries_to_kv
k_facts, v_facts = project_boundaries_to_kv(boundaries, layer, device)
```

**Note:** For 4-bit quantized models, the Python wrapper falls back to PyTorch `matmul` because bitsandbytes stores weights in a packed format that the raw CUDA kernel cannot access. The kernel is still built and available for non-quantized models.

### Fact Primitive

```python
@dataclass
class Fact:
    fact_id: str          # SHA-256(text)[:16]
    text: str             # Natural language statement
    boundary: np.ndarray  # (hidden_size,) float32 — ONLY stored tensor
    meta: dict            # edge_type, about_fact, etc.
    children: list[str]   # Child fact IDs
    sources: list[str]    # Parent fact IDs
    trust: float          # Cached trust score
    boundaries: np.ndarray | None = None  # (num_layers, hidden_size)
    boundary_emb: np.ndarray | None = None  # (hidden_size,) last-token embedding
```

**Narrative = Fact with `children=[fact_1, fact_2, ...]`**
**Edge = Fact with `meta["edge_type"] = "supports"`**
**Trust = Fact with `meta["edge_type"] = "trust", meta["score"]=0.95`**

### Ingestion Pipeline

```python
from pdga.fact.ingest import ingest_text_to_facts

facts = ingest_text_to_facts(
    model, tokenizer, text,
    output_dir=Path("./facts"),
    source_id="reuters",
    trust=0.95,
)
# Returns: [narrative, atomic_1, atomic_2, ..., trust_assertion]
```

For each sentence:
1. Forward through model layers 0..crystal_layer
2. Capture boundary = hidden state of LAST token at crystal_layer
3. Compute per-token per-layer K/V (unrotated) by projecting each layer's
   normed input through W_k/W_v with k_norm
4. Save as `.pdga` directory: `boundary.npy` + `text.txt` + `manifest.json`
   + `kv_cache.npy` (per-token unrotated K/V)

### Generation Engine

```python
from pdga.fact.generate import FactGenerator

gen = FactGenerator(model, tokenizer)

# Text-based generation (forward pass for each fact)
result = gen.generate(
    facts=[fact1, fact2, fact3],
    prompt="What happened at Davos?",
    max_new_tokens=100,
)

# Boundary-based generation (load cached K/V from disk)
result = gen.generate_from_boundaries(
    facts=[fact1, fact2, fact3],
    prompt="What happened at Davos?",
    max_new_tokens=100,
    cache_dir="./facts",
)
```

**Text-based KV generation (`generate`):**
1. For each selected fact, tokenize its text and run through the model
2. Populates a `DynamicCache` with the fact's full KV entries
3. Prompt tokens are forwarded with the pre-populated cache
4. Standard causal attention (explicit mask) ensures prompts see all fact positions
5. Decode autoregressively

**Boundary-based KV generation (`generate_from_boundaries`):**
1. Loads pre-computed per-token unrotated K/V from disk (`kv_cache.npy`)
2. Applies RoPE based on global position IDs
3. Inserts into `DynamicCache` directly — no forward pass for fact tokens
4. Prompt prefill + decode proceed as usual

The boundary-based approach avoids re-running the forward pass for fact tokens, trading ~2.8 MB/fact disk storage for ~20% faster generation. Per-token K/V is computed during ingestion by capturing each layer's input hidden state, applying `input_layernorm`, and projecting through `k_proj`/`v_proj` with `k_norm` applied. This matches the model's internal computation exactly, enabling factual recall with standard HF attention.

### Graph as Facts

```python
from pdga.fact.graph import FactGraph

graph = FactGraph(all_facts)
graph.deduplicate()          # exact + semantic boundary similarity
graph.create_edges()         # SUPPORTS, SAME_ENTITY, CONTRADICTS
graph.propagate_trust()      # source trust → facts + alignment boost
```

All graph operations produce **new Facts**:
- Deduplication creates `Fact(meta={"edge_type": "supports"})`
- Trust propagation creates `Fact(meta={"edge_type": "trust", "score": 0.95})`
- Contradictions create `Fact(meta={"edge_type": "contradicts"})`

## Model

Default: Qwen3-4B (unsloth bnb-4bit, cached at ~/.cache/huggingface/hub/)
GPU: Quadro T1000 (3GB VRAM)

## Storage Comparison

| | PDGA v1 | PDGA v2 | Savings |
|---|---|---|---|
| Per narrative | ~90 MB | ~10 KB | **9,000×** |
| 3 narratives (33 facts) | ~270 MB | ~330 KB | **800×** |
| Query with 10 facts | Load ~80 MB KV | Load ~100 KB boundaries, gen KV on-the-fly | **800×** |

## Benchmark Results

```bash
python3 examples/davos_benchmark_v2.py
```

| Phase | Time | Throughput |
|-------|------|------------|
| Ingestion | 23.4s | 41 facts, 445 KB |
| Generation | 33.6s | 100 tokens, 35 facts |
| Graph | 0.0s | 38 facts, 3 edges |
| **Total** | **57.0s** | |

Boundary-based generation (`generate_from_boundaries`) achieves ~22% faster
generation vs text-based (`generate`) by skipping per-fact forward passes,
at the cost of ~2.8 MB/fact disk storage.

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
from pdga.fact.primitive import Fact
from pdga.fact.generate import FactGenerator
gen = FactGenerator(model, tokenizer)
result = gen.generate_from_boundaries(facts, prompt, cache_dir='./davos')
"
```

## Design Decisions

1. **Everything is a Fact**: Narratives, edges, trust assertions, even the system itself. This unifies the data model and enables recursive reasoning (meta-facts about facts).

2. **Boundaries for retrieval, text for generation**: Boundaries (~10 KB) enable fast similarity search and deduplication. Text is used to regenerate full KV caches on-the-fly during generation. This gives compact storage without sacrificing recall.

3. **Custom CUDA kernel for boundary projection**: The kernel compiles and works, but 4-bit quantized models require a PyTorch fallback due to bitsandbytes' packed weight format. For non-quantized models, the CUDA kernel provides optimized boundary→K/V projection.

4. **On-the-fly KV generation**: Instead of storing KV caches on disk (~90 MB per narrative), we regenerate them from text during generation. This trades computation for storage and keeps the system lightweight.

5. **Fact-centric graph**: All relationships (SUPPORTS, CONTRADICTS, TRUST) are first-class Facts with their own boundaries and text. During generation, the model can load meta-facts alongside content facts, enabling trust-aware reasoning through attention.

## Known Limitations

1. **On-the-fly KV generation is slower than pre-computed**: ~30s for 35 facts with text-based generation. Boundary-based generation (`generate_from_boundaries`) with cached K/V is ~22% faster at ~25s. Still slower than fully pre-loaded KV caches.

2. **GPU memory constrained**: Qwen3-4B (3.4GB) on 3GB GPU leaves ~1.1GB headroom. Works for 300–500 token contexts. For longer contexts, use Qwen3-2B or larger GPU.

3. **kv_cache.npy storage**: ~2.8 MB per 20-token fact. 100 facts = ~280 MB. Manageable on modern drives but not as compact as ~10 KB boundary-only storage.
