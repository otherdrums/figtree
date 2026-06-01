# PDGA v2 — Fact-Centric Architecture

## Philosophy

Everything is a **Fact**. Narratives are facts containing other facts. Trust scores are facts. Graph edges are facts. Even the system itself is represented as facts.

Facts are stored as:
- **boundary.npy** — (hidden_size,) float32, ~10 KB. Compressed representation for retrieval and deduplication.
- **text.txt** — Natural language statement. Used to regenerate full KV cache on-the-fly during generation.
- **manifest.json** — Metadata (children, sources, trust, edge_type).

No pre-computed KV caches on disk. Storage is ~10 KB per fact (vs ~90 MB in v1).

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
│   ├── ingest.py       # Text → facts with boundary capture
│   ├── generate.py     # On-the-fly KV cache generation + standard attention
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
3. Save as `.pdga` directory: `boundary.npy` + `text.txt` + `manifest.json`

### Generation Engine

```python
from pdga.fact.generate import FactGenerator

gen = FactGenerator(model, tokenizer)
result = gen.generate(
    facts=[fact1, fact2, fact3],
    prompt="What happened at Davos?",
    max_new_tokens=100,
)
```

**On-the-fly KV generation:**
1. For each selected fact, tokenize its text and run through the model
2. This populates a `DynamicCache` with the fact's full KV entries
3. Prompt tokens are then forwarded with the pre-populated cache
4. Standard causal attention (explicit mask) ensures prompts see all fact positions
5. Decode autoregressively

**Why not boundaries for generation?** Boundaries are single 2560-d vectors captured at the crystal layer. They encode the full context of a fact in compressed form. However, on standard pretrained models with HF attention, projecting boundaries through W_k/W_v and using them as K/V entries does NOT produce factual recall. The model hasn't been trained to decode boundary residuals into specific facts. LARQL's custom Rust attention engine handles this differently. Until a custom attention kernel or fine-tuned model is available, on-the-fly KV generation from text is the pragmatic path.

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
```

## Design Decisions

1. **Everything is a Fact**: Narratives, edges, trust assertions, even the system itself. This unifies the data model and enables recursive reasoning (meta-facts about facts).

2. **Boundaries for retrieval, text for generation**: Boundaries (~10 KB) enable fast similarity search and deduplication. Text is used to regenerate full KV caches on-the-fly during generation. This gives compact storage without sacrificing recall.

3. **Custom CUDA kernel for boundary projection**: The kernel compiles and works, but 4-bit quantized models require a PyTorch fallback due to bitsandbytes' packed weight format. For non-quantized models, the CUDA kernel provides optimized boundary→K/V projection.

4. **On-the-fly KV generation**: Instead of storing KV caches on disk (~90 MB per narrative), we regenerate them from text during generation. This trades computation for storage and keeps the system lightweight.

5. **Fact-centric graph**: All relationships (SUPPORTS, CONTRADICTS, TRUST) are first-class Facts with their own boundaries and text. During generation, the model can load meta-facts alongside content facts, enabling trust-aware reasoning through attention.

## Known Limitations

1. **Boundary-only generation doesn't work on pretrained models**: Requires custom attention kernel or fine-tuned model. The CUDA kernel is built and ready for this future path.

2. **On-the-fly KV generation is slower than pre-computed**: ~30s for 35 facts vs ~25s for pre-loaded KV caches. Acceptable tradeoff for 800× storage savings.

3. **GPU memory constrained**: Qwen3-4B (3.4GB) on 3GB GPU leaves ~1.1GB headroom. Works for 300–500 token contexts. For longer contexts, use Qwen3-2B or larger GPU.
