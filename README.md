# PDGA — Parallel Delta Graph Architecture

A system for ingesting arbitrary text into minimal delta representations
that recreate the model's KV cache state. Multiple deltas are loaded
simultaneously during generation, each at full fidelity, enabling the
model to hold conflicting facts in parallel.

## Inspiration

Directly inspired by [**chrishayuk/larql**](https://github.com/chrishayuk/larql):

- **Boundary residuals** — LARQL's Apollo engine stores one residual vector per text window at the crystal layer. The Apollo 11 transcript demo (370K tokens) compresses to ~2.8MB of boundary residuals — 20,000× over full KV cache.
- **Crystal layer detection** — The model layer where the residual stream stabilizes, enabling early-layer skipping.
- **"The model IS the database"** — Model weights reorganized as queryable knowledge with patches as lightweight overlays.

PDGA extends these concepts with sparse novelty-gated detection, hybrid
fact+residual storage, multi-stream parallel generation, and a delta graph
architecture.

## Quick Start

```bash
pip install -e .

pdga ingest article.txt --trust 0.99 --tags summit_event
pdga list-deltas
pdga show <delta_id>
pdga generate "What happened?" --deltas abc123 --mode hybrid
pdga think "Analyze" --streams "conscious:d=abc123:st=0.5|explore:d=def456:st=0.9"
pdga retrieve "trade summit"
pdga graph link --source abc123 --edge-type contradicts --target def456
pdga stats
```

## Demo

```bash
python examples/run_demo.py
```

Two conflicting news articles are ingested with sparse novelty detection.
The demo compares three generation modes:

| Mode | Storage per article | What's sent to model | Quality |
|------|-------------------|---------------------|---------|
| replay | 372 tokens | Full article text | Article-accurate |
| hybrid | 119 tokens (32%) | Fact chunks only | Good grounding |
| residual | 14 vectors | Boundary residuals | Generic/semantic |

## Architecture

```
pdga/
├── delta/         Delta types, ContextDelta, .pdga format I/O
├── db/            SQLite delta registry and edges
├── graph/         Typed graph edge management
├── ingest/        Crystal layer detection, sparse novelty gating, fact classification
├── kernel/        Per-delta generation (replay / residual / hybrid modes)
├── retrieval/     LSH-based boundary residual index
└── cli/           Typer CLI
```

### Delta Format (.pdga)

```
delta.pdga/
├── boundaries.npy       Sparse boundary residuals at crystal layer (novelty-gated)
├── fact_tokens.npz      Token chunks for unknowable entities/numbers (hybrid only)
├── window_tokens.npz    Token IDs per window (LSH routing)
├── manifest.json        Model config, crystal layer, provenance
└── metadata.json        Trust, source URL, tags
```

## Model

Default: `Qwen/Qwen2.5-1.5B-Instruct` (28 layers, hidden_size=1536)  
Entity extraction: `urchade/gliner_small-v2.1` (dynamic labels, zero-shot NER)  
Tested on: Quadro T1000 (3GB VRAM)

## Credit

**chrishayuk/larql** — The boundary residual concept, crystal layer detection,
and "model IS the database" philosophy.  
**urchade/GLiNER** — Zero-shot NER for dynamic fact extraction (who/what/where/when/how much).
