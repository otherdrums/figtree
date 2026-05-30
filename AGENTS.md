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

# Generate with loaded deltas
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
- `pdga/cli/` — Typer CLI

## Model

Default: Qwen/Qwen2.5-1.5B-Instruct (cached at ~/.cache/huggingface/hub/)
GPU: Quadro T1000 (3GB VRAM)
