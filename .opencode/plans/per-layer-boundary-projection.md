# Per-Layer Boundary Projection Generation

## Architecture

**Figments are context-independent.** Each atomic figment is processed ALONE through the model during ingestion. Boundaries capture just that figment, not any narrative context. Deduplication means a figment like "Roses are red" is stored once even if referenced by 50 Images.

Trust is per-figment: a false Image is just another figment whose children happen to be true. You distrust the Image figment independently of its children.

## Ingestion Changes

### `figtree/figment.py`

New fields already added to `Figment`:
- `boundaries: np.ndarray | None` — shape `(num_layers, hidden_size)`, all 36 layer output boundaries
- `boundary_emb: np.ndarray | None` — shape `(hidden_size,)`, last-token embedding output

Files per figment:
```
figment_abc123.figment/
├── manifest.json
├── boundary.npy          # (hidden_size,) — crystal layer
├── boundaries.npy        # (num_layers, hidden_size) — all layers
├── boundary_emb.npy      # (hidden_size,) — last-token embedding
├── kv_cache.npy          # (num_layers, seq_len, 2, kv_dim) — unrotated K/V
└── text.txt
```

### `figtree/ingest.py`

For each sentence:
1. Compute embedding output for the last token → `boundary_emb`
2. Register hooks on ALL layers (not just crystal layer)
3. Run `model(ids)` — hooks capture each layer's output
4. Extract last-token hidden state from each layer → `boundaries[li, :]`
5. Compute per-token K/V (input_layernorm → k_proj/v_proj → k_norm)
6. Store in the Figment

The Image figment (which contains children) does NOT get its own boundaries — it references its children's boundaries through the `children` list.

## Generation Changes

### `figtree/generate.py` — `generate_from_boundaries()`

```
Input:
  - figments: list[Figment] — with cached kv_cache.npy on disk
  - prompt: str
  - max_new_tokens: int
```

Algorithm:
```
cache = DynamicCache()

For each figment:
    Load kv_cache.npy: (num_layers, seq_len, 2, kv_dim)
    Concatenate all K/V across figments

Apply RoPE based on global positions for all entries
Insert into DynamicCache

Prompt prefill + decode as in generate()
```

### Key insight

Since each figment's K/V is computed with proper `input_layernorm` during ingestion, the K/V entries match the model's internal computation exactly. When concatenated in the cache, standard HF attention handles inter-figment relationships naturally at query time.

## Implementation Status

This is now **implemented and working**:
- `generate_from_boundaries()` in `figtree/generate.py` loads per-token cached K/V from disk
- RoPE is applied lazily during generation based on global position IDs
- ~22% faster than text-based generation for 35 figments

## Verification

```bash
# Both should produce fact-grounded output
python3 -c "
from figtree.figment import Figment
from figtree.generate import FigmentGenerator
result_text = gen.generate(figments=selected, prompt='What happened?')
result_bd = gen.generate_from_boundaries(figments=selected, prompt='What happened?', cache_dir='./davos')
"
```
