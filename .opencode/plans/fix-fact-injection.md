# Fix Figment Injection in Figtree v2

## Problem

`figtree/generate.py` processes figments with **no causal attention mask** (`attention_mask=None`). This means tokens within a figment attend to **all** tokens including future ones, producing incorrect K/V entries. The model then generates garbage (repetition, nonsense).

## Root Cause

In `Qwen3Attention.forward` (modeling_qwen3.py:270-271), the `eager_attention_forward` only adds an attention mask when it's not `None`:

```python
if attention_mask is not None:
    attn_weights = attn_weights + attention_mask
```

With `attention_mask=None`, no mask is applied → bidirectional attention within each fact.

Additionally, each fact is processed individually through all 36 layers with a fresh `h = embed(...)`. Later facts can attend to earlier facts via the `DynamicCache`, but intra-fact attention is non-causal.

## Fix

Replace the per-figment loop with a **single concatenated figment sequence** processed through all layers with a **proper causal mask**.

### Old code (lines 63-112)

```python
cache = DynamicCache()
current_pos = 0
for figment in figments:
    figment_ids = self.tokenizer.encode(figment.text, add_special_tokens=False)
    ...
    for li in range(self.num_layers):
        ...
        h = layer(h, attention_mask=None, ...)  # BUG: no mask
    current_pos += figment_len
prompt_offset = current_pos
...
for li in range(self.num_layers):
    pe = rotary(h, prompt_pos_ids)  # recomputed per layer
    h = layer(h, attention_mask=attn_mask, ...)
```

### New code

```python
cache = DynamicCache()

# Concatenate all figment token IDs
all_figment_ids = []
for figment in figments:
    fid = self.tokenizer.encode(figment.text, add_special_tokens=False)
    if fid:
        all_figment_ids.extend(fid)

total_figment_len = len(all_figment_ids)

if total_figment_len > 0:
    figment_pos_ids = torch.arange(total_figment_len, device=device, dtype=torch.long).unsqueeze(0)

    # Causal mask: (1, 1, total_figment_len, total_figment_len)
    figment_mask = torch.full(
        (1, 1, total_figment_len, total_figment_len),
        float('-inf'), device=device, dtype=torch.float32,
    )
    for i in range(total_figment_len):
        figment_mask[:, :, i, :i + 1] = 0.0

    h = embed(torch.tensor([all_figment_ids], dtype=torch.long, device=device))
    pe_figments = rotary(h, figment_pos_ids)  # computed ONCE
    for li in range(self.num_layers):
        layer = self.model.model.layers[li]
        h = layer(
            h, attention_mask=figment_mask, position_ids=figment_pos_ids,
            position_embeddings=pe_figments, use_cache=True,
            past_key_values=cache,
        )

prompt_offset = total_figment_len
total_len = prompt_offset + P

# ── Prompt prefill (mask stays the same) ──
...
h = prompt_emb
pe_prompt = rotary(h, prompt_pos_ids)        # computed ONCE
for li in range(self.num_layers):
    layer = self.model.model.layers[li]
    h = layer(
        h, attention_mask=attn_mask, position_ids=prompt_pos_ids,
        position_embeddings=pe_prompt, use_cache=True,
        past_key_values=cache,
    )
```

Also optimize decode to compute `pe` once:

```python
pe_decode = rotary(h, pos_one)
for li in range(self.num_layers):
    ...
```

## Verification

Run `python3 examples/run_davos_v2.py all` and check:
1. Per-source generation produces relevant Davos content (not repetitive nonsense)
2. "All sources" query produces coherent text (not "444444...")
3. "Contradictions" query identifies actual disagreements

## Follow-up: Per-layer Boundary Projection (LARQL-like)

After the fix is verified, implement boundary-projection generation:

1. **Ingestion change**: Store boundaries at ALL layers (not just crystal layer)
2. **Generation change**: Project per-layer boundaries through W_k/W_v → K/V entries, pre-fill cache, generate
3. This enables lightweight storage + parallel generation from different figment combinations

See `figtree/kernel/boundary_project.py` and `figtree/kernel/boundary_project.cu` for the existing projection machinery.
