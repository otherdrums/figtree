"""Atomic fact extraction with absolute position preservation.

Extracts sentences from narrative text, then slices the pre-computed KV cache
to create individual fact KV caches with their original absolute positions
preserved. This enables:
- Loading arbitrary subsets of facts while maintaining position coherence
- Gaps in the narrative (pruned facts) are handled by the attention mask
- Deduplication across narratives (shared facts have identical KV caches)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.cache_utils import DynamicCache

from pdga.delta.cache_io import save_window_cache
from pdga.delta.base import DeltaManifest, Delta


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving approximate boundaries."""
    # Simple regex: split on period + space, but handle abbreviations roughly
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def merge_short_sentences(sentences: list[str], tokenizer, min_tokens: int = 10) -> list[dict]:
    """Merge very short sentences with neighbors to reach minimum token count.
    
    Returns list of dicts with 'text' and 'token_count'.
    """
    merged = []
    buffer = []
    buffer_tokens = []
    
    for sent in sentences:
        tokens = tokenizer.encode(sent, add_special_tokens=False)
        
        if len(tokens) < min_tokens and not buffer:
            # Start buffering short sentences
            buffer.append(sent)
            buffer_tokens.extend(tokens)
        elif buffer:
            # Continue buffering
            buffer.append(sent)
            buffer_tokens.extend(tokens)
            
            # If buffer is now long enough, finalize
            if len(buffer_tokens) >= min_tokens:
                merged.append({
                    "text": " ".join(buffer),
                    "tokens": buffer_tokens,
                })
                buffer = []
                buffer_tokens = []
        else:
            # Sentence is long enough on its own
            merged.append({
                "text": sent,
                "tokens": tokens,
            })
    
    # Don't forget trailing buffer
    if buffer:
        merged.append({
            "text": " ".join(buffer),
            "tokens": buffer_tokens,
        })
    
    return merged


def extract_kv_slice(full_cache: DynamicCache, start_pos: int, end_pos: int, layer_idx: int):
    """Extract K/V tensors for a specific position range from the full cache.
    
    Args:
        full_cache: The complete KV cache from the full narrative forward pass
        start_pos: Starting token position (inclusive)
        end_pos: Ending token position (exclusive)
        layer_idx: Which layer to extract from
    
    Returns:
        (keys, values) tensors with shape (1, num_kv_heads, end_pos-start_pos, head_dim)
    """
    layer = full_cache.layers[layer_idx]
    # Slice the position dimension (dim=2)
    keys = layer.keys[:, :, start_pos:end_pos, :].clone()
    values = layer.values[:, :, start_pos:end_pos, :].clone()
    return keys, values


def save_fact_cache(
    full_cache: DynamicCache,
    fact_dir: Path,
    start_pos: int,
    end_pos: int,
    num_layers: int,
) -> Path:
    """Save a fact's KV cache slice extracted from the full narrative cache.
    
    The K/V tensors preserve their original absolute positions from the narrative.
    """
    fact_dir = Path(fact_dir)
    fact_dir.mkdir(parents=True, exist_ok=True)
    
    state = {}
    for li in range(num_layers):
        k, v = extract_kv_slice(full_cache, start_pos, end_pos, li)
        state[f"layer_{li}_keys"] = k.cpu()
        state[f"layer_{li}_values"] = v.cpu()
    
    # Include position metadata so generator knows where this fact belongs
    state["_metadata"] = {
        "start_pos": start_pos,
        "end_pos": end_pos,
        "token_count": end_pos - start_pos,
    }
    
    out_path = fact_dir / "kv_cache.pt"
    torch.save(state, out_path, _use_new_zipfile_serialization=True)
    return out_path


def ingest_narrative_with_facts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    output_dir: Path,
    narrative_id: str,
    source_id: str,
    trust: float = 0.5,
    window_size: int = 500,
    min_fact_tokens: int = 10,
) -> dict:
    """Ingest a narrative and extract atomic facts with absolute-position KV caches.
    
    Process:
    1. Forward full narrative → capture complete KV cache
    2. Split into sentences → merge short ones
    3. For each fact, extract KV slice from full cache at its absolute positions
    4. Save narrative KV + individual fact KV caches
    
    Returns:
        dict with narrative_delta_id, fact_ids, and paths
    """
    device = model.device
    output_dir = Path(output_dir)
    num_layers = model.config.num_hidden_layers
    
    # Step 1: Tokenize full narrative
    full_tokens = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(full_tokens)
    
    if not full_tokens:
        raise ValueError("Text produced zero tokens")
    
    # Step 2: Forward full narrative, capture KV cache
    input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)
    pos_ids = torch.arange(0, total_tokens, device=device, dtype=torch.long).unsqueeze(0)
    
    h = model.get_input_embeddings()(input_ids)
    full_cache = DynamicCache()
    
    with torch.no_grad():
        for li in range(num_layers):
            layer = model.model.layers[li]
            pe = model.model.rotary_emb(h, pos_ids)
            h = layer(
                h, attention_mask=None, position_ids=pos_ids,
                position_embeddings=pe, use_cache=True,
                past_key_values=full_cache,
            )
    
    # Step 3: Save full narrative KV cache
    narrative_delta_id = Delta.generate_id(text[:10000])
    narrative_dir = output_dir / f"{narrative_delta_id}.pdga"
    narrative_dir.mkdir(parents=True, exist_ok=True)
    
    save_window_cache(full_cache, narrative_dir, 0, num_layers)
    
    # Step 4: Split into sentences and extract facts
    sentences = split_into_sentences(text)
    facts = merge_short_sentences(sentences, tokenizer, min_tokens=min_fact_tokens)
    
    # Step 5: Compute absolute positions for each fact
    fact_results = []
    current_pos = 0
    
    for i, fact in enumerate(facts):
        fact_tokens = fact["tokens"]
        start_pos = current_pos
        end_pos = current_pos + len(fact_tokens)
        
        # Create fact directory
        fact_id = f"{narrative_delta_id}_fact_{i:03d}"
        fact_dir = narrative_dir / "facts" / fact_id
        fact_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract KV slice from full cache
        save_fact_cache(full_cache, fact_dir, start_pos, end_pos, num_layers)
        
        # Save metadata
        fact_meta = {
            "fact_id": fact_id,
            "narrative_id": narrative_delta_id,
            "source_id": source_id,
            "text": fact["text"],
            "start_pos": start_pos,
            "end_pos": end_pos,
            "token_count": len(fact_tokens),
        }
        (fact_dir / "manifest.json").write_text(json.dumps(fact_meta, indent=2))
        
        fact_results.append({
            "fact_id": fact_id,
            "path": fact_dir,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "token_count": len(fact_tokens),
        })
        
        current_pos = end_pos
    
    # Step 6: Save narrative metadata
    narrative_meta = {
        "narrative_id": narrative_delta_id,
        "source_id": source_id,
        "source_key": source_id,  # Original source identifier (e.g., "pro_globalist")
        "text": text,
        "token_count": total_tokens,
        "num_facts": len(facts),
        "trust": trust,
    }
    (narrative_dir / "narrative.json").write_text(json.dumps(narrative_meta, indent=2))
    
    # Cleanup
    del full_cache
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "narrative_id": narrative_delta_id,
        "narrative_path": narrative_dir,
        "num_facts": len(facts),
        "facts": fact_results,
    }
