"""Streaming KV generation — progressive GPU loading with SDPA backend.

Supports loading multiple fact KV caches with absolute position preservation.
Facts can be loaded in arbitrary order; the generator maintains their original
narrative positions for correct attention computation.
"""

from __future__ import annotations

import gc
import time
import torch
from pathlib import Path
from typing import Union
from transformers.cache_utils import DynamicCache
from pdga.kernel.prompt import build_prompt_ids


class FactKVCache:
    """Represents one fact's KV cache with its absolute narrative position."""
    
    def __init__(self, path: Path):
        self.path = Path(path)
        self.state = torch.load(str(path), map_location="cpu", weights_only=True)
        
        # Read position metadata
        meta = self.state.get("_metadata", {})
        if meta:
            # Fact cache with absolute positions
            self.start_pos = meta.get("start_pos", 0)
            self.end_pos = meta.get("end_pos", 0)
            self.token_count = meta.get("token_count", 0)
        else:
            # Legacy full-narrative cache: infer from tensor shape
            k = self.state.get("layer_0_keys")
            if k is not None:
                self.token_count = k.shape[2]
                self.start_pos = 0
                self.end_pos = self.token_count
            else:
                self.token_count = 0
                self.start_pos = 0
                self.end_pos = 0
    
    def get_kv(self, layer_idx: int, device, dtype):
        """Get K/V tensors for a specific layer, moved to device."""
        k = self.state[f"layer_{layer_idx}_keys"].to(device=device, dtype=dtype)
        v = self.state[f"layer_{layer_idx}_values"].to(device=device, dtype=dtype)
        return k, v


class StreamingGenerator:
    """Generate using progressively loaded KV cache.

    Supports loading multiple facts with absolute position preservation.
    Facts can have gaps (missing positions); the attention mask handles this.
    """

    def __init__(
        self,
        model,
        kv_cache_paths: Union[Path, list[Path]],
    ):
        self.model = model
        self.num_layers = model.config.num_hidden_layers
        
        # Normalize to list
        if isinstance(kv_cache_paths, Path):
            kv_cache_paths = [kv_cache_paths]
        
        # Load all fact KV caches
        self.facts: list[FactKVCache] = []
        for path in kv_cache_paths:
            self.facts.append(FactKVCache(path))
        
        # Sort by absolute position
        self.facts.sort(key=lambda f: f.start_pos)
        
        # Compute total context length (max end_pos)
        if self.facts:
            self.max_pos = max(f.end_pos for f in self.facts)
            self.total_context_tokens = sum(f.token_count for f in self.facts)
        else:
            self.max_pos = 0
            self.total_context_tokens = 0
        
        gc.collect()

    def generate(
        self,
        tokenizer,
        prompt: str,
        injection_layer: int = 30,
        max_new_tokens: int = 80,
        sample_temp: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> dict:
        device = self.model.device
        dtype = self.model.dtype
        embed = self.model.get_input_embeddings()
        lm_head = self.model.lm_head
        final_norm = self.model.model.norm
        rotary = self.model.model.rotary_emb
        eos = tokenizer.eos_token_id
        
        # The prompt starts after the maximum narrative position
        offset = self.max_pos
        
        prompt_ids = build_prompt_ids(tokenizer, prompt, enable_thinking=False)
        P = len(prompt_ids)
        
        t0 = time.perf_counter()
        
        # ── Prepare ──────────────────────────────────────────────────────
        cache = DynamicCache()
        
        # Prefill prompt tokens
        h = embed(torch.tensor([prompt_ids], dtype=torch.long, device=device))
        pos_ids = torch.arange(offset, offset + P, device=device, dtype=torch.long).unsqueeze(0)
        
        # Build explicit causal mask that handles gaps
        # Total sequence length = max_pos + P (includes gaps)
        total_len = offset + P
        attn_mask = torch.full(
            (1, 1, P, total_len), -(2**15), device=device, dtype=dtype,
        )
        
        for i in range(P):
            prompt_pos = offset + i
            # Allow attention to all loaded fact positions
            for fact in self.facts:
                if fact.end_pos <= prompt_pos:
                    attn_mask[:, :, i, fact.start_pos:fact.end_pos] = 0.0
            # Allow attention to previous prompt tokens
            attn_mask[:, :, i, offset:offset + i + 1] = 0.0
        
        # ── Prefill with incremental KV loading ──────────────────────────
        # Build full-size KV cache with zeros, then overlay fact slices at absolute positions
        with torch.no_grad():
            for li in range(self.num_layers):
                # Get head dimension from first fact
                sample_k, _ = self.facts[0].get_kv(li, device, dtype)
                _, num_heads, _, head_dim = sample_k.shape
                
                # Create full-size cache filled with zeros
                full_k = torch.zeros(1, num_heads, self.max_pos, head_dim, device=device, dtype=dtype)
                full_v = torch.zeros(1, num_heads, self.max_pos, head_dim, device=device, dtype=dtype)
                
                # Overlay fact slices at their absolute positions
                for fact in self.facts:
                    k, v = fact.get_kv(li, device, dtype)
                    full_k[:, :, fact.start_pos:fact.end_pos, :] = k
                    full_v[:, :, fact.start_pos:fact.end_pos, :] = v
                
                cache.update(full_k, full_v, li)
                
                layer = self.model.model.layers[li]
                pe = rotary(h, pos_ids)
                h = layer(
                    h, attention_mask=attn_mask, position_ids=pos_ids,
                    position_embeddings=pe, use_cache=True,
                    past_key_values=cache,
                )
            
            h = final_norm(h)
            logits = lm_head(h[:, -1:, :])
        
        # Clean up after prefill
        del attn_mask
        gc.collect()
        torch.cuda.empty_cache()
        
        # ── Decode ───────────────────────────────────────────────────────
        gen_ids = list(prompt_ids)
        
        for step in range(max_new_tokens):
            with torch.no_grad():
                nxt = _sample(logits, sample_temp, top_k, top_p)
            tid = int(nxt.item()) if hasattr(nxt, "item") else int(nxt)
            if tid == eos:
                break
            gen_ids.append(tid)
            
            with torch.no_grad():
                tok_emb = embed(torch.tensor([[tid]], dtype=torch.long, device=device))
                cur_pos = offset + P + step
                pos_one = torch.tensor([[cur_pos]], device=device, dtype=torch.long)
                h = tok_emb
                
                for li in range(self.num_layers):
                    layer = self.model.model.layers[li]
                    pe = rotary(h, pos_one)
                    h = layer(
                        h, attention_mask=None, position_ids=pos_one,
                        position_embeddings=pe, use_cache=True,
                        past_key_values=cache,
                    )
                
                h = final_norm(h)
                logits = lm_head(h[:, -1:, :])
        
        elapsed = time.perf_counter() - t0
        ntok = len(gen_ids) - P
        text = tokenizer.decode(gen_ids[P:], skip_special_tokens=True)
        return {
            "generated_text": text,
            "num_tokens": ntok,
            "tokens_per_second": ntok / elapsed if elapsed > 0 else 0.0,
            "elapsed": elapsed,
            "path": "streaming",
            "window_tokens": self.total_context_tokens,
            "num_facts_loaded": len(self.facts),
            "max_position": self.max_pos,
        }


def _sample(logits, temperature, top_k, top_p):
    logits_f = logits.squeeze().float()
    if temperature <= 0:
        return torch.tensor(int(logits_f.argmax(dim=-1).item()))
    probs = torch.softmax(logits_f / temperature, dim=-1)
    if top_k > 0:
        vals, idx = torch.topk(probs, min(top_k, probs.size(-1)))
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask[idx] = True
        probs = probs * mask
    if top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cs = torch.cumsum(sp, dim=-1)
        cutoff = (cs > top_p).nonzero(as_tuple=True)
        if len(cutoff[0]) > 0:
            sp[cutoff[0][0] + 1:] = 0.0
            probs = torch.zeros_like(probs).scatter_(0, si, sp)
    probs = probs / probs.sum()
    return torch.multinomial(probs, 1).item()
