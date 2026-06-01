"""Generation engine v2: on-the-fly KV cache generation from fact text.

Pragmatic approach:
1. Facts store boundaries (for retrieval/dedup) + text (for KV generation)
2. During generation, selected facts have their text run through the model
3. Full KV caches are generated on-the-fly and used in standard attention
4. After generation, KV caches are freed (ephemeral)
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.cache_utils import DynamicCache

from pdga.fact.primitive import Fact


class FactGenerator:
    """Generate from facts by generating KV caches on-the-fly."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_layers = model.config.num_hidden_layers
        self.hidden_size = model.config.hidden_size
        self.device = model.device
        self.dtype = model.dtype
        self.eos = tokenizer.eos_token_id
        self.rotary = model.model.rotary_emb

    def generate(
        self,
        facts: list[Fact],
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> dict:
        device = self.device
        embed = self.model.get_input_embeddings()
        lm_head = self.model.lm_head
        final_norm = self.model.model.norm
        rotary = self.model.model.rotary_emb

        num_facts = len(facts)

        # Tokenize prompt
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        P = len(prompt_ids)

        t0 = time.perf_counter()

        with torch.no_grad():
            # ── Build combined KV cache from all facts (causal chain) ──
            cache = DynamicCache()

            all_fact_ids = []
            for fact in facts:
                fid = self.tokenizer.encode(fact.text, add_special_tokens=False)
                if fid:
                    all_fact_ids.extend(fid)

            total_fact_len = len(all_fact_ids)

            if total_fact_len > 0:
                fact_pos_ids = torch.arange(total_fact_len, device=device, dtype=torch.long).unsqueeze(0)

                fact_mask = torch.full(
                    (1, 1, total_fact_len, total_fact_len),
                    float('-inf'), device=device, dtype=torch.float32,
                )
                for i in range(total_fact_len):
                    fact_mask[:, :, i, :i + 1] = 0.0

                h = embed(torch.tensor([all_fact_ids], dtype=torch.long, device=device))
                pe_facts = rotary(h, fact_pos_ids)
                for li in range(self.num_layers):
                    layer = self.model.model.layers[li]
                    h = layer(
                        h, attention_mask=fact_mask, position_ids=fact_pos_ids,
                        position_embeddings=pe_facts, use_cache=True,
                        past_key_values=cache,
                    )

            prompt_offset = total_fact_len
            total_len = prompt_offset + P

            # ── Prefill prompt with cached fact K/V ──
            prompt_emb = embed(torch.tensor([prompt_ids], dtype=torch.long, device=device))
            prompt_pos_ids = torch.arange(prompt_offset, prompt_offset + P,
                                          device=device, dtype=torch.long).unsqueeze(0)

            attn_mask = torch.full((1, 1, P, total_len), float('-inf'),
                                   device=device, dtype=torch.float32)
            for i in range(P):
                attn_mask[:, :, i, :prompt_offset + i + 1] = 0.0

            h = prompt_emb
            pe_prompt = rotary(h, prompt_pos_ids)
            for li in range(self.num_layers):
                layer = self.model.model.layers[li]
                h = layer(
                    h, attention_mask=attn_mask, position_ids=prompt_pos_ids,
                    position_embeddings=pe_prompt, use_cache=True,
                    past_key_values=cache,
                )

            h = final_norm(h)
            logits = lm_head(h[:, -1:, :])

        # ── Decode ──
        gen_ids = list(prompt_ids)
        for step in range(max_new_tokens):
            with torch.no_grad():
                nxt = _sample(logits, temperature, top_k, top_p)
            tid = int(nxt.item()) if hasattr(nxt, 'item') else int(nxt)
            if tid == self.eos:
                break
            gen_ids.append(tid)

            with torch.no_grad():
                tok_emb = embed(torch.tensor([[tid]], dtype=torch.long, device=device))
                cur_pos = total_len + step
                pos_one = torch.tensor([[cur_pos]], device=device, dtype=torch.long)
                h = tok_emb

                pe_decode = rotary(h, pos_one)
                for li in range(self.num_layers):
                    layer = self.model.model.layers[li]
                    h = layer(
                        h, attention_mask=None, position_ids=pos_one,
                        position_embeddings=pe_decode, use_cache=True,
                        past_key_values=cache,
                    )

                h = final_norm(h)
                logits = lm_head(h[:, -1:, :])

        elapsed = time.perf_counter() - t0
        ntok = len(gen_ids) - P
        text = self.tokenizer.decode(gen_ids[P:], skip_special_tokens=True)

        del cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "generated_text": text,
            "num_tokens": ntok,
            "tokens_per_second": ntok / elapsed if elapsed > 0 else 0.0,
            "elapsed": elapsed,
            "num_facts": num_facts,
        }


    def generate_from_boundaries(
        self,
        facts: list[Fact],
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        cache_dir: str = "./facts",
    ) -> dict:
        """Generate using per-token cached K/V from disk.

        Each fact stores pre-computed unrotated K/V for every token at every
        layer (computed during ingestion). This loads the cached K/V, applies
        RoPE based on global positions, and populates the cache directly —
        no forward pass through the model for fact processing.

        Falls back to a clear error if facts lack cached K/V (re-ingest or
        use generate() instead).
        """
        device = self.device
        embed = self.model.get_input_embeddings()
        lm_head = self.model.lm_head
        final_norm = self.model.model.norm
        rotary = self.model.model.rotary_emb
        config = self.model.config
        cache_root = Path(cache_dir)

        num_kv_heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)

        # Load cached K/V for each fact
        all_k: list[torch.Tensor] = []
        all_v: list[torch.Tensor] = []
        for f in facts:
            fdir = cache_root / f"{f.fact_id}.pdga"
            kv_path = fdir / "kv_cache.npy"
            if not kv_path.exists():
                # Search recursively for the fact directory
                matches = list(cache_root.rglob(f"{f.fact_id}.pdga/kv_cache.npy"))
                if matches:
                    kv_path = matches[0]
                else:
                    raise FileNotFoundError(
                        f"No kv_cache.npy for fact {f.fact_id[:12]}... "
                        f"Re-ingest the text or use generate() instead."
                    )
            kv = np.load(str(kv_path))  # (num_layers, seq_len, 2, kv_dim)
            kv_t = torch.from_numpy(kv).to(device=device, dtype=self.dtype)
            all_k.append(kv_t[:, :, 0, :])  # (num_layers, seq_len, kv_dim)
            all_v.append(kv_t[:, :, 1, :])  # (num_layers, seq_len, kv_dim)

        total_tokens = sum(k.shape[1] for k in all_k)

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        P = len(prompt_ids)

        t0 = time.perf_counter()

        with torch.no_grad():
            # ── RoPE for all fact token positions ──
            pos_ids = torch.arange(total_tokens, device=device).unsqueeze(0)
            dummy = torch.zeros(1, total_tokens, self.hidden_size, device=device, dtype=self.dtype)
            cos, sin = rotary(dummy, pos_ids)

            # ── Populate KV cache ──
            cache = DynamicCache()
            for li in range(self.num_layers):
                k = torch.cat([kl[li] for kl in all_k], dim=0)  # (total_tokens, kv_dim)
                v = torch.cat([vl[li] for vl in all_v], dim=0)  # (total_tokens, kv_dim)

                k = k.unsqueeze(0)  # (1, total_tokens, kv_dim)
                v = v.unsqueeze(0)

                k = k.view(1, total_tokens, num_kv_heads, head_dim).transpose(1, 2)
                v = v.view(1, total_tokens, num_kv_heads, head_dim).transpose(1, 2)

                k, _ = _apply_rotary_pos_emb(k, k, cos, sin)
                cache.update(k, v, li)

            prompt_offset = total_tokens
            total_len = prompt_offset + P

            # ── Prefill prompt ──
            prompt_emb = embed(torch.tensor([prompt_ids], dtype=torch.long, device=device))
            prompt_pos_ids = torch.arange(prompt_offset, prompt_offset + P,
                                          device=device, dtype=torch.long).unsqueeze(0)

            attn_mask = torch.full((1, 1, P, total_len), float('-inf'),
                                   device=device, dtype=torch.float32)
            for i in range(P):
                attn_mask[:, :, i, :prompt_offset + i + 1] = 0.0

            h = prompt_emb
            pe_prompt = rotary(h, prompt_pos_ids)
            for li in range(self.num_layers):
                layer = self.model.model.layers[li]
                h = layer(
                    h, attention_mask=attn_mask, position_ids=prompt_pos_ids,
                    position_embeddings=pe_prompt, use_cache=True,
                    past_key_values=cache,
                )

            h = final_norm(h)
            logits = lm_head(h[:, -1:, :])

        # ── Decode ──
        gen_ids = list(prompt_ids)
        for step in range(max_new_tokens):
            with torch.no_grad():
                nxt = _sample(logits, temperature, top_k, top_p)
            tid = int(nxt.item()) if hasattr(nxt, 'item') else int(nxt)
            if tid == self.eos:
                break
            gen_ids.append(tid)

            with torch.no_grad():
                tok_emb = embed(torch.tensor([[tid]], dtype=torch.long, device=device))
                cur_pos = total_len + step
                pos_one = torch.tensor([[cur_pos]], device=device, dtype=torch.long)
                h = tok_emb

                pe_decode = rotary(h, pos_one)
                for li in range(self.num_layers):
                    layer = self.model.model.layers[li]
                    h = layer(
                        h, attention_mask=None, position_ids=pos_one,
                        position_embeddings=pe_decode, use_cache=True,
                        past_key_values=cache,
                    )

                h = final_norm(h)
                logits = lm_head(h[:, -1:, :])

        elapsed = time.perf_counter() - t0
        ntok = len(gen_ids) - P
        text = self.tokenizer.decode(gen_ids[P:], skip_special_tokens=True)

        del cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "generated_text": text,
            "num_tokens": ntok,
            "tokens_per_second": ntok / elapsed if elapsed > 0 else 0.0,
            "elapsed": elapsed,
            "num_facts": len(facts),
            "num_tokens_total": total_tokens,
        }

    # ────────────────────────────────────────────────


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


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
