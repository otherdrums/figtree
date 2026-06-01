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
            # ── Build combined KV cache from all facts ──
            cache = DynamicCache()
            current_pos = 0

            for fact in facts:
                fact_ids = self.tokenizer.encode(fact.text, add_special_tokens=False)
                if not fact_ids:
                    continue
                fact_len = len(fact_ids)
                fact_pos_ids = torch.arange(current_pos, current_pos + fact_len,
                                            device=device, dtype=torch.long).unsqueeze(0)

                h = embed(torch.tensor([fact_ids], dtype=torch.long, device=device))
                for li in range(self.num_layers):
                    layer = self.model.model.layers[li]
                    pe = rotary(h, fact_pos_ids)
                    h = layer(
                        h, attention_mask=None, position_ids=fact_pos_ids,
                        position_embeddings=pe, use_cache=True,
                        past_key_values=cache,
                    )

                current_pos += fact_len

            prompt_offset = current_pos
            total_len = prompt_offset + P

            # ── Prefill prompt with cached fact K/V ──
            prompt_emb = embed(torch.tensor([prompt_ids], dtype=torch.long, device=device))
            prompt_pos_ids = torch.arange(prompt_offset, prompt_offset + P,
                                          device=device, dtype=torch.long).unsqueeze(0)

            # Explicit causal mask: prompt sees all facts + previous prompts
            attn_mask = torch.full((1, 1, P, total_len), float('-inf'),
                                   device=device, dtype=torch.float32)
            for i in range(P):
                attn_mask[:, :, i, :prompt_offset + i + 1] = 0.0

            h = prompt_emb
            for li in range(self.num_layers):
                layer = self.model.model.layers[li]
                pe = rotary(h, prompt_pos_ids)
                h = layer(
                    h, attention_mask=attn_mask, position_ids=prompt_pos_ids,
                    position_embeddings=pe, use_cache=True,
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
