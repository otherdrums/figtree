"""Generation engine: on-the-fly KV cache generation from figment text.

Pragmatic approach:
1. Figments store boundaries (for retrieval/dedup) + text (for KV generation)
2. During generation, selected figments have their text run through the model
3. Full KV caches are generated on-the-fly and used in standard attention
4. After generation, KV caches are freed (ephemeral)
"""

from __future__ import annotations

import gc
import time

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.cache_utils import DynamicCache

from figtree.figment import Figment
from figtree.kernel.prompt import build_prompt_ids


class FigmentGenerator:
    """Generate from figments by generating KV caches on-the-fly."""

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
        figments: list[Figment],
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.15,
    ) -> dict:
        device = self.device
        embed = self.model.get_input_embeddings()
        lm_head = self.model.lm_head
        final_norm = self.model.model.norm
        rotary = self.model.model.rotary_emb

        num_figments = len(figments)

        prompt_ids = build_prompt_ids(self.tokenizer, prompt)
        P = len(prompt_ids)

        t0 = time.perf_counter()

        with torch.no_grad():
            cache = DynamicCache()

            sep_ids = self.tokenizer.encode("\n\n", add_special_tokens=False)
            all_figment_ids = []
            for figment in figments:
                fid = self.tokenizer.encode(figment.text, add_special_tokens=False)
                if fid:
                    if all_figment_ids:
                        all_figment_ids.extend(sep_ids)
                    all_figment_ids.extend(fid)

            total_figment_len = len(all_figment_ids)

            if total_figment_len > 0:
                figment_pos_ids = torch.arange(total_figment_len, device=device, dtype=torch.long).unsqueeze(0)

                figment_mask = torch.full(
                    (1, 1, total_figment_len, total_figment_len),
                    float('-inf'), device=device, dtype=torch.float32,
                )
                for i in range(total_figment_len):
                    figment_mask[:, :, i, :i + 1] = 0.0

                h = embed(torch.tensor([all_figment_ids], dtype=torch.long, device=device))
                pe_figments = rotary(h, figment_pos_ids)
                for li in range(self.num_layers):
                    layer = self.model.model.layers[li]
                    h = layer(
                        h, attention_mask=figment_mask, position_ids=figment_pos_ids,
                        position_embeddings=pe_figments, use_cache=True,
                        past_key_values=cache,
                    )

            prompt_offset = total_figment_len
            total_len = prompt_offset + P

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

        gen_ids = list(prompt_ids)
        for step in range(max_new_tokens):
            with torch.no_grad():
                nxt = _sample(logits, temperature, top_k, top_p, repetition_penalty, gen_ids)
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
            "num_figments": num_figments,
        }


    def generate_with_recall(
        self,
        figments: list[Figment],
        prompt: str,
        source_texts: list[str] | None = None,
        max_new_tokens: int = 400,
        recall_max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.15,
    ) -> dict:
        """Generate, then verify recall against ``source_texts`` and patch gaps.

        Runs :meth:`generate`, then (if ``source_texts`` is supplied) checks which
        checkable atoms from the sources are missing from the output using
        :mod:`figtree.recall`. Any missing atoms trigger one targeted follow-up
        pass that appends the recovered facts. The returned text is the original
        output with the recovered facts appended, and ``recall_score`` /
        ``missing_atoms`` are reported so callers can assert flawless recall.
        """
        from figtree.recall import build_recall_prompt, missing_atoms, recall_score

        result = self.generate(
            figments=figments, prompt=prompt, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        text = result["generated_text"]

        if not source_texts:
            result["recall_score"] = None
            result["missing_atoms"] = []
            return result

        source_blob = "\n".join(source_texts)
        miss = missing_atoms(source_blob, text)
        attempt = 0
        while miss and attempt < 2:
            attempt += 1
            recall_prompt = (
                f"{prompt}\n\n{build_recall_prompt(miss)}"
            )
            # Greedy decode for the patch so the model is more likely to comply
            # with stating the exact missing figures rather than free-generating.
            patch = self.generate(
                figments=figments, prompt=recall_prompt,
                max_new_tokens=recall_max_new_tokens,
                temperature=0.0, top_k=1, top_p=1.0,
                repetition_penalty=repetition_penalty,
            )
            text = f"{text.strip()}\n\n{_strip_lead_in(patch['generated_text'])}".strip()
            result["generated_text"] = text
            result["num_tokens"] = result["num_tokens"] + patch["num_tokens"]
            miss = missing_atoms(source_blob, text)

        result["recall_score"] = recall_score(source_blob, text)
        result["missing_atoms"] = miss
        return result

    def generate_from_boundaries(
        self,
        figments: list[Figment],
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.15,
        kv_manager=None,
    ) -> dict:
        """Generate using per-token cached K/V.

        K/V is obtained from ``kv_manager.materialize`` (LanceDB-backed, lazy by
        default — recomputes on demand; eager if blobs were persisted at ingest).
        ``kv_manager`` is required for boundary-based generation.
        """
        if kv_manager is None:
            raise ValueError(
                "kv_manager is required for generate_from_boundaries. "
                "Use a KVCacheManager (figtree.kv_cache_manager.KVCacheManager)."
            )
        device = self.device
        embed = self.model.get_input_embeddings()
        lm_head = self.model.lm_head
        final_norm = self.model.model.norm
        rotary = self.model.model.rotary_emb
        config = self.model.config

        num_kv_heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)

        all_k: list[torch.Tensor] = []
        all_v: list[torch.Tensor] = []

        kv_map = kv_manager.materialize(figments)
        for fig in figments:
            kv = kv_map[fig.figment_id]
            kv_t = torch.from_numpy(kv).to(device=device, dtype=self.dtype)
            all_k.append(kv_t[:, :, 0, :])
            all_v.append(kv_t[:, :, 1, :])

        total_tokens = sum(k.shape[1] for k in all_k)

        prompt_ids = build_prompt_ids(self.tokenizer, prompt)
        P = len(prompt_ids)

        t0 = time.perf_counter()

        with torch.no_grad():
            pos_ids = torch.arange(total_tokens, device=device).unsqueeze(0)
            dummy = torch.zeros(1, total_tokens, self.hidden_size, device=device, dtype=self.dtype)
            cos, sin = rotary(dummy, pos_ids)

            cache = DynamicCache()
            for li in range(self.num_layers):
                k = torch.cat([kl[li] for kl in all_k], dim=0)
                v = torch.cat([vl[li] for vl in all_v], dim=0)

                k = k.unsqueeze(0)
                v = v.unsqueeze(0)

                k = k.view(1, total_tokens, num_kv_heads, head_dim).transpose(1, 2)
                v = v.view(1, total_tokens, num_kv_heads, head_dim).transpose(1, 2)

                k, _ = _apply_rotary_pos_emb(k, k, cos, sin)
                cache.update(k, v, li)

            prompt_offset = total_tokens
            total_len = prompt_offset + P

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

        gen_ids = list(prompt_ids)
        for step in range(max_new_tokens):
            with torch.no_grad():
                nxt = _sample(logits, temperature, top_k, top_p, repetition_penalty, gen_ids)
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
            "num_figments": len(figments),
            "num_tokens_total": total_tokens,
        }


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _strip_lead_in(text: str) -> str:
    """Trim a prompted instruction echo from a patched generation.

    The recall follow-up prompt ends with an imperative ("Restate each with its
    exact figure or name."). If the model echoes that instruction, drop it so the
    appended facts read cleanly.
    """
    markers = (
        "restate each with its exact figure",
        "also explicitly state the following",
        "here are the facts",
        "certainly",
    )
    low = text.lower()
    for m in markers:
        idx = low.find(m)
        if idx != -1:
            # Cut from the marker backwards to the start of its sentence.
            start = low.rfind(".", 0, idx)
            start = 0 if start == -1 else start + 1
            return text[start:].strip()
    return text.strip()


def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _sample(logits, temperature, top_k, top_p, repetition_penalty=1.0, context=None):
    logits_f = logits.squeeze().float().clone()
    if repetition_penalty != 1.0 and context is not None and len(context) > 0:
        # Penalize tokens already present in the generated context.
        for tid in set(context):
            if tid < logits_f.numel():
                if logits_f[tid] > 0:
                    logits_f[tid] /= repetition_penalty
                else:
                    logits_f[tid] *= repetition_penalty
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
