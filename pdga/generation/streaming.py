"""Streaming KV generation — progressive GPU loading with SDPA backend.

Instead of monkey-patching attention, we load window KV from CPU RAM into the
DynamicCache one layer at a time.  The SDPA backend handles attention efficiently.
An explicit causal 4D mask handles the position offset (window tokens precede
prompt tokens).

All forward passes are wrapped in ``torch.no_grad()`` to avoid keeping 4-bit
dequantized weight buffers in the autograd graph (saves ~450 MB on GPU).
"""

from __future__ import annotations

import gc
import time
import torch
from pathlib import Path
from typing import Optional
from transformers.cache_utils import DynamicCache
from pdga.kernel.prompt import build_prompt_ids


class StreamingGenerator:
    """Generate using progressively loaded KV cache."""

    def __init__(self, model, kv_cache_path: Path, window_size: int = 300):
        self.model = model
        self.num_layers = model.config.num_hidden_layers
        self.window_size = None

        # Load full KV cache to CPU RAM
        state = torch.load(str(kv_cache_path), map_location="cpu", weights_only=True)
        self._window_k = []
        self._window_v = []
        for li in range(self.num_layers):
            self._window_k.append(state.get(f"layer_{li}_keys"))
            self._window_v.append(state.get(f"layer_{li}_values"))
        del state

        if self._window_k[0] is not None:
            self.window_size = self._window_k[0].shape[2]
            # Validate: all layers must have the same seq length
            for li in range(self.num_layers):
                if self._window_k[li].shape[2] != self.window_size:
                    raise ValueError(
                        f"Layer {li} KV has {self._window_k[li].shape[2]} positions, "
                        f"expected {self.window_size}"
                    )
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
        offset = self.window_size

        prompt_ids = build_prompt_ids(tokenizer, prompt, enable_thinking=False)
        P = len(prompt_ids)

        t0 = time.perf_counter()

        # ── Prepare ──────────────────────────────────────────────────────
        cache = DynamicCache()

        # Prefill prompt tokens
        h = embed(torch.tensor([prompt_ids], dtype=torch.long, device=device))
        pos_ids = torch.arange(offset, offset + P, device=device, dtype=torch.long).unsqueeze(0)

        # Explicit causal mask
        attn_mask = torch.full(
            (1, 1, P, offset + P), -(2**15), device=device, dtype=dtype,
        )
        for i in range(P):
            attn_mask[:, :, i, :offset + i + 1] = 0.0

        # ── Prefill with incremental KV loading ──────────────────────────
        with torch.no_grad():
            for li in range(self.num_layers):
                # Load this layer's window KV from CPU to GPU (one layer at a time)
                if self._window_k[li] is not None:
                    k = self._window_k[li].to(device=device, dtype=dtype)
                    v = self._window_v[li].to(device=device, dtype=dtype)
                    cache.update(k, v, li)

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
            "window_tokens": self.window_size,
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
