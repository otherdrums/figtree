"""PDGA generation engine — multi-delta generation with KV caching.

Architecture:
    Prefill: Each delta's [window_tokens, prompt_ids] → all layers → KV cache
    Decode:  Autoregressive generation using cached KVs

Supports mixed paths (compressed boundary-based and uncompressed text-based deltas)
and multi-delta parallel generation.
"""

from __future__ import annotations

import time
import torch
from transformers.cache_utils import DynamicCache

from pdga.kernel.prompt import build_prompt_ids


def generate(
    model,
    tokenizer,
    prompt: str,
    deltas: list[dict],
    injection_layer: int = 30,
    injection_topk: int = 8,
    injection_coefficient: float = 10.0,
    max_new_tokens: int = 300,
    sample_temp: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
) -> list[dict]:
    """Generate from multiple deltas in parallel with KV caching.

    Args:
        deltas: List of dicts, each with:
          - window_tokens: list[int] — token IDs of the article window
          - metadata: dict with 'delta_id', 'trust', 'source_url', 'tags', etc.
          - boundary: Optional[torch.Tensor] — for compressed path (experimental)
          - crystal_layer: Optional[int] — for compressed path

    Returns:
        List of dicts with: delta_id, trust, generated_text, source_url, tags,
        num_tokens, tokens_per_second, elapsed
    """
    device = model.device
    dtype = model.dtype
    embed = model.get_input_embeddings()
    lm_head = model.lm_head
    rotary = model.model.rotary_emb
    final_norm = model.model.norm
    num_layers = model.config.num_hidden_layers
    eos = tokenizer.eos_token_id

    N = len(deltas)
    if N == 0:
        return []

    # Build prompt with ChatML template
    prompt_ids = build_prompt_ids(tokenizer, prompt, enable_thinking=False)
    P = len(prompt_ids)

    # ── Prefill: each delta independently ──
    caches = [DynamicCache() for _ in range(N)]
    per_delta_logits = [None] * N
    per_delta_info = []

    for d in range(N):
        dd = deltas[d]
        w_tokens = dd.get("window_tokens", [])
        all_ids = w_tokens + prompt_ids
        L = len(all_ids)

        h = embed(torch.tensor([all_ids], dtype=torch.long, device=device))
        pos_ids = torch.arange(0, L, device=device, dtype=torch.long).unsqueeze(0)

        # Compute injection_delta from entries
        injection_delta = None
        entries_injected = 0
        inj_entries = dd.get("injection_entries", [])
        if inj_entries:
            entries_injected = min(len(inj_entries), injection_topk)
            selected = sorted(inj_entries, key=lambda x: -abs(x[1]))[:injection_topk]
            inj_ids = torch.tensor([e[0] for e in selected], device=device)
            inj_coeffs = torch.tensor([e[1] for e in selected], dtype=torch.float32, device=device).to(dtype)
            inj_embs = embed(inj_ids)
            injection_delta = (inj_embs * inj_coeffs.unsqueeze(-1) * injection_coefficient).sum(dim=0)

        # Compressed path: boundary at position 0 with zero RoPE
        use_compressed = dd.get("boundary") is not None
        crystal = dd.get("crystal_layer", 23)
        boundary = dd.get("boundary")

        if use_compressed and boundary is not None:
            # Build [BOS, prompt] — BOS will be replaced by boundary at crystal
            bos_id = tokenizer.bos_token_id or 1
            bos_emb = embed(torch.tensor([[bos_id]], dtype=torch.long, device=device))
            prompt_emb = embed(
                torch.tensor([prompt_ids], dtype=torch.long, device=device)
            )
            h = torch.cat([bos_emb, prompt_emb], dim=1)
            L = h.shape[1]
            pos_ids = torch.arange(0, L, device=device, dtype=torch.long).unsqueeze(0)

        cache = caches[d]
        for li in range(num_layers):
            layer = model.model.layers[li]
            pe = rotary(h, pos_ids)

            # Compressed: at crystal, swap BOS with boundary, zero pos 0 RoPE
            if use_compressed and boundary is not None and li == crystal:
                b_t = boundary.to(device=device, dtype=dtype)
                h[:, 0:1, :] = b_t.unsqueeze(0).unsqueeze(0)
                cos, sin = pe
                cos = cos.clone()
                sin = sin.clone()
                cos[:, 0:1, :] = 1.0
                sin[:, 0:1, :] = 0.0
                pe = (cos, sin)

            if use_compressed and boundary is not None and li > crystal:
                cos, sin = pe
                cos = cos.clone()
                sin = sin.clone()
                cos[:, 0:1, :] = 1.0
                sin[:, 0:1, :] = 0.0
                pe = (cos, sin)

            h = layer(
                h, attention_mask=None, position_ids=pos_ids,
                position_embeddings=pe, use_cache=True,
                past_key_values=cache,
            )

            if li == injection_layer and injection_delta is not None:
                h[:, -1:, :] = h[:, -1:, :] + injection_delta.unsqueeze(0).unsqueeze(0)

        h = final_norm(h)
        per_delta_logits[d] = lm_head(h[:, -1:, :])
        per_delta_info.append({
            "seq_len": L,
            "use_compressed": use_compressed,
            "crystal_layer": crystal,
            "boundary": boundary,
            "injection_delta": injection_delta,
            "entries_injected": entries_injected,
        })

    # ── Decode: autoregressive per-delta ──
    gen_ids = [list(prompt_ids) for _ in range(N)]
    active = [True] * N
    t0 = time.perf_counter()

    with torch.inference_mode():
        for step in range(max_new_tokens):
            if not any(active):
                break

            # Sample
            next_tokens = [None] * N
            for d in range(N):
                if not active[d]:
                    continue
                nxt = _sample(per_delta_logits[d], sample_temp, top_k, top_p)
                tid = int(nxt.item()) if hasattr(nxt, 'item') else int(nxt)
                if tid == eos:
                    active[d] = False
                else:
                    next_tokens[d] = tid
                    gen_ids[d].append(tid)

            if not any(active):
                break

            # Forward new tokens
            for d in range(N):
                if not active[d]:
                    continue
                tok = next_tokens[d]
                tok_emb = embed(torch.tensor([[tok]], dtype=torch.long, device=device))
                info = per_delta_info[d]
                cur_pos = info["seq_len"] + step
                pos_one = torch.tensor([[cur_pos]], device=device, dtype=torch.long)

                h = tok_emb
                cache = caches[d]
                for li in range(num_layers):
                    layer = model.model.layers[li]
                    pe = rotary(h, pos_one)
                    h = layer(
                        h, attention_mask=None, position_ids=pos_one,
                        position_embeddings=pe, use_cache=True,
                        past_key_values=cache,
                    )
                    if li == injection_layer and info["injection_delta"] is not None:
                        h[:, 0:1, :] = h[:, 0:1, :] + info["injection_delta"].unsqueeze(0).unsqueeze(0)

                h = final_norm(h)
                per_delta_logits[d] = lm_head(h[:, -1:, :])

    elapsed = time.perf_counter() - t0

    # ── Results ──
    results = []
    for d in range(N):
        meta = deltas[d].get("metadata", {})
        ntok = len(gen_ids[d]) - P
        tps = ntok / elapsed if elapsed > 0 else 0.0
        text = tokenizer.decode(gen_ids[d][P:], skip_special_tokens=True)
        results.append({
            "delta_id": meta.get("delta_id", f"delta_{d}"),
            "trust": meta.get("trust", 0.5),
            "generated_text": text,
            "source_url": meta.get("source_url", ""),
            "tags": meta.get("tags", []),
            "path": "compressed" if per_delta_info[d]["use_compressed"] else "uncompressed",
            "num_tokens": ntok,
            "tokens_per_second": tps,
            "elapsed": elapsed,
            "injection_layer": injection_layer,
            "entries_injected": per_delta_info[d]["entries_injected"],
        })
    return results


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
