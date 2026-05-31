"""Test compressed path with re-apply boundary at each layer crystal..N.

Tests:
1. Re-apply boundary at each layer (the new approach)
2. Boundary swap once (original approach, for comparison)
3. Uncompressed gold standard
4. Fact recall across all variants
"""

import torch
import time
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from pdga.kernel.prompt import build_prompt_ids
from pdga.apollo.compressed import compressed_generate, uncompressed_generate

MODEL_ID = "unsloth/Qwen3-4B-bnb-4bit"


def capture_boundary(model, tokenizer, text, crystal_layer):
    """Capture crystal-layer output at last token (boundary residual)."""
    ids = tokenizer.encode(text)
    device = model.device
    out = [None]

    def hook(mod, inp, o):
        x = o[0] if isinstance(o, tuple) else o
        out[0] = x[:, -1, :].detach().clone()

    target = model.model.layers[crystal_layer]
    h = target.register_forward_hook(hook)
    with torch.inference_mode():
        ids_t = torch.tensor([ids], dtype=torch.long, device=device)
        _ = model(input_ids=ids_t)
    h.remove()
    return out[0].squeeze(0), ids


def test_compressed_reapply():
    """Compare compressed (re-apply) vs uncompressed for factual recall."""
    print("=" * 80)
    print("COMPRESSED PATH — Re-apply Boundary at Each Layer")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Model: {model.config.num_hidden_layers}L, h={model.config.hidden_size}")

    article = (
        "GENEVA — Delegates from 47 nations reached a landmark trade agreement "
        "late Thursday, capping three weeks of intense negotiations at the Global "
        "Trade Summit. The pact covers digital services, carbon tariffs, and "
        "pharmaceutical patents. Secretary-General Maria Okonkwo hailed it as a "
        "turning point. Key provisions include a 3% digital services tax, carbon "
        "tariffs at $45 per ton, and pharmaceutical patent terms shortened from "
        "20 to 12 years. The agreement will be formally signed in Brussels next "
        "month. The IMF projects the deal could add $900 billion to global GDP."
    )

    prompt = "Summarize the key outcomes of the Global Trade Summit in Geneva:"
    prompt_ids = build_prompt_ids(tokenizer, prompt, enable_thinking=False)

    facts = [
        "47 nations", "Maria Okonkwo", "3%", "digital", "$45",
        "carbon", "pharmaceutical", "20 to 12", "Brussels", "$900",
    ]

    # ── Test at multiple crystal layers ──
    for crystal_layer in [23, 26, 29]:
        print(f"\n{'─'*80}")
        print(f"Crystal Layer L{crystal_layer}")
        print(f"{'─'*80}")

        boundary, window_ids = capture_boundary(model, tokenizer, article, crystal_layer)
        print(f"Boundary norm: {boundary.norm().item():.1f}  (window: {len(window_ids)} tokens)")

        injection_layer = min(crystal_layer + 7, 35)

        # Test 1: Compressed with re-apply
        print(f"\n  [Compressed, re-apply at each layer crystal..{model.config.num_hidden_layers-1}]")
        boundaries_t = boundary.unsqueeze(0)  # (1, hs)
        t0 = time.perf_counter()
        with torch.inference_mode():
            results_c = compressed_generate(
                model=model, tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                boundaries=boundaries_t,
                crystal_layer=crystal_layer,
                injection_layer=injection_layer,
                max_new_tokens=80, sample_temp=0.7,
            )
        elapsed_c = time.perf_counter() - t0
        text_c = results_c[0]["generated_text"]
        hits_c = sum(1 for f in facts if f.lower() in text_c.lower())
        print(f"  {hits_c}/{len(facts)} facts | {results_c[0]['num_tokens']} tok | {elapsed_c:.1f}s | {results_c[0]['tokens_per_second']:.1f} t/s")
        print(f"  {text_c[:200]}")

        gc.collect()
        torch.cuda.empty_cache()

        # Test 2: Uncompressed (gold standard)
        print("\n  [Uncompressed, full text context (gold)]")
        t0 = time.perf_counter()
        with torch.inference_mode():
            results_u = uncompressed_generate(
                model=model, tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                window_tokens_list=[window_ids[:200]],
                injection_layer=injection_layer,
                max_new_tokens=80, sample_temp=0.7,
            )
        elapsed_u = time.perf_counter() - t0
        text_u = results_u[0]["generated_text"]
        hits_u = sum(1 for f in facts if f.lower() in text_u.lower())
        print(f"  {hits_u}/{len(facts)} facts | {results_u[0]['num_tokens']} tok | {elapsed_u:.1f}s | {results_u[0]['tokens_per_second']:.1f} t/s")
        print(f"  {text_u[:200]}")

        gc.collect()
        torch.cuda.empty_cache()


def test_multi_window_boundaries():
    """Test with multiple window boundaries (one boundary per window)."""
    print("\n" + "=" * 80)
    print("COMPRESSED PATH — Multiple Window Boundaries")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    article = (
        "GENEVA — Delegates from 47 nations reached a landmark trade agreement "
        "late Thursday, capping three weeks of intense negotiations at the Global "
        "Trade Summit. The pact covers digital services, carbon tariffs, and "
        "pharmaceutical patents. Secretary-General Maria Okonkwo hailed it as a "
        "turning point. Key provisions include a 3% digital services tax, carbon "
        "tariffs at $45 per ton, and pharmaceutical patent terms shortened from "
        "20 to 12 years. The agreement will be formally signed in Brussels next "
        "month. Markets responded positively, with the S&P Global 1200 rising "
        "1.8% in after-hours trading. The IMF projects the deal could add $900 "
        "billion to global GDP over the next decade. Lead US negotiator "
        "Ambassador Sarah Chen called it a framework for shared prosperity."
    )
    prompt = "Summarize all specific facts from the trade summit article:"
    prompt_ids = build_prompt_ids(tokenizer, prompt, enable_thinking=False)

    all_ids = tokenizer.encode(article)
    window_size = 60
    windows = [all_ids[i:i+window_size] for i in range(0, len(all_ids), window_size)]

    print(f"Article: {len(all_ids)} tokens, split into {len(windows)} windows of {window_size} each")

    crystal_layer = 23
    injection_layer = 30

    # Capture boundary for each window
    boundaries = []
    for i, w in enumerate(windows):
        boundary_out = [None]
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            boundary_out[0] = o[:, -1, :].detach().clone()
        target = model.model.layers[crystal_layer]
        h = target.register_forward_hook(hook)
        with torch.inference_mode():
            ids_t = torch.tensor([w], dtype=torch.long, device=device)
            _ = model(input_ids=ids_t)
        h.remove()
        b = boundary_out[0].squeeze(0)
        boundaries.append(b)
        print(f"  Window {i}: {len(w)} tokens, boundary norm={b.norm().item():.1f}")

    # Test: one boundary vs all boundaries
    boundaries_t = torch.stack(boundaries, dim=0)

    # Try with just first boundary (representing first window)
    print("\n  [Compressed with 1 boundary (first window only)]")
    with torch.inference_mode():
        results = compressed_generate(
            model=model, tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            boundaries=boundaries_t[0:1],
            crystal_layer=crystal_layer,
            injection_layer=injection_layer,
            max_new_tokens=60, sample_temp=0.7,
        )
    print(f"  {results[0]['generated_text'][:250]}")

    gc.collect()
    torch.cuda.empty_cache()

    # Gold standard
    print("\n  [Uncompressed gold standard]")
    with torch.inference_mode():
        results = uncompressed_generate(
            model=model, tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            window_tokens_list=[all_ids[:200]],
            injection_layer=injection_layer,
            max_new_tokens=60, sample_temp=0.7,
        )
    print(f"  {results[0]['generated_text'][:250]}")


def test_speed_comparison():
    """Benchmark compressed vs uncompressed speed at various token counts."""
    print("\n" + "=" * 80)
    print("COMPRESSED vs UNCOMPRESSED — Speed Comparison")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    article = "Delegates from 47 nations reached a landmark trade agreement. " * 8
    prompt = "Summarize:"
    prompt_ids = build_prompt_ids(tokenizer, prompt, enable_thinking=False)
    window_ids = tokenizer.encode(article)

    crystal_layer = 23
    injection_layer = 30
    boundary, _ = capture_boundary(model, tokenizer, article[:500], crystal_layer)
    boundaries_t = boundary.unsqueeze(0)

    print(f"\n{'Context':<15} {'Path':<14} {'Tokens':>8} {'Time':>8} {'t/s':>8} {'Cache(KB)':>10}")
    print("-" * 65)

    for ctx_size in [50, 100, 200]:
        # Compressed
        t0 = time.perf_counter()
        with torch.inference_mode():
            results = compressed_generate(
                model=model, tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                boundaries=boundaries_t,
                crystal_layer=crystal_layer,
                injection_layer=injection_layer,
                max_new_tokens=50, sample_temp=0.7,
            )
        elapsed_c = time.perf_counter() - t0
        ntok_c = results[0]["num_tokens"]

        # Estimate KV cache size: 1 boundary token × num_layers × (K_per_layer + V_per_layer)
        # Qwen3-4B: 36 layers, 8 KV heads, 128 head_dim, bf16
        kv_bytes = 36 * 2 * 8 * 128 * 2  # 36 layers * K+V * 8 heads * 128 dim * 2 bytes
        comp_kv_kb = kv_bytes / 1024

        # Uncompressed
        ctx = window_ids[:ctx_size]
        t0 = time.perf_counter()
        with torch.inference_mode():
            results = uncompressed_generate(
                model=model, tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                window_tokens_list=[ctx],
                injection_layer=injection_layer,
                max_new_tokens=50, sample_temp=0.7,
            )
        elapsed_u = time.perf_counter() - t0
        ntok_u = results[0]["num_tokens"]
        ucomp_kv_bytes = ctx_size * 36 * 2 * 8 * 128 * 2
        ucomp_kv_kb = ucomp_kv_bytes / 1024

        print(f"{ctx_size:<15} {'compressed':<14} {ntok_c:>8} {elapsed_c:>8.1f}s {ntok_c/elapsed_c:>7.1f} {comp_kv_kb:>10.0f}")
        print(f"{ctx_size:<15} {'uncompressed':<14} {ntok_u:>8} {elapsed_u:>8.1f}s {ntok_u/elapsed_u:>7.1f} {ucomp_kv_kb:>10.0f}")
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import sys
    test = sys.argv[1] if len(sys.argv) > 1 else "all"

    if test in ("facts", "all"):
        test_compressed_reapply()
    if test in ("multi", "all"):
        test_multi_window_boundaries()
    if test in ("speed", "all"):
        test_speed_comparison()
