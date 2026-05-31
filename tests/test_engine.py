"""Test and benchmark the generation engine — compressed vs uncompressed paths.

Validates:
1. Factual recall from boundary residual (compressed path)
2. Factual recall from full text context (uncompressed gold standard)
3. Speed comparison
4. Multi-delta parallelism
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pdga.kernel.prompt import build_prompt_ids
from pdga.generation.engine import generate


MODEL_ID = "unsloth/Qwen3-4B-bnb-4bit"
CRYSTAL = 23
INJECTION = 30


def _load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _capture_boundary(model, tokenizer, text: str, crystal_layer: int) -> torch.Tensor:
    """Forward text through model, capture crystal-layer output at last token."""
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
    return out[0].squeeze(0)


def test_single_delta():
    """Compressed vs uncompressed on one delta with known facts."""
    print("=" * 80)
    print("generation engine — Single Delta Test")
    print("=" * 80)

    model, tokenizer = _load_model()
    print(f"Model: {model.config.num_hidden_layers}L, h={model.config.hidden_size}")

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
    prompt = (
        "Summarize the key outcomes of the Global Trade Summit in Geneva. "
        "List every specific number, name, provision, and fact mentioned in the source text."
    )
    prompt_ids = build_prompt_ids(tokenizer, prompt, enable_thinking=False)
    window_ids = tokenizer.encode(article)

    boundary = _capture_boundary(model, tokenizer, article, CRYSTAL)
    print(f"\nArticle: {len(window_ids)} tokens")
    print(f"Boundary norm: {boundary.norm().item():.2f}")

    # ── Compressed ─────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("COMPRESSED (boundary residual, no window tokens)")
    print("─" * 80)

    t0 = time.perf_counter()
    with torch.inference_mode():
        results = generate(
            model=model, tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            deltas=[{
                "path": "compressed",
                "boundary": boundary,
                "injection_delta": None,
            }],
            crystal_layer=CRYSTAL, injection_layer=INJECTION,
            max_new_tokens=100, sample_temp=0.7,
        )
    elapsed_c = time.perf_counter() - t0
    r = results[0]
    text_c = r["generated_text"]
    print(f"({r['num_tokens']} tok, {elapsed_c:.1f}s, {r['tokens_per_second']:.1f} t/s)")
    print(text_c[:600])

    # ── Uncompressed ───────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("UNCOMPRESSED (full text, gold standard)")
    print("─" * 80)

    t0 = time.perf_counter()
    with torch.inference_mode():
        results = generate(
            model=model, tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            deltas=[{
                "path": "uncompressed",
                "window_tokens": window_ids,
                "injection_delta": None,
            }],
            crystal_layer=CRYSTAL, injection_layer=INJECTION,
            max_new_tokens=100, sample_temp=0.7,
        )
    elapsed_u = time.perf_counter() - t0
    r = results[0]
    text_u = r["generated_text"]
    print(f"({r['num_tokens']} tok, {elapsed_u:.1f}s, {r['tokens_per_second']:.1f} t/s)")
    print(text_u[:600])

    # ── Fact Check ─────────────────────────────────────────────────────────
    facts = [
        "47 nations", "Maria Okonkwo", "3%", "digital services",
        "$45", "carbon", "pharmaceutical", "20 to 12",
        "Brussels", "S&P", "1.8%", "$900 billion", "Sarah Chen",
    ]
    print("\n" + "─" * 80)
    print("FACT CHECK")
    print("─" * 80)
    print(f"{'Fact':<25} {'Compressed':>12} {'Gold':>10}")
    print("-" * 50)
    c_hits = 0
    u_hits = 0
    for f in facts:
        in_c = "YES" if f.lower() in text_c.lower() else "no"
        in_u = "YES" if f.lower() in text_u.lower() else "no"
        if in_c == "YES":
            c_hits += 1
        if in_u == "YES":
            u_hits += 1
        print(f"{f:<25} {in_c:>12} {in_u:>10}")
    print("-" * 50)
    print(f"{'TOTAL':<25} {f'{c_hits}/{len(facts)}':>12} {f'{u_hits}/{len(facts)}':>10}")
    print(f"\nSpeed: compressed={results[0]['tokens_per_second']:.1f} t/s, "
          f"gold={r.get('tokens_per_second', 0):.1f} t/s")


def test_multi_delta():
    """Multi-delta parallel generation — 2 articles with contradictory facts."""
    print("\n" + "=" * 80)
    print("generation engine — Multi-Delta Test")
    print("=" * 80)

    model, tokenizer = _load_model()

    article_a = (
        "GENEVA — Delegates from 47 nations reached a landmark trade agreement. "
        "Secretary-General Maria Okonkwo hailed it as a turning point. "
        "Key provisions: 3% digital services tax, carbon tariffs at $45/ton, "
        "patent terms shortened from 20 to 12 years. US Ambassador Sarah Chen "
        "called it a framework for shared prosperity. IMF projects $900B GDP boost."
    )
    article_b = (
        "GENEVA — Only 34 nations signed the watered-down communiqué. The US "
        "and China walked out, calling the 3% digital services tax 'punitive.' "
        "Maria Okonkwo acknowledged the deal 'falls short.' Carbon tariffs set "
        "at just $15/ton after lobbying. IMF warned deal might add only $250B."
    )
    prompt = (
        "Produce a comprehensive report: What happened at the Global Trade "
        "Summit in Geneva? Include every specific detail from the source — "
        "all agreements, provisions, numbers, names, dates, reactions, and outcomes."
    )
    prompt_ids = build_prompt_ids(tokenizer, prompt, enable_thinking=False)

    b_a = _capture_boundary(model, article_a, CRYSTAL)
    b_b = _capture_boundary(model, article_b, CRYSTAL)

    print("\nGenerating from 2 deltas in parallel (compressed path)...")
    t0 = time.perf_counter()
    with torch.inference_mode():
        results = generate(
            model=model, tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            deltas=[
                {"path": "compressed", "boundary": b_a, "injection_delta": None},
                {"path": "compressed", "boundary": b_b, "injection_delta": None},
            ],
            crystal_layer=CRYSTAL, injection_layer=INJECTION,
            max_new_tokens=100, sample_temp=0.7,
        )
    elapsed = time.perf_counter() - t0

    for i, r in enumerate(results):
        label = "Article A (pro-deal)" if i == 0 else "Article B (skeptical)"
        print(f"\n── {label} ──")
        print(f"({r['num_tokens']} tokens)")
        print(r["generated_text"][:500])

    total_tokens = sum(r["num_tokens"] for r in results)
    print(f"\nTotal: {total_tokens} tokens in {elapsed:.1f}s "
          f"({total_tokens/elapsed:.1f} t/s)")


def test_multi_with_uncompressed():
    """Multi-delta: one compressed, one uncompressed (mixed paths)."""
    print("\n" + "=" * 80)
    print("generation engine — Mixed Path Test (compressed + uncompressed)")
    print("=" * 80)

    model, tokenizer = _load_model()

    article = (
        "GENEVA — Delegates from 47 nations reached a landmark trade agreement. "
        "Key provisions: 3% digital services tax, $45/ton carbon tariffs, "
        "patent terms from 20 to 12 years. Maria Okonkwo hailed it as a turning "
        "point. $900B GDP boost projected by IMF."
    )
    window_ids = tokenizer.encode(article)
    boundary = _capture_boundary(model, tokenizer, article, CRYSTAL)

    prompt = "Summarize the key outcomes of the trade summit:"
    prompt_ids = build_prompt_ids(tokenizer, prompt, enable_thinking=False)

    window_ids = tokenizer.encode(article)
    boundary = _capture_boundary(model, tokenizer, article, CRYSTAL)

    results = generate(
        model=model, tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        deltas=[
            {"path": "compressed", "boundary": boundary, "injection_delta": None},
            {"path": "uncompressed", "window_tokens": window_ids, "injection_delta": None},
        ],
        crystal_layer=CRYSTAL, injection_layer=INJECTION,
        max_new_tokens=80, sample_temp=0.7,
    )

    for i, r in enumerate(results):
        label = "Compressed" if i == 0 else "Uncompressed (gold)"
        print(f"\n── {label} ({r['num_tokens']} tok, {r['tokens_per_second']:.1f} t/s) ──")
        print(r["generated_text"][:400])


def benchmark_speed():
    """Benchmark multi-delta speed at various batch sizes."""
    print("\n" + "=" * 80)
    print("generation engine — Speed Benchmark")
    print("=" * 80)

    model, tokenizer = _load_model()

    article = (
        "Delegates from 47 nations reached a landmark trade agreement. "
        "Key provisions: 3% digital services tax, $45/ton carbon tariffs, "
        "patent terms from 20 to 12 years. Maria Okonkwo hailed it."
    )
    prompt = "Summarize the key outcomes of the trade summit:"
    prompt_ids = build_prompt_ids(tokenizer, prompt, enable_thinking=False)
    window_ids = tokenizer.encode(article)
    boundary = _capture_boundary(model, tokenizer, article, CRYSTAL)

    print(f"\n{'N':>3} {'Path':<14} {'Tokens':>8} {'Time(s)':>8} {'t/s':>8} {'per-delta t/s':>14}")
    print("-" * 62)

    for n in [1, 2, 3, 4]:
        for path in ["compressed", "uncompressed"]:
            deltas = []
            for _ in range(n):
                if path == "compressed":
                    deltas.append({"path": "compressed", "boundary": boundary.clone()})
                else:
                    deltas.append({"path": "uncompressed", "window_tokens": list(window_ids)})

            t0 = time.perf_counter()
            with torch.inference_mode():
                results = generate(
                    model=model, tokenizer=tokenizer,
                    prompt_ids=prompt_ids, deltas=deltas,
                    crystal_layer=CRYSTAL, injection_layer=INJECTION,
                    max_new_tokens=50, sample_temp=0.7,
                )
            elapsed = time.perf_counter() - t0
            total_tok = sum(r["num_tokens"] for r in results)
            tps = total_tok / elapsed if elapsed > 0 else 0
            per_delta = tps / n if n > 0 else 0
            print(f"{n:>3} {path:<14} {total_tok:>8} {elapsed:>8.2f} {tps:>8.1f} {per_delta:>14.1f}")

    print()
    print("Note: Compressed path uses 1 boundary token instead of ~100 window tokens.")
    print("The speedup comes from reduced prefill KV cache (1 vs 100 positions cached).")


if __name__ == "__main__":
    import sys
    test = sys.argv[1] if len(sys.argv) > 1 else "all"

    if test == "single" or test == "all":
        test_single_delta()
    if test == "multi" or test == "all":
        test_multi_delta()
    if test == "mixed" or test == "all":
        test_multi_with_uncompressed()
    if test == "bench" or test == "all":
        benchmark_speed()
