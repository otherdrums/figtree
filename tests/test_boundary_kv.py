"""Test boundary-kv generation — ingest with KV cache, generate with instant prefill.

Validates:
1. KV cache capture during ingestion
2. Boundary-kv generation (load cached KV, skip prefill)
3. Factual recall comparison vs uncompressed gold standard
4. Speed benchmark: prefill elimination
"""

import shutil
import time
import torch
import gc
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from pdga.ingest.text import ingest_text
from pdga.apollo.boundary_kv import generate_boundary_kv
from pdga.apollo.compressed import uncompressed_generate
from pdga.kernel.prompt import build_prompt_ids
from pdga.delta.cache_io import list_window_caches, get_cache_size_per_window

MODEL_ID = "unsloth/Qwen3-4B-bnb-4bit"
TEST_DIR = Path("/tmp/pdga_boundary_kv_test")


def test_ingest_with_kv_cache():
    """Test that ingestion captures and stores KV caches."""
    print("=" * 80)
    print("BOUNDARY-KV — Ingest with KV Cache")
    print("=" * 80)

    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

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
        "20 to 12 years. The IMF projects $900 billion GDP boost."
    )

    print(f"\nArticle: {len(tokenizer.encode(article))} tokens")
    print(f"Model: {model.config.num_hidden_layers}L, h={model.config.hidden_size}")

    # Ingest with window_size=50 to get multiple windows
    delta = ingest_text(
        model=model, tokenizer=tokenizer, text=article,
        output_dir=TEST_DIR, window_size=80, trust=0.99,
        tags=["test"],
    )

    # Check KV cache files
    windows = list_window_caches(TEST_DIR / f"{delta.delta_id}.pdga")
    print(f"\nDelta: {delta.delta_id}")
    print(f"Windows: {delta.num_windows}")
    print(f"KV cache windows stored: {windows}")

    # Check file sizes
    for w in windows:
        cache_path = TEST_DIR / f"{delta.delta_id}.pdga" / f"kv_cache_w{w}.pt"
        size_mb = cache_path.stat().st_size / 1024 / 1024
        print(f"  Window {w}: {size_mb:.1f} MB")

    # Estimate
    est = get_cache_size_per_window(80, model.config.num_hidden_layers)
    print(f"  Expected per window: {est / 1024 / 1024:.1f} MB (80 tokens × 36L × K+V × 8 heads × 128 dim × 2B)")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return delta, windows


def test_boundary_kv_generation(delta):
    """Test generation using pre-computed KV cache."""
    print("\n" + "=" * 80)
    print("BOUNDARY-KV — Generation (instant prefill)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    delta_path = TEST_DIR / f"{delta.delta_id}.pdga"
    windows = list_window_caches(delta_path)
    # Only use first window for RoPE safety (LARQL uses LSH to pick best window)
    use_windows = windows[:1]

    prompt = "Summarize the key facts and provisions from the article:"
    prompt_ids = build_prompt_ids(tokenizer, prompt, enable_thinking=False)

    facts = ["47", "Maria", "3%", "$45", "20 to 12", "digital", "carbon", "pharmaceutical", "$900"]

    # ── Boundary-KV generation ────────────────────────────────────────────
    print(f"\n[Boundary-KV, {len(use_windows)} window loaded from disk]")
    t0 = time.perf_counter()
    with torch.inference_mode():
        results = generate_boundary_kv(
            model=model, tokenizer=tokenizer,
            prompt=prompt,
            delta_paths=[delta_path],
            window_indices=[use_windows],
            injection_layer=29,
            max_new_tokens=100, sample_temp=0.7,
        )
    elapsed_kv = time.perf_counter() - t0
    r_kv = results[0]
    hits_kv = sum(1 for f in facts if f.lower() in r_kv["generated_text"].lower())

    print(f"  {hits_kv}/{len(facts)} facts | {r_kv['num_tokens']} tok | {elapsed_kv:.1f}s | {r_kv['tokens_per_second']:.1f} t/s")
    print(f"  Generated: {r_kv['generated_text'][:300]}")

    gc.collect()
    torch.cuda.empty_cache()

    # ── Uncompressed gold standard ────────────────────────────────────────
    window_ids = tokenizer.encode(
        "GENEVA — Delegates from 47 nations reached a landmark trade agreement "
        "late Thursday. Secretary-General Maria Okonkwo hailed it as a turning "
        "point. Key provisions: 3% digital services tax, $45/ton carbon tariffs, "
        "pharmaceutical patent terms from 20 to 12 years. $900B GDP boost projected."
    )

    print(f"\n[Uncompressed gold standard, full text context]")
    t0 = time.perf_counter()
    with torch.inference_mode():
        results = uncompressed_generate(
            model=model, tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            window_tokens_list=[window_ids],
            injection_layer=29,
            max_new_tokens=100, sample_temp=0.7,
        )
    elapsed_u = time.perf_counter() - t0
    r = results[0]
    hits_u = sum(1 for f in facts if f.lower() in r["generated_text"].lower())

    print(f"  {hits_u}/{len(facts)} facts | {r['num_tokens']} tok | {elapsed_u:.1f}s | {r['tokens_per_second']:.1f} t/s")
    print(f"  Generated: {r['generated_text'][:300]}")

    # ── Speed comparison ──────────────────────────────────────────────────
    print(f"\n[Speed Comparison]")
    print(f"  Boundary-KV:   {elapsed_kv:.1f}s total  ({r_kv['tokens_per_second']:.1f} t/s)")
    print(f"  Uncompressed:  {elapsed_u:.1f}s total  ({r.get('tokens_per_second', 0):.1f} t/s)")
    if elapsed_u > 0:
        print(f"  Speedup:       {elapsed_u / elapsed_kv:.2f}x")

    # ── Fact comparison ───────────────────────────────────────────────────
    print(f"\n[Fact Recall]")
    print(f"  Boundary-KV:  {hits_kv}/{len(facts)}")
    print(f"  Uncompressed: {hits_u}/{len(facts)}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return hits_kv, hits_u


def test_multi_delta_boundary_kv():
    """Test with two articles ingested and generated in parallel."""
    print("\n" + "=" * 80)
    print("BOUNDARY-KV — Multi-Delta Test")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    article_a = (
        "Delegates from 47 nations reached a landmark trade agreement. "
        "Maria Okonkwo hailed it as a turning point. Key provisions: "
        "3% digital services tax and $45/ton carbon tariffs."
    )
    article_b = (
        "Only 34 nations signed the watered-down communiqué. The US "
        "and China walked out. Carbon tariffs reduced to $15/ton."
    )

    dir_a = TEST_DIR / "delta_a"
    dir_b = TEST_DIR / "delta_b"
    for d in [dir_a, dir_b]:
        if d.exists():
            shutil.rmtree(d)

    delta_a = ingest_text(model, tokenizer, article_a, dir_a, window_size=50, trust=0.99)
    delta_b = ingest_text(model, tokenizer, article_b, dir_b, window_size=50, trust=0.50)

    prompt = "Summarize what happened at the summit:"
    wa = list_window_caches(dir_a / f"{delta_a.delta_id}.pdga")
    wb = list_window_caches(dir_b / f"{delta_b.delta_id}.pdga")
    # Only use first window per delta (RoPE-safe for boundary-kv)
    wa_use = wa[:1] if wa else []
    wb_use = wb[:1] if wb else []

    print(f"\nGenerating from 2 deltas ({len(wa_use)}+{len(wb_use)} cached windows)...")
    t0 = time.perf_counter()
    with torch.inference_mode():
        results = generate_boundary_kv(
            model=model, tokenizer=tokenizer,
            prompt=prompt,
            delta_paths=[
                dir_a / f"{delta_a.delta_id}.pdga",
                dir_b / f"{delta_b.delta_id}.pdga",
            ],
            window_indices=[wa_use, wb_use],
            injection_layer=29,
            max_new_tokens=60, sample_temp=0.7,
        )
    elapsed = time.perf_counter() - t0

    for i, r in enumerate(results):
        label = "Article A" if i == 0 else "Article B"
        print(f"\n── {label} ({r['num_tokens']} tok, {r['windows_loaded']} windows loaded) ──")
        print(r["generated_text"][:300])

    total_tok = sum(r["num_tokens"] for r in results)
    print(f"\nTotal: {total_tok} tokens in {elapsed:.1f}s ({total_tok/elapsed:.1f} t/s)")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import sys
    test = sys.argv[1] if len(sys.argv) > 1 else "all"

    if test in ("ingest", "all"):
        delta, wins = test_ingest_with_kv_cache()
    if test in ("generate", "all"):
        # Re-run ingest since cleanup happens
        if test == "generate":
            if TEST_DIR.exists():
                shutil.rmtree(TEST_DIR)
            TEST_DIR.mkdir(parents=True)
            delta, _ = test_ingest_with_kv_cache()
        test_boundary_kv_generation(delta)
    if test in ("multi", "all"):
        test_multi_delta_boundary_kv()
    if test in ("clean",):
        shutil.rmtree(TEST_DIR)
        print("Cleaned up test directory")
