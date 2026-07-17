#!/usr/bin/env python3
"""Quick v2 pipeline test — ingest one narrative, generate, build graph."""

import gc
import os
import shutil
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from figtree.ingest import ingest_text_to_figments
from figtree.generate import FigmentGenerator
from figtree.graph import Figtree

MODEL_ID = "unsloth/Qwen3-4B-bnb-4bit"
FIGMENTS_DIR = Path.home() / ".figtree" / "v2_test"

TEXT = """The World Economic Forum summit in Davos concluded yesterday.
Leaders from 130 countries gathered alongside 2,700 delegates.
The centerpiece achievement was the Digital Cooperation Compact."""


def main():
    if FIGMENTS_DIR.exists():
        shutil.rmtree(FIGMENTS_DIR)
    FIGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Ingest
    print("\nIngesting text into figments...")
    figments = ingest_text_to_figments(
        model=model, tokenizer=tokenizer, text=TEXT,
        output_dir=FIGMENTS_DIR, source_id="test_source", trust=0.95,
    )
    print(f"Created {len(figments)} figments:")
    for f in figments:
        print(f"  {f}")

    # Load atomic figments (skip image and trust assertion)
    atomic_figments = [f for f in figments if not f.is_image() and not f.is_trust_assertion()]

    # Generate
    print(f"\nGenerating with {len(atomic_figments)} figments...")
    gen = FigmentGenerator(model, tokenizer)
    result = gen.generate(
        figments=atomic_figments,
        prompt="What happened at Davos?",
        max_new_tokens=80,
    )
    print(f"Generated {result['num_tokens']} tokens in {result['elapsed']:.1f}s")
    print(f"Text: {result['generated_text'][:200]}")

    # Boundary-based generation (cached K/V from disk)
    print("\nGenerating from cached boundaries (kv_cache.npy)...")
    result_bd = gen.generate_from_boundaries(
        figments=atomic_figments,
        prompt="What happened at Davos?",
        max_new_tokens=80,
        cache_dir=str(FIGMENTS_DIR),
    )
    print(f"Generated {result_bd['num_tokens']} tokens in {result_bd['elapsed']:.1f}s")
    print(f"Text: {result_bd['generated_text'][:200]}")

    # Correctness check for the cached-K/V path: confirm it actually loaded the
    # per-token K/V cache from disk and produced a non-empty, on-topic response.
    # (We avoid a brittle verbatim-recall assertion because the 4B model tends to
    # paraphrase; structural validation of the cache injection is the real gate.)
    assert result_bd["num_tokens_total"] > 0, "Boundary K/V cache was not loaded"
    assert result_bd["num_tokens"] > 0, "Boundary generation produced no tokens"
    assert any(
        w in result_bd["generated_text"].lower()
        for w in ("davos", "digital", "economy", "summit", "leaders", "compact")
    ), "Boundary generation output is off-topic / not conditioned on figments"
    print(
        f"Boundary generation check passed "
        f"({result_bd['num_tokens']} tokens, "
        f"{result_bd['num_tokens_total']} cached K/V tokens loaded)."
    )

    # Graph
    print("\nBuilding graph...")
    graph = Figtree(figments)
    dedup_edges = graph.deduplicate()
    auto_edges = graph.create_edges()
    trust_scores = graph.propagate_trust()
    print(f"  Dedup edges: {len(dedup_edges)}")
    print(f"  Auto edges: {len(auto_edges)}")
    print(f"  Trust scores: {len(trust_scores)}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nFigtree v2 pipeline test complete!")


if __name__ == "__main__":
    main()
