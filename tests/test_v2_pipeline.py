#!/usr/bin/env python3
"""Quick v2 pipeline test — ingest one narrative, generate, build graph."""

import gc
import os
import shutil
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pdga.fact.ingest import ingest_text_to_facts
from pdga.fact.generate import FactGenerator
from pdga.fact.graph import FactGraph

MODEL_ID = "unsloth/Qwen3-4B-bnb-4bit"
DELTAS_DIR = Path.home() / ".pdga" / "v2_test"

# Sample narrative
TEXT = """The World Economic Forum summit in Davos concluded yesterday.
Leaders from 130 countries gathered alongside 2,700 delegates.
The centerpiece achievement was the Digital Cooperation Compact."""


def main():
    if DELTAS_DIR.exists():
        shutil.rmtree(DELTAS_DIR)
    DELTAS_DIR.mkdir(parents=True, exist_ok=True)

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
    print("\nIngesting text into facts...")
    facts = ingest_text_to_facts(
        model=model, tokenizer=tokenizer, text=TEXT,
        output_dir=DELTAS_DIR, source_id="test_source", trust=0.95,
    )
    print(f"Created {len(facts)} facts:")
    for f in facts:
        print(f"  {f}")

    # Load atomic facts (skip narrative and trust assertion)
    atomic_facts = [f for f in facts if not f.is_narrative() and not f.is_trust_assertion()]

    # Generate
    print(f"\nGenerating with {len(atomic_facts)} facts...")
    gen = FactGenerator(model, tokenizer)
    result = gen.generate(
        facts=atomic_facts,
        prompt="What happened at Davos?",
        max_new_tokens=80,
    )
    print(f"Generated {result['num_tokens']} tokens in {result['elapsed']:.1f}s")
    print(f"Text: {result['generated_text'][:200]}")

    # Graph
    print("\nBuilding graph...")
    graph = FactGraph(facts)
    dedup_edges = graph.deduplicate()
    auto_edges = graph.create_edges()
    trust_scores = graph.propagate_trust()
    print(f"  Dedup edges: {len(dedup_edges)}")
    print(f"  Auto edges: {len(auto_edges)}")
    print(f"  Trust scores: {len(trust_scores)}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nv2 pipeline test complete!")


if __name__ == "__main__":
    main()
