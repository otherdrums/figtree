#!/usr/bin/env python3
"""Quick v2 pipeline test — ingest to LanceDB, generate (text + boundary),
build graph. Also exercises LanceDB compression and lazy KV recompute.

Run with FIGTREE_TEST_S3_URI=s3://bucket/path to also exercise remote storage.
"""

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
from figtree.lancedb_store import connect
from figtree.kv_cache_manager import KVCacheManager

MODEL_ID = "unsloth/Qwen3-4B-bnb-4bit"
FIGMENTS_DIR = Path.home() / ".figtree" / "v2_test"
STORE_URI = os.environ.get("FIGTREE_TEST_S3_URI", str(FIGMENTS_DIR) + ".lance")

TEXT = """The World Economic Forum summit in Davos concluded yesterday.
Leaders from 130 countries gathered alongside 2,700 delegates.
The centerpiece achievement was the Digital Cooperation Compact."""


def main():
    if Path(STORE_URI).exists():
        shutil.rmtree(STORE_URI)
    Path(STORE_URI).parent.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    store = connect(STORE_URI)
    kv_manager = KVCacheManager(
        model, tokenizer, kv_root=str(Path(STORE_URI).with_suffix("")) + "_kv",
        mode="lazy",
    )

    # Ingest into LanceDB (lazy: no K/V persisted).
    print("\nIngesting text into LanceDB figments...")
    figments = ingest_text_to_figments(
        model=model, tokenizer=tokenizer, text=TEXT,
        source_id="test_source", trust=0.95,
        store=store, kv_manager=kv_manager, compute_kv=False,
    )
    print(f"Created {len(figments)} figments in {STORE_URI}")
    assert store.count() == len(figments), "LanceDB row count mismatch"

    # Reload atomic figments from the store (round-trip + compression).
    atomic_figments = [f for f in store.by_source("test_source")
                       if not f.is_image() and not f.is_trust_assertion()]
    assert atomic_figments, "No atomic figments loaded from store"
    # Compression sanity: boundary vectors round-trip exactly (order-independent).
    orig_map = {f.figment_id: f for f in figments
                if not f.is_image() and not f.is_trust_assertion()}
    assert set(orig_map) == {f.figment_id for f in atomic_figments}, \
        "Figment id set mismatch after store round-trip"
    for fid, loaded in zip(orig_map, atomic_figments):
        assert orig_map[fid].boundary.shape == loaded.boundary.shape
        assert orig_map[fid].boundary.dtype == loaded.boundary.dtype

    # Generate (text path)
    print(f"\nGenerating with {len(atomic_figments)} figments...")
    gen = FigmentGenerator(model, tokenizer)
    result = gen.generate(
        figments=atomic_figments,
        prompt="What happened at Davos?",
        max_new_tokens=200,
    )
    print(f"Generated {result['num_tokens']} tokens in {result['elapsed']:.1f}s")

    # Boundary-based generation: lazy recompute via kv_manager (no eager blob).
    print("\nGenerating from boundaries (lazy K/V recompute)...")
    result_bd = gen.generate_from_boundaries(
        figments=atomic_figments,
        prompt="What happened at Davos?",
        max_new_tokens=200,
        kv_manager=kv_manager,
    )
    print(f"Generated {result_bd['num_tokens']} tokens in {result_bd['elapsed']:.1f}s")

    assert result_bd["num_tokens_total"] > 0, "Boundary K/V cache was not loaded"
    assert result_bd["num_tokens"] > 0, "Boundary generation produced no tokens"
    entities = ("davos", "digital", "economy", "summit", "leaders", "compact",
                "cooperation", "130", "countries")
    assert any(w in result_bd["generated_text"].lower() for w in entities), \
        "Boundary generation output is off-topic / not conditioned on figments"
    text_hits = sum(w in result["generated_text"].lower() for w in entities)
    bd_hits = sum(w in result_bd["generated_text"].lower() for w in entities)
    assert bd_hits >= 1, "Boundary path recalled no source entities"
    print(
        f"Boundary generation check passed "
        f"({result_bd['num_tokens']} tokens, "
        f"{result_bd['num_tokens_total']} K/V tokens; "
        f"entities text={text_hits} boundary={bd_hits})."
    )

    # Flawless-recall verification: generate_with_recall must reproduce every
    # checkable atom (here: 130, 2,700) from the source text, patching any gaps.
    all_atomic = [f for f in atomic_figments if not f.is_image() and not f.is_trust_assertion()]
    recall_res = gen.generate_with_recall(
        figments=all_atomic,
        prompt="What happened at Davos? Include every number and detail.",
        source_texts=[TEXT],
        max_new_tokens=200,
    )
    print(f"Recall score: {recall_res['recall_score']:.2f} "
          f"(missing: {recall_res['missing_atoms']})")
    assert recall_res["recall_score"] >= 1.0, \
        f"Recall not flawless: missing {recall_res['missing_atoms']}"

    # Graph + persisted trust via store (idempotent).
    print("\nBuilding graph...")
    graph = Figtree(store.all(), store=store)
    dedup_edges = graph.deduplicate()
    auto_edges = graph.create_edges()
    trust_scores = graph.propagate_trust(store=store)  # idempotent + persisted
    print(f"  Dedup edges: {len(dedup_edges)}")
    print(f"  Auto edges: {len(auto_edges)}")
    print(f"  Trust scores: {len(trust_scores)}")

    # P3a: trust Figment is persisted in the store and re-runnable.
    trust_fig = store.get("trust:test_source")
    assert trust_fig is not None, "Trust Figment was not persisted to store"
    assert trust_fig.meta.get("edge_type") == "trust", "Persisted trust type wrong"
    assert "rationale" in trust_fig.meta, "Persisted trust missing rationale"
    # Idempotent re-run overwrites the same row.
    graph.propagate_trust(store=store)
    assert store.get("trust:test_source").figment_id == "trust:test_source"
    print(f"  Persisted trust Figment: score={trust_fig.meta.get('score')}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nFigtree v2 pipeline test complete!")


if __name__ == "__main__":
    main()
