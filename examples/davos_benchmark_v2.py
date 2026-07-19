#!/usr/bin/env python3
"""Figtree Davos v2 Benchmark."""

import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.table import Table

from figtree.ingest import ingest_text_to_figments
from figtree.generate import FigmentGenerator
from figtree.graph import Figtree
from figtree.lancedb_store import connect
from figtree.kv_cache_manager import KVCacheManager

console = Console()
MODEL_ID = "unsloth/Qwen3-4B-bnb-4bit"
FIGMENTS_DIR = Path(__file__).parent / "davos_figments_v2"
STORE_URI = os.environ.get("FIGTREE_STORE_URI", str(FIGMENTS_DIR) + ".lance")
NARRATIVES_DIR = Path(__file__).parent / "davos_narratives"

SOURCES = {
    "pro_globalist": {"name": "Reuters-style", "trust": 0.95},
    "anti_globalist": {"name": "Guardian-style", "trust": 0.60},
    "conspiracy": {"name": "Fringe Blog", "trust": 0.15},
}


def benchmark():
    results = {}

    if Path(STORE_URI).exists():
        shutil.rmtree(STORE_URI)
    Path(STORE_URI).parent.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Loading model...[/bold]")
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

    console.print("\n[bold]Phase 1: Ingestion[/bold]")
    t0 = time.perf_counter()
    total_figments = 0
    for key in SOURCES:
        text = (NARRATIVES_DIR / f"{key}.txt").read_text().strip()
        figments = ingest_text_to_figments(
            model=model, tokenizer=tokenizer, text=text,
            source_id=key, store=store,
            kv_manager=kv_manager, compute_kv=False,
            trust=SOURCES[key]["trust"], min_chars=20,
        )
        total_figments += len(figments)

    ingest_time = time.perf_counter() - t0
    import subprocess
    size_out = subprocess.run(
        ["du", "-sk", STORE_URI], capture_output=True, text=True
    ).stdout.split()[0]
    total_size = int(size_out) * 1024
    results["ingest"] = {
        "time": ingest_time,
        "figments": total_figments,
        "size_kb": total_size / 1024,
    }
    console.print(f"  {total_figments} figments in {ingest_time:.1f}s, {total_size/1024:.1f} KB (LanceDB)")

    console.print("\n[bold]Phase 2: Generation[/bold]")
    gen = FigmentGenerator(model, tokenizer)
    all_atomic = []
    for key in SOURCES:
        for f in store.by_source(key):
            if not f.is_image() and not f.is_trust_assertion():
                all_atomic.append(f)

    t0 = time.perf_counter()
    result = gen.generate(
        figments=all_atomic,
        prompt="What happened at Davos?",
        max_new_tokens=100,
    )
    gen_time = time.perf_counter() - t0
    results["generate"] = {
        "time": gen_time,
        "tokens": result["num_tokens"],
        "figments": len(all_atomic),
    }
    console.print(f"  {result['num_tokens']} tokens in {gen_time:.1f}s")
    console.print(f"  Figments loaded: {len(all_atomic)}")

    console.print("\n[bold]Phase 3: Graph[/bold]")
    all_figments = store.all()

    t0 = time.perf_counter()
    graph = Figtree(all_figments, store=store)
    graph.deduplicate()
    graph.create_edges()
    graph.propagate_trust(store=store)
    graph_time = time.perf_counter() - t0

    results["graph"] = {
        "time": graph_time,
        "figments": len([f for f in all_figments if not f.is_edge() and not f.is_trust_assertion()]),
        "edges": len([f for f in all_figments if f.is_edge()]),
    }
    console.print(f"  Graph built in {graph_time:.1f}s")

    console.print("\n[bold]Benchmark Summary[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Phase")
    table.add_column("Time (s)", justify="right")
    table.add_column("Throughput")
    table.add_row("Ingestion", f"{results['ingest']['time']:.1f}",
                    f"{results['ingest']['figments']} figments, {results['ingest']['size_kb']:.1f} KB")
    table.add_row("Generation", f"{results['generate']['time']:.1f}",
                    f"{results['generate']['tokens']} tokens, {results['generate']['figments']} figments")
    table.add_row("Graph", f"{results['graph']['time']:.1f}",
                    f"{results['graph']['figments']} figments, {results['graph']['edges']} edges")
    total = sum(r["time"] for r in results.values())
    table.add_row("Total", f"{total:.1f}", "")
    console.print(table)


if __name__ == "__main__":
    benchmark()
