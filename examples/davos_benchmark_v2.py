#!/usr/bin/env python3
"""PDGA Davos v2 Benchmark."""

import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.table import Table

from pdga.fact.ingest import ingest_text_to_facts
from pdga.fact.generate import FactGenerator
from pdga.fact.graph import FactGraph
from pdga.fact.primitive import Fact

console = Console()
MODEL_ID = "unsloth/Qwen3-4B-bnb-4bit"
DELTAS_DIR = Path(__file__).parent / "davos_deltas_v2"
NARRATIVES_DIR = Path(__file__).parent / "davos_narratives"

SOURCES = {
    "pro_globalist": {"name": "Reuters-style", "trust": 0.95},
    "anti_globalist": {"name": "Guardian-style", "trust": 0.60},
    "conspiracy": {"name": "Fringe Blog", "trust": 0.15},
}


def benchmark():
    results = {}

    if DELTAS_DIR.exists():
        shutil.rmtree(DELTAS_DIR)
    DELTAS_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    console.print("[bold]Loading model...[/bold]")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Phase 1: Ingestion
    console.print("\n[bold]Phase 1: Ingestion[/bold]")
    t0 = time.perf_counter()
    total_facts = 0
    total_size = 0
    for key in SOURCES:
        text = (NARRATIVES_DIR / f"{key}.txt").read_text().strip()
        facts = ingest_text_to_facts(
            model=model, tokenizer=tokenizer, text=text,
            output_dir=DELTAS_DIR / key, source_id=key,
            trust=SOURCES[key]["trust"], min_chars=20,
        )
        total_facts += len(facts)
        size = sum(f.stat().st_size for f in (DELTAS_DIR / key).rglob("*") if f.is_file())
        total_size += size

    ingest_time = time.perf_counter() - t0
    results["ingest"] = {
        "time": ingest_time,
        "facts": total_facts,
        "size_kb": total_size / 1024,
    }
    console.print(f"  {total_facts} facts in {ingest_time:.1f}s, {total_size/1024:.1f} KB")

    # Phase 2: Generation
    console.print("\n[bold]Phase 2: Generation[/bold]")
    gen = FactGenerator(model, tokenizer)
    all_atomic = []
    for key in SOURCES:
        dirs = sorted((DELTAS_DIR / key).glob("*.pdga"))
        for d in dirs:
            f = Fact.load(d)
            if not f.is_narrative() and not f.is_trust_assertion():
                all_atomic.append(f)

    t0 = time.perf_counter()
    result = gen.generate(
        facts=all_atomic,
        prompt="What happened at Davos?",
        max_new_tokens=100,
    )
    gen_time = time.perf_counter() - t0
    results["generate"] = {
        "time": gen_time,
        "tokens": result["num_tokens"],
        "facts": len(all_atomic),
    }
    console.print(f"  {result['num_tokens']} tokens in {gen_time:.1f}s")
    console.print(f"  Facts loaded: {len(all_atomic)}")

    # Phase 3: Graph
    console.print("\n[bold]Phase 3: Graph[/bold]")
    all_facts = []
    for key in SOURCES:
        dirs = sorted((DELTAS_DIR / key).glob("*.pdga"))
        for d in dirs:
            all_facts.append(Fact.load(d))

    t0 = time.perf_counter()
    graph = FactGraph(all_facts)
    graph.deduplicate()
    graph.create_edges()
    graph.propagate_trust()
    graph_time = time.perf_counter() - t0

    results["graph"] = {
        "time": graph_time,
        "facts": len([f for f in all_facts if not f.is_edge() and not f.is_trust_assertion()]),
        "edges": len([f for f in all_facts if f.is_edge()]),
    }
    console.print(f"  Graph built in {graph_time:.1f}s")

    # Summary
    console.print("\n[bold]Benchmark Summary[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Phase")
    table.add_column("Time (s)", justify="right")
    table.add_column("Throughput")
    table.add_row("Ingestion", f"{results['ingest']['time']:.1f}",
                    f"{results['ingest']['facts']} facts, {results['ingest']['size_kb']:.1f} KB")
    table.add_row("Generation", f"{results['generate']['time']:.1f}",
                    f"{results['generate']['tokens']} tokens, {results['generate']['facts']} facts")
    table.add_row("Graph", f"{results['graph']['time']:.1f}",
                    f"{results['graph']['facts']} facts, {results['graph']['edges']} edges")
    total = sum(r["time"] for r in results.values())
    table.add_row("Total", f"{total:.1f}", "")
    console.print(table)


if __name__ == "__main__":
    benchmark()
