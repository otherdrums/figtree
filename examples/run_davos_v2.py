#!/usr/bin/env python3
"""PDGA Davos Demo v2 — three narratives, one event, multiple perspectives.

Usage:
    python3 run_davos_v2.py ingest
    python3 run_davos_v2.py generate
    python3 run_davos_v2.py graph
    python3 run_davos_v2.py all
"""

import gc
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich.text import Text

from pdga.fact.ingest import ingest_text_to_facts
from pdga.fact.generate import FactGenerator
from pdga.fact.graph import FactGraph
from pdga.fact.primitive import Fact

console = Console()
MODEL_ID = "unsloth/Qwen3-4B-bnb-4bit"
DELTAS_DIR = Path(__file__).parent / "davos_deltas_v2"
NARRATIVES_DIR = Path(__file__).parent / "davos_narratives"

SOURCES = {
    "pro_globalist": {"name": "Reuters-style", "trust": 0.95, "color": "green"},
    "anti_globalist": {"name": "Guardian-style", "trust": 0.60, "color": "yellow"},
    "conspiracy": {"name": "Fringe Blog", "trust": 0.15, "color": "red"},
}


def banner(title: str, dim: str = ""):
    console.print()
    console.print(Rule(f"[bold blue]{title}[/bold blue]"))
    if dim:
        console.print(f"[dim]{dim}[/dim]")


def do_ingest():
    banner("PDGA Davos v2 — Ingestion", "Three narratives → atomic facts + boundaries")
    console.print("[bold]Model:[/bold] Qwen3-4B (36L, h=2560)")
    console.print("[bold]GPU:[/bold] " +
                  (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))

    if DELTAS_DIR.exists():
        shutil.rmtree(DELTAS_DIR)
    DELTAS_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    console.print("  Loaded: {} layers, h={}".format(
        model.config.num_hidden_layers, model.config.hidden_size))

    banner("Ingesting...", "Boundary capture per sentence (~10 KB/fact)")

    for key in SOURCES:
        text = (NARRATIVES_DIR / f"{key}.txt").read_text().strip()
        source = SOURCES[key]
        token_count = len(tokenizer.encode(text))
        console.print(f"  {key} ({source['name']}, trust={source['trust']}) — {token_count} tokens...")

        facts = ingest_text_to_facts(
            model=model, tokenizer=tokenizer, text=text,
            output_dir=DELTAS_DIR / key, source_id=key,
            trust=source["trust"], min_chars=20,
        )

        # Count sizes
        total_size = sum(f.stat().st_size for f in (DELTAS_DIR / key).rglob("*") if f.is_file())
        console.print(f"    {len(facts)} facts, {total_size / 1024:.1f} KB total")

    console.print("\n[bold green]Ingestion complete.[/bold green] Run: python3 run_davos_v2.py generate")


def do_generate():
    banner("PDGA Davos v2 — Generation", "On-the-fly KV from fact text")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    gc.collect()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    console.print("  Loaded: {} layers, h={}".format(
        model.config.num_hidden_layers, model.config.hidden_size))

    gen = FactGenerator(model, tokenizer)

    # Load all facts per source
    source_facts: dict[str, list[Fact]] = {}
    for key in SOURCES:
        fact_dirs = sorted((DELTAS_DIR / key).glob("*.pdga"))
        facts = [Fact.load(d) for d in fact_dirs]
        # Filter to atomic facts only (not narrative or trust)
        atomic = [f for f in facts if not f.is_narrative() and not f.is_trust_assertion()]
        source_facts[key] = atomic

    # Query 1: Per-source generation
    console.print("\n[bold underline green]── QUERY 1: Per-Source Generation ──[/bold underline green]")
    for key, facts in source_facts.items():
        source = SOURCES[key]
        t0 = time.perf_counter()
        result = gen.generate(
            facts=facts,
            prompt="What happened at Davos?",
            max_new_tokens=150,
        )
        elapsed = time.perf_counter() - t0

        color = source["color"]
        title = Text(f"{source['name']} (trust={source['trust']})", style=color)
        title.append(f"  {result['num_tokens']} tokens  {elapsed:.1f}s", style="dim")
        console.print(Panel(title, border_style=color))
        console.print(result["generated_text"][:500])
        console.print()

        gc.collect()
        torch.cuda.empty_cache()

    # Query 2: Cross-source agreement
    console.print("\n[bold underline yellow]── QUERY 2: What Do Sources Agree On? ──[/bold underline yellow]")
    all_facts = []
    for facts in source_facts.values():
        all_facts.extend(facts)

    t0 = time.perf_counter()
    result = gen.generate(
        facts=all_facts,
        prompt="Based on all sources, what facts do they agree on?",
        max_new_tokens=200,
    )
    console.print(Panel(f"All sources ({result['num_tokens']} tokens)", border_style="blue"))
    console.print(result["generated_text"])
    console.print()

    # Query 3: Disagreements
    console.print("\n[bold underline red]── QUERY 3: Where Do Sources Disagree? ──[/bold underline red]")
    result = gen.generate(
        facts=all_facts,
        prompt="What are the major disagreements between the different perspectives?",
        max_new_tokens=200,
    )
    console.print(Panel(f"Contradictions ({result['num_tokens']} tokens)", border_style="red"))
    console.print(result["generated_text"])
    console.print()

    console.print("[bold green]✓[/bold green] Davos v2 generation complete")


def do_graph():
    banner("PDGA Davos v2 — Graph", "Deduplication + edges + trust propagation")

    all_facts = []
    for key in SOURCES:
        fact_dirs = sorted((DELTAS_DIR / key).glob("*.pdga"))
        for d in fact_dirs:
            all_facts.append(Fact.load(d))

    graph = FactGraph(all_facts)

    # Deduplicate
    dedup_edges = graph.deduplicate()
    auto_edges = graph.create_edges()
    trust_scores = graph.propagate_trust()

    console.print(f"  Facts: {len([f for f in all_facts if not f.is_edge() and not f.is_trust_assertion()])}")
    console.print(f"  Dedup edges: {len(dedup_edges)}")
    console.print(f"  Auto edges: {len(auto_edges)}")
    console.print(f"  Trust scores updated: {len(trust_scores)}")

    # Top trusted facts
    top = graph.get_top_facts(10)
    console.print("\n[bold underline green]── Top Trusted Facts ──[/bold underline green]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Trust", width=8, justify="right")
    table.add_column("Fact ID", style="dim", width=16)
    table.add_column("Text")
    for f in top:
        text = f.text[:100]
        table.add_row(f"{f.trust:.2f}", f.fact_id[:14], text)
    console.print(table)

    console.print("\n[bold green]✓[/bold green] Graph complete")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "ingest":
        do_ingest()
    elif cmd == "generate":
        do_generate()
    elif cmd == "graph":
        do_graph()
    elif cmd == "all":
        do_ingest()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        do_generate()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        do_graph()
    else:
        print("Usage: python3 run_davos_v2.py [ingest|generate|graph|all]")
