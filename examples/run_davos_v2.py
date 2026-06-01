#!/usr/bin/env python3
"""Figtree Davos Demo v2 — three narratives, one event, multiple perspectives.

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

from figtree.ingest import ingest_text_to_figments
from figtree.generate import FigmentGenerator
from figtree.graph import Figtree
from figtree.figment import Figment

console = Console()
MODEL_ID = "unsloth/Qwen3-4B-bnb-4bit"
FIGMENTS_DIR = Path(__file__).parent / "davos_figments_v2"
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
    banner("Figtree Davos v2 — Ingestion", "Three narratives -> atomic figments + boundaries")
    console.print("[bold]Model:[/bold] Qwen3-4B (36L, h=2560)")
    console.print("[bold]GPU:[/bold] " +
                  (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))

    if FIGMENTS_DIR.exists():
        shutil.rmtree(FIGMENTS_DIR)
    FIGMENTS_DIR.mkdir(parents=True, exist_ok=True)

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

    banner("Ingesting...", "Boundary capture per sentence (~10 KB/figment)")

    for key in SOURCES:
        text = (NARRATIVES_DIR / f"{key}.txt").read_text().strip()
        source = SOURCES[key]
        token_count = len(tokenizer.encode(text))
        console.print(f"  {key} ({source['name']}, trust={source['trust']}) — {token_count} tokens...")

        figments = ingest_text_to_figments(
            model=model, tokenizer=tokenizer, text=text,
            output_dir=FIGMENTS_DIR / key, source_id=key,
            trust=source["trust"], min_chars=20,
        )

        total_size = sum(f.stat().st_size for f in (FIGMENTS_DIR / key).rglob("*") if f.is_file())
        console.print(f"    {len(figments)} figments, {total_size / 1024:.1f} KB total")

    console.print("\n[bold green]Ingestion complete.[/bold green] Run: python3 run_davos_v2.py generate")


def do_generate():
    banner("Figtree Davos v2 — Generation", "On-the-fly KV from figment text")

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

    gen = FigmentGenerator(model, tokenizer)

    source_figments: dict[str, list[Figment]] = {}
    for key in SOURCES:
        figment_dirs = sorted((FIGMENTS_DIR / key).glob("*.figment"))
        figments = [Figment.load(d) for d in figment_dirs]
        atomic = [f for f in figments if not f.is_image() and not f.is_trust_assertion()]
        source_figments[key] = atomic

    console.print("\n[bold underline green]── QUERY 1: Per-Source Generation ──[/bold underline green]")
    for key, figments in source_figments.items():
        source = SOURCES[key]
        t0 = time.perf_counter()
        result = gen.generate(
            figments=figments,
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

    console.print("\n[bold underline yellow]── QUERY 2: What Do Sources Agree On? ──[/bold underline yellow]")
    all_figments = []
    for figments in source_figments.values():
        all_figments.extend(figments)

    t0 = time.perf_counter()
    result = gen.generate(
        figments=all_figments,
        prompt="Based on all sources, what facts do they agree on?",
        max_new_tokens=200,
    )
    console.print(Panel(f"All sources ({result['num_tokens']} tokens)", border_style="blue"))
    console.print(result["generated_text"])
    console.print()

    console.print("\n[bold underline red]── QUERY 3: Where Do Sources Disagree? ──[/bold underline red]")
    result = gen.generate(
        figments=all_figments,
        prompt="What are the major disagreements between the different perspectives?",
        max_new_tokens=200,
    )
    console.print(Panel(f"Contradictions ({result['num_tokens']} tokens)", border_style="red"))
    console.print(result["generated_text"])
    console.print()

    console.print("[bold green]Figtree Davos v2 generation complete[/bold green]")


def do_graph():
    banner("Figtree Davos v2 — Graph", "Deduplication + edges + trust propagation")

    all_figments = []
    for key in SOURCES:
        figment_dirs = sorted((FIGMENTS_DIR / key).glob("*.figment"))
        for d in figment_dirs:
            all_figments.append(Figment.load(d))

    graph = Figtree(all_figments)

    dedup_edges = graph.deduplicate()
    auto_edges = graph.create_edges()
    trust_scores = graph.propagate_trust()

    console.print(f"  Figments: {len([f for f in all_figments if not f.is_edge() and not f.is_trust_assertion()])}")
    console.print(f"  Dedup edges: {len(dedup_edges)}")
    console.print(f"  Auto edges: {len(auto_edges)}")
    console.print(f"  Trust scores updated: {len(trust_scores)}")

    top = graph.get_top_facts(10)
    console.print("\n[bold underline green]── Top Trusted Figments ──[/bold underline green]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Trust", width=8, justify="right")
    table.add_column("Figment ID", style="dim", width=16)
    table.add_column("Text")
    for f in top:
        text = f.text[:100]
        table.add_row(f"{f.trust:.2f}", f.figment_id[:14], text)
    console.print(table)

    console.print("\n[bold green]Graph complete[/bold green]")


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
