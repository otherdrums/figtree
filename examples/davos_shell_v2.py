#!/usr/bin/env python3
"""PDGA Davos v2 Interactive Shell.

Usage:
    python3 davos_shell_v2.py
"""

import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pdga.fact.primitive import Fact
from pdga.fact.generate import FactGenerator
from pdga.fact.graph import FactGraph

console = Console()
MODEL_ID = "unsloth/Qwen3-4B-bnb-4bit"
DELTAS_DIR = Path(__file__).parent / "davos_deltas_v2"

SOURCES = {
    "pro_globalist": {"name": "Reuters-style", "trust": 0.95, "color": "green"},
    "anti_globalist": {"name": "Guardian-style", "trust": 0.60, "color": "yellow"},
    "conspiracy": {"name": "Fringe Blog", "trust": 0.15, "color": "red"},
}


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_all_facts() -> list[Fact]:
    facts = []
    for key in SOURCES:
        fact_dirs = sorted((DELTAS_DIR / key).glob("*.pdga"))
        for d in fact_dirs:
            facts.append(Fact.load(d))
    return facts


def retrieve_facts(query: str, facts: list[Fact], limit: int = 10) -> list[Fact]:
    query_words = set(query.lower().split())
    scored = []
    for f in facts:
        if f.is_edge() or f.is_trust_assertion():
            continue
        text_words = set(f.text.lower().split())
        score = len(query_words & text_words)
        if score > 0:
            scored.append((score, f))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [f for _, f in scored[:limit]]


def main():
    if not DELTAS_DIR.exists() or not any(DELTAS_DIR.iterdir()):
        console.print("[bold red]Error:[/bold red] No ingested deltas found.")
        console.print("Run: python3 run_davos_v2.py ingest")
        sys.exit(1)

    console.print("[bold blue]PDGA Davos v2 Interactive Shell[/bold blue]")
    console.print("Loading model...")
    model, tokenizer = load_model()
    gen = FactGenerator(model, tokenizer)
    all_facts = load_all_facts()
    graph = FactGraph(all_facts)
    graph.propagate_trust()

    console.print(f"Ready. {len(all_facts)} facts loaded. Type /help for commands.\n")

    while True:
        try:
            query = console.input("[bold]query> [/bold]").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query == "/quit":
            break
        elif query == "/help":
            console.print("Commands: /trust, /sources, /facts <key>, /all, /quit")
            continue
        elif query == "/trust":
            top = graph.get_top_facts(10)
            table = Table(show_header=True, header_style="bold")
            table.add_column("Trust", width=8, justify="right")
            table.add_column("Source", width=15)
            table.add_column("Text")
            for f in top:
                src = f.meta.get("source_id", "unknown")[:14]
                table.add_row(f"{f.trust:.2f}", src, f.text[:80])
            console.print(table)
            continue
        elif query == "/sources":
            for key in SOURCES:
                dirs = list((DELTAS_DIR / key).glob("*.pdga"))
                atomic = [Fact.load(d) for d in dirs if not Fact.load(d).is_narrative() and not Fact.load(d).is_trust_assertion()]
                console.print(f"  {key}: {len(atomic)} facts")
            continue
        elif query.startswith("/facts "):
            key = query.split(" ", 1)[1].strip()
            if key not in SOURCES:
                console.print(f"[red]Unknown: {key}[/red]")
                continue
            dirs = sorted((DELTAS_DIR / key).glob("*.pdga"))
            for i, d in enumerate(dirs[:5], 1):
                f = Fact.load(d)
                if not f.is_narrative() and not f.is_trust_assertion():
                    console.print(f"  {i}. {f.text[:80]}")
            continue
        elif query == "/all":
            facts = [f for f in all_facts if not f.is_narrative() and not f.is_trust_assertion()]
            console.print(f"[dim]Loading all {len(facts)} facts...[/dim]")
        else:
            facts = retrieve_facts(query, all_facts)
            if not facts:
                console.print("[yellow]No relevant facts.[/yellow]")
                continue
            console.print(f"[dim]Retrieved {len(facts)} facts[/dim]")

        gc.collect()
        torch.cuda.empty_cache()

        result = gen.generate(facts, query, max_new_tokens=100)
        console.print(Panel(f"Generated ({result['num_tokens']} tokens)", border_style="blue"))
        console.print(result["generated_text"])
        console.print()

        gc.collect()
        torch.cuda.empty_cache()

    console.print("Goodbye.")


if __name__ == "__main__":
    main()
