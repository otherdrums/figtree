#!/usr/bin/env python3
"""Figtree Davos v2 Interactive Shell.

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

from figtree.figment import Figment
from figtree.generate import FigmentGenerator
from figtree.graph import Figtree

console = Console()
MODEL_ID = "unsloth/Qwen3-4B-bnb-4bit"
FIGMENTS_DIR = Path(__file__).parent / "davos_figments_v2"

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


def load_all_figments() -> list[Figment]:
    figments = []
    for key in SOURCES:
        figment_dirs = sorted((FIGMENTS_DIR / key).glob("*.figment"))
        for d in figment_dirs:
            figments.append(Figment.load(d))
    return figments


def retrieve_figments(query: str, figments: list[Figment], limit: int = 10) -> list[Figment]:
    query_words = set(query.lower().split())
    scored = []
    for f in figments:
        if f.is_edge() or f.is_trust_assertion():
            continue
        text_words = set(f.text.lower().split())
        score = len(query_words & text_words)
        if score > 0:
            scored.append((score, f))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [f for _, f in scored[:limit]]


def main():
    if not FIGMENTS_DIR.exists() or not any(FIGMENTS_DIR.iterdir()):
        console.print("[bold red]Error:[/bold red] No ingested figments found.")
        console.print("Run: python3 run_davos_v2.py ingest")
        sys.exit(1)

    console.print("[bold blue]Figtree Davos v2 Interactive Shell[/bold blue]")
    console.print("Loading model...")
    model, tokenizer = load_model()
    gen = FigmentGenerator(model, tokenizer)
    all_figments = load_all_figments()
    graph = Figtree(all_figments)
    graph.propagate_trust()

    console.print(f"Ready. {len(all_figments)} figments loaded. Type /help for commands.\n")

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
            console.print("Commands: /trust, /sources, /figments <key>, /all, /quit")
            continue
        elif query == "/trust":
            top = graph.get_top_figments(10)
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
                dirs = list((FIGMENTS_DIR / key).glob("*.figment"))
                atomic = [Figment.load(d) for d in dirs if not Figment.load(d).is_image() and not Figment.load(d).is_trust_assertion()]
                console.print(f"  {key}: {len(atomic)} figments")
            continue
        elif query.startswith("/figments "):
            key = query.split(" ", 1)[1].strip()
            if key not in SOURCES:
                console.print(f"[red]Unknown: {key}[/red]")
                continue
            dirs = sorted((FIGMENTS_DIR / key).glob("*.figment"))
            for i, d in enumerate(dirs[:5], 1):
                f = Figment.load(d)
                if not f.is_image() and not f.is_trust_assertion():
                    console.print(f"  {i}. {f.text[:80]}")
            continue
        elif query == "/all":
            figments = [f for f in all_figments if not f.is_image() and not f.is_trust_assertion()]
            console.print(f"[dim]Loading all {len(figments)} figments...[/dim]")
        else:
            figments = retrieve_figments(query, all_figments)
            if not figments:
                console.print("[yellow]No relevant figments.[/yellow]")
                continue
            console.print(f"[dim]Retrieved {len(figments)} figments[/dim]")

        gc.collect()
        torch.cuda.empty_cache()

        result = gen.generate(figments, query, max_new_tokens=100)
        console.print(Panel(f"Generated ({result['num_tokens']} tokens)", border_style="blue"))
        console.print(result["generated_text"])
        console.print()

        gc.collect()
        torch.cuda.empty_cache()

    console.print("Goodbye.")


if __name__ == "__main__":
    main()
