#!/usr/bin/env python3
"""Davos + prompt-learning demo (manual §7 validation on the real model).

1. Ingest a small base corpus (Davos-style news).
2. Teach a series of single statements (name, policy, allergy).
3. Query by role intersection — answers can only be correct if the teachings
   were retained, identity-linked, and still in the graph after the store is
   reloaded fresh (original prompts are NOT in the window).

Usage:
    python3 examples/davos_learn_demo.py
"""

import os
import shutil
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.panel import Panel

from figtree import (
    FigmentGenerator,
    connect,
    ingest_text_to_figments,
    learned_facts,
    retrieve_by_roles,
    teach,
)

console = Console()
MODEL_ID = "unsloth/Qwen3-4B-bnb-4bit"
STORE_URI = str(Path(__file__).parent / "davos_learn.lance")

BASE = """The World Economic Forum summit in Davos concluded yesterday.
Leaders from 130 countries gathered alongside 2,700 delegates.
The centerpiece achievement was the Digital Cooperation Compact."""

TEACHINGS = [
    "My daughter's name is June and she is allergic to cashews.",
    "Never use the legacy auth endpoint again.",
    "My name is Alex. Please remember that.",
]

QUERIES = [
    "What is the user's name?",
    "What should I do instead of the legacy auth endpoint?",
    "What is the user's daughter allergic to?",
]


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


def main():
    if Path(STORE_URI).exists():
        shutil.rmtree(STORE_URI)

    console.print("[bold blue]Loading model...[/bold blue]")
    model, tokenizer = load_model()
    gen = FigmentGenerator(model, tokenizer)
    store = connect(STORE_URI)

    console.print("\n[bold]1. Ingest base corpus[/bold]")
    figments = ingest_text_to_figments(
        model, tokenizer, BASE, source_id="demo", trust=0.9, store=store,
    )
    console.print(f"   {len(figments)} figments ingested")

    console.print("\n[bold]2. Teach single statements[/bold]")
    for stmt in TEACHINGS:
        res = teach(model, tokenizer, stmt, store, session_id="user", trust=0.95)
        console.print(f"   roles={res['roles']}")
        if res["superseded"]:
            console.print(f"   [yellow]superseded={res['superseded']}[/yellow]")

    console.print("\n[bold]3. Reload store fresh (prompts NOT in window)[/bold]")
    store = connect(STORE_URI)
    facts = learned_facts(store)
    console.print(f"   {len(facts)} learned facts persisted:")
    for f in facts:
        console.print(f"   {f.meta.get('role', '?'):12s} {f.text}")

    console.print("\n[bold]4. Role-intersection queries (rule-based, CPU)[/bold]")
    for query in QUERIES:
        hits = retrieve_by_roles(query, store)
        console.print(f"\n[bold]Query:[/bold] {query}")
        if not hits:
            console.print("   [red]No learned statement retrieved.[/red]")
            continue
        for f in hits:
            console.print(f"   [dim][{f.meta.get('role_match')}][/dim] {f.text}")
        result = gen.generate(hits, query, max_new_tokens=60)
        console.print(Panel(result["generated_text"], border_style="blue"))


if __name__ == "__main__":
    main()
