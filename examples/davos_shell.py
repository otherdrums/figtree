#!/usr/bin/env python3
"""PDGA Davos Interactive Shell

Query the multi-perspective knowledge base interactively.

Usage:
    python3 davos_shell.py

Commands:
    /trust           Show top trusted facts
    /sources         List loaded narratives
    /facts <id>      Show facts from a narrative
    /quit            Exit
"""

import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pdga.db.store import DeltaDB
from pdga.generation.streaming import StreamingGenerator
from pdga.graph.trust import TrustPropagator

console = Console()
MODEL_ID = "unsloth/Qwen3-4B-bnb-4bit"
DELTAS_DIR = Path(__file__).parent / "davos_deltas"

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


def get_fact_paths(narrative_dir: Path) -> list[Path]:
    facts_dir = narrative_dir / "facts"
    if not facts_dir.exists():
        return []
    return sorted(facts_dir.glob("*/kv_cache.pt"))


def get_all_fact_paths() -> list[Path]:
    paths = []
    for pdga_dir in DELTAS_DIR.glob("*.pdga"):
        paths.extend(get_fact_paths(pdga_dir))
    return paths


def retrieve_facts(query: str, db: DeltaDB, limit: int = 15) -> list[dict]:
    """Simple keyword-based retrieval from fact database."""
    query_words = set(query.lower().split())
    facts = db.list_all(delta_type="fact")
    scored = []
    for fact in facts:
        text = fact.get("source_text", "").lower()
        score = len(query_words & set(text.split()))
        if score > 0:
            scored.append((score, fact))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [f for _, f in scored[:limit]]


def format_fact_prompt(facts: list[dict]) -> str:
    """Format retrieved facts for the prompt."""
    lines = []
    for i, f in enumerate(facts, 1):
        trust = f.get("trust", 0.5)
        text = f.get("source_text", "")[:200]
        lines.append(f"[{i}] (trust={trust:.2f}) {text}")
    return "\n".join(lines)


def main():
    if not DELTAS_DIR.exists() or not any(DELTAS_DIR.glob("*.pdga")):
        console.print("[bold red]Error:[/bold red] No ingested deltas found.")
        console.print("Run: python3 run_davos_demo.py ingest")
        sys.exit(1)

    console.print("[bold blue]PDGA Davos Interactive Shell[/bold blue]")
    console.print("Loading model...")
    model, tokenizer = load_model()
    db = DeltaDB()
    trust_prop = TrustPropagator(db)

    # Build narrative map
    narrative_map = {}
    for path in DELTAS_DIR.glob("*.pdga"):
        narrative_json = path / "narrative.json"
        if narrative_json.exists():
            meta = json.loads(narrative_json.read_text())
            source_key = meta.get("source_key", "")
            for key in SOURCES:
                if key == source_key:
                    narrative_map[key] = path
                    break

    console.print("Ready. Type your query or /help for commands.\n")

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
            console.print("Commands:")
            console.print("  /trust      - Show top trusted facts")
            console.print("  /sources    - List loaded narratives")
            console.print("  /facts <id> - Show facts from a narrative")
            console.print("  /all        - Load all facts (slow)")
            console.print("  /quit       - Exit")
            continue
        elif query == "/trust":
            facts = trust_prop.rank_facts(10)
            table = Table(show_header=True, header_style="bold")
            table.add_column("Trust", width=8, justify="right")
            table.add_column("Source", width=15)
            table.add_column("Text")
            for f in facts:
                tags = json.loads(f.get("tags", "[]"))
                source = next((SOURCES.get(t, {}).get("name", t) for t in tags if t in SOURCES), "unknown")
                text = f.get("source_text", "")[:120]
                table.add_row(f"{f.get('trust', 0.5):.2f}", source, text)
            console.print(table)
            continue
        elif query == "/sources":
            for key, path in narrative_map.items():
                meta = json.loads((path / "narrative.json").read_text())
                num_facts = len(get_fact_paths(path))
                console.print(f"  {key}: {SOURCES[key]['name']} "
                             f"(trust={SOURCES[key]['trust']}, {num_facts} facts)")
            continue
        elif query.startswith("/facts "):
            key = query.split(" ", 1)[1].strip()
            if key not in narrative_map:
                console.print(f"[red]Unknown source: {key}[/red]")
                continue
            paths = get_fact_paths(narrative_map[key])
            console.print(f"Facts from {SOURCES[key]['name']} ({len(paths)} total):")
            for i, p in enumerate(paths[:5], 1):
                manifest = p.parent / "manifest.json"
                if manifest.exists():
                    meta = json.loads(manifest.read_text())
                    console.print(f"  {i}. {meta['text'][:100]}")
            if len(paths) > 5:
                console.print(f"  ... and {len(paths) - 5} more")
            continue
        elif query == "/all":
            fact_paths = get_all_fact_paths()
            console.print(f"[dim]Loading all {len(fact_paths)} facts...[/dim]")
        else:
            # Normal query: retrieve relevant facts
            retrieved = retrieve_facts(query, db)
            if not retrieved:
                console.print("[yellow]No relevant facts found. Try a different query.[/yellow]")
                continue
            fact_paths = []
            for f in retrieved:
                path = Path(f["path"])
                kv = path / "kv_cache.pt"
                if kv.exists():
                    fact_paths.append(kv)
            console.print(f"[dim]Retrieved {len(fact_paths)} relevant facts[/dim]")

        # Generate response
        if not fact_paths:
            continue

        gc.collect()
        torch.cuda.empty_cache()

        gen = StreamingGenerator(model, fact_paths)
        
        # Build trust-aware prompt
        if query == "/all":
            prompt = f"Based on all available information, answer this question: {query}"
        else:
            facts_text = format_fact_prompt(retrieved)
            prompt = (
                f"Based on the following facts with associated trust scores, "
                f"answer this question: {query}\n\n"
                f"Facts:\n{facts_text}\n\n"
                f"Answer:"
            )

        r = gen.generate(tokenizer, prompt=prompt, max_new_tokens=200, sample_temp=0.7)
        
        # Display result
        console.print(Panel(f"Generated ({r['num_tokens']} tokens, {r['tokens_per_second']:.1f} t/s)",
                           border_style="blue"))
        console.print(r["generated_text"])
        console.print()

        del gen
        gc.collect()
        torch.cuda.empty_cache()

    console.print("Goodbye.")
    db.close()


if __name__ == "__main__":
    main()
