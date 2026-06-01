#!/usr/bin/env python3
"""PDGA Davos Demo — three narratives, one event, multiple perspectives.

Phase 1 (ingest):  python3 run_davos_demo.py ingest
Phase 2 (generate): python3 run_davos_demo.py generate
"""

import gc
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich.text import Text

from pdga.ingest.text import ingest_text
from pdga.db.store import DeltaDB
from pdga.delta.cache_io import list_window_caches
from pdga.generation.streaming import StreamingGenerator
from pdga.generation.retrieval import retrieve_top_windows

console = Console()
MODEL_ID = "unsloth/Qwen3-4B-bnb-4bit"
PDGA_HOME = Path.home() / ".pdga"
DELTAS_DIR = Path(__file__).parent / "davos_deltas"
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


def check_facts(text: str, facts: list[str]):
    t = text.lower()
    hits = []
    for f in facts:
        hits.append(f.lower() in t)
    return sum(hits), hits


def do_ingest():
    banner("PDGA Davos — Ingestion Phase",
           "Three narratives → ContextDelta + KV cache")
    console.print("[bold]Model:[/bold] Qwen3-4B (36L, h=2560)")
    console.print("[bold]GPU:[/bold] " +
                  (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))

    if PDGA_HOME.exists():
        shutil.rmtree(PDGA_HOME)
    if DELTAS_DIR.exists():
        shutil.rmtree(DELTAS_DIR)
    DELTAS_DIR.mkdir(parents=True, exist_ok=True)

    # Load all three narratives
    narratives = {}
    for key in SOURCES:
        path = NARRATIVES_DIR / f"{key}.txt"
        text = path.read_text().strip()
        narratives[key] = text
        color = SOURCES[key]["color"]
        console.print(Panel(text, title=f"{key}", border_style=color))

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

    banner("Ingesting...", "Each narrative's full KV cache computed and stored.")
    db = DeltaDB()

    for key, text in narratives.items():
        source = SOURCES[key]
        token_count = len(tokenizer.encode(text))
        console.print(f"  {key} ({source['name']}, trust={source['trust']}) — {token_count} tokens...")
        
        delta = ingest_text(
            model=model, tokenizer=tokenizer, text=text,
            output_dir=DELTAS_DIR, window_size=500,
            trust=source["trust"], tags=["davos", key],
        )
        delta_dir = delta.path
        wins = list_window_caches(delta_dir)
        kv_size = sum((delta_dir / f"kv_cache_w{w}.pt").stat().st_size
                      for w in wins) / 1024
        
        db.register(
            delta_id=delta.delta_id, delta_type="context",
            path=str(delta_dir),
            base_model=model.config._name_or_path or MODEL_ID,
            source_text=text, trust=source["trust"],
            num_windows=delta.num_windows, tags=["davos", key],
        )

        pdga_size = sum(f.stat().st_size for f in delta_dir.rglob("*")
                        if f.is_file()) / 1024
        console.print("    windows={}  kv_caches={}  kv={:.0f}KB  total={:.0f}KB  crystal=L{}".format(
            delta.num_windows, len(wins), kv_size, pdga_size,
            delta.manifest.crystal_layer))

    db.close()
    console.print("\n[bold green]Ingestion complete.[/bold green] Run: python3 run_davos_demo.py generate")


def do_generate():
    banner("PDGA Davos — Generation Phase",
           "Load KV caches → multi-perspective generation")

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

    paths = sorted(DELTAS_DIR.glob("*.pdga"))
    
    # Map paths to narrative keys by reading metadata
    import json
    narrative_map = {}
    for path in paths:
        meta = json.loads((path / "metadata.json").read_text())
        tags = meta.get("tags", [])
        for key in SOURCES:
            if key in tags:
                narrative_map[key] = path
                break

    # ── Query 1: Full narrative load per source ──────────────────────────
    console.print("\n[bold underline green]── QUERY 1: Per-Source Generation ──[/bold underline green]")
    
    queries = [
        ("What happened at Davos?", "neutral"),
    ]
    
    for query_text, query_type in queries:
        console.print(f"\n[bold]Query:[/bold] {query_text}")
        
        results = []
        t0 = time.perf_counter()
        
        for key, path in narrative_map.items():
            source = SOURCES[key]
            wins = list_window_caches(path)
            win_paths = [path / f"kv_cache_w{w}.pt" for w in wins]
            
            gen = StreamingGenerator(model, win_paths)
            r = gen.generate(
                tokenizer,
                prompt=f"Based on the provided context, answer this question: {query_text}",
                max_new_tokens=300, sample_temp=0.7,
            )
            r["source"] = source["name"]
            r["trust"] = source["trust"]
            r["key"] = key
            results.append(r)
            del gen
            gc.collect()
            torch.cuda.empty_cache()
        
        elapsed = time.perf_counter() - t0
        console.print(f"  Generated {sum(r['num_tokens'] for r in results)} tokens in {elapsed:.1f}s\n")
        
        # Display outputs
        for r in results:
            color = SOURCES[r["key"]]["color"]
            title = Text(f"{r['source']} (trust={r['trust']})", style=color)
            title.append(f"  {r['num_tokens']} tokens  {r['tokens_per_second']:.1f} t/s", style="dim")
            console.print(Panel(title, border_style=color))
            console.print(r["generated_text"][:600])
            console.print()

    # ── Query 2: Cross-source comparison ────────────────────────────────
    console.print("\n[bold underline yellow]── QUERY 2: What Do Sources Agree On? ──[/bold underline yellow]")
    
    # Load all three and generate
    all_paths = []
    for key, path in narrative_map.items():
        wins = list_window_caches(path)
        all_paths.extend([path / f"kv_cache_w{w}.pt" for w in wins])
    
    gen = StreamingGenerator(model, all_paths)
    r = gen.generate(
        tokenizer,
        prompt="Based on all the provided sources, what facts do they all agree on? List only points that appear in multiple sources.",
        max_new_tokens=400, sample_temp=0.7,
    )
    console.print(Panel(f"All sources combined  ({r['num_tokens']} tokens)", border_style="blue"))
    console.print(r["generated_text"])
    console.print()
    del gen
    gc.collect()
    torch.cuda.empty_cache()

    # ── Query 3: Where do they disagree? ──────────────────────────────────
    console.print("\n[bold underline red]── QUERY 3: Where Do Sources Disagree? ──[/bold underline red]")
    
    gen = StreamingGenerator(model, all_paths)
    r = gen.generate(
        tokenizer,
        prompt="Based on all the provided sources, what are the major disagreements or contradictions between the different perspectives?",
        max_new_tokens=400, sample_temp=0.7,
    )
    console.print(Panel(f"Contradictions  ({r['num_tokens']} tokens)", border_style="red"))
    console.print(r["generated_text"])
    console.print()
    del gen
    gc.collect()
    torch.cuda.empty_cache()

    console.print()
    console.print("[bold green]✓[/bold green] Davos multi-perspective generation complete")

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "ingest":
        do_ingest()
    elif cmd == "generate":
        do_generate()
    elif cmd == "all":
        do_ingest()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        do_generate()
