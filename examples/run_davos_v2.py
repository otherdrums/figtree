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
from rich.rule import Rule

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
    from datetime import datetime

    banner("Figtree Davos v2 — Generation", "On-the-fly KV from figment text")
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(__file__).parent / f"davos_run_{run_ts}.log"

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

    # -- Source narratives --
    banner("Input Narratives", "Full source texts used in this run")
    source_texts = {}
    for key in SOURCES:
        text = (NARRATIVES_DIR / f"{key}.txt").read_text().strip()
        source_texts[key] = text
        source = SOURCES[key]
        color = source["color"]
        console.print(f"\n[bold {color}]── {key} ({source['name']}, trust={source['trust']}) ──[/bold {color}]")
        console.print(text)

    # -- Load figments --
    source_figments: dict[str, list[Figment]] = {}
    for key in SOURCES:
        figment_dirs = sorted((FIGMENTS_DIR / key).glob("*.figment"))
        figments = [Figment.load(d) for d in figment_dirs]
        atomic = [f for f in figments if not f.is_image() and not f.is_trust_assertion()]
        source_figments[key] = atomic

    all_figments = []
    for figments in source_figments.values():
        all_figments.extend(figments)

    # -- Log file header --
    with open(log_path, "w") as f:
        f.write("=" * 72 + "\n")
        f.write("Figtree Davos v2 — Generation Log\n")
        f.write(f"Run: {datetime.now().isoformat()}\n")
        f.write("=" * 72 + "\n\n")
        for key in SOURCES:
            source = SOURCES[key]
            f.write(f"── {key} ({source['name']}, trust={source['trust']}) ──\n")
            f.write(source_texts[key] + "\n\n")
        f.write("\n")

    # -- Helper: run + log a query --
    def run_query(label: str, figments: list[Figment], prompt: str, max_new_tokens: int = 150):
        console.print(f"\n[bold]Prompt:[/bold] {prompt}")
        t0 = time.perf_counter()
        result = gen.generate(
            figments=figments, prompt=prompt, max_new_tokens=max_new_tokens,
        )
        elapsed = time.perf_counter() - t0
        ntok = result["num_tokens"]
        console.print(f"\n[bold]Output ({ntok} tokens, {elapsed:.1f}s):[/bold]")
        console.print(result["generated_text"])
        console.print()

        with open(log_path, "a") as f:
            f.write(f"── {label} ──\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Tokens: {ntok}, Elapsed: {elapsed:.1f}s\n")
            f.write(result["generated_text"] + "\n\n")

        gc.collect()
        torch.cuda.empty_cache()

    # -- QUERY 1: Per-Source --
    console.print("\n[bold underline green]── QUERY 1: Per-Source Generation ──[/bold underline green]")
    for key, figments in source_figments.items():
        source = SOURCES[key]
        console.print(f"\n[bold {source['color']}]── {source['name']} (trust={source['trust']}) ──[/bold {source['color']}]")
        run_query(source["name"], figments, "What happened at Davos?", max_new_tokens=150)

    # -- QUERY 2: Agreement --
    console.print("\n[bold underline yellow]── QUERY 2: What Do Sources Agree On? ──[/bold underline yellow]")
    run_query("Agreement", all_figments,
              "Based on all sources, what facts do they agree on?", max_new_tokens=200)

    # -- QUERY 3: Disagreement --
    console.print("\n[bold underline red]── QUERY 3: Where Do Sources Disagree? ──[/bold underline red]")
    run_query("Disagreement", all_figments,
              "What are the major disagreements between the different perspectives?",
              max_new_tokens=200)

    console.print("\n[bold green]Figtree Davos v2 generation complete[/bold green]")
    console.print(f"[dim]Log saved to: {log_path}[/dim]")


def do_graph():
    from datetime import datetime

    banner("Figtree Davos v2 — Graph", "Deduplication + edges + trust propagation")
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(__file__).parent / f"davos_graph_{run_ts}.log"

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

    top = graph.get_top_figments(10)
    console.print("\n[bold underline green]── Top Trusted Figments ──[/bold underline green]")
    for i, f in enumerate(top, 1):
        console.print(f"\n[bold]{i}.[/bold] trust={f.trust:.2f}  id={f.figment_id}")
        console.print(f"  {f.text}")
    console.print()

    with open(log_path, "w") as f:
        f.write("=" * 72 + "\n")
        f.write("Figtree Davos v2 — Graph Log\n")
        f.write(f"Run: {datetime.now().isoformat()}\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Figments: {len(all_figments)}\n")
        f.write(f"Dedup edges: {len(dedup_edges)}\n")
        f.write(f"Auto edges: {len(auto_edges)}\n")
        f.write(f"Trust scores updated: {len(trust_scores)}\n\n")
        f.write("── Top Trusted Figments ──\n")
        for i, fig in enumerate(top, 1):
            f.write(f"\n{i}. trust={fig.trust:.2f}  id={fig.figment_id}\n")
            f.write(f"   {fig.text}\n")
        f.write("\n")

    console.print(f"[bold green]Graph complete[/bold green]")
    console.print(f"[dim]Log saved to: {log_path}[/dim]")


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
