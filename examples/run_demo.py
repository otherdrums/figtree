#!/usr/bin/env python3
"""PDGA Conflicting News Demo — per-delta independent generation with trust."""

import os, shutil, sqlite3
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule

console = Console()

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PDGA_HOME = Path.home() / ".pdga"
DELTAS_DIR = Path(__file__).parent.parent / "deltas"
EXAMPLES_DIR = Path(__file__).parent / "conflicting_news"
TRUST_A, TRUST_B = 0.99, 0.50


def banner(title: str):
    console.print()
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]"))


# ── Step 0: Clean setup ─────────────────────────────────────────────────────
banner("PDGA — Parallel Delta Graph Architecture — Demo")
console.print("[dim]Two conflicting news articles stored as ContextDeltas.[/dim]")
console.print("[dim]Each delta generates independently at full fidelity.[/dim]")
console.print()
console.print(f"[bold]Model:[/bold] {MODEL_ID}")
console.print(f"[bold]GPU:[/bold] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
console.print(f"[bold]Storage:[/bold] Sparse boundary residuals (novelty-gated, ~8-12% of token positions)")
console.print(f"[bold]Retrieval:[/bold] Cosine-LSH over boundary residual vectors")
console.print(f"[bold]Generation:[/bold] Token replay → KV cache (reference); boundary-residual seeding (CUDA path)")

if PDGA_HOME.exists():
    shutil.rmtree(PDGA_HOME)
if DELTAS_DIR.exists():
    shutil.rmtree(DELTAS_DIR)
DELTAS_DIR.mkdir(parents=True, exist_ok=True)

# ── Step 1: Show input texts ────────────────────────────────────────────────
banner("Step 1 — Input: Two Conflicting News Articles")

for fname, trust, color in [
    ("article_a.txt", TRUST_A, "green"),
    ("article_b.txt", TRUST_B, "red"),
]:
    text = (EXAMPLES_DIR / fname).read_text()
    label = f"Article {'A' if 'a' in fname else 'B'} — trust={trust:.0%}"
    console.print(Panel(text.strip(), title=label, border_style=color))

# ── Step 2: Load model ──────────────────────────────────────────────────────
banner("Step 2 — Load Model")
device = "cuda" if torch.cuda.is_available() else "cpu"
console.print(f"[dim]{MODEL_ID} on {device}[/dim]")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map=device, trust_remote_code=True,
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
console.print(f"[green]Loaded: {model.config.num_hidden_layers} layers, hidden_size={model.config.hidden_size}[/green]")

# ── Step 3: Ingest ──────────────────────────────────────────────────────────
banner("Step 3 — Ingest: Text → ContextDelta (crystal layer auto-detected)")

from pdga.ingest.text import ingest_text
from pdga.db.store import DeltaDB
from pdga.retrieval.lsh import create_lsh_for_model

db = DeltaDB()
lsh = create_lsh_for_model(model.config.hidden_size)
delta_ids = {}

for fname, trust in [("article_a.txt", TRUST_A), ("article_b.txt", TRUST_B)]:
    text = (EXAMPLES_DIR / fname).read_text()
    console.print(f"  Ingesting {fname}...", end="")
    delta = ingest_text(
        model=model, tokenizer=tokenizer, text=text,
        output_dir=DELTAS_DIR, window_size=200, novel_fraction=0.05,
        trust=trust, tags=["summit_event"],
    )
    delta_ids[fname] = delta.delta_id
    db.register(
        delta_id=delta.delta_id, delta_type="context",
        path=str(delta.path),
        base_model=model.config._name_or_path or MODEL_ID,
        source_text=text, trust=trust,
        num_windows=delta.num_windows, tags=["summit_event"],
    )
    lsh.insert(delta.delta_id, delta.boundaries)
    size_kb = sum(f.stat().st_size for f in delta.path.rglob("*") if f.is_file()) / 1024
    console.print(f" [green]δID={delta.delta_id}  "
                  f"boundaries={delta.boundaries.shape[0]}  "
                  f"crystal=L{delta.manifest.crystal_layer}  "
                  f"size={size_kb:.1f}KB[/green]")

# ── Step 4: Graph ───────────────────────────────────────────────────────────
banner("Step 4 — Graph: Link Deltas")

from pdga.graph.edges import EdgeType, EdgeOps
ops = EdgeOps(db)
ops.add(delta_ids["article_a.txt"], EdgeType.CONTRADICTS, delta_ids["article_b.txt"])
ops.add(delta_ids["article_a.txt"], EdgeType.ABOUT_SAME_EVENT, delta_ids["article_b.txt"])
console.print(f"  {delta_ids['article_a.txt']} --[contradicts]--> {delta_ids['article_b.txt']}")
console.print(f"  {delta_ids['article_a.txt']} --[about_same_event]--> {delta_ids['article_b.txt']}")

# ── Step 5: Show delta contents with novel token mapping ──────────────────
banner("Step 5 — Delta Contents: Novelty-Gated Residuals + Token Mapping")

import numpy as np
from pdga.delta.io import load_delta

for fname, trust, color in [("article_a.txt", TRUST_A, "green"), ("article_b.txt", TRUST_B, "red")]:
    did = delta_ids[fname]
    delta = load_delta(Path(db.get(did)["path"]))
    text = db.get(did)["source_text"]
    all_ids = []
    for w in range(delta.num_windows):
        all_ids.extend(delta.get_window_tokens(w))
    total = len(all_ids)
    novel = delta.boundaries.shape[0]

    console.print(f"\n[bold]{fname} (trust={trust:.0%}) — δID={did}[/bold]")
    table = Table()
    table.add_column("Metric", style="dim"); table.add_column("Value")
    table.add_row("Novel positions", f"[green]{novel}[/green] / {total} ([bold]{novel/max(total,1)*100:.1f}%[/bold])")
    table.add_row("Boundary vectors", f"( {novel}, , {delta.manifest.hidden_size} ) f32")
    table.add_row("Crystal layer", f"L{delta.manifest.crystal_layer}")
    table.add_row("PDA-size", f"{sum(f.stat().st_size for f in delta.path.rglob('*') if f.is_file())/1024:.1f}KB")
    table.add_row("Fact chunks", str(len(delta.fact_tokens)) if hasattr(delta, 'fact_tokens') and delta.fact_tokens else "0")
    console.print(table)

    console.print(f"  [dim]Dynamic labels:[/dim] {', '.join(delta.dynamic_labels[:15])}")
    if len(delta.dynamic_labels) > 15:
        console.print(f"  [dim]               [/dim] {', '.join(delta.dynamic_labels[15:])}")

    # Show what the novel positions correspond to in the text
    console.print("\n  [dim]Top 8 novel positions mapped to text:[/dim]")
    for rank, pos in enumerate(delta.boundary_positions[:8]):
        if pos < len(all_ids):
            ctx_ids = all_ids[max(0,pos-2):pos+5]
            ctx = tokenizer.decode(ctx_ids)
            console.print(f"    pos={pos:3d}  [{ctx}]")

# ── Step 6: Generate — three mode comparison ──────────────────────────────
banner("Step 6 — Generate: Residual, Replay, and Hybrid Comparison")

from pdga.kernel.reference import (
    generate as gen_replay,
    generate_from_residuals as gen_residual,
    generate_hybrid as gen_hybrid,
)

did_a, did_b = delta_ids["article_a.txt"], delta_ids["article_b.txt"]
delta_a = load_delta(Path(db.get(did_a)["path"]))
delta_b = load_delta(Path(db.get(did_b)["path"]))

query = "What happened at the Global Trade Summit in Geneva? Give the specific details from your source."

console.print(f"\n[bold]Query:[/bold] [yellow]{query}[/yellow]")
console.print(f"[bold]Prompt to model:[/bold] just the query above — no article text prepended.")

for label, gen_fn, gen_mode in [
    ("RESIDUAL (boundary-only)", gen_residual, "residual"),
    ("HYBRID (fact tokens + residuals)", gen_hybrid, "hybrid"),
    ("REPLAY (full text, reference)", gen_replay, "replay"),
]:
    console.print(f"\n[bold underline]── {gen_mode.upper()} MODE ──[/bold underline]")

    if gen_mode == "residual":
        console.print(f"[dim]Source: {delta_a.boundaries.shape[0] + delta_b.boundaries.shape[0]} boundary residual vectors only[/dim]")
    elif gen_mode == "hybrid":
        console.print(f"[dim]Source: {len(delta_a.fact_tokens) + len(delta_b.fact_tokens)} GLiNER fact chunks "
                      f"+ {delta_a.boundaries.shape[0] + delta_b.boundaries.shape[0]} boundary residuals[/dim]")
        console.print(f"[dim]Dynamic labels: {', '.join(delta_a.dynamic_labels[:12])}...[/dim]")
    else:
        console.print("[dim]Source: full article token IDs (all 372 tokens)[/dim]")

    results = gen_fn(model=model, tokenizer=tokenizer, prompt=query,
                     deltas=[delta_a, delta_b], max_new_tokens=150, sample_temp=0.7)

    results.sort(key=lambda r: r["trust"], reverse=True)
    for r in results:
        tp = f"{r['trust']:.0%}"
        tc = "green" if r["trust"] >= 0.8 else "red"
        title = f"{r['delta_id']}  trust: [{tc}]{tp}[/{tc}]"
        console.print(Panel(r["generated_text"].strip() or "(empty)", title=title, border_style=tc))

# ── Step 7: Multi-stream ────────────────────────────────────────────────────
banner("Step 7 — Think: Multi-Stream Parallel Generation")

from pdga.kernel.stream import StreamConfig
from pdga.kernel.gather import think as think_fn

deltas_map = {did_a: delta_a, did_b: delta_b}
streams = [
    StreamConfig(id="conscious", delta_ids=[did_a, did_b],
                 delta_temps={}, sample_temp=0.5, conscious=True),
    StreamConfig(id="explore", delta_ids=[did_a, did_b],
                 delta_temps={}, sample_temp=0.9),
]

result = think_fn(model=model, tokenizer=tokenizer,
                  prompt=query, streams=streams,
                  deltas_map=deltas_map, max_new_tokens=150)

for r in result.streams:
    label = "CONSCIOUS" if r.is_conscious else "subconscious"
    style = "bold green" if r.is_conscious else "dim"
    console.print(f"\n[{style}]──── {label}: {r.stream_id} (τ={r.sample_temp}) ────[/{style}]")
    for dr in r.delta_results:
        tp = f"{dr['trust']:.0%}"
        tc = "green" if float(dr["trust"]) >= 0.8 else "red"
        title = f"{dr['delta_id']}  trust: [{tc}]{tp}[/{tc}]"
        console.print(Panel(dr["generated_text"].strip() or "(empty)",
                            title=title, border_style=tc))

# ── Done ────────────────────────────────────────────────────────────────────
banner("Pipeline Complete")
console.print("[bold green]✓[/bold green] Text → Boundary Residuals → .pdga → LSH → Graph → Generate")
console.print()
console.print("[bold]PDGA representation:[/bold]")
console.print("  • Sparse boundary residuals at crystal layer (L18) — novelty-gated")
console.print("  • Pairwise cosine distance identifies positions the model can't reproduce")
console.print(f"  • Only ~{delta_a.boundaries.shape[0]/sum(len(t) for t in delta_a.window_tokens)*100:.0f}% of token positions stored as residual vectors")
console.print("  • Window token IDs stored for LSH routing, NOT for generation")
console.print("  • Original text stored only in DB metadata ([bold]pdga show[/bold]), never sent to model")
console.print()
console.print("[bold]Credit:[/bold] Boundary residual concept from [bold]chrishayuk/larql[/bold]")
console.print("  LARQL's Apollo engine demonstrated that a single residual vector at the crystal")
console.print("  layer encodes sufficient context for fact retrieval (Apollo 11 transcript demo).")
console.print("  PDGA extends this with GLiNER-based entity extraction, model-driven semantic")
console.print("  spans, and a parallel delta graph architecture.")
db.close()
