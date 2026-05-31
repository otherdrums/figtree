#!/usr/bin/env python3
"""PDGA Conflicting News Demo — per-delta independent generation with trust."""

import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule

console = Console()

MODEL_ID = "unsloth/Qwen3-4B-bnb-4bit"
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
console.print("[dim]Each delta generates independently at full fidelity (sovereign).[/dim]")
console.print("[dim]Generation via residual stream injection (LARQL Apollo-style).[/dim]")
console.print()
console.print("[bold]Model:[/bold] Qwen3-4B (unsloth bnb-4bit) — 36 layers, hidden_size=2560")
console.print(f"[bold]GPU:[/bold] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
console.print("[bold]Storage:[/bold] 1 boundary residual per window (L23 crystal) + GLiNER entities")
console.print("[bold]Generation:[/bold] 4 modes — replay, hybrid, residuals, inject")

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
    device_map="auto", trust_remote_code=True,
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
console.print(f"[green]Loaded: {model.config.num_hidden_layers} layers, hidden_size={model.config.hidden_size}[/green]")

# ── Step 3: Ingest ──────────────────────────────────────────────────────────
banner("Step 3 — Ingest: Text → ContextDelta (crystal + injection layers detected)")

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
    inj_shape = delta.injection_deltas.shape if delta.injection_deltas is not None else (0, 0)
    console.print(f" [green]δID={delta.delta_id}  "
                  f"windows={delta.num_windows}  "
                  f"crystal=L{delta.manifest.crystal_layer}  "
                  f"inject=L{delta.manifest.injection_layer}  "
                  f"size={size_kb:.1f}KB  "
                  f"deltas={inj_shape}[/green]")

# ── Step 4: Graph ───────────────────────────────────────────────────────────
banner("Step 4 — Graph: Link Deltas")

from pdga.graph.edges import EdgeType, EdgeOps
ops = EdgeOps(db)
ops.add(delta_ids["article_a.txt"], EdgeType.CONTRADICTS, delta_ids["article_b.txt"])
ops.add(delta_ids["article_a.txt"], EdgeType.ABOUT_SAME_EVENT, delta_ids["article_b.txt"])
console.print(f"  {delta_ids['article_a.txt']} --[contradicts]--> {delta_ids['article_b.txt']}")
console.print(f"  {delta_ids['article_a.txt']} --[about_same_event]--> {delta_ids['article_b.txt']}")

# ── Step 5: Show delta structure ────────────────────────────────────────
banner("Step 5 — Delta Structure: Boundaries + Injection Deltas")

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

    console.print(f"\n[bold]{fname} (trust={trust:.0%}) — δID={did}[/bold]")
    table = Table()
    table.add_column("Metric", style="dim"); table.add_column("Value")
    table.add_row("Windows", str(delta.num_windows))
    table.add_row("Boundary residuals", f"( {delta.num_windows}, {delta.manifest.hidden_size} ) f32")
    table.add_row("Injection tokens", f"({delta.num_windows}, {delta.injection_token_ids.shape[1]})" if delta.injection_token_ids is not None else "none")
    table.add_row("Crystal layer", f"L{delta.manifest.crystal_layer}")
    table.add_row("Injection layer", f"L{delta.manifest.injection_layer}")
    table.add_row("PDGA size", f"{sum(f.stat().st_size for f in delta.path.rglob('*') if f.is_file())/1024:.1f}KB")
    table.add_row("Fact chunks", str(len(delta.fact_tokens)) if hasattr(delta, 'fact_tokens') and delta.fact_tokens else "0")
    console.print(table)

    console.print(f"  [dim]Dynamic labels:[/dim] {', '.join(delta.dynamic_labels[:15])}")
    if len(delta.dynamic_labels) > 15:
        console.print(f"  [dim]               [/dim] {', '.join(delta.dynamic_labels[15:])}")

    console.print("\n  [dim]Window injection tokens (boundary residual → top-k embedding matches):[/dim]")
    for w in range(min(delta.num_windows, 8)):
        ctx_ids = delta.get_window_tokens(w)[:30]
        ctx = tokenizer.decode(ctx_ids)
        if delta.injection_token_ids is not None and delta.injection_coefficients is not None:
            tok_ids = delta.injection_token_ids[w]
            coeffs = delta.injection_coefficients[w]
            tokens_str = ", ".join(
                f"'{tokenizer.decode([int(t)])}'({float(c):.2f})"
                for t, c in zip(tok_ids, coeffs)
            )
            console.print(f"    W{w}: tokens=[{tokens_str}]  ctx=[{ctx}...]")
        else:
            d_norm = float(np.linalg.norm(delta.injection_deltas[w])) if delta.injection_deltas is not None else 0.0
            console.print(f"    W{w}: ||residual||={d_norm:.2f}  [{ctx}...]")

# ── Step 6: Generate — three mode comparison ──────────────────────────────
banner("Step 6 — Generate: Injection, Hybrid, and Replay Comparison")

from pdga.kernel.reference import generate as gen_replay, generate_hybrid as gen_hybrid
from pdga.kernel.inject import generate_from_injection as gen_inject
from pdga.kernel.residual_inject import generate_from_residuals as gen_residual

did_a, did_b = delta_ids["article_a.txt"], delta_ids["article_b.txt"]
delta_a = load_delta(Path(db.get(did_a)["path"]))
delta_b = load_delta(Path(db.get(did_b)["path"]))

query = "/no_think What happened at the Global Trade Summit in Geneva? Give the specific details from your source."

console.print(f"\n[bold]Query:[/bold] [yellow]{query}[/yellow]")
console.print("[bold]Prompt to model:[/bold] just the query above — no article text prepended.")

for label, gen_fn, gen_mode in [
    ("RESIDUALS (CUDA direct injection at crystal layer)", gen_residual, "residuals"),
    ("HYBRID (GLiNER fact token chunks)", gen_hybrid, "hybrid"),
    ("REPLAY (full text reference)", gen_replay, "replay"),
    ("INJECT  (token embedding hook)", gen_inject, "inject"),
]:
    console.print(f"\n[bold underline]── {gen_mode.upper()} MODE ──[/bold underline]")

    if gen_mode == "residuals":
        if delta_a.boundaries is not None:
            console.print(f"[dim]Source: {delta_a.boundaries.shape[0] + delta_b.boundaries.shape[0]} boundary residuals injected at L{delta_a.manifest.crystal_layer}[/dim]")
        console.print("[dim]Forward hook + CUDA/Triton kernel adds deltas to residual stream[/dim]")
    elif gen_mode == "inject":
        tok_a = delta_a.injection_token_ids.shape[0] if delta_a.injection_token_ids is not None else 0
        tok_b = delta_b.injection_token_ids.shape[0] if delta_b.injection_token_ids is not None else 0
        console.print(f"[dim]Source: {tok_a + tok_b} windows → GLiNER entity tokens injected at L{delta_a.manifest.injection_layer}[/dim]")
        console.print("[dim]LARQL-style token embedding perturbation via forward pre_hook[/dim]")
    elif gen_mode == "hybrid":
        console.print(f"[dim]Source: {len(delta_a.fact_tokens) + len(delta_b.fact_tokens)} GLiNER fact chunks[/dim]")
    else:
        console.print("[dim]Source: full article token IDs[/dim]")

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
                  deltas_map=deltas_map, max_new_tokens=150, mode="inject")

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
console.print("[bold green]✓[/bold green] Text → Boundary Residuals → GLiNER Entities → .pdga → LSH → Graph → Generate")
console.print()
console.print("[bold]PDGA representation:[/bold]")
console.print("  • 1 boundary residual per window at crystal layer — for LSH retrieval")
console.print("  • Boundary residuals injected directly at crystal layer via CUDA/Triton kernel")
console.print("  • Residuals mode: forward hook + per-position injection (no token replay)")
console.print("  • Hybrid mode: GLiNER fact token chunks prepended to prompt (best compressed)")
console.print("  • Inject mode: token embedding perturbation at injection layer (experimental)")
console.print("  • Replay mode: full token replay (reference gold standard)")
console.print("  • Each delta generates independently (sovereign), trust is metadata only")
console.print()
console.print("[bold]Qwen3-4B results:[/bold] Replay and Hybrid modes produce specific fact recall.")
console.print("  Residuals and Inject modes do not — boundary residual alone insufficient even at 2560 dims.")
console.print("  LARQL's Apollo uses per-token VecInjectEntry with token IDs, not full residual vectors.")
console.print()
console.print("[bold]Credit:[/bold] Boundary residual concept from [bold]chrishayuk/larql[/bold]")
console.print("  PDGA extends LARQL with CUDA/Triton kernel injection, GLiNER entity extraction,")
console.print("  multi-delta graph architecture, and multiple generation modes.")
db.close()
