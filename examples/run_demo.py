#!/usr/bin/env python3
# ruff: noqa: E402
"""PDGA Conflicting News Demo — Apollo residual injection with proper Qwen3 handling."""

import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich.text import Text

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
console.print("[dim]Generation via Apollo residual injection — no KV caching, no text prepending.[/dim]")
console.print()
console.print("[bold]Model:[/bold] Qwen3-4B (unsloth bnb-4bit) — 36 layers, h=2560, head_dim=128")
console.print(f"[bold]GPU:[/bold] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
console.print("[bold]ChatML:[/bold] enable_thinking=False — empty closed <think> block pre-seeded")
console.print("[bold]Generation:[/bold] 3 modes — Apollo (residual injection), Replay (gold), Think (multi-stream)")

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
    MODEL_ID, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map="auto", trust_remote_code=True,
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
console.print(f"[green]Loaded: {model.config.num_hidden_layers} layers, hidden_size={model.config.hidden_size}[/green]")

# ── Step 3: Ingest ──────────────────────────────────────────────────────────
banner("Step 3 — Ingest: Text → ContextDelta (crystal + injection layers)")

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
    raw_kb = len(text.encode()) / 1024
    inj_count = delta.injection_token_ids.shape[1] if delta.injection_token_ids is not None else 0
    inj_total = delta.injection_token_ids.size if delta.injection_token_ids is not None else 0
    console.print(f" [green]δID={delta.delta_id}  "
                  f"windows={delta.num_windows}  "
                  f"crystal=L{delta.manifest.crystal_layer}  "
                  f"inj_tokens={inj_total} (max {inj_count}/window)  "
                  f"size={size_kb:.1f}KB ({raw_kb/size_kb:.0f}x compression)[/green]")

# ── Step 4: Graph ───────────────────────────────────────────────────────────
banner("Step 4 — Graph: Link Deltas")

from pdga.graph.edges import EdgeType, EdgeOps
ops = EdgeOps(db)
ops.add(delta_ids["article_a.txt"], EdgeType.CONTRADICTS, delta_ids["article_b.txt"])
ops.add(delta_ids["article_a.txt"], EdgeType.ABOUT_SAME_EVENT, delta_ids["article_b.txt"])
console.print(f"  {delta_ids['article_a.txt']} --[contradicts]--> {delta_ids['article_b.txt']}")
console.print(f"  {delta_ids['article_a.txt']} --[about_same_event]--> {delta_ids['article_b.txt']}")

# ── Step 5: Show delta structure ────────────────────────────────────────────
banner("Step 5 — Delta Structure: Boundaries + Injection Tokens")

from pdga.delta.io import load_delta

for fname, trust, color in [("article_a.txt", TRUST_A, "green"), ("article_b.txt", TRUST_B, "red")]:
    did = delta_ids[fname]
    delta = load_delta(Path(db.get(did)["path"]))
    raw_kb = len(db.get(did).get("source_text", "").encode()) / 1024
    size_kb = sum(f.stat().st_size for f in delta.path.rglob("*") if f.is_file()) / 1024

    console.print(f"\n[bold]{fname} (trust={trust:.0%}) — δID={did}[/bold]")
    table = Table()
    table.add_column("Metric", style="dim")
    table.add_column("Value")
    table.add_row("Windows", str(delta.num_windows))
    table.add_row("Boundary residuals", f"({delta.num_windows}, {delta.manifest.hidden_size}) f32")
    inj_count = delta.injection_token_ids.shape[1] if delta.injection_token_ids is not None else 0
    inj_total = delta.injection_token_ids.size if delta.injection_token_ids is not None else 0
    table.add_row("Injection tokens", f"{inj_total} ({inj_count}/window max)")
    table.add_row("Crystal layer", f"L{delta.manifest.crystal_layer}")
    table.add_row("Injection layer", f"L{delta.manifest.injection_layer}")
    table.add_row("PDGA size", f"{size_kb:.1f}KB ({raw_kb/size_kb:.0f}x compression)")
    console.print(table)

    if delta.dynamic_labels:
        console.print(f"  [dim]Labels:[/dim] {', '.join(delta.dynamic_labels[:15])}")
        if len(delta.dynamic_labels) > 15:
            console.print(f"  [dim]       [/dim] {', '.join(delta.dynamic_labels[15:])}")

# ── Step 6: Generate — Apollo, Replay, Think ────────────────────────────────
banner("Step 6 — Generate: Apollo Residual Injection vs Replay Gold Standard")

from pdga.kernel.residual_inject import generate_from_residuals
from pdga.kernel.reference import generate as gen_replay

did_a, did_b = delta_ids["article_a.txt"], delta_ids["article_b.txt"]
delta_a = load_delta(Path(db.get(did_a)["path"]))
delta_b = load_delta(Path(db.get(did_b)["path"]))

prompt = "What happened at the Global Trade Summit in Geneva? Give the specific details from your source."

console.print(f"\n[bold]Query:[/bold] [yellow]{prompt}[/yellow]")
console.print("[bold]Prompt format:[/bold] ChatML with enable_thinking=False — no reasoning, direct answer")

# ── Apollo mode ─────────────────────────────────────────────────────────────
console.print("\n[bold underline green]── APOLLO MODE (residual injection, no KV cache) ──[/bold underline green]")
console.print("[dim]Full forward re-run per step. Boundary residual carries all context.[/dim]")
console.print(f"[dim]Crystal layer: L{delta_a.manifest.crystal_layer}  |  Injection coeff: 0.75–3.0[/dim]")

for coeff in [0.75, 1.5, 3.0]:
    console.print(f"\n[bold]── coeff = {coeff} ──[/bold]")
    results = generate_from_residuals(
        model=model, tokenizer=tokenizer, prompt=prompt,
        deltas=[delta_a, delta_b], max_new_tokens=50,
        sample_temp=0.7, injection_coefficient=coeff, use_chat_template=True,
    )
    results.sort(key=lambda r: r["trust"], reverse=True)
    for r in results:
        tp = f"{r['trust']:.0%}"
        tc = "green" if r["trust"] >= 0.8 else "red"
        tps = r.get("tokens_per_second", 0)
        elapsed = r.get("elapsed", 0)
        tokens = r.get("num_tokens", 0)
        title = Text(f"{r['delta_id']}  trust={tp}  ", style=tc)
        title.append(f"{tps:.1f}t/s  {tokens}tok  {elapsed:.1f}s", style="dim")
        console.print(Panel(r["generated_text"].strip() or "(empty)",
                            title=title, border_style=tc))

# ── Replay mode ─────────────────────────────────────────────────────────────
console.print("\n[bold underline blue]── REPLAY MODE (full-text gold standard) ──[/bold underline blue]")
console.print("[dim]Full article tokens prepended. Shows what the model can do with full context.[/dim]")

results_r = gen_replay(
    model=model, tokenizer=tokenizer, prompt=prompt,
    deltas=[delta_a, delta_b], max_new_tokens=60, sample_temp=0.7,
)
results_r.sort(key=lambda r: r["trust"], reverse=True)
for r in results_r:
    tp = f"{r['trust']:.0%}"
    tc = "green" if r["trust"] >= 0.8 else "red"
    title = Text(f"{r['delta_id']}  trust={tp}", style=tc)
    console.print(Panel(r["generated_text"].strip() or "(empty)",
                        title=title, border_style=tc))

# ── Step 7: Multi-stream Think ──────────────────────────────────────────────
banner("Step 7 — Think: Multi-Stream Apollo Generation")

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
                  prompt=prompt, streams=streams,
                  deltas_map=deltas_map, max_new_tokens=80, mode="residuals")

for r in result.streams:
    label = "CONSCIOUS" if r.is_conscious else "subconscious"
    style = "bold green" if r.is_conscious else "dim"
    console.print(f"\n[{style}]──── {label}: {r.stream_id} (τ={r.sample_temp}) ────[/{style}]")
    for dr in r.delta_results:
        tp = f"{dr['trust']:.0%}"
        tc = "green" if float(dr["trust"]) >= 0.8 else "red"
        title = Text(f"{dr['delta_id']}  trust={tp}", style=tc)
        console.print(Panel(dr["generated_text"].strip() or "(empty)",
                            title=title, border_style=tc))

# ── Done ────────────────────────────────────────────────────────────────────
banner("Pipeline Complete")
console.print("[bold green]✓[/bold green] Text → Boundaries + GLiNER → .pdga → Apollo Engine → Generate")
console.print()
console.print("[bold]Architecture:[/bold]")
console.print("  • 1 boundary residual per window at crystal layer (L23) — carries full context")
console.print("  • GLiNER entity token embeddings as injection delta — biases output toward facts")
console.print("  • Apollo forward: dummy→boundary swap at crystal, causal SDPA, all-layers forward")
console.print("  • Full forward re-run at each decode step — no KV cache")
console.print("  • O(N × seq_len²) complexity — ~450ms/step on Quadro T1000 for Qwen3-4B")
console.print()
console.print("[bold]What works:[/bold]")
console.print("  • [green]Apollo[/green] — boundary swap + injection produces factual claims (correct location, trade context)")
console.print("  • [green]Replay[/green] — full-text reference, confirms model capability with full context")
console.print("  • [green]Think[/green] — multi-stream generation with different temperatures")
console.print()
console.print("[bold]What was dropped:[/bold]")
console.print("  • [red]Hybrid[/red] — GLiNER chunks produce placeholder responses on Qwen3")
console.print("  • [red]Inject[/red] — token embedding hook never achieved factual recall")
console.print("  • [red]Residuals v1[/red] — bidirectional attention caused representation collapse")
console.print()
console.print("[bold]Performance:[/bold]")
console.print("  • ~2 t/s for early steps, ~1 t/s for later steps (seq_len grows)")
console.print("  • C++/CUDA Apollo engine (pdga/apollo/) is the path to production speed")
console.print("  • GPTQ/AWQ 4-bit matmul kernels + fused attention can achieve 3-5 t/s")
console.print()
console.print("[bold]Credit:[/bold] Boundary residual concept from [bold]chrishayuk/larql[/bold]")
console.print("  PDGA extends LARQL with causal-attention forward, GLiNER entity extraction,")
console.print("  multi-delta graph architecture, CUDA injection kernels, and ChatML integration.")
db.close()
