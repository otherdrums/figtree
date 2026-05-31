#!/usr/bin/env python3
# ruff: noqa: E402
"""PDGA Conflicting News Demo — multi-delta generation with factual recall."""

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
    console.print(Rule(f"[bold dark_cyan]{title}[/bold dark_cyan]"))


# ── Step 0 ────────────────────────────────────────────────────────────────────
banner("PDGA — Parallel Delta Graph Architecture — Demo")
console.print("[dim]Two conflicting news articles → ContextDeltas → multi-delta generation.[/dim]")
console.print("[dim]Each delta is sovereign — generates independently at full fidelity.[/dim]")
console.print()
console.print("[bold]Model:[/bold] Qwen3-4B (unsloth bnb-4bit) — 36 layers, h=2560, head_dim=128")
console.print(f"[bold]GPU:[/bold] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

if PDGA_HOME.exists():
    shutil.rmtree(PDGA_HOME)
if DELTAS_DIR.exists():
    shutil.rmtree(DELTAS_DIR)
DELTAS_DIR.mkdir(parents=True, exist_ok=True)

# ── Step 1 ────────────────────────────────────────────────────────────────────
banner("Step 1 — Input: Two Conflicting News Articles")

for fname, trust, color in [
    ("article_a.txt", TRUST_A, "green"),
    ("article_b.txt", TRUST_B, "red"),
]:
    text = (EXAMPLES_DIR / fname).read_text()
    label = f"Article {'A' if 'a' in fname else 'B'} — trust={trust:.0%}"
    console.print(Panel(text.strip(), title=label, border_style=color))

# ── Step 2 ────────────────────────────────────────────────────────────────────
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

# ── Step 3 ────────────────────────────────────────────────────────────────────
banner("Step 3 — Ingest: Text → ContextDelta")

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
    inj_total = delta.injection_token_ids.size if delta.injection_token_ids is not None else 0
    console.print(f" [green]δID={delta.delta_id}  "
                  f"windows={delta.num_windows}  "
                  f"crystal=L{delta.manifest.crystal_layer}  "
                  f"inj_tokens={inj_total}  "
                  f"size={size_kb:.1f}KB[/green]")

# ── Step 4 ────────────────────────────────────────────────────────────────────
banner("Step 4 — Graph: Link Deltas")

from pdga.graph.edges import EdgeType, EdgeOps
ops = EdgeOps(db)
ops.add(delta_ids["article_a.txt"], EdgeType.CONTRADICTS, delta_ids["article_b.txt"])
ops.add(delta_ids["article_a.txt"], EdgeType.ABOUT_SAME_EVENT, delta_ids["article_b.txt"])
console.print(f"  {delta_ids['article_a.txt']} --[contradicts]--> {delta_ids['article_b.txt']}")
console.print(f"  {delta_ids['article_a.txt']} --[about_same_event]--> {delta_ids['article_b.txt']}")

# ── Step 5 ────────────────────────────────────────────────────────────────────
banner("Step 5 — Delta Structure: Windows + Boundaries + Injection Entries")

from pdga.delta.io import load_delta

for fname, trust, color in [("article_a.txt", TRUST_A, "green"), ("article_b.txt", TRUST_B, "red")]:
    did = delta_ids[fname]
    delta = load_delta(Path(db.get(did)["path"]))
    size_kb = sum(f.stat().st_size for f in delta.path.rglob("*") if f.is_file()) / 1024

    console.print(f"\n[bold]{fname} (trust={trust:.0%}) — δID={did}[/bold]")
    table = Table()
    table.add_column("Metric", style="dim")
    table.add_column("Value")
    table.add_row("Windows", str(delta.num_windows))
    table.add_row("Boundary residuals", f"({delta.num_windows}, {delta.manifest.hidden_size}) f32")
    inj_total = delta.injection_token_ids.size if delta.injection_token_ids is not None else 0
    table.add_row("Injection entries", str(inj_total))
    table.add_row("Crystal layer", f"L{delta.manifest.crystal_layer}")
    table.add_row("Injection layer", f"L{delta.manifest.injection_layer}")
    table.add_row("PDGA size", f"{size_kb:.1f}KB")
    console.print(table)

    if delta.dynamic_labels:
        labels = delta.dynamic_labels
        console.print(f"  [dim]GLiNER labels:[/dim] {', '.join(labels[:12])}")
        for i in range(12, len(labels), 12):
            console.print(f"  [dim]              [/dim] {', '.join(labels[i:i+12])}")

# ── Step 6 ────────────────────────────────────────────────────────────────────
banner("Step 6 — Generate: Corrected Injection vs Replay Gold Standard")

from pdga.kernel.corrected import generate_multi_corrected
from pdga.kernel.reference import generate as gen_replay

did_a, did_b = delta_ids["article_a.txt"], delta_ids["article_b.txt"]
delta_a = load_delta(Path(db.get(did_a)["path"]))
delta_b = load_delta(Path(db.get(did_b)["path"]))

# Use the auto-detected injection layer from the manifest
inj_a = delta_a.manifest.injection_layer
inj_b = delta_b.manifest.injection_layer
# Both articles detected the same layer (same model), take the consensus
inj_layer = inj_a if inj_a == inj_b else max(inj_a, inj_b)

prompt = "What happened at the Global Trade Summit in Geneva? Give the specific details from your source."

console.print(f"\n[bold]Query:[/bold] [yellow]{prompt}[/yellow]")
console.print(f"[bold]Injection layer:[/bold] [bold green]L{inj_layer}[/bold green] — auto-detected via residual compression analysis")
console.print(f"[dim]  Article A detected L{inj_a}, Article B detected L{inj_b}[/dim]")

# ── Corrected injection mode ──────────────────────────────────────────────────
console.print(f"\n[bold underline green]── CORRECTED INJECTION (L{inj_layer}, coeff=10.0, top-K=8, 120 tokens) ──[/bold underline green]")
console.print("[dim]Window tokens + prompt as context.  Entry routing by query-window token overlap.[/dim]")
console.print("[dim]Injection at L{inj_layer}: sum of top-8 entry embeddings × 10.0 added to residual stream.[/dim]")

results_c = generate_multi_corrected(
    model=model, tokenizer=tokenizer, prompt=prompt,
    deltas=[delta_a, delta_b], max_new_tokens=120,
    sample_temp=0.7, injection_layer=inj_layer,
    injection_coefficient=10.0, injection_topk=8,
)
results_c.sort(key=lambda r: r["trust"], reverse=True)
for r in results_c:
    tp = f"{r['trust']:.0%}"
    tc = "green" if r["trust"] >= 0.8 else "red"
    tps = r.get("tokens_per_second", 0)
    elapsed = r.get("elapsed", 0)
    tokens = r.get("num_tokens", 0)
    entries = r.get("entries_injected", 0)
    title = Text(f"{r['delta_id']}  trust={tp}  ", style=tc)
    title.append(f"{tps:.1f}t/s  {tokens}tok  {entries} entries  {elapsed:.0f}s", style="dim")
    console.print(Panel(r["generated_text"].strip() or "(empty)", title=title, border_style=tc))

# ── Replay mode ───────────────────────────────────────────────────────────────
console.print("\n[bold underline blue]── REPLAY (full-text gold standard, 120 tokens) ──[/bold underline blue]")
console.print("[dim]Full article tokens prepended.  Confirms model can recall all specifics.[/dim]")

results_r = gen_replay(
    model=model, tokenizer=tokenizer, prompt=prompt,
    deltas=[delta_a, delta_b], max_new_tokens=120, sample_temp=0.7,
)
results_r.sort(key=lambda r: r["trust"], reverse=True)
for r in results_r:
    tp = f"{r['trust']:.0%}"
    tc = "green" if r["trust"] >= 0.8 else "red"
    title = Text(f"{r['delta_id']}  trust={tp}", style=tc)
    text = r["generated_text"].strip() or "(empty)"
    console.print(Panel(text, title=title, border_style=tc))

# ── Done ──────────────────────────────────────────────────────────────────────
banner("Pipeline Complete")
console.print("[bold green]✓[/bold green] Text → Boundaries + GLiNER → .pdga → Corrected Injection → Generate")
console.print()
console.print("[bold]How it works:[/bold]")
console.print(f"  • Crystal layer L{delta_a.manifest.crystal_layer}: where boundary residuals stabilise")
console.print(f"  • Injection layer L{inj_layer}: where residual compression peaks —")
console.print("    model has maximally organised the article's facts but hasn't converged to generic output yet")
console.print("  • Detect: run 2 article windows + 1 calibration text through all layers,")
console.print("    find peak of within-article compression ÷ cross-context separation")
console.print("  • Inject: top-8 entry embeddings (routed by query-window token overlap) × 10.0")
console.print("  • Context: matched window tokens (200) + prompt tokens ≈ 230 tokens")
console.print("  • KV cache: window+prompt prefilled once per delta, decode steps use cache")
console.print()
console.print("[bold]Credit:[/bold] Boundary residual concept from [bold]chrishayuk/larql[/bold]")
db.close()
