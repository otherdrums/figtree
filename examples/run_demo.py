#!/usr/bin/env python3
"""PDGA Boundary-KV Demo — two-phase pipeline.

Phase 1 (ingest):  python3 run_demo.py ingest
Phase 2 (generate): python3 run_demo.py generate

Two separate processes ensure clean GPU memory for generation.
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
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich.text import Text

from pdga.ingest.text import ingest_text
from pdga.db.store import DeltaDB
from pdga.delta.cache_io import list_window_caches
from pdga.generation.streaming import StreamingGenerator

console = Console()
MODEL_ID = "unsloth/Qwen3-4B-bnb-4bit"
PDGA_HOME = Path.home() / ".pdga"
DELTAS_DIR = Path(__file__).parent / "deltas"
EXAMPLES_DIR = Path(__file__).parent / "conflicting_news"


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
    banner("PDGA Boundary-KV — Ingestion Phase",
           "Text → ContextDelta + full-article KV cache")
    console.print("[bold]Model:[/bold] Qwen3-4B (36L, h=2560)")
    console.print("[bold]GPU:[/bold] " +
                  (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))

    if PDGA_HOME.exists():
        shutil.rmtree(PDGA_HOME)
    if DELTAS_DIR.exists():
        shutil.rmtree(DELTAS_DIR)
    DELTAS_DIR.mkdir(parents=True, exist_ok=True)

    article_a = (EXAMPLES_DIR / "article_a.txt").read_text().strip()
    article_b = (EXAMPLES_DIR / "article_b.txt").read_text().strip()

    for name, text, color in [
        ("Article A (pro-deal)", article_a, "green"),
        ("Article B (skeptical)", article_b, "red"),
    ]:
        console.print(Panel(text, title=name, border_style=color))

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

    banner("Ingesting...", "Each article's full KV cache computed and stored.")
    db = DeltaDB()

    for name, text, trust in [("a", article_a, 0.99), ("b", article_b, 0.50)]:
        token_count = len(tokenizer.encode(text))
        console.print(f"  Article {name.upper()} ({token_count} tokens)...")
        delta = ingest_text(
            model=model, tokenizer=tokenizer, text=text,
            output_dir=DELTAS_DIR, window_size=500,
            trust=trust, tags=["summit_event"],
        )
        delta_dir = delta.path
        wins = list_window_caches(delta_dir)
        kv_size = sum((delta_dir / "kv_cache_w{}.pt".format(w)).stat().st_size
                      for w in wins) / 1024
        db.register(
            delta_id=delta.delta_id, delta_type="context",
            path=str(delta_dir),
            base_model=model.config._name_or_path or MODEL_ID,
            source_text=text, trust=trust,
            num_windows=delta.num_windows, tags=["summit_event"],
        )

        pdga_size = sum(f.stat().st_size for f in delta_dir.rglob("*")
                        if f.is_file()) / 1024
        console.print("    windows={}  kv_caches={}  kv={:.0f}KB  total={:.0f}KB  crystal=L{}".format(
            delta.num_windows, len(wins), kv_size, pdga_size,
            delta.manifest.crystal_layer))

    # Register with DB (used by LSH retrieval later)
    db.close()
    console.print("\n[bold green]Ingestion complete.[/bold green] Run: python3 run_demo.py generate")


def do_generate():
    banner("PDGA Boundary-KV — Generation Phase",
           "Load KV caches from disk → instant prefill → multi-delta decode")

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

    console.print("\n[bold underline green]── STREAMING GENERATION (full article, 600 tokens) ──[/bold underline green]")
    console.print("[dim]Full article KV cache in system RAM → progressive GPU loading → SDPA[/dim]")

    gc.collect()
    torch.cuda.empty_cache()

    # Streaming generation: full article context via progressive KV loading
    paths = sorted(DELTAS_DIR.glob("*.pdga"))
    wa = list_window_caches(paths[0])[:1]
    wb = list_window_caches(paths[1])[:1]

    results = []
    t0 = time.perf_counter()
    for i, (path, wins) in enumerate([(paths[0], wa), (paths[1], wb)]):
        kv_path = path / f"kv_cache_w{wins[0]}.pt"
        gen = StreamingGenerator(model, kv_path)
        r = gen.generate(
            tokenizer,
            prompt="Quote verbatim every sentence from the text that contains a name, number, dollar amount, percentage, or location. Do not summarize or rephrase — copy the exact words from the source text:",
            max_new_tokens=600, sample_temp=0.7,
        )
        results.append(r)
        del gen
        gc.collect()
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    total_tok = sum(r["num_tokens"] for r in results)
    console.print(f"  Generated: {total_tok} tokens in {elapsed:.1f}s ({total_tok/elapsed:.1f} t/s)\n")

    # Display output
    for i, r in enumerate(results):
        label = "Article A (pro-deal)" if i == 0 else "Article B (skeptical)"
        trust = "99%" if i == 0 else "50%"
        tc = "green" if i == 0 else "red"
        title = Text("{} — trust={}".format(label, trust), style=tc)
        title.append("  {} tokens  {:.1f} t/s".format(
            r["num_tokens"], r["tokens_per_second"]), style="dim")
        console.print(Panel(title, border_style=tc))
        console.print(r["generated_text"])
        console.print()

    # Fact check
    text_a = results[0]["generated_text"]
    text_b = results[1]["generated_text"]

    facts_a = [
        "47 nations", "landmark", "turning point", "multilateral cooperation",
        "Maria Okonkwo", "digital services", "3%", "carbon tariffs",
        "$45", "pharmaceutical", "20 to 12", "$12 billion",
        "climate adaptation", "Brussels", "$900 billion", "global GDP",
        "Sarah Chen", "shared prosperity", "S&P", "1.8%",
    ]
    facts_b = [
        "United States", "China", "walked out", "$900 billion",
        "Maria Okonkwo", "$45", "S&P", "1.7%",
        "digital taxation", "carbon tariffs", "pharmaceutical",
        "IMF", "Global Trade Summit", "Geneva",
    ]

    for label, txt, facts in [("Article A", text_a, facts_a),
                               ("Article B", text_b, facts_b)]:
        hits, found = check_facts(txt, facts)
        table = Table(title="{} — {}/{} facts".format(label, hits, len(facts)))
        table.add_column("Fact", style="dim")
        table.add_column("Found")
        rows = sorted(zip(facts, found), key=lambda x: (not x[1], x[0]))
        for f, ok in rows:
            table.add_row(f, "[green]✓[/green]" if ok else "[red]✗[/red]")
        console.print(table)
        console.print()

    # Sovereignty check
    a_only = {"47 nations", "$12 billion", "Brussels", "1.8%",
               "shared prosperity", "multilateral cooperation",
               "landmark", "turning point", "Sarah Chen", "climate adaptation"}
    b_only = {"walked out", "1.7%", "IMF"}

    contam_a = [f for f in b_only if f.lower() in text_a.lower()]
    contam_b = [f for f in a_only if f.lower() in text_b.lower()]

    if contam_a:
        console.print("[red]⚠  Article A output leaks B-facts: {}[/red]".format(", ".join(contam_a)))
    else:
        console.print("[green]✓  Article A clean — no cross-contamination[/green]")

    if contam_b:
        console.print("[red]⚠  Article B output leaks A-facts: {}[/red]".format(", ".join(contam_b)))
    else:
        console.print("[green]✓  Article B clean — no cross-contamination[/green]")

    console.print()
    console.print("[bold]Key contradiction highlights:[/bold]")
    for label, a_fact, b_fact in [
        ("Signatories", "47 nations" in text_a, "34 nations" in text_b),
        ("Carbon tariff", "$45" in text_a, "$15" in text_b),
        ("GDP impact", "$900 billion" in text_a, "$250 billion" in text_b),
        ("US/China", "", "walked out" in text_b),
    ]:
        a_s = "[green]✓[/green]" if a_fact else "[dim]-[/dim]"
        b_s = "[green]✓[/green]" if b_fact else "[dim]-[/dim]"
        console.print("  {:<15}  A: {}    B: {}".format(label, a_s, b_s))

    console.print()
    console.print("[bold green]✓[/bold green] Boundary-KV engine complete — instant prefill from stored KV caches")
    console.print("[dim]Credit: LARQL boundary-kv engine concept from chrishayuk/larql[/dim]")

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
        # Force GPU reset after ingestion
        torch.cuda.synchronize()
        do_generate()
