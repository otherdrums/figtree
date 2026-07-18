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

    store_uri = os.environ.get("FIGTREE_STORE_URI", str(FIGMENTS_DIR) + ".lance")
    compute_kv = os.environ.get("FIGTREE_COMPUTE_KV", "0") == "1"
    console.print(f"[bold]Store:[/bold] {store_uri}  [bold]compute_kv:[/bold] {compute_kv}")

    if Path(store_uri).exists():
        shutil.rmtree(store_uri)
    Path(store_uri).parent.mkdir(parents=True, exist_ok=True)

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

    from figtree.lancedb_store import connect
    from figtree.kv_cache_manager import KVCacheManager

    store = connect(store_uri)
    kv_manager = KVCacheManager(
        model, tokenizer, kv_root=str(Path(store_uri).with_suffix("")) + "_kv",
        mode="eager" if compute_kv else "lazy",
    )

    banner("Ingesting...", "Boundary capture per sentence (~10 KB/figment)")

    for key in SOURCES:
        text = (NARRATIVES_DIR / f"{key}.txt").read_text().strip()
        source = SOURCES[key]
        token_count = len(tokenizer.encode(text))
        console.print(f"  {key} ({source['name']}, trust={source['trust']}) — {token_count} tokens...")

        figments = ingest_text_to_figments(
            model=model, tokenizer=tokenizer, text=text,
            output_dir=None, source_id=key, store=store,
            kv_manager=kv_manager, compute_kv=compute_kv,
            trust=source["trust"], min_chars=20,
        )

        console.print(f"    {len(figments)} figments persisted to {store_uri}")

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

    store_uri = os.environ.get("FIGTREE_STORE_URI", str(FIGMENTS_DIR) + ".lance")
    compute_kv = os.environ.get("FIGTREE_COMPUTE_KV", "0") == "1"
    from figtree.lancedb_store import connect
    from figtree.kv_cache_manager import KVCacheManager

    store = connect(store_uri)
    kv_manager = KVCacheManager(
        model, tokenizer, kv_root=str(Path(store_uri).with_suffix("")) + "_kv",
        mode="eager" if compute_kv else "lazy",
    )

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

    # -- Load figments from the LanceDB store --
    source_figments: dict[str, list[Figment]] = {}
    for key in SOURCES:
        figments = store.by_source(key)
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
        run_query(source["name"], figments, "What happened at Davos? Recount the key details, figures, and claims from this source's narrative.", max_new_tokens=400)

    # -- QUERY 2: Agreement (trust-aware) --
    console.print("\n[bold underline yellow]── QUERY 2: What Do Sources Agree On? (trust-aware) ──[/bold underline yellow]")
    _run_trust_aware(
        "Agreement", "Based on all sources, what facts do they agree on?", gen,
        source_figments, log_path,
    )

    # -- QUERY 3: Disagreement (trust-aware) --
    console.print("\n[bold underline red]── QUERY 3: Where Do Sources Disagree? (trust-aware) ──[/bold underline red]")
    _run_trust_aware(
        "Disagreement", "What are the major disagreements between the different perspectives?",
        gen, source_figments, log_path,
    )

    # -- QUERY 4: Boundary-based generation (cached K/V) --
    console.print("\n[bold underline blue]── QUERY 4: Boundary-Based Generation (cached K/V) ──[/bold underline blue]")
    for key, figments in source_figments.items():
        source = SOURCES[key]
        console.print(f"\n[bold {source['color']}]── {source['name']} (trust={source['trust']}) ──[/bold {source['color']}]")
        try:
            result_bd = gen.generate_from_boundaries(
                figments=figments, prompt="What happened at Davos? Recount the key details, figures, and claims from this source's narrative.",
                max_new_tokens=400, kv_manager=kv_manager,
            )
            console.print(f"\n[bold]Output ({result_bd['num_tokens']} tokens, {result_bd['elapsed']:.1f}s):[/bold]")
            console.print(result_bd["generated_text"])
            console.print()
            with open(log_path, "a") as f:
                f.write(f"── {source['name']} (boundary-based) ──\n")
                f.write(f"Tokens: {result_bd['num_tokens']}, Elapsed: {result_bd['elapsed']:.1f}s\n")
                f.write(result_bd["generated_text"] + "\n\n")
        except FileNotFoundError as e:
            console.print(f"[yellow]Skipped: {e}[/yellow]")

    console.print("\n[bold green]Figtree Davos v2 generation complete[/bold green]")
    console.print(f"[dim]Log saved to: {log_path}[/dim]")


def _build_graph() -> "Figtree":
    """Load all ingested figments and compute (persisted) source-based trust."""
    from figtree.graph import Figtree
    from figtree.lancedb_store import connect

    store_uri = os.environ.get("FIGTREE_STORE_URI", str(FIGMENTS_DIR) + ".lance")
    store = connect(store_uri)
    all_figments = store.all()
    graph = Figtree(all_figments, store=store)
    graph.propagate_trust(store=store)  # idempotent, store-persisted
    return graph


def _run_trust_aware(
    query_label: str,
    query: str,
    gen: "FigmentGenerator",
    source_figments_map: dict[str, list[Figment]],
    log_path: Path,
) -> None:
    """Recall all perspectives on a query and explain credibility, then generate.

    The credibility relationships (which sources agree / contradict) are injected
    into the generation prompt so the model cannot fabricate cross-source agreement
    that the sources never state. Only figments from sources that actually
    corroborate each other are fed for agreement-style queries; all recalled
    perspectives are fed for disagreement-style queries.
    """
    graph = _build_graph()
    ctx = graph.build_trust_aware_context(query)

    console.print("\n[bold]Credibility context:[/bold]")
    console.print(ctx["rationale"])
    with open(log_path, "a") as f:
        f.write(f"── {query_label} (trust-aware context) ──\n")
        f.write(ctx["rationale"] + "\n\n")

    sources = list(ctx["by_source"].keys())
    # Sources that genuinely corroborate at least one other recalled source.
    agreeing_sources = {
        s for s in sources
        if set(ctx["by_source"][s].get("agreeing", [])) & set(sources)
    }
    contradicting_sources = {
        s for s in sources
        if set(ctx["by_source"][s].get("contradicting", [])) & set(sources)
    }

    if query_label.lower().startswith("agreement"):
        # These narratives are constructed to conflict, so genuine same-stance
        # alignment is rare. Feed every recalled perspective but constrain the
        # model to only facts literally present in >=2 sources' text; the weak
        # "agreeing" set (same explicit non-neutral stance) is shown for honesty.
        use_sources = sorted(sources)
        scoped_note = (
            f"Sources that genuinely AGREE (same explicit stance) on this topic: "
            f"{sorted(agreeing_sources) or 'none'}. "
            f"Sources that CONTRADICT each other: {sorted(contradicting_sources) or 'none'}. "
            f"Most sources merely discuss the same TOPICS (overlap), which is NOT "
            f"agreement. List only facts that appear verbatim or identically in "
            f"the text of at least two sources. Do NOT infer shared conclusions "
            f"(e.g. 'all endorsed', 'all issued voluntary frameworks') that are "
            f"stated by only one source. If the only shared facts are surface "
            f"details (e.g. the event occurred, a fund was announced), say so."
        )
    else:
        # Disagreement query: feed every recalled perspective and let the model
        # contrast them, grounded by the explicit contradiction relationships.
        use_sources = sorted(sources)
        scoped_note = (
            f"Sources that CONTRADICT each other: {sorted(contradicting_sources) or 'none'}. "
            f"Sources that genuinely AGREE: {sorted(agreeing_sources) or 'none'}. "
            f"Only describe disagreements that are explicitly present in the "
            f"sources' text; do not invent disagreement where the texts align."
        )

    # Collect the figments from the scoped sources that are relevant to the query.
    qwords = set(query.lower().split())
    all_relevant: list[Figment] = []
    for src in use_sources:
        for fig in source_figments_map.get(src, []):
            if qwords & set(fig.text.lower().split()):
                all_relevant.append(fig)

    grounded_prompt = (
        f"{scoped_note}\n\n"
        f"{query}\n\n"
        f"Instructions: base every claim strictly on the provided source text. "
        f"Do not invent facts, dates, or agreements that are not in the text."
    )

    result = gen.generate(figments=all_relevant, prompt=grounded_prompt, max_new_tokens=450)
    console.print(f"\n[bold]Output ({result['num_tokens']} tokens):[/bold]")
    console.print(result["generated_text"])
    console.print()
    with open(log_path, "a") as f:
        f.write(f"── {query_label} ──\nPrompt: {grounded_prompt}\n")
        f.write(f"Tokens: {result['num_tokens']}\n")
        f.write(result["generated_text"] + "\n\n")


def do_graph():
    from datetime import datetime

    banner("Figtree Davos v2 — Graph", "Deduplication + edges + trust propagation")
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(__file__).parent / f"davos_graph_{run_ts}.log"

    store_uri = os.environ.get("FIGTREE_STORE_URI", str(FIGMENTS_DIR) + ".lance")
    from figtree.lancedb_store import connect
    store = connect(store_uri)
    all_figments = store.all()

    graph = Figtree(all_figments, store=store)

    dedup_edges = graph.deduplicate()
    auto_edges = graph.create_edges()
    trust_scores = graph.propagate_trust(store=store)  # idempotent + persisted

    console.print(f"  Figments: {len([f for f in all_figments if not f.is_edge() and not f.is_trust_assertion()])}")
    console.print(f"  Dedup edges: {len(dedup_edges)}")
    console.print(f"  Auto edges: {len(auto_edges)}")
    console.print(f"  Trust scores updated: {len(trust_scores)}")

    analysis = graph.analyze_sources()
    console.print("\n[bold underline green]── Source Credibility (trust-aware) ──[/bold underline green]")
    for src, info in sorted(analysis.items(), key=lambda kv: kv[1]["adjusted_trust"], reverse=True):
        console.print(f"\n[bold]{src}[/bold]  adjusted_trust={info['adjusted_trust']:.2f}  base={info['base_trust']:.2f}")
        console.print(f"  {info['rationale']}")
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
        f.write("── Source Credibility ──\n")
        for src, info in sorted(analysis.items(), key=lambda kv: kv[1]["adjusted_trust"], reverse=True):
            f.write(f"\n{src}  adjusted_trust={info['adjusted_trust']:.2f}  base={info['base_trust']:.2f}\n")
            f.write(f"  {info['rationale']}\n")
        f.write("\n")

    console.print("[bold green]Graph complete[/bold green]")
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
