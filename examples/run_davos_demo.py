#!/usr/bin/env python3
"""PDGA Davos Demo — three narratives, one event, multiple perspectives.

Phase 1 (ingest):  python3 run_davos_demo.py ingest
Phase 2 (generate): python3 run_davos_demo.py generate
"""

import gc
import json
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

from pdga.ingest.facts import ingest_narrative_with_facts
from pdga.db.store import DeltaDB
from pdga.generation.streaming import StreamingGenerator
from pdga.graph.dedup import FactDeduper, make_embed_fn
from pdga.graph.auto_edges import AutoEdgeGenerator
from pdga.graph.trust import TrustPropagator

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

    banner("Ingesting...", "Full narrative KV + atomic fact extraction with absolute positions.")
    db = DeltaDB()

    for key, text in narratives.items():
        source = SOURCES[key]
        token_count = len(tokenizer.encode(text))
        console.print(f"  {key} ({source['name']}, trust={source['trust']}) — {token_count} tokens...")
        
        result = ingest_narrative_with_facts(
            model=model, tokenizer=tokenizer, text=text,
            output_dir=DELTAS_DIR, narrative_id=key, source_id=key,
            trust=source["trust"], min_fact_tokens=10,
        )
        
        narrative_dir = result["narrative_path"]
        
        # Register narrative
        db.register(
            delta_id=result["narrative_id"], delta_type="narrative",
            path=str(narrative_dir),
            base_model=model.config._name_or_path or MODEL_ID,
            source_text=text, trust=source["trust"],
            num_windows=1, tags=["davos", key],
        )

        # Register each fact as a delta
        for f in result["facts"]:
            db.register(
                delta_id=f["fact_id"], delta_type="fact",
                path=str(narrative_dir / "facts" / f["fact_id"]),
                base_model=model.config._name_or_path or MODEL_ID,
                source_text=f["text"], trust=source["trust"],
                num_windows=0, tags=["davos", key, "fact"],
            )

        pdga_size = sum(f.stat().st_size for f in narrative_dir.rglob("*")
                        if f.is_file()) / 1024
        console.print("    facts={}  total={:.0f}KB".format(
            result["num_facts"], pdga_size))
        
        # Show extracted facts
        for f in result["facts"]:
            console.print(f"      fact {f['fact_id']}: pos {f['start_pos']}-{f['end_pos']} ({f['token_count']} tokens)")

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
    
    # Helper: get all fact paths for a narrative
    def get_fact_paths(narrative_dir):
        facts_dir = narrative_dir / "facts"
        if not facts_dir.exists():
            return []
        return sorted(facts_dir.glob("*/kv_cache.pt"))
    
    # Map paths to narrative keys by reading narrative.json
    narrative_map = {}
    for path in paths:
        narrative_json = path / "narrative.json"
        if narrative_json.exists():
            meta = json.loads(narrative_json.read_text())
            source_key = meta.get("source_key", "")
            for key in SOURCES:
                if key == source_key:
                    narrative_map[key] = path
                    break

    # ── Query 1: Per-Source Generation (load all facts from each narrative) ─
    console.print("\n[bold underline green]── QUERY 1: Per-Source Generation ──[/bold underline green]")
    console.print("[dim]Loading all facts from each narrative with absolute positions preserved[/dim]")
    
    queries = [
        ("What happened at Davos?", "neutral"),
    ]
    
    for query_text, query_type in queries:
        console.print(f"\n[bold]Query:[/bold] {query_text}")
        
        results = []
        t0 = time.perf_counter()
        
        for key, path in narrative_map.items():
            source = SOURCES[key]
            fact_paths = get_fact_paths(path)
            
            gen = StreamingGenerator(model, fact_paths)
            r = gen.generate(
                tokenizer,
                prompt=f"Based on the provided context, answer this question: {query_text}",
                max_new_tokens=300, sample_temp=0.7,
            )
            r["source"] = source["name"]
            r["trust"] = source["trust"]
            r["key"] = key
            r["num_facts"] = len(fact_paths)
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
            title.append(f"  {r['num_tokens']} tokens  {r['tokens_per_second']:.1f} t/s  {r['num_facts']} facts", style="dim")
            console.print(Panel(title, border_style=color))
            console.print(r["generated_text"][:600])
            console.print()

    # ── Query 2: Cross-source comparison ────────────────────────────────
    console.print("\n[bold underline yellow]── QUERY 2: What Do Sources Agree On? ──[/bold underline yellow]")
    console.print("[dim]Loading all facts from all narratives combined[/dim]")
    
    # Load all facts from all narratives
    all_fact_paths = []
    for key, path in narrative_map.items():
        all_fact_paths.extend(get_fact_paths(path))
    
    gen = StreamingGenerator(model, all_fact_paths)
    r = gen.generate(
        tokenizer,
        prompt="Based on all the provided sources, what facts do they all agree on? List only points that appear in multiple sources.",
        max_new_tokens=400, sample_temp=0.7,
    )
    console.print(Panel(f"All sources combined  ({r['num_tokens']} tokens, {len(all_fact_paths)} facts)", border_style="blue"))
    console.print(r["generated_text"])
    console.print()
    del gen
    gc.collect()
    torch.cuda.empty_cache()

    # ── Query 3: Where do they disagree? ──────────────────────────────────
    console.print("\n[bold underline red]── QUERY 3: Where Do Sources Disagree? ──[/bold underline red]")
    
    gen = StreamingGenerator(model, all_fact_paths)
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


def do_graph():
    banner("PDGA Davos — Graph & Deduplication",
           "Find shared facts, contradictions, and auto-generate edges")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    console.print("  Loaded model for embedding generation")

    paths = sorted(DELTAS_DIR.glob("*.pdga"))

    # Load all facts
    deduper = FactDeduper(
        embed_fn=make_embed_fn(model, tokenizer),
        semantic_threshold=0.92,
    )

    all_facts = []
    for path in paths:
        facts = deduper.load_narrative_facts(path)
        all_facts.extend(facts)
        console.print(f"  Loaded {len(facts)} facts from {path.name}")

    # Deduplicate
    console.print("\n[bold]Deduplicating facts...[/bold]")
    canonical = deduper.deduplicate(all_facts)

    shared = deduper.get_shared_facts()
    console.print(f"  Total canonical facts: {len(canonical)}")
    console.print(f"  Shared across sources: {len(shared)}")

    # Display shared facts
    if shared:
        console.print("\n[bold underline green]── Shared Facts ──[/bold underline green]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Canonical ID", style="dim", width=20)
        table.add_column("Text")
        table.add_column("Sources", width=30)
        for sf in shared[:10]:
            sources = ", ".join(
                f"{s['narrative_id'][:8]}..." for s in sf["sources"]
            )
            text = sf["text"][:120] + "..." if len(sf["text"]) > 120 else sf["text"]
            table.add_row(sf["canonical_id"][:16], text, sources)
        console.print(table)

    # Display unique facts per source
    for key in SOURCES:
        unique = deduper.get_unique_facts(key)
        if unique:
            console.print(f"\n[bold underline {SOURCES[key]['color']}]── Unique to {key} ──[/bold underline {SOURCES[key]['color']}]")
            for uf in unique[:5]:
                console.print(f"  • {uf['text'][:100]}")
            if len(unique) > 5:
                console.print(f"  ... and {len(unique) - 5} more")

    # Generate graph edges
    console.print("\n[bold]Generating graph edges...[/bold]")
    db = DeltaDB()
    edge_gen = AutoEdgeGenerator(db)
    edge_gen.generate_all(paths, canonical, deduper.fact_to_canonical)

    # Display edges summary
    from pdga.graph.edges import EdgeType
    for etype in EdgeType:
        count = db.conn.execute(
            "SELECT COUNT(*) FROM edges WHERE edge_type=?", (etype.value,)
        ).fetchone()[0]
        if count > 0:
            console.print(f"  {etype.value}: {count} edges")

    # Sample edges
    console.print("\n[bold underline blue]── Sample Edges ──[/bold underline blue]")
    rows = db.conn.execute(
        "SELECT * FROM edges LIMIT 20"
    ).fetchall()
    table = Table(show_header=True, header_style="bold")
    table.add_column("Source", width=16)
    table.add_column("Type", width=15)
    table.add_column("Target", width=16)
    table.add_column("Weight", width=8)
    for row in rows:
        table.add_row(row[0][:14], row[2], row[1][:14], f"{row[3]:.2f}")
    console.print(table)

    # Build narrative_map for trust display
    narrative_map_graph = {}
    for path in paths:
        narrative_json = path / "narrative.json"
        if narrative_json.exists():
            meta = json.loads(narrative_json.read_text())
            source_key = meta.get("source_key", "")
            for key in SOURCES:
                if key == source_key:
                    narrative_map_graph[key] = path
                    break

    # Trust propagation
    console.print("\n[bold]Propagating trust scores...[/bold]")
    trust_prop = TrustPropagator(db)
    trust_scores = trust_prop.propagate()
    console.print(f"  Updated trust for {len(trust_scores)} facts")

    # Display top trusted facts
    top_facts = trust_prop.rank_facts(10)
    console.print("\n[bold underline green]── Top Trusted Facts ──[/bold underline green]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Trust", width=8, justify="right")
    table.add_column("Fact ID", style="dim", width=20)
    table.add_column("Text")
    for f in top_facts:
        text = f.get("source_text", "")[:100]
        table.add_row(f"{f.get('trust', 0.5):.2f}", f["delta_id"][:16], text)
    console.print(table)

    # Narrative-level trust
    console.print("\n[bold underline yellow]── Narrative Trust Scores ──[/bold underline yellow]")
    for key in SOURCES:
        if key not in narrative_map_graph:
            continue
        path = narrative_map_graph[key]
        narrative_json = path / "narrative.json"
        if narrative_json.exists():
            meta = json.loads(narrative_json.read_text())
            narr_trust = trust_prop.get_narrative_trust(meta["narrative_id"])
            original = SOURCES[key]["trust"]
            delta = narr_trust - original
            color = "green" if delta > 0 else "red" if delta < 0 else "white"
            console.print(f"  {SOURCES[key]['name']}: {original:.2f} → {narr_trust:.2f} "
                         f"([{color}]{delta:+.2f}[/{color}])")

    db.close()
    console.print("\n[bold green]✓[/bold green] Graph generation complete")


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
