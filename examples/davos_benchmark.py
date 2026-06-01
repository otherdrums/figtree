#!/usr/bin/env python3
"""PDGA Davos Benchmark — measure end-to-end pipeline performance.

Usage:
    python3 davos_benchmark.py
"""

import gc
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.table import Table

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
    "pro_globalist": {"name": "Reuters-style", "trust": 0.95},
    "anti_globalist": {"name": "Guardian-style", "trust": 0.60},
    "conspiracy": {"name": "Fringe Blog", "trust": 0.15},
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


def benchmark():
    results = {}
    
    # Cleanup
    if PDGA_HOME.exists():
        shutil.rmtree(PDGA_HOME)
    if DELTAS_DIR.exists():
        shutil.rmtree(DELTAS_DIR)
    DELTAS_DIR.mkdir(parents=True, exist_ok=True)

    # Load narratives
    narratives = {}
    for key in SOURCES:
        text = (NARRATIVES_DIR / f"{key}.txt").read_text().strip()
        narratives[key] = text

    # Phase 1: Ingestion
    console.print("[bold]Phase 1: Ingestion[/bold]")
    model, tokenizer = load_model()
    db = DeltaDB()

    t0 = time.perf_counter()
    total_facts = 0
    total_size = 0
    for key, text in narratives.items():
        result = ingest_narrative_with_facts(
            model=model, tokenizer=tokenizer, text=text,
            output_dir=DELTAS_DIR, narrative_id=key, source_id=key,
            trust=SOURCES[key]["trust"], min_fact_tokens=10,
        )
        total_facts += result["num_facts"]
        narrative_dir = result["narrative_path"]
        pdga_size = sum(f.stat().st_size for f in narrative_dir.rglob("*") if f.is_file())
        total_size += pdga_size
        
        # Register in DB
        db.register(
            delta_id=result["narrative_id"], delta_type="narrative",
            path=str(narrative_dir),
            base_model=model.config._name_or_path or MODEL_ID,
            source_text=text, trust=SOURCES[key]["trust"],
            num_windows=1, tags=["davos", key],
        )
        for f in result["facts"]:
            db.register(
                delta_id=f["fact_id"], delta_type="fact",
                path=str(f["path"]),
                base_model=model.config._name_or_path or MODEL_ID,
                source_text=f["text"], trust=SOURCES[key]["trust"],
                num_windows=0, tags=["davos", key, "fact"],
            )

    ingest_time = time.perf_counter() - t0
    results["ingest"] = {
        "time": ingest_time,
        "narratives": len(narratives),
        "facts": total_facts,
        "size_mb": total_size / (1024 * 1024),
    }
    console.print(f"  Ingested {total_facts} facts from {len(narratives)} narratives in {ingest_time:.1f}s")
    console.print(f"  Total size: {results['ingest']['size_mb']:.1f} MB")

    # Phase 2: Generation
    console.print("\n[bold]Phase 2: Generation[/bold]")
    
    def get_fact_paths(narrative_dir):
        facts_dir = narrative_dir / "facts"
        if not facts_dir.exists():
            return []
        return sorted(facts_dir.glob("*/kv_cache.pt"))

    all_fact_paths = []
    for path in DELTAS_DIR.glob("*.pdga"):
        all_fact_paths.extend(get_fact_paths(path))

    t0 = time.perf_counter()
    gen = StreamingGenerator(model, all_fact_paths)
    r = gen.generate(
        tokenizer,
        prompt="What happened at Davos?",
        max_new_tokens=200, sample_temp=0.7,
    )
    gen_time = time.perf_counter() - t0
    results["generate"] = {
        "time": gen_time,
        "tokens": r["num_tokens"],
        "tps": r["tokens_per_second"],
        "facts_loaded": len(all_fact_paths),
    }
    console.print(f"  Generated {r['num_tokens']} tokens in {gen_time:.1f}s ({r['tokens_per_second']:.1f} t/s)")
    console.print(f"  Facts loaded: {len(all_fact_paths)}")
    del gen
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 3: Graph
    console.print("\n[bold]Phase 3: Graph & Deduplication[/bold]")
    
    t0 = time.perf_counter()
    deduper = FactDeduper(embed_fn=make_embed_fn(model, tokenizer), semantic_threshold=0.92)
    all_facts = []
    for path in DELTAS_DIR.glob("*.pdga"):
        all_facts.extend(deduper.load_narrative_facts(path))
    canonical = deduper.deduplicate(all_facts)
    
    edge_gen = AutoEdgeGenerator(db)
    edge_gen.generate_all(list(DELTAS_DIR.glob("*.pdga")), canonical, deduper.fact_to_canonical)
    
    trust_prop = TrustPropagator(db)
    trust_prop.propagate()
    
    graph_time = time.perf_counter() - t0
    
    # Count edges
    edge_counts = {}
    for row in db.conn.execute("SELECT edge_type, COUNT(*) FROM edges GROUP BY edge_type").fetchall():
        edge_counts[row[0]] = row[1]
    
    results["graph"] = {
        "time": graph_time,
        "canonical_facts": len(canonical),
        "shared_facts": len(deduper.get_shared_facts()),
        "edges": edge_counts,
    }
    console.print(f"  Deduplicated {len(all_facts)} facts into {len(canonical)} canonical in {graph_time:.1f}s")
    console.print(f"  Shared facts: {results['graph']['shared_facts']}")
    console.print(f"  Edges: {edge_counts}")

    # Summary table
    console.print("\n[bold]Benchmark Summary[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Phase")
    table.add_column("Time (s)", justify="right")
    table.add_column("Throughput")
    
    table.add_row(
        "Ingestion",
        f"{results['ingest']['time']:.1f}",
        f"{results['ingest']['facts']} facts, {results['ingest']['size_mb']:.1f} MB",
    )
    table.add_row(
        "Generation",
        f"{results['generate']['time']:.1f}",
        f"{results['generate']['tokens']} tokens @ {results['generate']['tps']:.1f} t/s",
    )
    table.add_row(
        "Graph",
        f"{results['graph']['time']:.1f}",
        f"{results['graph']['canonical_facts']} canonical, {sum(edge_counts.values())} edges",
    )
    
    total_time = sum(r["time"] for r in results.values())
    table.add_row("Total", f"{total_time:.1f}", "")
    
    console.print(table)
    
    db.close()


if __name__ == "__main__":
    benchmark()
