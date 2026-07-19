#!/usr/bin/env python3
"""Conventional RAG baseline for the Davos multi-narrative task.

Baseline = the simplest thing Figtree competes with: chunk each narrative into
sentences, embed with the **same** crystal-layer boundary extraction Figtree
uses (so embedding parity holds), store in memory, retrieve top-k by cosine,
stuff into the same Qwen3 ChatML prompt, and generate. No K/V replay, no trust
graph, no boundary-engineered attention.

Run:
    python3 examples/rag_baseline_davos.py
    FIGTREE_COMPUTE_KV=1 python3 examples/rag_baseline_davos.py   # not used here
"""

import os
import shutil
import tempfile
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer

from figtree.ingest import ingest_text_to_figments
from figtree.generate import FigmentGenerator
from figtree.lancedb_store import connect

from davos_eval import QUERIES, SOURCES, contradiction_aware, fidelity_score, vram_peak_mb

console = Console()
MODEL_ID = "unsloth/Qwen3-4B-bnb-4bit"
NARRATIVES_DIR = Path(__file__).parent / "davos_narratives"
K = 8  # retrieve top-k sentences


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    gen = FigmentGenerator(model, tokenizer)

    # 1) Embed each narrative's sentences with the same boundary extraction.
    chunks: list[dict] = []
    tmp = tempfile.mkdtemp(prefix="figtree_rag_")
    try:
        store = connect(tmp)
        for key in SOURCES:
            text = (NARRATIVES_DIR / f"{key}.txt").read_text().strip()
            figs = ingest_text_to_figments(
                model=model, tokenizer=tokenizer, text=text,
                source_id=key, store=store, kv_manager=None, compute_kv=False,
                trust=SOURCES[key]["trust"], min_chars=20,
            )
            for f in figs:
                if f.is_image() or f.is_trust_assertion():
                    continue
                chunks.append({
                    "text": f.text,
                    "boundary": np.asarray(f.boundary, dtype=np.float32),
                    "source": key,
                })
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    console.print(f"[bold]RAG baseline:[/bold] {len(chunks)} sentence chunks, top-k={K}")

    # 2) Retrieve + generate per query.
    table = Table(show_header=True, header_style="bold")
    table.add_column("Query")
    table.add_column("Fidelity (avg)", justify="right")
    table.add_column("Contradiction aware", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Latency (s)", justify="right")

    faith_sum = 0.0
    faith_n = 0
    contra_hits = 0
    tok_total = 0
    t0_all = time.perf_counter()

    for qid, query in QUERIES.items():
        # cosine retrieval over all chunks
        q_vec = np.zeros_like(chunks[0]["boundary"])
        for c in chunks:
            q_vec = q_vec + c["boundary"]
        q_vec /= len(chunks)
        ranked = sorted(chunks, key=lambda c: cosine(q_vec, c["boundary"]), reverse=True)
        top = ranked[:K]
        context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in top)

        t0 = time.perf_counter()
        result = gen.generate(figments=[], prompt=f"{context}\n\n{query}", max_new_tokens=120)
        dt = time.perf_counter() - t0

        out = result["text"]
        # fidelity: average over sources that contributed to the retrieved set
        sources_in = {c["source"] for c in top}
        if sources_in:
            fs = sum(fidelity_score(out, s) for s in sources_in) / len(sources_in)
        else:
            fs = 0.0
        faith_sum += fs
        faith_n += 1
        if contradiction_aware(out):
            contra_hits += 1
        tok_total += result["num_tokens"]

        table.add_row(
            qid, f"{fs:.2f}", "yes" if contradiction_aware(out) else "no",
            str(result["num_tokens"]), f"{dt:.1f}",
        )
        console.print(f"\n[bold]{qid}[/bold]: {query}")
        console.print(f"  {out[:280]}")

    total_dt = time.perf_counter() - t0_all
    vram = vram_peak_mb()
    table.add_row("—", f"{faith_sum / max(1, faith_n):.2f}",
                  f"{contra_hits}/{faith_n}", str(tok_total), f"{total_dt:.1f}")
    console.print("\n[bold]RAG baseline summary[/bold]")
    console.print(table)
    console.print(f"[dim]VRAM peak: {vram:.0f} MB[/dim]" if vram else "[dim]CPU run[/dim]")


if __name__ == "__main__":
    main()
