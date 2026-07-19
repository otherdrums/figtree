#!/usr/bin/env python3
"""Side-by-side comparison: Figtree vs conventional RAG on the Davos task.

Runs the conventional RAG baseline and the Figtree pipeline (ingest + generate
+ graph) against the same narratives and queries, then prints a comparison
table of fidelity, contradiction awareness, VRAM and latency.

Run:
    python3 examples/davos_compare.py
"""

from rich.console import Console

from rag_baseline_davos import main as rag_main
from davos_benchmark_v2 import benchmark as figtree_benchmark

console = Console()


def main():
    console.rule("[bold blue]Conventional RAG baseline[/bold blue]")
    rag_main()

    console.rule("[bold blue]Figtree (boundary + KV + trust graph)[/bold blue]")
    figtree_benchmark()

    console.rule("[bold blue]How to read this[/bold blue]")
    console.print(
        "Both use the same model (Qwen3-4B 4-bit) and same Davos narratives.\n"
        "RAG: retrieve top-k sentences by cosine, stuff + generate. No K/V replay, no trust graph.\n"
        "Figtree: per-figment K/V replay into attention + propagated source trust in the prompt.\n"
        "Compare fidelity (figures reproduced), contradiction awareness, VRAM peak, and latency.\n"
        "Claims are intentionally modest — see AGENTS.md Known Limitations."
    )


if __name__ == "__main__":
    main()
