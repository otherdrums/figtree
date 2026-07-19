#!/usr/bin/env python3
"""Figtree command-line interface.

Thin Typer wrapper around the Davos v2 demo phases
(``examples/run_davos_v2.py``). Exposes ingest / generate / graph / benchmark
as installable console commands so the package is runnable after
``pip install -e .``.

Usage:
    figtree ingest
    figtree generate
    figtree graph
    figtree all
    figtree benchmark
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import typer  # noqa: E402

from examples.run_davos_v2 import (  # noqa: E402
    do_generate,
    do_graph,
    do_ingest,
)

app = typer.Typer(
    add_completion=False,
    help="Figtree — grow coherent Images from Figments.",
)


@app.command()
def ingest():
    """Ingest the Davos narratives into the LanceDB store."""
    do_ingest()


@app.command()
def generate():
    """Generate answers from ingested figments."""
    do_generate()


@app.command()
def graph():
    """Build the figment graph and propagate trust."""
    do_graph()


@app.command()
def all():  # noqa: A001
    """Run ingest + generate + graph end to end."""
    do_ingest()
    do_generate()
    do_graph()


@app.command()
def benchmark():
    """Run the Davos v2 benchmark (ingest / generate / graph timings)."""
    from examples.davos_benchmark_v2 import benchmark as _benchmark

    _benchmark()


@app.command()
def compare():
    """Compare Figtree vs a conventional RAG baseline on the Davos task."""
    from examples.davos_compare import main as _compare

    _compare()


if __name__ == "__main__":
    app()
