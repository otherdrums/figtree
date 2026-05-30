"""PDGA CLI — Parallel Delta Graph Architecture command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from pdga import __version__

app = typer.Typer(
    name="pdga",
    help="Parallel Delta Graph Architecture — context-aware multi-delta generation",
    no_args_is_help=True,
)

console = Console()
_state = {"model": None, "tokenizer": None, "lsh": None}


def _load_model(model_id: str | None = None, use_4bit: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_id is None:
        model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    if _state["model"] is not None:
        return _state["model"], _state["tokenizer"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[dim]Loading {model_id} on {device}...[/dim]")

    kwargs = dict(trust_remote_code=True)
    kwargs["torch_dtype"] = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map=device, **kwargs,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _state["model"] = model
    _state["tokenizer"] = tokenizer
    return model, tokenizer


def _get_db():
    from pdga.db.store import DeltaDB
    return DeltaDB()


def _get_lsh(model):
    from pdga.retrieval.lsh import create_lsh_for_model
    if _state["lsh"] is None:
        _state["lsh"] = create_lsh_for_model(model.config.hidden_size)
    return _state["lsh"]


# ── ingest ──────────────────────────────────────────────────────────────

@app.command()
def ingest(
    file: Path = typer.Argument(..., help="Text file to ingest"),
    tags: str = typer.Option("", help="Comma-separated tags"),
    trust: float = typer.Option(0.5, help="Source trust factor [0.0-1.0]"),
    source_url: str = typer.Option("", help="Source URL or identifier"),
    window_size: int = typer.Option(200, help="Tokens per window"),
    novel_fraction: float = typer.Option(0.05, help="Fraction of positions to retain as novel (0.0-1.0, lower = sparser)"),
    model_id: str = typer.Option(
        "Qwen/Qwen2.5-1.5B-Instruct", help="Model to use for ingestion"
    ),
    crystal_layer: int = typer.Option(-1, help="Crystal layer override (-1 = auto)"),
    output_dir: str = typer.Option("deltas", help="Output directory for .pdga files"),
):
    """Ingest a text file into a ContextDelta."""
    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    text = file.read_text()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    model, tokenizer = _load_model(model_id)

    from pdga.ingest.text import ingest_text
    from pdga.db.store import DeltaDB
    from pdga.retrieval.lsh import create_lsh_for_model

    crystal = None if crystal_layer < 0 else crystal_layer

    console.print(f"[bold]Ingesting:[/bold] {file.name} ({len(text)} chars)")
    console.print(f"  trust={trust}, tags={tag_list}, window_size={window_size}, novel_fraction={novel_fraction}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    delta = ingest_text(
        model=model,
        tokenizer=tokenizer,
        text=text,
        output_dir=out_path,
        window_size=window_size,
        novel_fraction=novel_fraction,
        trust=trust,
        source_url=source_url,
        tags=tag_list,
        crystal_layer=crystal,
    )

    with DeltaDB() as db:
        db.register(
            delta_id=delta.delta_id,
            delta_type="context",
            path=str(delta.path),
            base_model=model.config._name_or_path or model_id,
            source_text=text,
            trust=trust,
            num_windows=delta.num_windows,
            tags=tag_list,
        )

    lsh = create_lsh_for_model(model.config.hidden_size)
    lsh.insert(delta.delta_id, delta.boundaries)

    console.print()
    table = Table(title=f"Delta Created: {delta.delta_id}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("ID", delta.delta_id)
    table.add_row("Type", "context")
    table.add_row("Windows", str(delta.num_windows))
    table.add_row("Crystal layer", str(delta.manifest.crystal_layer))
    table.add_row("Window size", str(window_size))
    table.add_row("Hidden size", str(delta.hidden_size))
    table.add_row("Trust", str(trust))
    table.add_row("Tags", ", ".join(tag_list))
    table.add_row("Path", str(delta.path))
    total_tokens = sum(len(w) for w in delta.window_tokens)
    table.add_row("Novel positions", f"{delta.boundaries.shape[0]} / {total_tokens} ({delta.boundaries.shape[0]/max(total_tokens,1)*100:.1f}%)")
    console.print(table)


# ── list ────────────────────────────────────────────────────────────────

@app.command()
def list_deltas(
    delta_type: str = typer.Option("", help="Filter: context, weight"),
    tags: str = typer.Option("", help="Filter by comma-separated tags"),
    limit: int = typer.Option(50, help="Max results"),
):
    """List stored deltas."""
    with _get_db() as db:
        if tags:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]
            results = db.list_by_tags(tag_list, delta_type or None)
        else:
            results = db.list_all(delta_type or None, limit)

    if not results:
        console.print("[dim]No deltas found.[/dim]")
        return

    table = Table(title="Stored Deltas")
    table.add_column("ID", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Model", style="dim")
    table.add_column("Windows", justify="right")
    table.add_column("Trust", justify="right")
    table.add_column("Tags", style="yellow")
    table.add_column("Created", style="dim")

    for r in results:
        table.add_row(
            r["delta_id"],
            r["delta_type"],
            r["base_model"][:30],
            str(r["num_windows"]),
            f"{r['trust']:.2f}",
            ", ".join(json.loads(r.get("tags", "[]"))[:3]),
            r["created_at"][:19],
        )

    console.print(table)


# ── generate ────────────────────────────────────────────────────────────

@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Prompt or query text"),
    deltas: str = typer.Option("", help="Comma-separated delta IDs to load"),
    model_id: str = typer.Option(
        "Qwen/Qwen2.5-1.5B-Instruct", help="Model for generation"
    ),
    max_tokens: int = typer.Option(256, help="Max new tokens"),
    temperature: float = typer.Option(0.7, help="Sampling temperature"),
    top_k: int = typer.Option(50, help="Top-K sampling"),
    top_p: float = typer.Option(0.95, help="Top-P sampling"),
    show_deltas: bool = typer.Option(False, help="Show loaded delta info"),
    mode: str = typer.Option("replay", help="Generation mode: replay | hybrid | residuals | inject"),
):
    """Generate text with context from stored deltas."""
    model, tokenizer = _load_model(model_id)
    delta_list = [d.strip() for d in deltas.split(",") if d.strip()]

    loaded_deltas = []
    if delta_list:
        from pdga.delta.io import load_delta

        for did in delta_list:
            with _get_db() as db:
                entry = db.get(did)
            if entry is None:
                console.print(f"[red]Delta not found: {did}[/red]")
                raise typer.Exit(1)

            delta = load_delta(Path(entry["path"]))
            loaded_deltas.append(delta)

            if show_deltas:
                console.print(
                    f"[dim]Loaded: {did} (trust={entry['trust']:.2f}, "
                    f"windows={delta.num_windows})[/dim]"
                )

    if not loaded_deltas:
        console.print("[dim]No deltas specified. Generating without context.[/dim]")
        return

    if mode == "residuals":
        from pdga.kernel.residual_inject import generate_from_residuals as gen
    elif mode == "inject":
        from pdga.kernel.inject import generate_from_injection as gen
    elif mode == "hybrid":
        from pdga.kernel.reference import generate_hybrid as gen
    else:
        from pdga.kernel.reference import generate as gen

    results = gen(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        deltas=loaded_deltas,
        max_new_tokens=max_tokens,
        sample_temp=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    results.sort(key=lambda r: r["trust"], reverse=True)

    for r in results:
        trust_pct = f"{r['trust']:.0%}"
        trust_color = "green" if r["trust"] >= 0.8 else ("yellow" if r["trust"] >= 0.5 else "red")
        gen_mode = r.get("mode", mode)
        title = f"{r['delta_id']}  —  trust: [{trust_color}]{trust_pct}[/{trust_color}]  —  [{gen_mode}]"
        if r["source_url"]:
            title += f"  —  {r['source_url']}"
        if r["tags"]:
            title += f"  —  {' '.join('#' + t for t in r['tags'])}"

        console.print(Panel(r["generated_text"] or "(empty)", title=title,
                            border_style=trust_color))


# ── think ───────────────────────────────────────────────────────────────

@app.command()
def think(
    prompt: str = typer.Argument(..., help="Prompt or query text"),
    streams: str = typer.Option(
        "",
        help="Stream spec: 'id:d=a,b:dt=0.8,0.9:st=0.7|id2:d=c:dt=0.5:st=1.0'",
    ),
    model_id: str = typer.Option(
        "Qwen/Qwen2.5-1.5B-Instruct", help="Model for generation"
    ),
    max_tokens: int = typer.Option(256, help="Max new tokens per stream"),
):
    """Run multi-stream parallel generation."""
    if not streams:
        console.print(
            "[red]--streams required. Format: id:d=a,b:dt=0.8,0.9:st=0.7[/red]"
        )
        raise typer.Exit(1)

    model, tokenizer = _load_model(model_id)

    from pdga.kernel.stream import StreamConfig
    from pdga.kernel.gather import think as think_fn
    from pdga.delta.io import load_delta

    stream_configs = []
    deltas_map = {}

    for spec in streams.split("|"):
        parts = spec.split(":")
        if len(parts) < 3:
            console.print(f"[red]Invalid stream spec: {spec}[/red]")
            raise typer.Exit(1)

        sid = parts[0]
        delta_ids = []
        delta_temps = {}

        for part in parts[1:]:
            if part.startswith("d="):
                delta_ids = part[2:].split(",")
            elif part.startswith("dt="):
                temps = [float(t) for t in part[3:].split(",")]
                for i, did in enumerate(delta_ids):
                    if i < len(temps):
                        delta_temps[did] = temps[i]
            elif part.startswith("st="):
                sample_temp = float(part[3:])

        if "sample_temp" not in locals():
            sample_temp = 0.7

        stream_configs.append(StreamConfig(
            id=sid,
            delta_ids=delta_ids,
            delta_temps=delta_temps,
            sample_temp=sample_temp,
        ))

    if stream_configs:
        stream_configs[0].conscious = True

    for cfg in stream_configs:
        for did in cfg.delta_ids:
            if did not in deltas_map:
                with _get_db() as db:
                    entry = db.get(did)
                if entry is None:
                    console.print(f"[red]Delta not found: {did}[/red]")
                    raise typer.Exit(1)
                deltas_map[did] = load_delta(Path(entry["path"]))

    console.print(f"[bold]Thinking with {len(stream_configs)} stream(s)...[/bold]")

    result = think_fn(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        streams=stream_configs,
        deltas_map=deltas_map,
        max_new_tokens=max_tokens,
    )

    for r in result.streams:
        label = "CONSCIOUS" if r.is_conscious else "subconscious"
        style = "bold green" if r.is_conscious else "dim"
        console.print(f"\n[{style}]──── {label}: {r.stream_id} "
                      f"(τ_sample={r.sample_temp}) ────[/{style}]")

        for dr in r.delta_results:
            trust_pct = f"{dr['trust']:.0%}"
            trust_color = "green" if dr["trust"] >= 0.8 else ("yellow" if dr["trust"] >= 0.5 else "red")
            title = f"{dr['delta_id']}  trust: [{trust_color}]{trust_pct}[/{trust_color}]"
            console.print(Panel(dr["generated_text"] or "(empty)",
                                title=title, border_style=trust_color))


# ── retrieve ────────────────────────────────────────────────────────────

@app.command()
def retrieve(
    query: str = typer.Argument(..., help="Query text to find relevant deltas"),
    top_k: int = typer.Option(10, help="Max candidates to return"),
    model_id: str = typer.Option(
        "Qwen/Qwen2.5-1.5B-Instruct", help="Model for embedding"
    ),
):
    """Find relevant context deltas for a query using LSH."""
    model, tokenizer = _load_model(model_id)
    lsh = _get_lsh(model)

    import torch
    tokens = tokenizer.encode(query, add_special_tokens=True)
    input_tensor = torch.tensor([tokens], dtype=torch.long, device=model.device)
    embed = model.get_input_embeddings()

    with torch.inference_mode():
        query_emb = embed(input_tensor).mean(dim=1).squeeze(0).float().cpu().numpy()

    candidates = lsh.query(query_emb, top_k=top_k)

    if not candidates:
        console.print("[dim]No matching deltas found.[/dim]")
        return

    table = Table(title=f"LSH Retrieval Results for: {query[:60]}...")
    table.add_column("Delta ID", style="cyan")
    table.add_column("Window", justify="right")
    table.add_column("Trust", justify="right")
    table.add_column("Tags", style="yellow")

    with _get_db() as db:
        for did, widx in candidates:
            entry = db.get(did)
            if entry:
                table.add_row(
                    did,
                    str(widx),
                    f"{entry['trust']:.2f}",
                    ", ".join(json.loads(entry.get("tags", "[]"))[:3]),
                )

    console.print(table)

    lsh_stats = lsh.stats()
    console.print(f"[dim]LSH: {lsh_stats['indexed_deltas']} deltas indexed, "
                  f"{lsh_stats['unique_buckets']} buckets[/dim]")


# ── graph ───────────────────────────────────────────────────────────────

@app.command()
def graph(
    action: str = typer.Argument(..., help="Action: link, show, unlink"),
    source: str = typer.Option("", help="Source delta ID"),
    edge_type: str = typer.Option("", help="Edge type (contradicts, compatible, ...)"),
    target: str = typer.Option("", help="Target delta ID"),
):
    """Manage delta graph edges."""
    from pdga.graph.edges import EdgeType, EdgeOps

    with _get_db() as db:
        ops = EdgeOps(db)

        if action == "link":
            if not source or not edge_type or not target:
                console.print("[red]--source, --edge-type, and --target required[/red]")
                raise typer.Exit(1)

            try:
                et = EdgeType(edge_type)
            except ValueError:
                valid = ", ".join(e.value for e in EdgeType)
                console.print(f"[red]Invalid edge type. Valid: {valid}[/red]")
                raise typer.Exit(1)

            ops.add(source, et, target)
            console.print(f"[green]Linked: {source} --[{et.value}]--> {target}[/green]")

        elif action == "show":
            if source:
                edges = ops.find_related(source)
            else:
                with _get_db() as db2:
                    rows = db2.conn.execute("SELECT * FROM edges LIMIT 50").fetchall()
                    edges = [
                        {"source_id": r[0], "target_id": r[1], "edge_type": r[2], "weight": r[3]}
                        for r in rows
                    ]

            if not edges:
                console.print("[dim]No edges found.[/dim]")
                return

            table = Table(title="Graph Edges")
            table.add_column("Source", style="cyan")
            table.add_column("Type", style="magenta")
            table.add_column("Target", style="green")
            for e in edges:
                table.add_row(e["source_id"], e["edge_type"], e["target_id"])
            console.print(table)

        elif action == "unlink":
            if not source or not edge_type or not target:
                console.print("[red]--source, --edge-type, and --target required[/red]")
                raise typer.Exit(1)

            try:
                et = EdgeType(edge_type)
            except ValueError:
                raise typer.Exit(1)

            ops.remove(source, et, target)
            console.print(f"[yellow]Unlinked: {source} --[{et.value}]--> {target}[/yellow]")

        else:
            console.print(f"[red]Unknown action: {action}. Use: link, show, unlink[/red]")


# ── show ────────────────────────────────────────────────────────────────

@app.command()
def show(
    delta_id: str = typer.Argument(..., help="Delta ID to display"),
):
    """Show full details and source text of a stored delta."""
    import json

    with _get_db() as db:
        entry = db.get(delta_id)

    if entry is None:
        console.print(f"[red]Delta not found: {delta_id}[/red]")
        raise typer.Exit(1)

    from pdga.delta.io import load_delta
    from pathlib import Path

    delta = load_delta(Path(entry["path"]))

    table = Table(title=f"Delta: {delta_id}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Type", entry["delta_type"])
    table.add_row("Model", entry["base_model"])
    table.add_row("Crystal layer", str(delta.manifest.crystal_layer))
    table.add_row("Injection layer", str(delta.manifest.injection_layer))
    table.add_row("Windows", str(entry["num_windows"]))
    table.add_row("Trust", f"{entry['trust']:.2f}")
    table.add_row("Tags", ", ".join(json.loads(entry.get("tags", "[]"))))
    table.add_row("Created", entry["created_at"])
    table.add_row("Path", entry["path"])
    table.add_row("Source URL", entry.get("source_url", ""))
    console.print(table)

    if entry.get("source_text"):
        console.print()
        console.print(Panel(
            entry["source_text"],
            title=f"Source Text ({entry['delta_id']})",
            border_style="dim",
        ))

    edges = db.conn.execute(
        "SELECT * FROM edges WHERE source_id=? OR target_id=?",
        (delta_id, delta_id),
    ).fetchall()
    if edges:
        console.print()
        edge_table = Table(title="Graph Edges")
        edge_table.add_column("Source", style="cyan")
        edge_table.add_column("Type", style="magenta")
        edge_table.add_column("Target", style="green")
        for e in edges:
            edge_table.add_row(e[0], e[1], e[2])
        console.print(edge_table)


# ── stats ───────────────────────────────────────────────────────────────

@app.command()
def stats():
    """Show PDGA database and index statistics."""
    with _get_db() as db:
        total = db.count()
        context = db.count("context")
        weight = db.count("weight")
        edge_count = db.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

    lsh_stats = {"indexed_deltas": 0, "unique_buckets": 0, "total_entries": 0}
    import sqlite3
    from pdga.db.schema import get_db_path
    lsh_conn = sqlite3.connect(str(get_db_path()))
    try:
        lsh_total = lsh_conn.execute("SELECT COUNT(*) FROM lsh_tables").fetchone()[0]
        lsh_deltas = lsh_conn.execute("SELECT COUNT(DISTINCT delta_id) FROM lsh_tables").fetchone()[0]
        lsh_buckets = lsh_conn.execute("SELECT COUNT(DISTINCT bucket_key || ':' || table_idx) FROM lsh_tables").fetchone()[0]
        lsh_stats = {"indexed_deltas": lsh_deltas, "unique_buckets": lsh_buckets, "total_entries": lsh_total}
    finally:
        lsh_conn.close()

    table = Table(title="PDGA System Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total deltas", str(total))
    table.add_row("Context deltas", str(context))
    table.add_row("Weight deltas", str(weight))
    table.add_row("Graph edges", str(edge_count))
    table.add_row("LSH indexed", str(lsh_stats["indexed_deltas"]))
    table.add_row("LSH buckets", str(lsh_stats["unique_buckets"]))
    table.add_row("LSH entries", str(lsh_stats["total_entries"]))
    console.print(table)


# ── version ─────────────────────────────────────────────────────────────

@app.command()
def version():
    """Show PDGA version."""
    console.print(f"[bold]pdga[/bold] v{__version__}")


if __name__ == "__main__":
    app()
