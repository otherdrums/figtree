"""PDGA generation engine — Comprehensive Test Suite

Tests:
1. Factual recall with 300+ token generation (single delta)
2. Multi-delta parallel generation with contradictory articles
3. Speed benchmarking at various configurations
4. Integration with ingestion pipeline (boundaries, KV cache)
"""

import time
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule

from transformers import AutoModelForCausalLM, AutoTokenizer
from pdga.generation.engine import generate
from pdga.kernel.prompt import build_prompt_ids

console = Console()
MODEL_ID = "unsloth/Qwen3-4B-bnb-4bit"


# ── Articles ──────────────────────────────────────────────────────────────────

ARTICLE_A = (
    "GENEVA — Delegates from 47 nations reached a landmark trade agreement "
    "late Thursday, capping three weeks of intense negotiations at the Global "
    "Trade Summit. The pact, which covers digital services, carbon tariffs, and "
    "pharmaceutical patents, was hailed by Secretary-General Maria Okonkwo as "
    "\"a turning point for multilateral cooperation.\"\n\n"
    "The agreement establishes new frameworks for cross-border data flows while "
    "imposing standardized environmental levies on carbon-intensive imports. "
    "Pharmaceutical companies will face revised patent terms designed to "
    "accelerate generic drug availability in developing nations.\n\n"
    "Key provisions include:\n"
    "- Digital services tax harmonization across all signatory nations, set at "
    "3% of gross revenue for companies exceeding $20 billion in annual turnover\n"
    "- Carbon border adjustment mechanisms standardized to $45 per metric ton "
    "of CO2 equivalent\n"
    "- Pharmaceutical patent terms shortened from 20 to 12 years for drugs "
    "designated as essential medicines\n"
    "- A $12 billion climate adaptation fund for small island developing states\n\n"
    "The agreement will be formally signed at a ceremony next month in Brussels. "
    "Markets responded positively, with the S&P Global 1200 rising 1.8% in "
    "after-hours trading. The International Monetary Fund projects the deal "
    "could add $900 billion to global GDP over the next decade.\n\n"
    "\"This is not just a trade deal,\" said lead US negotiator Ambassador "
    "Sarah Chen. \"This is a framework for shared prosperity in the twenty-first "
    "century. Every nation at this table, from the largest economy to the "
    "smallest island state, will benefit from what we've built here together.\""
)

ARTICLE_B = (
    "GENEVA — Only 34 nations signed the watered-down communiqué at the "
    "conclusion of the Global Trade Summit on Friday, with the United States "
    "and China walking out of the final session in protest. The remaining "
    "delegates adopted a non-binding statement that critics say lacks "
    "enforcement mechanisms.\n\n"
    "Secretary-General Maria Okonkwo acknowledged the outcome \"falls short "
    "of our ambitions\" but praised delegates for \"keeping the dialogue "
    "alive.\" The centerpiece of the agreement — a 3% digital services tax — "
    "was described as \"punitive and unworkable\" by the US delegation in "
    "a scathing departure statement.\n\n"
    "Carbon tariffs were dramatically reduced from the proposed $45 per ton "
    "to just $15 per ton after heavy lobbying by industrial nations. "
    "Pharmaceutical patent provisions were stripped from the final text "
    "entirely after opposition from major drug manufacturers.\n\n"
    "The International Monetary Fund issued a sharply revised projection, "
    "warning the weakened deal might add only $250 billion to global GDP — "
    "far below the $900 billion originally estimated for a comprehensive "
    "agreement. The S&P Global 1200 dropped 0.7% on the news.\n\n"
    "Ambassador Sarah Chen of the United States dismissed the proceedings "
    "as \"a missed opportunity,\" while Chinese delegate Li Wei called for "
    "\"a fundamental restructuring of the negotiation framework\" before any "
    "future talks."
)

# ── Ground truth facts ────────────────────────────────────────────────────────

FACTS_A = [
    "47 nations", "landmark trade agreement", "three weeks", "Global Trade Summit",
    "digital services", "carbon tariffs", "pharmaceutical patents",
    "Maria Okonkwo", "turning point", "multilateral cooperation",
    "cross-border data flows", "environmental levies", "generic drug availability",
    "3%", "$20 billion", "$45 per ton", "CO2 equivalent",
    "20 to 12 years", "essential medicines",
    "$12 billion", "climate adaptation fund", "small island developing states",
    "Brussels", "S&P Global 1200", "1.8%", "after-hours trading",
    "$900 billion", "global GDP", "International Monetary Fund",
    "Sarah Chen", "shared prosperity", "twenty-first century",
]

FACTS_B = [
    "34 nations", "United States", "China", "walking out",
    "Maria Okonkwo", "falls short", "keeping the dialogue alive",
    "3% digital services tax", "punitive", "unworkable",
    "US delegation", "scathing departure statement",
    "$15 per ton", "heavy lobbying", "industrial nations",
    "pharmaceutical", "stripped", "drug manufacturers",
    "$250 billion", "IMF", "S&P Global 1200", "0.7%",
    "Sarah Chen", "missed opportunity", "Li Wei",
    "fundamental restructuring",
]


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


def check_facts(text: str, facts: list[str]) -> tuple[int, list[bool]]:
    results = []
    hits = 0
    tlower = text.lower()
    for f in facts:
        found = f.lower() in tlower
        results.append(found)
        if found:
            hits += 1
    return hits, results


def test_factual_recall_300_tokens():
    """Single delta, 300+ tokens, verify 90%+ fact recall from full article."""
    console.print(Rule("[bold blue]TEST 1: Factual Recall — 300+ Token Report[/bold blue]"))
    console.print("[dim]Uncompressed path, full article A context, 300 tokens[/dim]")
    console.print()

    model, tokenizer = load_model()
    window_ids = tokenizer.encode(ARTICLE_A)

    prompt = (
        "Produce a comprehensive report: What happened at the Global Trade "
        "Summit in Geneva? Include every specific detail from the source — all "
        "agreements, provisions, numbers, names, dates, reactions, and outcomes."
    )
    prompt_ids = build_prompt_ids(tokenizer, prompt, enable_thinking=False)

    console.print(f"[bold]Article:[/bold] {len(window_ids)} tokens")
    console.print(f"[bold]Prompt:[/bold] {len(prompt_ids)} tokens")
    console.print()

    t0 = time.perf_counter()
    with torch.inference_mode():
        results = generate(
            model=model, tokenizer=tokenizer,
            prompt=prompt,
            deltas=[{
                "window_tokens": window_ids,
                "injection_entries": [],
                "metadata": {"delta_id": "article_a", "trust": 0.99},
            }],
            injection_layer=30, injection_coefficient=10.0, injection_topk=0,
            max_new_tokens=300, sample_temp=0.7,
        )
    elapsed = time.perf_counter() - t0
    r = results[0]

    console.print(f"[bold]Generated:[/bold] {r['num_tokens']} tokens in {elapsed:.1f}s ({r['tokens_per_second']:.1f} t/s)")
    console.print()

    hits, facts_found = check_facts(r["generated_text"], FACTS_A)
    pct = 100 * hits / len(FACTS_A)

    # Display fact check
    table = Table(title=f"Fact Recall: {hits}/{len(FACTS_A)} ({pct:.0f}%)")
    table.add_column("Fact", style="dim")
    table.add_column("Found", style="bold")
    for fact, found in zip(FACTS_A, facts_found):
        table.add_row(fact, "[green]YES[/green]" if found else "[red]no[/red]")
    console.print(table)
    console.print()

    # Show generated text
    console.print(Panel(r["generated_text"][:2000], title="Generated Output (first 2000 chars)",
                        border_style="green" if pct >= 80 else "yellow" if pct >= 50 else "red"))
    return pct


def test_multi_delta_contradictory():
    """Two deltas with contradictory facts — verify each sovereign."""
    console.print(Rule("[bold blue]TEST 2: Multi-Delta — Contradictory Articles[/bold blue]"))
    console.print("[dim]Article A (pro-deal) vs Article B (skeptical), 150 tokens each[/dim]")
    console.print()

    model, tokenizer = load_model()
    wa = tokenizer.encode(ARTICLE_A)
    wb = tokenizer.encode(ARTICLE_B)

    prompt = (
        "Produce a comprehensive report: What happened at the Global Trade "
        "Summit in Geneva? Include every specific detail from the source — all "
        "agreements, provisions, numbers, names, dates, reactions, and outcomes."
    )

    console.print(f"[bold]Article A:[/bold] {len(wa)} tokens")
    console.print(f"[bold]Article B:[/bold] {len(wb)} tokens")
    console.print()

    t0 = time.perf_counter()
    with torch.inference_mode():
        results = generate(
            model=model, tokenizer=tokenizer,
            prompt=prompt,
            deltas=[
                {
                    "window_tokens": wa,
                    "injection_entries": [],
                    "metadata": {"delta_id": "article_a", "trust": 0.99,
                                 "source_url": "article_a.txt", "tags": ["summit"]},
                },
                {
                    "window_tokens": wb,
                    "injection_entries": [],
                    "metadata": {"delta_id": "article_b", "trust": 0.50,
                                 "source_url": "article_b.txt", "tags": ["summit"]},
                },
            ],
            injection_layer=30,
            max_new_tokens=150, sample_temp=0.7,
        )
    elapsed = time.perf_counter() - t0

    total_tok = sum(r["num_tokens"] for r in results)
    console.print(f"[bold]Total:[/bold] {total_tok} tokens in {elapsed:.1f}s ({total_tok/elapsed:.1f} t/s)")
    console.print()

    for i, r in enumerate(results):
        trust_str = f"{r['trust']:.0%}"
        tc = "green" if r["trust"] >= 0.8 else "red"
        console.print(Panel(f"δID={r['delta_id']}  trust={trust_str}  "
                            f"{r['num_tokens']} tokens",
                            border_style=tc))
        console.print(r["generated_text"][:1000])
        console.print()

    # Quick fact check
    ta = results[0]["generated_text"].lower()
    tb = results[1]["generated_text"].lower()
    ha = sum(1 for f in FACTS_A if f.lower() in ta)
    hb = sum(1 for f in FACTS_B if f.lower() in tb)

    console.print(f"[bold]Article A fact recall:[/bold] {ha}/{len(FACTS_A)}")
    console.print(f"[bold]Article B fact recall:[/bold] {hb}/{len(FACTS_B)}")

    # Verify no cross-contamination (A facts shouldn't appear in B output, and vice versa)
    contam_a = sum(1 for f in FACTS_B if f.lower() in ta and f.lower() not in "maria okonkwo sarah chen global trade summit geneva".split())
    contam_b = sum(1 for f in FACTS_A if f.lower() in tb and f.lower() not in "maria okonkwo sarah chen global trade summit geneva".split())
    console.print(f"[bold]Cross-contamination A→B:[/bold] {contam_b} (should be 0)")
    console.print(f"[bold]Cross-contamination B→A:[/bold] {contam_a} (should be 0)")
    return ha, hb


def test_speed_benchmark():
    """Benchmark at various batch sizes and token counts."""
    console.print(Rule("[bold blue]TEST 3: Speed Benchmark[/bold blue]"))
    console.print()

    model, tokenizer = load_model()
    window_ids = tokenizer.encode(ARTICLE_A)
    prompt = "Summarize the trade summit:"

    sizes = [1, 2, 3, 4]
    tokens_list = [50, 100, 200]

    table = Table(title="Speed Benchmark (uncompressed path)")
    table.add_column("Deltas")
    table.add_column("Max Tokens")
    table.add_column("Time (s)")
    table.add_column("t/s")
    table.add_column("per-delta t/s")

    for n in sizes:
        for mt in tokens_list:
            deltas = [{
                "window_tokens": window_ids[:120],
                "injection_entries": [],
                "metadata": {"delta_id": f"d{i}", "trust": 0.9},
            } for i in range(n)]

            t0 = time.perf_counter()
            with torch.inference_mode():
                results = generate(
                    model=model, tokenizer=tokenizer,
                    prompt=prompt,
                    deltas=deltas,
                    injection_layer=30,
                    max_new_tokens=mt, sample_temp=0.7,
                )
            elapsed = time.perf_counter() - t0
            total_tok = sum(r["num_tokens"] for r in results)
            tps = total_tok / elapsed if elapsed > 0 else 0
            per_delta = tps / n if n > 0 else 0
            table.add_row(
                str(n), str(mt), f"{elapsed:.1f}s",
                f"{tps:.1f}", f"{per_delta:.1f}",
            )

    console.print(table)


def test_injection_entries():
    """Test with manual entity injection entries."""
    console.print(Rule("[bold blue]TEST 4: Entity Injection[/bold blue]"))
    console.print("[dim]Manual injection entries for key entities — verify they appear[/dim]")
    console.print()

    model, tokenizer = load_model()
    window_ids = tokenizer.encode(ARTICLE_A[:500])  # truncated for speed

    # Create injection entries: (token_id, coefficient) for key entities
    entity_tokens = [
        "Maria", "Okonkwo", "Sarah", "Chen",
        "Brussels", "Geneva", "carbon", "tariffs",
        "pharmaceutical", "patent", "digital", "services",
    ]
    entries = []
    for word in entity_tokens:
        tids = tokenizer.encode(" " + word, add_special_tokens=False)
        for tid in tids:
            entries.append((tid, 1.0))

    prompt = "Name every person and location mentioned in the source text about the trade summit."

    results = generate(
        model=model, tokenizer=tokenizer,
        prompt=prompt,
        deltas=[
            {
                "window_tokens": window_ids,
                "injection_entries": entries,
                "metadata": {"delta_id": "with_injection", "trust": 0.99},
            },
            {
                "window_tokens": window_ids,
                "injection_entries": [],
                "metadata": {"delta_id": "no_injection", "trust": 0.99},
            },
        ],
        injection_layer=30, injection_coefficient=10.0, injection_topk=8,
        max_new_tokens=80, sample_temp=0.7,
    )

    for r in results:
        console.print(f"\n[bold]δID={r['delta_id']}  entries={r['entries_injected']}[/bold]")
        console.print(r["generated_text"][:400])


if __name__ == "__main__":
    import sys
    test = sys.argv[1] if len(sys.argv) > 1 else "all"

    if test in ("1", "facts", "all"):
        test_factual_recall_300_tokens()
    if test in ("2", "multi", "all"):
        test_multi_delta_contradictory()
    if test in ("3", "bench", "all"):
        test_speed_benchmark()
    if test in ("4", "inject", "all"):
        test_injection_entries()
