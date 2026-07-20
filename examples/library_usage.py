#!/usr/bin/env python3
"""Minimal, domain-neutral example of using FigTree as a library.

This script demonstrates the core Figment substrate with no news / Davos framing:
ingest a short encyclopedic paragraph, store it, retrieve by id and by boundary
similarity, and generate a faithful answer. Run it with a CUDA (or CPU) capable
machine; it downloads the reference 4-bit model on first use.

    python3 examples/library_usage.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from figtree import (
    FigmentGenerator,
    FigmentStore,
    connect,
    ingest_text_to_figments,
    load_model,
)

PARAGRAPH = (
    "The Library of Alexandria was founded in the 3rd century BCE in Egypt. "
    "It is estimated to have held between 40,000 and 400,000 scrolls. "
    "The main collection was destroyed by fire in 48 BCE during the civil war."
)


def main() -> None:
    print("Loading model (first run downloads weights)...")
    model, tokenizer = load_model()

    # 1. Ingest text into atomic Figments with boundaries + K/V capture.
    tmp = Path(tempfile.mkdtemp(prefix="figtree_lib_"))
    store: FigmentStore = connect(tmp / "library.lance")
    figments = ingest_text_to_figments(
        model=model,
        tokenizer=tokenizer,
        text=PARAGRAPH,
        source_id="encyclopedia",
        store=store,
        trust=0.9,
    )
    print(f"Ingested {len(figments)} figments into {store}.")

    # 2. Retrieve by ID and by boundary similarity (store is queryable).
    first = figments[0]
    fetched = store.get(first.figment_id)
    print(f"Fetched by id: {fetched.text[:60]!r}")

    # Use a figment's own boundary as a query vector -> it should rank first.
    near = store.search(first.boundary, k=3)
    print(f"Nearest match: {near[0][0].text[:60]!r} (distance {near[0][1]:.3f})")

    atomic = [f for f in figments if not f.is_image() and not f.is_trust_assertion()]

    # 3. Faithful generation: recall every figure verbatim in one pass.
    gen = FigmentGenerator(model, tokenizer)
    result = gen.generate_faithful(
        figments=atomic,
        prompt=(
            "List EVERY figure from the text verbatim as a bullet list: each "
            "number, percent, year, and named entity. Do not summarize."
        ),
        source_texts=[PARAGRAPH],
        max_new_tokens=200,
    )
    print("\nGenerated (faithful recall):")
    print(result["generated_text"])
    print(f"\nRecall score: {result['recall_score']:.2f}")


if __name__ == "__main__":
    main()
