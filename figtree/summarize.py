"""Hierarchical Figments: give higher-level Images their own summarized boundary.

An Image figment normally just links its children. By default its ``boundary``
is copied from the first child (cheap, but not a real summary). When
``summarize_images=True`` is passed to ingestion, this module generates a short
natural-language summary of the children and re-derives a genuine image-level
boundary from that summary, so coarse retrieval can match the Image as a whole
rather than only its first sentence.

Opt-in (default off) to protect low-VRAM users on the 3 GB target card: summary
generation is one extra forward pass + one boundary pass per ingested text.
"""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from figtree.figment import Figment
from figtree.ingest import boundary_for_text
from figtree.kernel.prompt import build_prompt_ids


def summarize_image(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    children: list[Figment],
    max_new_tokens: int = 64,
    crystal_layer: int | None = None,
) -> tuple[str, object, object, object]:
    """Summarize ``children`` texts and return (summary, boundary, boundaries, emb).

    The summary is generated with the model's chat template (thinking off), then
    its boundary vectors are extracted via :func:`boundary_for_text` so the Image
    figment carries a real summarized representation.
    """
    child_texts = [c.text for c in children if c.text]
    if not child_texts:
        raise ValueError("Cannot summarize an Image with no child text.")

    joined = "\n".join(f"- {t}" for t in child_texts)
    prompt = (
        "Summarize the following related statements into a single concise "
        f"statement that preserves the key facts:\n{joined}"
    )
    prompt_ids = build_prompt_ids(tokenizer, prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)

    summary_text = ""
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.15,
        )
        gen_ids = out[0][len(prompt_ids):]
        summary_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    if not summary_text:
        summary_text = joined  # fall back to concatenation if generation is empty

    boundary, boundaries, emb = boundary_for_text(
        model, tokenizer, summary_text, crystal_layer=crystal_layer
    )
    return summary_text, boundary, boundaries, emb
