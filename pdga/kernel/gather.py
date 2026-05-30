"""Multi-stream parallel generation orchestrator.

Each stream runs its own set of deltas independently. Within a stream,
each delta generates at full fidelity. Streams produce side-by-side
panels of per-delta results with metadata attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from pdga.delta.context import ContextDelta
from pdga.kernel.reference import generate as generate_single
from pdga.kernel.stream import StreamConfig


@dataclass
class StreamResult:
    stream_id: str
    is_conscious: bool
    sample_temp: float
    delta_temps: dict[str, float]
    delta_results: list[dict]


@dataclass
class ThinkResult:
    prompt: str
    streams: list[StreamResult]
    conscious_output: str

    @property
    def subconscious_outputs(self) -> list[StreamResult]:
        return [s for s in self.streams if not s.is_conscious]


def think(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    streams: list[StreamConfig],
    deltas_map: dict[str, ContextDelta],
    max_new_tokens: int = 256,
    top_k: int = 50,
    top_p: float = 0.95,
) -> ThinkResult:
    """Run multiple generation streams in parallel.

    Each stream generates from its assigned deltas independently at full
    fidelity. Streams differ only in sampling temperature — the model
    generates from each delta's full context.
    """
    if not any(s.conscious for s in streams):
        if streams:
            streams[0].conscious = True

    results: list[StreamResult] = []

    for stream_cfg in streams:
        stream_deltas = []
        for did in stream_cfg.delta_ids:
            if did in deltas_map:
                stream_deltas.append(deltas_map[did])

        delta_results = []
        for delta in stream_deltas:
            outputs = generate_single(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                deltas=[delta],
                max_new_tokens=max_new_tokens,
                sample_temp=stream_cfg.sample_temp,
                top_k=top_k,
                top_p=top_p,
            )
            delta_results.extend(outputs)

        results.append(StreamResult(
            stream_id=stream_cfg.id,
            is_conscious=stream_cfg.conscious,
            sample_temp=stream_cfg.sample_temp,
            delta_temps=stream_cfg.delta_temps,
            delta_results=delta_results,
        ))

    conscious = next(
        (s for s in results if s.is_conscious),
        results[0] if results else None,
    )

    conscious_text = ""
    if conscious and conscious.delta_results:
        conscious_text = "\n\n".join(
            r["generated_text"] for r in conscious.delta_results
        )

    return ThinkResult(
        prompt=prompt,
        streams=results,
        conscious_output=conscious_text,
    )
