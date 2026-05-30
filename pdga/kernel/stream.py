"""Stream state management for PDGA generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from pdga.delta.context import ContextDelta


@dataclass
class StreamConfig:
    id: str
    delta_ids: list[str] = field(default_factory=list)
    delta_temps: dict[str, float] = field(default_factory=dict)
    sample_temp: float = 0.7
    conscious: bool = False


@dataclass
class StreamState:
    """Runtime state for one generation stream.

    Manages the loaded deltas, their pre-built KV caches, the seeded
    past_key_values, and the active generation state.
    """

    stream_config: StreamConfig
    deltas: list[ContextDelta] = field(default_factory=list)
    delta_kv_caches: list[dict] = field(default_factory=list)
    past_key_values: list | None = None
    generated_tokens: list[int] = field(default_factory=list)

    @property
    def stream_id(self) -> str:
        return self.stream_config.id

    @property
    def is_conscious(self) -> bool:
        return self.stream_config.conscious

    def get_delta_temp(self, delta_id: str) -> float:
        return self.stream_config.delta_temps.get(delta_id, 1.0)

    def output_text(self, tokenizer) -> str:
        return tokenizer.decode(self.generated_tokens)
