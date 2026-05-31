"""Kernel module — generation and state management."""

from pdga.kernel.reference import generate, generate_hybrid
from pdga.kernel.inject import generate_from_injection
from pdga.kernel.gather import think, ThinkResult, StreamResult
from pdga.kernel.stream import StreamConfig, StreamState
from pdga.kernel.prompt import build_prompt_ids
from pdga.kernel.multi import generate_multi

__all__ = [
    "generate",
    "generate_hybrid",
    "generate_from_injection",
    "generate_multi",
    "think",
    "ThinkResult",
    "StreamResult",
    "StreamConfig",
    "StreamState",
    "build_prompt_ids",
]
