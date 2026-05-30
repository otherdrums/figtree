"""Kernel module — generation and state management."""

from pdga.kernel.reference import generate, generate_hybrid
from pdga.kernel.inject import generate_from_injection
from pdga.kernel.residual_inject import generate_from_residuals
from pdga.kernel.gather import think, ThinkResult, StreamResult
from pdga.kernel.stream import StreamConfig, StreamState

__all__ = [
    "generate",
    "generate_hybrid",
    "generate_from_residuals",
    "generate_from_injection",
    "think",
    "ThinkResult",
    "StreamResult",
    "StreamConfig",
    "StreamState",
]
