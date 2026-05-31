"""PDGA generation engine — multi-delta generation with KV caching.

Provides:
  - generate: unified multi-delta generator with KV caching
  - compressed_generate: compressed path using boundary residuals
  - uncompressed_generate: standard path with full text context
  - ModelAdapter: model-agnostic forward pass wrapper
  - detect_model_features: detect model capabilities
"""

from pdga.generation.engine import generate
from pdga.generation.compressed import (
    compressed_generate,
    uncompressed_generate,
    ModelAdapter,
    detect_model_features,
)

__all__ = [
    "generate",
    "compressed_generate",
    "uncompressed_generate",
    "ModelAdapter",
    "detect_model_features",
]
