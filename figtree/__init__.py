"""Figtree — grow coherent Images from Figments.

Everything is a Figment. An Image is a Figment with children.

Public API (importable for use in higher-level projects):

    from figtree import (
        Figment, FigmentGenerator, Figtree, ingest_text_to_figments,
        connect, KVCacheManager, recall_score, missing_atoms, extract_atoms,
        summarize_image,
    )
"""

from __future__ import annotations

from figtree.figment import Figment
from figtree.ingest import ingest_text_to_figments
from figtree.generate import FigmentGenerator
from figtree.graph import Figtree
from figtree.lancedb_store import FigmentStore, connect
from figtree.kv_cache_manager import KVCacheManager
from figtree.model import load_model
from figtree.recall import extract_atoms, missing_atoms, recall_score
from figtree.summarize import summarize_image

try:  # version is single-sourced from package metadata
    from importlib.metadata import version as _version, PackageNotFoundError

    try:
        __version__ = _version("figtree")
    except PackageNotFoundError:  # running from a source checkout without install
        __version__ = "0.2.1"
except Exception:  # pragma: no cover
    __version__ = "0.2.1"

__all__ = [
    "Figment",
    "FigmentGenerator",
    "Figtree",
    "FigmentStore",
    "ingest_text_to_figments",
    "connect",
    "KVCacheManager",
    "load_model",
    "extract_atoms",
    "missing_atoms",
    "recall_score",
    "summarize_image",
    "__version__",
]
