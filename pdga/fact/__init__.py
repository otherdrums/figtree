"""PDGA v2 — Fact-centric architecture.

Everything is a Fact. Boundaries are the only stored representation.
Custom CUDA kernel projects boundaries through W_k/W_v for direct KV injection.
"""

from pdga.fact.primitive import Fact
from pdga.fact.ingest import ingest_text_to_facts
from pdga.fact.generate import FactGenerator
from pdga.fact.graph import FactGraph

__all__ = ["Fact", "ingest_text_to_facts", "FactGenerator", "FactGraph"]
