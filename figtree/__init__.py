"""Figtree — grow coherent Images from Figments.

Everything is a Figment. An Image is a Figment with children.
"""

from figtree.figment import Figment
from figtree.ingest import ingest_text_to_figments
from figtree.generate import FigmentGenerator
from figtree.graph import Figtree

__all__ = ["Figment", "ingest_text_to_figments", "FigmentGenerator", "Figtree"]
__version__ = "0.2.0"
