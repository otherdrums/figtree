"""Delta base types and manifest schema."""

from __future__ import annotations

import json
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

DeltaType = Literal["context", "weight"]


@dataclass
class DeltaManifest:
    version: str = "0.1.0"
    delta_id: str = ""
    delta_type: DeltaType = "context"
    base_model_id: str = ""
    hidden_size: int = 0
    num_layers: int = 0
    crystal_layer: int = 0
    window_size: int = 200
    num_windows: int = 0
    injection_layer: int = 0
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DeltaManifest":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Delta(ABC):
    """Abstract base for all delta types (context, weight)."""

    manifest: DeltaManifest
    path: Optional[Path] = None

    @abstractmethod
    def save(self, path: Path) -> None:
        """Write delta to a .pdga directory."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "Delta":
        """Load delta from a .pdga directory."""
        ...

    @property
    def delta_id(self) -> str:
        return self.manifest.delta_id

    @staticmethod
    def generate_id(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())
