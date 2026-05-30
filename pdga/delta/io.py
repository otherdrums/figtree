"""PDGA format I/O utilities."""

from pathlib import Path
from typing import Optional

from pdga.delta.base import Delta
from pdga.delta.context import ContextDelta


def load_delta(path: Path) -> Delta:
    """Load any delta type from a .pdga directory."""
    from pdga.delta.base import _read_json

    manifest = _read_json(path / "manifest.json")
    delta_type = manifest.get("delta_type", "context")

    if delta_type == "context":
        return ContextDelta.load(path)
    elif delta_type == "weight":
        raise NotImplementedError("WeightDelta loading not yet implemented")
    else:
        raise ValueError(f"Unknown delta type: {delta_type}")


def save_delta(delta: Delta, output_dir: Path) -> None:
    """Save a delta to a .pdga directory."""
    name = f"{delta.manifest.delta_id}.pdga"
    path = output_dir / name
    delta.save(path)
