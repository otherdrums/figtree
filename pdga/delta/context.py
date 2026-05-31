"""ContextDelta — boundary residual representation of ingested text."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from pdga.delta.base import Delta, DeltaManifest, _write_json, _read_json


class ContextDelta(Delta):
    """Represents ingested text as boundary residuals at the crystal layer.

    One boundary residual per window (last token at crystal layer) plus
    optional KV cache files for instant prefill during generation.
    """

    def __init__(
        self,
        manifest: DeltaManifest,
        boundaries: np.ndarray,
        window_tokens: list[list[int]],
        boundary_positions: Optional[list[int]] = None,
        fact_tokens: Optional[list[list[int]]] = None,
        dynamic_labels: Optional[list[str]] = None,
        source_text: str = "",
        trust: float = 0.5,
        source_url: str = "",
        tags: Optional[list[str]] = None,
        kv_projections: Optional[dict] = None,
    ):
        self.manifest = manifest
        self.boundaries = boundaries
        self.window_tokens = window_tokens
        self.boundary_positions = boundary_positions or []
        self.fact_tokens = fact_tokens or []
        self.dynamic_labels = dynamic_labels or []
        self.source_text = source_text
        self.trust = trust
        self.source_url = source_url
        self.tags = tags or []
        self.kv_projections = kv_projections
        self.path = None

        if manifest.delta_type != "context":
            manifest.delta_type = "context"
        manifest.num_windows = len(window_tokens)

    @property
    def is_hybrid(self) -> bool:
        return len(self.fact_tokens) > 0

    @property
    def num_windows(self) -> int:
        return len(self.window_tokens)

    @property
    def hidden_size(self) -> int:
        return self.manifest.hidden_size

    def get_boundary(self, window_idx: int) -> np.ndarray:
        """Return the boundary residual for a specific window."""
        if window_idx < 0 or window_idx >= self.num_windows:
            raise IndexError(f"Window {window_idx} out of range [0, {self.num_windows})")
        return self.boundaries[window_idx]

    def get_window_tokens(self, window_idx: int) -> list[int]:
        """Return token IDs for a specific window."""
        return self.window_tokens[window_idx]

    def save(self, path: Path) -> None:
        """Write ContextDelta to a .pdga directory.

        If path doesn't end with the delta ID directory, creates it.
        """
        if path.name != f"{self.manifest.delta_id}.pdga":
            path = path / f"{self.manifest.delta_id}.pdga"
        path.mkdir(parents=True, exist_ok=True)

        _write_json(path / "manifest.json", self.manifest.to_dict())
        np.save(path / "boundaries.npy", self.boundaries)

        tokens_dict = {str(i): np.array(t, dtype=np.int32) for i, t in enumerate(self.window_tokens)}
        np.savez(path / "window_tokens.npz", **tokens_dict)

        metadata = {
            "trust": self.trust,
            "source_url": self.source_url,
            "tags": self.tags,
            "boundary_positions": self.boundary_positions,
            "dynamic_labels": self.dynamic_labels,
        }
        _write_json(path / "metadata.json", metadata)

        if self.fact_tokens:
            chunks_dict = {str(i): np.array(c, dtype=np.int32) for i, c in enumerate(self.fact_tokens)}
            np.savez(path / "fact_tokens.npz", **chunks_dict)

        if self.kv_projections is not None:
            kv_path = path / "kv_projections.npz"
            np.savez(kv_path, **self.kv_projections)

        self.path = path

    @classmethod
    def load(cls, path: Path) -> "ContextDelta":
        """Load ContextDelta from a .pdga directory."""
        manifest = DeltaManifest.from_dict(_read_json(path / "manifest.json"))
        boundaries = np.load(path / "boundaries.npy", mmap_mode="r")
        tokens_npz = np.load(path / "window_tokens.npz", allow_pickle=True)
        window_tokens = [tokens_npz[str(i)].tolist() for i in range(manifest.num_windows)]

        metadata = {}
        meta_path = path / "metadata.json"
        if meta_path.exists():
            metadata = _read_json(meta_path)

        kv = None
        kv_path = path / "kv_projections.npz"
        if kv_path.exists():
            kv = dict(np.load(kv_path, allow_pickle=True))

        fact_tokens = []
        ft_path = path / "fact_tokens.npz"
        if ft_path.exists():
            ft_data = np.load(ft_path, allow_pickle=True)
            for key in sorted(ft_data.files, key=lambda k: int(k)):
                fact_tokens.append(ft_data[key].tolist())

        delta = cls(
            manifest=manifest,
            boundaries=boundaries,
            window_tokens=window_tokens,
            boundary_positions=metadata.get("boundary_positions", []),
            fact_tokens=fact_tokens if fact_tokens else None,
            dynamic_labels=metadata.get("dynamic_labels", []),
            source_text=metadata.get("source_text", ""),
            trust=metadata.get("trust", 0.5),
            source_url=metadata.get("source_url", ""),
            tags=metadata.get("tags", []),
            kv_projections=kv,
        )
        delta.path = path
        return delta
