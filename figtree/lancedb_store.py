"""LanceDB-backed storage for Figments.

Figments are persisted as rows in a LanceDB table. This enables:

- Compression (zstd for text/metadata, lz4 for hot columns, Lance's automatic
  dictionary/FSST encodings) configured at table creation.
- Remote / object storage backends (``s3://``, ``gs://``, ``az://``) by passing
  an object-store URI plus ``storage_options``.
- ANN similarity search over the ``boundary`` vector column.

K/V caches are intentionally NOT stored inside the Lance row (they are large,
variable-shape tensors). Each row carries ``has_kv_cache`` and ``kv_uri``
pointing at an external blob managed by ``figtree.kv_cache_manager``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import lancedb
    from lancedb.db import DBConnection
    from lancedb.pydantic import LanceModel, Vector
except Exception:  # pragma: no cover - import guard for type checking
    lancedb = None
    DBConnection = Any
    LanceModel = object

    class Vector:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

from figtree.figment import Figment

DEFAULT_TABLE = "figments"
DEFAULT_LOCAL_URI = "./figtree.lance"

# Compression strategy for writes. Text/metadata get zstd (ratio); small/hot
# string columns get lz4. Lance still applies automatic dictionary/FSST for
# string columns and bit-packing for numeric columns.
COMPRESSION_OVERRIDES: dict[str, Any] = {
    "columns": {
        "figment_id": "lz4",
        "source_id": "lz4",
        "edge_type": "lz4",
        "kv_uri": "lz4",
        "text": "zstd",
        "meta": "zstd",
    },
}

# Storage options applied at connection / table-creation time.
STORAGE_OPTIONS: dict[str, str] = {
    "new_table_data_storage_version": "stable",
}

# Per-column compression overrides, expanded into storage_options at connect
# time (Lance keys: "columns.<name>.compression"). Text/metadata -> zstd (ratio);
# small/hot string columns -> lz4. Lance still applies automatic
# dictionary/FSST for string columns and bit-packing for numerics.
for _col, _comp in COMPRESSION_OVERRIDES.get("columns", {}).items():
    STORAGE_OPTIONS[f"columns.{_col}.compression"] = _comp


def _schema_for(hidden_size: int) -> type[LanceModel]:
    """Build a LanceModel schema bound to the model's hidden size."""

    class FigmentRecord(LanceModel):
        figment_id: str
        text: str
        source_id: str = ""
        edge_type: str | None = None
        trust: float = 0.5
        is_image: bool = False
        has_kv_cache: bool = False
        kv_uri: str | None = None
        children: list[str] = []
        sources: list[str] = []
        # meta is stored as a JSON string for schema portability; converted to a
        # dict on read (see _from_record).
        meta_json: str = "{}"
        boundary: Vector(hidden_size)
        # Flattened (num_layers * hidden_size,) for reconstruction; nullable.
        boundaries: list[float] | None = None
        boundary_emb: list[float] | None = None

    return FigmentRecord


def connect(
    uri: str | Path = DEFAULT_LOCAL_URI,
    storage_options: dict[str, str] | None = None,
    table_name: str = DEFAULT_TABLE,
) -> "FigmentStore":
    """Open (or lazily create) a FigmentStore at ``uri``.

    ``uri`` may be a local directory or an object-store URI (``s3://...``).
    """
    if lancedb is None:
        raise ImportError("lancedb is required for LanceDB storage; install with `pip install lancedb`")
    opts = dict(STORAGE_OPTIONS)
    if storage_options:
        opts.update(storage_options)
    db: DBConnection = lancedb.connect(str(uri), storage_options=opts)
    return FigmentStore(db, table_name=table_name)


class FigmentStore:
    """Thin wrapper over a LanceDB table holding Figment records."""

    def __init__(self, db: DBConnection, table_name: str = DEFAULT_TABLE):
        self.db = db
        self.table_name = table_name
        self._table = None
        self._hidden_size: int | None = None

    # -- table lifecycle ------------------------------------------------- #
    def _ensure_table(self, hidden_size: int):
        if self._table is not None:
            return
        schema = _schema_for(hidden_size)
        if self.table_name in self.db.table_names():
            self._table = self.db.open_table(self.table_name)
            vec_field = self._table.schema.field("boundary")
            existing = vec_field.type.list_size
            if existing != hidden_size:
                raise ValueError(
                    f"Existing table boundary dim {existing} != model hidden_size {hidden_size}. "
                    "Use a different uri or migrate the table."
                )
            self._hidden_size = existing
        else:
            self._table = self.db.create_table(
                self.table_name, schema=schema, mode="create"
            )
            self._hidden_size = hidden_size

    @property
    def table(self):
        if self._table is None:
            if self.table_name in self.db.table_names():
                self._table = self.db.open_table(self.table_name)
            else:
                raise RuntimeError(
                    "Table not created yet; call upsert/get with a Figment first "
                    "(which supplies the hidden size) or call _ensure_table()."
                )
        return self._table

    # -- conversions ----------------------------------------------------- #
    @staticmethod
    def _to_record(f: Figment, hidden_size: int) -> dict[str, Any]:
        # Trust/edge figments may carry a 1-D placeholder boundary; normalize to
        # the model's hidden size so the vector column stays consistent.
        b = f.boundary.astype(np.float32)
        if b.shape[0] != hidden_size:
            b = np.resize(b, (hidden_size,)).astype(np.float32)
        rec: dict[str, Any] = {
            "figment_id": f.figment_id,
            "text": f.text,
            "source_id": f.meta.get("source_id", ""),
            "edge_type": f.meta.get("edge_type"),
            "trust": float(f.trust),
            "is_image": f.is_image(),
            "has_kv_cache": False,
            "kv_uri": None,
            "children": list(f.children),
            "sources": list(f.sources),
            "meta_json": json.dumps(dict(f.meta), default=str),
            "boundary": b.tolist(),
            "boundaries": (
                f.boundaries.astype(np.float32).ravel().tolist()
                if f.boundaries is not None else None
            ),
            "boundary_emb": (
                f.boundary_emb.astype(np.float32).tolist()
                if f.boundary_emb is not None else None
            ),
        }
        if "kv_uri" in f.meta:
            rec["kv_uri"] = f.meta["kv_uri"]
            rec["has_kv_cache"] = bool(f.meta.get("has_kv_cache", False))
        return rec

    @classmethod
    def _from_record(cls, rec: dict[str, Any]) -> Figment:
        boundary = np.asarray(rec["boundary"], dtype=np.float32)
        boundaries = rec.get("boundaries")
        boundary_emb = rec.get("boundary_emb")
        boundaries_arr = (
            np.asarray(boundaries, dtype=np.float32).reshape(-1, boundary.shape[0])
            if boundaries is not None else None
        )
        emb_arr = np.asarray(boundary_emb, dtype=np.float32) if boundary_emb is not None else None
        meta = json.loads(rec.get("meta_json") or "{}")
        if rec.get("kv_uri"):
            meta["kv_uri"] = rec["kv_uri"]
            meta["has_kv_cache"] = bool(rec.get("has_kv_cache", False))
        return Figment(
            figment_id=rec["figment_id"],
            text=rec["text"],
            boundary=boundary,
            boundaries=boundaries_arr,
            boundary_emb=emb_arr,
            meta=meta,
            children=list(rec.get("children") or []),
            sources=list(rec.get("sources") or []),
            trust=float(rec.get("trust", 0.5)),
        )

    # -- CRUD ------------------------------------------------------------ #
    def upsert(self, figments: list[Figment], hidden_size: int | None = None) -> None:
        if not figments:
            return
        # Resolve the model hidden size: prefer an explicit arg, else the size
        # already established by the table, else fall back to the first figment.
        if hidden_size is None and self._hidden_size is None and self.table_name in self.db.table_names():
            # Open existing table to learn its vector dimension.
            self._table = self.db.open_table(self.table_name)
            vec_field = self._table.schema.field("boundary")
            self._hidden_size = vec_field.type.list_size
        hs = hidden_size or self._hidden_size or figments[0].hidden_size
        self._ensure_table(hs)
        existing_ids = {r["figment_id"] for r in self.table.search().select(["figment_id"]).to_list()}
        for f in figments:
            rec = self._to_record(f, hs)
            fid = rec["figment_id"]
            if fid in existing_ids:
                # Idempotent overwrite: delete the old row, then append the new
                # one. (merge_insert/update mishandle the vector column in this
                # LanceDB version, so delete+add is the reliable path.)
                self._table.delete(f"figment_id = '{fid}'")
            self._table.add([rec], mode="append")

    def upsert_one(self, f: Figment, hidden_size: int | None = None) -> None:
        self.upsert([f], hidden_size=hidden_size)

    def get(self, figment_id: str) -> Figment | None:
        tbl = self.table
        rows = tbl.search().where(f"figment_id = '{figment_id}'").limit(1).to_list()
        if not rows:
            return None
        return self._from_record(rows[0])

    def delete(self, figment_id: str) -> None:
        self.table.delete(f"figment_id = '{figment_id}'")

    def all(self) -> list[Figment]:
        tbl = self.table
        return [self._from_record(r) for r in tbl.search().select(
            ["figment_id", "text", "source_id", "edge_type", "trust", "is_image",
             "has_kv_cache", "kv_uri", "children", "sources", "meta_json",
             "boundary", "boundaries", "boundary_emb"]
        ).to_list()]

    def by_source(self, source_id: str) -> list[Figment]:
        rows = (
            self.table.search()
            .where(f"source_id = '{source_id}'")
            .to_list()
        )
        return [self._from_record(r) for r in rows]

    def search(self, vector: np.ndarray, k: int = 8) -> list[tuple[Figment, float]]:
        """ANN search by boundary vector; returns (figment, score) pairs."""
        tbl = self.table
        vec = np.asarray(vector, dtype=np.float32).tolist()
        rows = tbl.search(vec).limit(k).to_list()
        out = []
        for r in rows:
            out.append((self._from_record(r), float(r.get("_distance", 0.0))))
        return out

    def count(self) -> int:
        return self.table.count_rows()

    def set_kv_ref(self, figment_id: str, kv_uri: str, hidden_size: int) -> None:
        """Record that this figment's K/V lives at ``kv_uri``."""
        self._ensure_table(hidden_size)
        self._table.update(
            where=f"figment_id = '{figment_id}'",
            values={"has_kv_cache": True, "kv_uri": kv_uri},
        )
