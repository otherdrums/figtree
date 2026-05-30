"""Graph edge types and operations."""

from __future__ import annotations

import json
from enum import Enum
from typing import Optional

from pdga.db.store import DeltaDB


class EdgeType(str, Enum):
    CONTRADICTS = "contradicts"
    ABOUT_SAME_EVENT = "about_same_event"
    COMPATIBLE = "compatible"
    DERIVED_FROM = "derived_from"
    ENHANCES = "enhances"
    SAME_SOURCE = "same_source"
    REQUIRES = "requires"


class EdgeOps:
    """Edge CRUD backed by SQLite."""

    def __init__(self, db: DeltaDB):
        self.db = db

    def add(
        self,
        source_id: str,
        edge_type: EdgeType,
        target_id: str,
        weight: float = 1.0,
        metadata: dict | None = None,
    ) -> None:
        self.db.conn.execute(
            """INSERT OR REPLACE INTO edges
               (source_id, target_id, edge_type, weight, metadata)
               VALUES (?, ?, ?, ?, ?)""",
            (
                source_id,
                target_id,
                edge_type.value,
                weight,
                json.dumps(metadata or {}),
            ),
        )
        self.db.conn.commit()

    def remove(
        self, source_id: str, edge_type: EdgeType, target_id: str
    ) -> None:
        self.db.conn.execute(
            "DELETE FROM edges WHERE source_id=? AND target_id=? AND edge_type=?",
            (source_id, target_id, edge_type.value),
        )
        self.db.conn.commit()

    def get(
        self, source_id: str, edge_type: EdgeType | None = None
    ) -> list[dict]:
        if edge_type:
            rows = self.db.conn.execute(
                "SELECT * FROM edges WHERE source_id=? AND edge_type=?",
                (source_id, edge_type.value),
            ).fetchall()
        else:
            rows = self.db.conn.execute(
                "SELECT * FROM edges WHERE source_id=?",
                (source_id,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def find_contradictions(self, delta_id: str) -> list[dict]:
        rows = self.db.conn.execute(
            """SELECT * FROM edges
               WHERE (source_id=? OR target_id=?)
                 AND edge_type='contradicts'""",
            (delta_id, delta_id),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def find_related(self, delta_id: str, edge_type: EdgeType | None = None) -> list[dict]:
        query = "SELECT * FROM edges WHERE (source_id=? OR target_id=?)"
        params = (delta_id, delta_id)
        if edge_type:
            query += " AND edge_type=?"
            params = (delta_id, delta_id, edge_type.value)
        rows = self.db.conn.execute(query, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def _row_to_dict(self, row: tuple) -> dict:
        return {
            "source_id": row[0],
            "target_id": row[1],
            "edge_type": row[2],
            "weight": row[3],
            "metadata": row[4],
        }
