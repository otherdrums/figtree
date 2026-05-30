"""Delta registry — CRUD operations on the SQLite delta database."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pdga.db.schema import get_db_path, init_db


class DeltaDB:
    """Manages the delta registry, edges, and LSH tables in SQLite."""

    def __init__(self, db_path: Path | None = None):
        if db_path is None:
            db_path = get_db_path()
        self.db_path = db_path
        self.conn = init_db(db_path)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def register(
        self,
        delta_id: str,
        delta_type: str,
        path: str,
        base_model: str,
        source_text: str = "",
        trust: float = 0.5,
        num_windows: int = 0,
        tags: list[str] | None = None,
        lsh_buckets: dict | None = None,
    ) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO deltas
               (delta_id, delta_type, path, base_model, source_text, trust,
                num_windows, tags, lsh_buckets, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                delta_id,
                delta_type,
                path,
                base_model,
                source_text[:2000],
                trust,
                num_windows,
                json.dumps(tags or []),
                json.dumps(lsh_buckets or {}),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

    def get(self, delta_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM deltas WHERE delta_id = ? AND pruned = 0",
            (delta_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def list_all(self, delta_type: str | None = None, limit: int = 100) -> list[dict]:
        query = "SELECT * FROM deltas WHERE pruned = 0"
        params = ()
        if delta_type:
            query += " AND delta_type = ?"
            params = (delta_type,)
        query += " ORDER BY created_at DESC LIMIT ?"
        params = params + (limit,)
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def list_by_tags(
        self, tags: list[str], delta_type: str | None = None
    ) -> list[dict]:
        result = []
        for row in self.conn.execute(
            "SELECT * FROM deltas WHERE pruned = 0 ORDER BY created_at DESC"
        ).fetchall():
            entry = self._row_to_dict(row)
            entry_tags = json.loads(entry.get("tags", "[]"))
            if any(t in entry_tags for t in tags):
                if delta_type is None or entry["delta_type"] == delta_type:
                    result.append(entry)
        return result

    def update_trust(self, delta_id: str, trust: float) -> None:
        self.conn.execute(
            "UPDATE deltas SET trust = ? WHERE delta_id = ?",
            (max(0.0, min(1.0, trust)), delta_id),
        )
        self.conn.commit()

    def update_lsh(self, delta_id: str, lsh_buckets: dict) -> None:
        self.conn.execute(
            "UPDATE deltas SET lsh_buckets = ? WHERE delta_id = ?",
            (json.dumps(lsh_buckets), delta_id),
        )
        self.conn.commit()

    def delete(self, delta_id: str) -> None:
        self.conn.execute("DELETE FROM lsh_tables WHERE delta_id = ?", (delta_id,))
        self.conn.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?",
                          (delta_id, delta_id))
        self.conn.execute("DELETE FROM deltas WHERE delta_id = ?", (delta_id,))
        self.conn.commit()

    def mark_pruned(self, delta_id: str, pruned_by: str) -> None:
        self.conn.execute(
            "UPDATE deltas SET pruned = 1, pruned_by = ? WHERE delta_id = ?",
            (pruned_by, delta_id),
        )
        self.conn.commit()

    def count(self, delta_type: str | None = None) -> int:
        query = "SELECT COUNT(*) FROM deltas WHERE pruned = 0"
        if delta_type:
            query += " AND delta_type = ?"
            return self.conn.execute(query, (delta_type,)).fetchone()[0]
        return self.conn.execute(query).fetchone()[0]

    def _row_to_dict(self, row: tuple) -> dict:
        cols = [
            "delta_id", "delta_type", "path", "base_model", "source_text",
            "trust", "lsh_buckets", "pruned", "pruned_by", "num_windows",
            "tags", "created_at",
        ]
        return dict(zip(cols, row))
