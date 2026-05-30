"""SQLite schema for PDGA delta graph database."""

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS deltas (
    delta_id    TEXT PRIMARY KEY,
    delta_type  TEXT NOT NULL,
    path        TEXT NOT NULL,
    base_model  TEXT NOT NULL,
    source_text TEXT DEFAULT '',
    trust       REAL DEFAULT 0.5,
    lsh_buckets TEXT DEFAULT '{}',
    pruned      INTEGER DEFAULT 0,
    pruned_by   TEXT,
    num_windows INTEGER DEFAULT 0,
    tags        TEXT DEFAULT '[]',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    edge_type   TEXT NOT NULL,
    weight      REAL DEFAULT 1.0,
    metadata    TEXT DEFAULT '{}',
    PRIMARY KEY (source_id, target_id, edge_type),
    FOREIGN KEY (source_id) REFERENCES deltas(delta_id),
    FOREIGN KEY (target_id) REFERENCES deltas(delta_id)
);

CREATE TABLE IF NOT EXISTS lsh_tables (
    table_idx   INTEGER NOT NULL,
    bucket_key  TEXT NOT NULL,
    delta_id    TEXT NOT NULL,
    window_idx  INTEGER NOT NULL,
    boundary_hash TEXT DEFAULT '',
    PRIMARY KEY (table_idx, bucket_key, delta_id, window_idx),
    FOREIGN KEY (delta_id) REFERENCES deltas(delta_id)
);

CREATE TABLE IF NOT EXISTS cogitation_runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    bucket_key  TEXT NOT NULL,
    table_idx   INTEGER NOT NULL,
    num_inputs  INTEGER DEFAULT 0,
    output_delta_id TEXT,
    started_at  TEXT,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def get_db_path() -> Path:
    pdga_home = Path.home() / ".pdga"
    pdga_home.mkdir(parents=True, exist_ok=True)
    return pdga_home / "db.sqlite"


def init_db(db_path: Path | None = None) -> sqlite3.Connection:
    if db_path is None:
        db_path = get_db_path()

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(CREATE_TABLES)

    row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()

    return conn
