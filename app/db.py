"""SQLite schema and small helpers for the job queue and the lesson list (CLAUDE.md §12).

Why SQLite: one machine, one worker, a handful of jobs — a file database needs no server and
survives restarts. WAL mode lets the API read while the worker writes.

Tables:
- ``jobs``: one row per uploaded video; the single worker takes the oldest ``queued`` job.
- ``lessons``: one row per finished report (pipeline output or sample lesson).
Lesson ids are long random tokens, so only people who have the link can open a lesson.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    status       TEXT NOT NULL CHECK (status IN ('queued', 'processing', 'done', 'failed')),
    stage        TEXT,
    progress     REAL NOT NULL DEFAULT 0,
    error_key    TEXT,
    params_json  TEXT NOT NULL DEFAULT '{}',
    upload_path  TEXT,
    lesson_id    TEXT,
    created_at   TEXT NOT NULL,
    started_at   TEXT,
    finished_at  TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs (status, created_at);

CREATE TABLE IF NOT EXISTS lessons (
    id           TEXT PRIMARY KEY,
    student      TEXT NOT NULL,
    label        TEXT NOT NULL,
    is_sample    INTEGER NOT NULL DEFAULT 0,
    duration_s   REAL,
    report_path  TEXT NOT NULL,
    video_path   TEXT,
    created_at   TEXT NOT NULL
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as con:
        con.executescript(SCHEMA)


@contextmanager
def session(db_path: Path) -> Iterator[sqlite3.Connection]:
    """A connection that commits on success and always closes."""
    con = connect(db_path)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def queue_length(db_path: Path) -> int:
    """Jobs waiting or running — shown by /health."""
    with session(db_path) as con:
        return con.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'processing')").fetchone()[0]
