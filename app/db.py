"""SQLite schema and small helpers for the job queue and the lesson list (CLAUDE.md §12).

Why SQLite: one machine, one worker, a handful of jobs — a file database needs no server and
survives restarts. WAL mode lets the API read while the worker writes.

Tables:
- ``jobs``: one row per uploaded video; the single worker takes the oldest ``queued`` job.
- ``lessons``: one row per finished report (pipeline output or sample lesson).
Lesson and job ids are long random tokens, so only people who have the link can open a lesson.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
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


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    return con


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with session(db_path) as con:
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


# ---- jobs -------------------------------------------------------------------------------------------

def queue_length(db_path: Path) -> int:
    """Jobs waiting or running — shown by /health."""
    with session(db_path) as con:
        return con.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'processing')").fetchone()[0]


def create_job(db_path: Path, job_id: str, params: dict, upload_path: Path) -> None:
    with session(db_path) as con:
        con.execute("INSERT INTO jobs (id, status, stage, params_json, upload_path, created_at) "
                    "VALUES (?, 'queued', 'queued', ?, ?, ?)", (job_id, json.dumps(params), str(upload_path), now()))


def claim_next_job(db_path: Path) -> sqlite3.Row | None:
    """Atomically move the oldest queued job to 'processing' and return it (FIFO)."""
    with session(db_path) as con:
        return con.execute(
            "UPDATE jobs SET status='processing', stage='starting', progress=0, started_at=? "
            "WHERE id = (SELECT id FROM jobs WHERE status='queued' ORDER BY created_at, rowid LIMIT 1) "
            "RETURNING *", (now(),)).fetchone()


def requeue_interrupted(db_path: Path) -> int:
    """After a restart, jobs that were running go back to the queue (CLAUDE.md §12)."""
    with session(db_path) as con:
        return con.execute("UPDATE jobs SET status='queued', stage='queued', progress=0 "
                           "WHERE status='processing'").rowcount


def update_progress(db_path: Path, job_id: str, stage: str, progress: float) -> None:
    with session(db_path) as con:
        con.execute("UPDATE jobs SET stage=?, progress=? WHERE id=?", (stage, round(progress, 3), job_id))


def finish_job(db_path: Path, job_id: str, lesson_id: str) -> None:
    with session(db_path) as con:
        con.execute("UPDATE jobs SET status='done', stage='done', progress=1, lesson_id=?, finished_at=? WHERE id=?",
                    (lesson_id, now(), job_id))


def fail_job(db_path: Path, job_id: str, error_key: str) -> None:
    with session(db_path) as con:
        con.execute("UPDATE jobs SET status='failed', stage='failed', error_key=?, finished_at=? WHERE id=?",
                    (error_key, now(), job_id))


def get_job(db_path: Path, job_id: str) -> dict | None:
    with session(db_path) as con:
        row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return None
        ahead = 0
        if row["status"] == "queued":
            ahead = con.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'processing') AND "
                                "(created_at < ? OR (created_at = ? AND rowid < (SELECT rowid FROM jobs WHERE id=?)))",
                                (row["created_at"], row["created_at"], job_id)).fetchone()[0]
        return {"status": row["status"], "stage": row["stage"], "progress": row["progress"],
                "queue_position": ahead, "error_key": row["error_key"], "lesson_id": row["lesson_id"]}


# ---- lessons ----------------------------------------------------------------------------------------

def add_lesson(db_path: Path, lesson: dict, report_path: Path, video_path: Path, is_sample: bool = False) -> None:
    with session(db_path) as con:
        con.execute("INSERT OR REPLACE INTO lessons (id, student, label, is_sample, duration_s, report_path, "
                    "video_path, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (lesson["id"], lesson["student"], lesson["label"], int(is_sample), lesson.get("duration_s"),
                     str(report_path), str(video_path), lesson.get("created_at") or now()))


def list_lessons(db_path: Path, ids: list[str]) -> list[dict]:
    """Sample lessons plus the requested ids (the lessons this browser uploaded), newest first."""
    marks = ",".join("?" * len(ids)) or "''"
    with session(db_path) as con:
        rows = con.execute(f"SELECT id, student, label, is_sample, duration_s, created_at FROM lessons "
                           f"WHERE is_sample = 1 OR id IN ({marks}) ORDER BY created_at DESC", ids).fetchall()
    return [dict(r) | {"is_sample": bool(r["is_sample"])} for r in rows]


def get_lesson(db_path: Path, lesson_id: str) -> dict | None:
    with session(db_path) as con:
        row = con.execute("SELECT * FROM lessons WHERE id=?", (lesson_id,)).fetchone()
    return dict(row) if row else None
