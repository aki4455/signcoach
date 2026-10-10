"""GET /health — liveness plus deployment evidence.

The deploy workflow polls this endpoint after a restart until ``git_sha`` equals the commit it
just checked out; that is how we prove that a push to ``main`` really went live.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.db import queue_length
from app.settings import get_settings

router = APIRouter()


@router.get("/health")
def health() -> dict:
    s = get_settings()
    return {
        "ok": True,
        "git_sha": s.git_sha,
        "model": s.cfg["detect"]["model"],
        "device": s.device,
        "queue_length": queue_length(s.db_path),
    }
