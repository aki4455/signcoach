"""GET /api/jobs/{id} — what the upload page polls while a video is processed (CLAUDE.md §12)."""

from __future__ import annotations

from fastapi import APIRouter

from app import db
from app.api.errors import ApiError
from app.settings import get_settings

router = APIRouter(prefix="/api/jobs")


@router.get("/{job_id}")
def job_status(job_id: str) -> dict:
    job = db.get_job(get_settings().db_path, job_id)
    if job is None:
        raise ApiError("error.not_found", 404)
    return job
