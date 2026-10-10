"""Chunked, resumable video upload (CLAUDE.md §12).

Why chunks: dashcam files are hundreds of MB and phone connections drop. The browser sends the
file in 8 MB pieces; if the connection breaks it asks which pieces arrived (GET) and sends only
the missing ones. `complete` checks consent, joins the pieces into the job folder and queues the
job. Client file names are never used in paths — only a sanitised extension is kept.
"""

from __future__ import annotations

import json
import re
import secrets
import shutil
import time
from pathlib import Path

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app import db
from app.api.errors import ApiError
from app.settings import get_settings

router = APIRouter(prefix="/api/uploads")
UPLOAD_ID = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


class InitBody(BaseModel):
    filename: str = Field(max_length=255)
    size_bytes: int


class CompleteBody(BaseModel):
    student: str = Field(min_length=1, max_length=80)
    lesson_label: str = Field(default="", max_length=120)
    camera_height_m: float | None = Field(default=None, ge=0.5, le=3.0)
    hfov_deg: float | None = Field(default=None, ge=30, le=170)
    consent: bool


def _dir(upload_id: str) -> Path:
    if not UPLOAD_ID.match(upload_id):
        raise ApiError("error.upload_incomplete", 404)
    d = get_settings().uploads_dir / upload_id
    if not (d / "meta.json").is_file():
        raise ApiError("error.upload_incomplete", 404)
    return d


def _meta(d: Path) -> dict:
    return json.loads((d / "meta.json").read_text(encoding="utf-8"))


def _received(d: Path) -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in d.glob("chunk_*.part"))


def _prune_stale(uploads_dir: Path, max_age_h: float) -> None:
    cutoff = time.time() - max_age_h * 3600
    for d in uploads_dir.iterdir():
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)


@router.post("/init")
def init_upload(body: InitBody) -> dict:
    cfg = get_settings().cfg
    if body.size_bytes <= 0:
        raise ApiError("error.not_a_video")
    if body.size_bytes > cfg["limits"]["max_upload_bytes"]:
        raise ApiError("error.too_large", 413)
    ext = Path(body.filename).suffix.lower()
    ext = ext if re.fullmatch(r"\.[a-z0-9]{1,6}", ext) else ".bin"
    chunk = cfg["upload"]["chunk_bytes"]
    uploads_dir = get_settings().uploads_dir
    _prune_stale(uploads_dir, cfg["upload"]["stale_after_hours"])
    upload_id = secrets.token_urlsafe(16)
    d = uploads_dir / upload_id
    d.mkdir(parents=True)
    meta = {"size_bytes": body.size_bytes, "chunk_bytes": chunk, "n_chunks": -(-body.size_bytes // chunk), "ext": ext}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return {"upload_id": upload_id, "chunk_bytes": chunk}


@router.put("/{upload_id}/chunk")
async def put_chunk(upload_id: str, index: int, request: Request) -> dict:
    d = _dir(upload_id)
    meta = _meta(d)
    if not 0 <= index < meta["n_chunks"]:
        raise ApiError("error.upload_incomplete")
    expected = meta["chunk_bytes"] if index < meta["n_chunks"] - 1 else \
        meta["size_bytes"] - meta["chunk_bytes"] * (meta["n_chunks"] - 1)
    data = await request.body()
    if len(data) != expected:
        raise ApiError("error.upload_incomplete")
    tmp = d / f"chunk_{index}.tmp"
    tmp.write_bytes(data)
    tmp.rename(d / f"chunk_{index}.part")  # a chunk only "exists" once it is complete
    return {"received": index}


@router.get("/{upload_id}")
def upload_status(upload_id: str) -> dict:
    d = _dir(upload_id)
    meta = _meta(d)
    return {"received": _received(d), "size_bytes": meta["size_bytes"], "chunk_bytes": meta["chunk_bytes"]}


@router.post("/{upload_id}/complete")
def complete_upload(upload_id: str, body: CompleteBody, request: Request) -> dict:
    s = get_settings()
    if not body.consent:
        raise ApiError("error.consent_required")
    d = _dir(upload_id)
    meta = _meta(d)
    if _received(d) != list(range(meta["n_chunks"])):
        raise ApiError("error.upload_incomplete")
    if db.queue_length(s.db_path) >= s.cfg["limits"]["max_queued_jobs"]:
        raise ApiError("error.queue_full", 429)

    job_id = secrets.token_urlsafe(16)
    job_dir = s.jobs_dir / job_id
    job_dir.mkdir(parents=True)
    upload_path = job_dir / f"upload{meta['ext']}"
    with upload_path.open("wb") as out:
        for i in range(meta["n_chunks"]):
            with (d / f"chunk_{i}.part").open("rb") as part:
                shutil.copyfileobj(part, out)
    shutil.rmtree(d, ignore_errors=True)
    db.create_job(s.db_path, job_id, body.model_dump(exclude={"consent"}), upload_path)
    worker = getattr(request.app.state, "worker", None)
    if worker is not None:
        worker.notify()
    return {"job_id": job_id}
