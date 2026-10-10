"""The single background worker: runs queued jobs one at a time, oldest first (CLAUDE.md §12).

Why one thread: there is one GPU, and the models (sign detector, vehicle detector, face detector)
are loaded once and kept in memory. The worker sleeps until the API signals a new job (or polls
every few seconds). Every failure becomes a user-friendly error key — never a stack trace on the
website; the trace goes to the server log.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from pathlib import Path

from app import db
from app.settings import Settings
from pipeline.catalogue import Catalogue
from pipeline.ingest import IngestError
from pipeline.run import JobInput, JobTimeout, Models, process_job

log = logging.getLogger("signcoach.worker")


class Worker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._wake = threading.Event()
        self._models: Models | None = None
        self._thread = threading.Thread(target=self._loop, name="signcoach-worker", daemon=True)

    def start(self) -> None:
        n = db.requeue_interrupted(self.settings.db_path)
        if n:
            log.warning("re-queued %d job(s) interrupted by a restart", n)
        self._thread.start()

    def notify(self) -> None:
        """Called by the API after a new job is queued."""
        self._wake.set()

    def _loop(self) -> None:
        while True:
            try:
                row = db.claim_next_job(self.settings.db_path)
            except Exception:  # database briefly locked etc. — try again shortly
                log.exception("could not claim a job")
                row = None
            if row is None:
                self._wake.wait(timeout=3.0)
                self._wake.clear()
                continue
            self._run(row)

    def _load_models(self) -> Models:
        if self._models is None:
            s = self.settings
            self._models = Models.load(s.cfg, s.models_dir, s.device)
            log.info("models loaded on %s", s.device)
        return self._models

    def _run(self, row) -> None:
        s = self.settings
        job_id = row["id"]
        params = json.loads(row["params_json"])
        lesson_id = secrets.token_urlsafe(16)
        try:
            models = self._load_models()
            report = process_job(
                JobInput(job_id=job_id, lesson_id=lesson_id, upload_path=Path(row["upload_path"]),
                         student=params["student"], label=params["lesson_label"], created_at=row["created_at"],
                         camera_height_m=params.get("camera_height_m"), hfov_deg=params.get("hfov_deg")),
                s.cfg, models, Catalogue.load(), s.media_dir, s.jobs_dir, s.models_dir, s.git_sha,
                lambda stage, p: db.update_progress(s.db_path, job_id, stage, p),
                deadline=time.monotonic() + s.cfg["job"]["timeout_s"])
            media = s.media_dir / lesson_id
            db.add_lesson(s.db_path, report.lesson, media / "report.json", media / "annotated.mp4")
            db.finish_job(s.db_path, job_id, lesson_id)
            log.info("job %s done -> lesson %s (%.1fs)", job_id, lesson_id, report.processing["total_seconds"])
        except IngestError as e:
            db.fail_job(s.db_path, job_id, e.error_key)
        except JobTimeout:
            db.fail_job(s.db_path, job_id, "error.timeout")
        except Exception:
            log.exception("job %s failed", job_id)
            db.fail_job(s.db_path, job_id, "error.processing_failed")
