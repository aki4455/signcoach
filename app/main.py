"""FastAPI application: API routes, the processing worker, media files and the built frontend.

Run locally:  uv run uvicorn app.main:app --reload --port 8001
In production systemd starts uvicorn from the deploy checkout (deploy/signcoach.service).

- /api/...    uploads, jobs, lessons (see app/api/)
- /media/...  blurred lesson media (annotated.mp4, poster, thumbnails); served by StaticFiles,
              which answers HTTP Range requests, so the video player can seek
- /voice/...  pre-generated English voice prompts (assets/voice, scripts/generate_voice.py)
- /health     liveness + deployed git SHA
- anything else: the single-page frontend from web/dist (index.html for client-side routes),
  or a placeholder page before the first frontend build.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api import health, jobs, lessons, uploads
from app.api.errors import ApiError, api_error_handler
from app.db import init_db
from app.settings import REPO_DIR, get_settings
from app.worker import Worker

WEB_DIST = REPO_DIR / "web" / "dist"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

PLACEHOLDER = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>SignCoach</title></head>
<body style="font-family: system-ui, sans-serif; max-width: 40rem; margin: 3rem auto; padding: 0 1rem">
<h1>SignCoach</h1><p>AI co-instructor for driving schools. The web app is being built.</p>
<p><a href="/health">Service health</a></p></body></html>"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    init_db(settings.db_path)
    if os.environ.get("SIGNCOACH_WORKER", "on") != "off":  # tests switch the worker off
        app.state.worker = Worker(settings)
        app.state.worker.start()
    yield


app = FastAPI(title="SignCoach", lifespan=lifespan)
app.add_exception_handler(ApiError, api_error_handler)
for module in (health, uploads, jobs, lessons):
    app.include_router(module.router)
app.mount("/media", StaticFiles(directory=get_settings().media_dir, check_dir=False), name="media")
app.mount("/voice", StaticFiles(directory=REPO_DIR / "assets" / "voice", check_dir=False), name="voice")


@app.get("/{path:path}", include_in_schema=False)
def frontend(path: str):
    """Serve the built single-page app (or the placeholder before the first build)."""
    if path.startswith(("api/", "media/", "voice/")):
        return HTMLResponse("Not found", status_code=404)
    candidate = (WEB_DIST / path).resolve()
    if path and candidate.is_file() and WEB_DIST.resolve() in candidate.parents:
        return FileResponse(candidate)
    index = WEB_DIST / "index.html"
    if index.is_file():
        return FileResponse(index)
    return HTMLResponse(PLACEHOLDER)
