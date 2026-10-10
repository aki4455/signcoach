"""FastAPI application: API routes, media files and the built frontend.

Run locally:  uv run uvicorn app.main:app --reload
In production systemd starts uvicorn from the deploy checkout (deploy/signcoach.service).

The frontend is the Vite build in ``web/dist``. Any path that is not an API or media route gets
``index.html`` so client-side routes like /lessons/<id> work on reload. Before the first frontend
build, a small placeholder page is served instead.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse

from app.api import health
from app.db import init_db
from app.settings import REPO_DIR, get_settings

WEB_DIST = REPO_DIR / "web" / "dist"

PLACEHOLDER = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>SignCoach</title></head>
<body style="font-family: system-ui, sans-serif; max-width: 40rem; margin: 3rem auto; padding: 0 1rem">
<h1>SignCoach</h1><p>AI co-instructor for driving schools. The web app is being built.</p>
<p><a href="/health">Service health</a></p></body></html>"""


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    init_db(settings.db_path)
    yield


app = FastAPI(title="SignCoach", lifespan=lifespan)
app.include_router(health.router)


@app.get("/{path:path}", include_in_schema=False)
def frontend(path: str):
    """Serve the built single-page app (or the placeholder before the first build)."""
    if path.startswith(("api/", "media/")):
        return HTMLResponse("Not found", status_code=404)
    candidate = (WEB_DIST / path).resolve()
    if path and candidate.is_file() and WEB_DIST.resolve() in candidate.parents:
        return FileResponse(candidate)
    index = WEB_DIST / "index.html"
    if index.is_file():
        return FileResponse(index)
    return HTMLResponse(PLACEHOLDER)
