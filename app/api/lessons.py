"""Lessons and their reports (CLAUDE.md §12).

There is no public list of everyone's uploads: `/api/lessons` returns the sample lessons plus the
lessons whose ids the browser sends (the ones it uploaded itself). Lesson ids are unguessable.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app import db
from app.api.errors import ApiError
from app.settings import get_settings

router = APIRouter(prefix="/api/lessons")


@router.get("")
def list_lessons(ids: str = "") -> list[dict]:
    wanted = [i for i in ids.split(",") if i][:50]
    return db.list_lessons(get_settings().db_path, wanted)


@router.get("/{lesson_id}/report")
def lesson_report(lesson_id: str):
    lesson = db.get_lesson(get_settings().db_path, lesson_id)
    if lesson is None or not Path(lesson["report_path"]).is_file():
        raise ApiError("error.not_found", 404)
    return FileResponse(lesson["report_path"], media_type="application/json")
