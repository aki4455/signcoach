"""Chunked upload API (CLAUDE.md §12): resume information, consent, completeness, job creation."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.settings import get_settings


def upload(client, data: bytes, chunk: int):
    r = client.post("/api/uploads/init", json={"filename": "lesson.MP4", "size_bytes": len(data)})
    assert r.status_code == 200
    uid = r.json()["upload_id"]
    for i in range(0, len(data), chunk):
        assert client.put(f"/api/uploads/{uid}/chunk", params={"index": i // chunk},
                          content=data[i:i + chunk]).status_code == 200
    return uid


def test_chunked_upload_creates_a_queued_job(video, monkeypatch):
    monkeypatch.setitem(get_settings().cfg["upload"], "chunk_bytes", 20_000)
    data = video.read_bytes()
    with TestClient(app) as client:
        uid = upload(client, data, 20_000)
        status = client.get(f"/api/uploads/{uid}").json()
        assert status["received"] == list(range(-(-len(data) // 20_000)))

        body = {"student": "Aziz", "lesson_label": "Lesson 1", "consent": False}
        r = client.post(f"/api/uploads/{uid}/complete", json=body)
        assert r.status_code == 400 and r.json()["error_key"] == "error.consent_required"

        r = client.post(f"/api/uploads/{uid}/complete", json=body | {"consent": True})
        assert r.status_code == 200
        job = client.get(f"/api/jobs/{r.json()['job_id']}").json()
        assert job["status"] == "queued" and job["queue_position"] >= 0


def test_incomplete_upload_cannot_complete(video, monkeypatch):
    monkeypatch.setitem(get_settings().cfg["upload"], "chunk_bytes", 20_000)
    with TestClient(app) as client:
        r = client.post("/api/uploads/init", json={"filename": "a.mp4", "size_bytes": 50_000})
        uid = r.json()["upload_id"]
        client.put(f"/api/uploads/{uid}/chunk", params={"index": 0}, content=b"x" * 20_000)
        r = client.post(f"/api/uploads/{uid}/complete", json={"student": "A", "consent": True})
        assert r.json()["error_key"] == "error.upload_incomplete"


def test_wrong_chunk_size_and_too_large_files_are_refused():
    with TestClient(app) as client:
        too_big = get_settings().cfg["limits"]["max_upload_bytes"] + 1
        assert client.post("/api/uploads/init", json={"filename": "a.mp4", "size_bytes": too_big}).status_code == 413
        uid = client.post("/api/uploads/init", json={"filename": "a.mp4", "size_bytes": 10}).json()["upload_id"]
        assert client.put(f"/api/uploads/{uid}/chunk", params={"index": 0}, content=b"123").status_code == 400


def test_lessons_list_never_exposes_other_peoples_lessons():
    with TestClient(app) as client:
        assert client.get("/api/lessons", params={"ids": "someone-elses-id"}).json() == []
        assert client.get("/api/lessons/unknown/report").status_code == 404
