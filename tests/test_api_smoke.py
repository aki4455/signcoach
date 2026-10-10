"""Smoke test of the web app: it starts, /health answers, and the front page is served."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_health_reports_status_and_commit():
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["ok"] is True
    assert body["device"] == "cpu"
    assert body["queue_length"] == 0
    assert body["git_sha"]


def test_front_page_is_html():
    with TestClient(app) as client:
        r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_unknown_api_path_is_404():
    with TestClient(app) as client:
        assert client.get("/api/does-not-exist").status_code == 404
