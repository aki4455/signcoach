"""Whole pipeline on a synthetic clip (CLAUDE.md M1 acceptance): a video with no signs still gives a
valid report with honest warnings, a playable annotated video without audio, and no leftovers.

Marked slow: it needs the real vehicle and face model weights (the deploy workflow skips it).
"""

from __future__ import annotations

import json
import subprocess
import time

import pytest

from app.settings import get_settings
from pipeline.catalogue import Catalogue
from pipeline.run import JobInput, Models, process_job
from pipeline.types import Report

pytestmark = pytest.mark.slow


def test_clip_without_signs_gives_valid_report(video, tmp_path):
    s = get_settings()
    if not (s.models_dir / "yolo11n.pt").exists():
        pytest.skip("model weights not available")
    replay = tmp_path / "empty.json"
    replay.write_text(json.dumps({"model": "none", "detections": []}))
    models = Models.load(s.cfg, s.models_dir, "cpu", replay)
    upload = tmp_path / "jobs" / "j1" / "upload.mp4"
    upload.parent.mkdir(parents=True)
    upload.write_bytes(video.read_bytes())

    report = process_job(JobInput("j1", "L1", upload, "Test", "e2e", "2026-10-10T00:00:00Z"), s.cfg, models,
                         Catalogue.load(), tmp_path / "media", tmp_path / "jobs", s.models_dir, "test",
                         lambda stage, p: None, time.monotonic() + 600)

    out = tmp_path / "media" / "L1"
    Report.model_validate_json((out / "report.json").read_text())
    assert report.signs == [] and report.events == []
    assert {"warning.no_signs", "warning.speed_not_checked"} <= set(report.quality_warnings)
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                              str(out / "annotated.mp4")], capture_output=True, text=True).stdout.split()
    assert streams == ["video"]                         # audio stripped
    assert (out / "poster.jpg").exists()
    assert not (tmp_path / "jobs" / "j1").exists()      # unblurred material deleted
