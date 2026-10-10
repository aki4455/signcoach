"""Ingest (CLAUDE.md §10.1): probe, limits, normalization rules, quality warnings."""

from __future__ import annotations

import subprocess

import pytest

from app.settings import get_settings
from pipeline import ingest
from tests.conftest import make_video

CFG = get_settings().cfg


def streams(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    return out.split()


def test_probe_reads_basic_facts(video):
    info = ingest.probe(video)
    assert info.fps == 25 and (info.width, info.height) == (640, 360) and 2.9 < info.duration_s < 3.2


def test_normalize_keeps_lower_fps_and_drops_audio(video, tmp_path):
    out = ingest.normalize(video, tmp_path / "n.mp4", ingest.probe(video), CFG)
    assert out.fps == 25                      # never raised to 30: duplicates would fake a standing car
    assert streams(tmp_path / "n.mp4") == ["h264,video"]


def test_normalize_caps_high_fps(tmp_path):
    src = make_video(tmp_path / "fast.mp4", seconds=2, fps=60, audio=False)
    assert ingest.normalize(src, tmp_path / "n.mp4", ingest.probe(src), CFG).fps == 30


def test_too_long_video_is_rejected(video):
    cfg = CFG | {"limits": CFG["limits"] | {"max_video_seconds": 1}}
    with pytest.raises(ingest.IngestError) as e:
        ingest.validate(ingest.probe(video), cfg)
    assert e.value.error_key == "error.too_long"


def test_non_video_file_is_rejected(tmp_path):
    bad = tmp_path / "notes.mp4"
    bad.write_text("this is not a video")
    with pytest.raises(ingest.IngestError) as e:
        ingest.probe(bad)
    assert e.value.error_key == "error.not_a_video"


def test_dark_short_clip_gets_warnings(tmp_path):
    dark = make_video(tmp_path / "dark.mp4", seconds=2, source="color=c=black", audio=False)
    warnings = ingest.quality_warnings(dark, ingest.probe(dark), CFG)
    assert "warning.low_light" in warnings and "warning.short_clip" in warnings
