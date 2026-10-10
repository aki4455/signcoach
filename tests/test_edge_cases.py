"""Edge cases from CLAUDE.md §18 at the ingest step: every odd upload ends in a usable normalized
video with honest warnings, or in a clear error — never a crash. Inputs are synthetic (ffmpeg)."""

from __future__ import annotations

import subprocess

import pytest

from app.settings import get_settings
from pipeline import ingest

CFG = get_settings().cfg


def ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", *args], check=True)


def test_iphone_style_portrait_hevc_mov(tmp_path):
    land = tmp_path / "land.mov"
    ff("-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=1", "-c:v", "libx265", "-tag:v", "hvc1",
       "-x265-params", "log-level=error", str(land))
    portrait = tmp_path / "IMG_0001.MOV"
    ff("-display_rotation", "90", "-i", str(land), "-c", "copy", str(portrait))
    info = ingest.probe(portrait)
    assert (info.width, info.height) == (720, 1280) and info.codec == "hevc"
    out = ingest.normalize(portrait, tmp_path / "n.mp4", info, CFG)
    assert (out.width, out.height) == (720, 1280)                      # rotation applied
    assert "warning.portrait" in ingest.quality_warnings(tmp_path / "n.mp4", out, CFG)


def test_hdr_hlg_video_is_tone_mapped(tmp_path):
    hdr = tmp_path / "hdr.mov"
    ff("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=1", "-c:v", "libx265", "-pix_fmt", "yuv420p10le",
       "-x265-params", "log-level=error:colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc", str(hdr))
    info = ingest.probe(hdr)
    assert info.is_hdr
    out = ingest.normalize(hdr, tmp_path / "n.mp4", info, CFG)
    assert out.color_transfer in (None, "bt709") and out.codec == "h264"


def test_4k_60fps_is_downscaled_and_capped(tmp_path):
    src = tmp_path / "4k60.mp4"
    ff("-f", "lavfi", "-i", "testsrc2=size=3840x2160:rate=60:duration=1", "-c:v", "libx264", "-preset", "ultrafast", str(src))
    out = ingest.normalize(src, tmp_path / "n.mp4", ingest.probe(src), CFG)
    assert out.width == 1920 and out.fps == 30


def test_longer_than_three_minutes_is_rejected_clearly(tmp_path):
    src = tmp_path / "long.mp4"
    ff("-f", "lavfi", "-i", "color=c=gray:size=64x36:rate=5:duration=190", "-c:v", "libx264", "-preset", "ultrafast", str(src))
    with pytest.raises(ingest.IngestError) as e:
        ingest.validate(ingest.probe(src), CFG)
    assert e.value.error_key == "error.too_long" and e.value.params["max_minutes"] == 3.0


def test_audio_only_file_is_not_a_video(tmp_path):
    src = tmp_path / "voice_memo.mp4"
    ff("-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "aac", str(src))
    with pytest.raises(ingest.IngestError) as e:
        ingest.probe(src)
    assert e.value.error_key == "error.not_a_video"
