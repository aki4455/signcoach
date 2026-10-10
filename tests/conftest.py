"""Shared pytest setup: every test run gets its own throw-away data folder.

Tests must never touch the production database or media, so SIGNCOACH_DATA points to a temp
folder and SIGNCOACH_INSTANCE is "test". The device is CPU (the GPU may be training), and the
background worker is off (tests that need processing call the pipeline directly). Model weights
are read from the real models folder if it exists (only "slow" tests need them).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

os.environ["SIGNCOACH_DATA"] = tempfile.mkdtemp(prefix="signcoach-test-")
os.environ["SIGNCOACH_INSTANCE"] = "test"
os.environ["SIGNCOACH_DEVICE"] = "cpu"
os.environ["SIGNCOACH_WORKER"] = "off"
REAL_MODELS = Path("~/signcoach-data/models").expanduser()
if REAL_MODELS.is_dir():
    os.environ.setdefault("SIGNCOACH_MODELS", str(REAL_MODELS))


def make_video(path: Path, seconds: float = 3, fps: int = 25, size: str = "640x360", source: str = "testsrc2",
               audio: bool = True) -> Path:
    """A small synthetic test video made with ffmpeg (never a real recording)."""
    sep = ":" if "=" in source else "="  # "testsrc2=size=..." vs "color=c=black:size=..."
    cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i", f"{source}{sep}size={size}:rate={fps}:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac", "-shortest"]
    subprocess.run(cmd + ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.fixture
def video(tmp_path) -> Path:
    return make_video(tmp_path / "clip.mp4")
