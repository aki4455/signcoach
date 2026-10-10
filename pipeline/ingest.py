"""Ingest: inspect an uploaded video, reject what we cannot process, and normalize the rest (CLAUDE.md §10.1).

Why normalize: uploads come in every format (dashcam MP4/AVI, iPhone HEVC .mov with HDR, MKV...).
Every later stage reads one predictable file instead: H.264, constant frame rate, at most 1920 px
wide, no audio, rotation applied, standard SDR colours. Its timeline is the time base of the whole
report (all event times are seconds from its first frame).

Two rules matter for the speed estimate later:
- the frame rate is never raised (fps = min(source, cap)); duplicated frames would look like the car
  standing still;
- HDR video is tone-mapped to SDR; otherwise colours wash out and red signs fade.

Errors are raised as `IngestError(error_key)` with an i18n key the website can show as-is.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

HDR_TRANSFERS = {"arib-std-b67", "smpte2084"}  # HLG and PQ (iPhone HDR, Dolby Vision)


class IngestError(Exception):
    """A problem with the uploaded file that the user should see (error_key is an i18n key)."""

    def __init__(self, error_key: str, **params):
        super().__init__(error_key)
        self.error_key = error_key
        self.params = params


@dataclass(frozen=True)
class VideoInfo:
    duration_s: float
    fps: float
    width: int            # after rotation
    height: int           # after rotation
    rotation: int         # degrees, from the display matrix / rotate tag
    codec: str
    creation_time: str | None
    color_transfer: str | None

    @property
    def is_hdr(self) -> bool:
        return self.color_transfer in HDR_TRANSFERS


def _run(cmd: list[str], timeout: float = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def probe(path: Path) -> VideoInfo:
    """Read duration, frame rate, size, rotation, codec and colour info with ffprobe."""
    r = _run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)], 60)
    if r.returncode != 0:
        raise IngestError("error.not_a_video")
    data = json.loads(r.stdout or "{}")
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"
                  and s.get("disposition", {}).get("attached_pic", 0) == 0), None)
    if video is None:
        raise IngestError("error.not_a_video")

    fps = 0.0
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            rate = Fraction(video.get(key, "0/1"))
            if rate > 0:
                fps = float(rate)
                break
        except (ValueError, ZeroDivisionError):
            continue
    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0)

    rotation = 0
    for side in video.get("side_data_list", []):
        if "rotation" in side:
            rotation = int(round(float(side["rotation"])))
    if not rotation and "rotate" in video.get("tags", {}):
        rotation = int(video["tags"]["rotate"])
    width, height = int(video.get("width", 0)), int(video.get("height", 0))
    if abs(rotation) % 180 == 90:
        width, height = height, width

    tags = data.get("format", {}).get("tags", {}) | video.get("tags", {})
    if fps <= 0 or duration <= 0 or width <= 0 or height <= 0:
        raise IngestError("error.not_a_video")
    return VideoInfo(duration_s=duration, fps=fps, width=width, height=height, rotation=rotation,
                     codec=video.get("codec_name", "?"), creation_time=tags.get("creation_time"),
                     color_transfer=video.get("color_transfer"))


def validate(info: VideoInfo, cfg: dict) -> None:
    """Reject videos over the length limit with a clear message."""
    max_s = cfg["limits"]["max_video_seconds"]
    if info.duration_s > max_s + 0.5:
        raise IngestError("error.too_long", max_minutes=round(max_s / 60, 1))


def normalize(src: Path, dst: Path, info: VideoInfo, cfg: dict) -> VideoInfo:
    """Re-encode to H.264, CFR, fps <= cap (never raised), width <= max, no audio, SDR. Returns the new info."""
    ing = cfg["ingest"]
    fps_out = min(info.fps, float(ing["fps_cap"]))
    filters = []
    if info.is_hdr and ing.get("tonemap_hdr", True):
        # HDR (HLG/PQ) -> linear light -> tone-map -> SDR BT.709
        filters += ["zscale=t=linear:npl=100", "format=gbrpf32le", "zscale=p=bt709",
                    "tonemap=tonemap=hable:desat=0", "zscale=t=bt709:m=bt709:r=tv"]
    filters += [f"scale='min({ing['max_width']},iw)':-2", f"fps={fps_out:.6f}", "format=yuv420p"]
    cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(src), "-map", "0:v:0", "-an", "-sn", "-dn",
           "-vf", ",".join(filters), "-fps_mode", "cfr", "-c:v", "libx264", "-preset", ing["preset"],
           "-crf", str(ing["crf"]), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    r = _run(cmd, timeout=1800)
    if r.returncode != 0 or not dst.exists():
        raise IngestError("error.not_a_video")
    return probe(dst)


def quality_warnings(path: Path, info: VideoInfo, cfg: dict) -> list[str]:
    """Cheap checks on a few sampled frames -> i18n warning keys (portrait, dark, blurry, short)."""
    q = cfg["quality"]
    warnings = []
    if info.height > info.width:
        warnings.append("warning.portrait")
    if info.duration_s < cfg["ingest"]["min_duration_s"]:
        warnings.append("warning.short_clip")
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    lumas, sharpness = [], []
    for i in np.linspace(0, n - 1, num=min(q["sample_frames"], n)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lumas.append(float(gray.mean()))
        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    cap.release()
    if lumas and np.median(lumas) < q["low_light_luma"]:
        warnings.append("warning.low_light")
    if sharpness and np.median(sharpness) < q["blur_laplacian_var"]:
        warnings.append("warning.blurry")
    return warnings
