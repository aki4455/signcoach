"""Render the annotated, privacy-blurred lesson video, the thumbnails and the poster (CLAUDE.md §10.10).

This is the second decode pass (§10.2): the video is decoded again at 720p and the same analysis
frame rate, so frame i here is frame i of the analysis pass. For every frame we:
1. blur faces (YuNet on this frame), people's heads and licence-plate bands (boxes from the
   analysis pass, scaled down) — privacy first, before anything is drawn or saved;
2. draw boxes and labels for relevant signs (irrelevant ones in grey only in debug mode);
3. draw the HUD: speed limit in force and our speed ("est." because it comes from video);
4. draw a subtitle line while an event's voice prompt would be playing.
Frames are piped as raw pixels into ffmpeg (libx264, +faststart, no audio). Thumbnails and the
poster are cut from blurred frames only, so nothing unblurred ever leaves the job folder.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from pipeline import frames
from pipeline.privacy import FaceDetector, blur_frame
from pipeline.types import MotionSample, RuleState
from pipeline.vehicles import ObjectBox

GREEN, GREY, WHITE, RED, DARK = (60, 200, 60), (160, 160, 160), (255, 255, 255), (40, 40, 230), (25, 25, 25)
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class RenderJob:
    normalized: Path
    width: int                                   # analysis (normalized) frame size
    height: int
    fps: float
    objects_by_frame: list[list[ObjectBox]]      # analysis pass: people and vehicles per frame
    sign_boxes: dict[int, list[tuple[tuple, str, bool]]]  # frame -> [(box, label, relevant)]
    timeline: list[RuleState]
    motion: list[MotionSample]
    subtitles: list[tuple[float, float, str]]    # (t_start, t_end, text)
    thumbnails: dict[int, list[tuple[str, tuple]]] = field(default_factory=dict)  # frame -> [(file, box)]
    poster_frame: int = 0


def _state_at(timeline: list[RuleState], t: float) -> RuleState:
    current = timeline[0]
    for s in timeline:
        if s.t <= t:
            current = s
    return current


def _speed_at(motion: list[MotionSample], t: float) -> MotionSample | None:
    if not motion:
        return None
    m = min(motion, key=lambda m: abs(m.t - t))
    return m if abs(m.t - t) <= 1.0 else None


def _text(frame, text, org, scale, color, thick=1):
    cv2.putText(frame, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def _draw_sign(frame, box, label, color):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
    ty = max(th + 6, y1 - 4)
    cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 8, ty + 2), color, -1)
    _text(frame, label, (x1 + 4, ty - 3), 0.5, DARK)


def _draw_hud(frame, state: RuleState, sample: MotionSample | None):
    limit = f"{state.speed_limit_kmh} km/h" if state.speed_limit_kmh else "unknown"
    over = False
    if sample is not None and sample.speed_kmh is not None:
        speed = f"{sample.speed_kmh:.0f} km/h est."
        over = state.speed_limit_kmh is not None and sample.speed_kmh > state.speed_limit_kmh
    else:
        speed = "not measured"
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (292, 78), DARK, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    _text(frame, f"Limit: {limit}", (24, 38), 0.6, WHITE, 1)
    _text(frame, f"Speed: {speed}", (24, 66), 0.6, RED if over else WHITE, 2 if over else 1)


def _draw_subtitle(frame, text):
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.65, 2)
    x = max(10, (w - tw) // 2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 12, h - 52 - th), (x + tw + 12, h - 30), DARK, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    _text(frame, text, (x, h - 40), 0.65, WHITE, 2)


def _crop(frame, box, min_side=128):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(min_side, 2.2 * max(x2 - x1, y2 - y1))
    H, W = frame.shape[:2]
    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    return frame[b:int(min(H, cy + side / 2)), a:int(min(W, cx + side / 2))]


def render(job: RenderJob, out_dir: Path, log_path: Path, faces: FaceDetector | None, cfg: dict,
           check_deadline: Callable[[], None], progress: Callable[[float], None]) -> dict[str, str]:
    """Write annotated.mp4, poster.jpg and thumbs/*.jpg into out_dir. Returns {thumbnail file: path}."""
    out_cfg, privacy_cfg = cfg["output"], cfg["privacy"]
    w, h = frames.output_size(job.width, job.height, out_cfg["height"])
    scale = h / job.height
    (out_dir / "thumbs").mkdir(parents=True, exist_ok=True)
    log = open(log_path, "wb")  # ffmpeg messages stay in the private job folder
    enc = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
         "-r", f"{job.fps:.6f}", "-i", "-", "-an", "-c:v", "libx264", "-preset", out_cfg["preset"],
         "-crf", str(out_cfg["crf"]), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(out_dir / "annotated.mp4")], stdin=subprocess.PIPE, stderr=log)
    saved: dict[str, str] = {}
    n = max(1, len(job.objects_by_frame))
    try:
        for idx, t, frame in frames.iter_frames(job.normalized, job.width, job.height, job.fps, out_cfg["height"]):
            check_deadline()
            objects = job.objects_by_frame[idx] if idx < len(job.objects_by_frame) else []
            blur_frame(frame, faces, objects, scale, privacy_cfg)            # 1. privacy first

            for name, box in job.thumbnails.get(idx, []):                    # thumbnails: blurred, no overlay
                crop = _crop(frame, [v * scale for v in box])
                if crop.size:
                    cv2.imwrite(str(out_dir / "thumbs" / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    saved[name] = f"thumbs/{name}"

            for box, label, relevant in job.sign_boxes.get(idx, []):         # 2. signs
                if relevant or out_cfg.get("debug_draw_irrelevant"):
                    _draw_sign(frame, [v * scale for v in box], label, GREEN if relevant else GREY)
            _draw_hud(frame, _state_at(job.timeline, t), _speed_at(job.motion, t))  # 3. HUD
            for t0, t1, text in job.subtitles:                               # 4. voice prompt subtitles
                if t0 <= t < t1:
                    _draw_subtitle(frame, text)
                    break

            if idx == job.poster_frame:
                cv2.imwrite(str(out_dir / "poster.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            enc.stdin.write(np.ascontiguousarray(frame).tobytes())
            if idx % 15 == 0:
                progress(min(1.0, idx / n))
    finally:
        enc.stdin.close()
        enc.wait()
        log.close()
    if enc.returncode != 0:
        raise RuntimeError(f"ffmpeg encoder failed with code {enc.returncode}")
    return saved
