"""Decode the normalized video at the analysis rate, with exact timestamps (CLAUDE.md §10.2).

ffmpeg decodes and streams raw BGR frames through a pipe. We keep every `step`-th frame of the
(constant-frame-rate) normalized video, with step = round(video fps / analysis_fps): 30 fps -> every
2nd frame (15 fps), 25 fps -> every 2nd (12.5 fps), 10 fps -> every frame. Keeping whole frames
(instead of resampling to exactly 15 fps) keeps the time between two analysed frames constant,
which the speed estimate depends on: resampling 25 fps to 15 fps would alternate gaps of 0.04 s
and 0.08 s. Frame i is at t = i * step / video_fps.

Both pipeline passes use the same step, so frame indices line up: the analysis pass (full
resolution) and the render pass (720p).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterator

import numpy as np


def analysis_step(video_fps: float, target_fps: float) -> int:
    """Keep every n-th frame so that the analysed rate is as close as possible to target_fps."""
    return max(1, round(video_fps / target_fps))


def output_size(width: int, height: int, out_height: int | None) -> tuple[int, int]:
    """Frame size after optional downscaling to `out_height` (width kept even, aspect preserved)."""
    if not out_height or out_height >= height:
        return width, height
    w = int(round(width * out_height / height / 2)) * 2
    return w, out_height


def iter_frames(path: Path, width: int, height: int, video_fps: float, step: int,
                out_height: int | None = None) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_idx, t_seconds, BGR frame) for every `step`-th frame of the video."""
    w, h = output_size(width, height, out_height)
    filters = [f"framestep={step}"] if step > 1 else []
    if (w, h) != (width, height):
        filters.append(f"scale={w}:{h}")
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path)]
    cmd += (["-vf", ",".join(filters)] if filters else []) + ["-fps_mode", "passthrough",
                                                             "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    frame_bytes = w * h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_bytes * 2)
    try:
        idx = 0
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield idx, idx * step / video_fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
            idx += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
