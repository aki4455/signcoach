"""Decode the normalized video at a fixed analysis rate, with exact timestamps (CLAUDE.md §10.2).

ffmpeg does the decoding and the frame selection (`fps` filter) and streams raw BGR frames through
a pipe. Because the normalized video has a constant frame rate starting at t=0, frame number i of
this stream is at exactly t = i / fps. Both pipeline passes use this iterator with the same fps,
so frame indices line up: the analysis pass (full resolution) and the render pass (720p).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterator

import numpy as np


def output_size(width: int, height: int, out_height: int | None) -> tuple[int, int]:
    """Frame size after optional downscaling to `out_height` (width kept even, aspect preserved)."""
    if not out_height or out_height >= height:
        return width, height
    w = int(round(width * out_height / height / 2)) * 2
    return w, out_height


def iter_frames(path: Path, width: int, height: int, fps: float,
                out_height: int | None = None) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_idx, t_seconds, BGR frame) at `fps` frames per second."""
    w, h = output_size(width, height, out_height)
    vf = f"fps={fps:.6f}" + (f",scale={w}:{h}" if (w, h) != (width, height) else "")
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vf", vf,
           "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    frame_bytes = w * h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_bytes * 2)
    try:
        idx = 0
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield idx, idx / fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
            idx += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
