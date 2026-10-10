"""Sign tracker: links the detections of one physical sign across analysed frames.

Why not ByteTrack (the tracker built into Ultralytics)? ByteTrack joins a new box to a track only
when the two boxes overlap on screen. We analyse 12.5–15 frames per second, and at 60–80 km/h a sign
close to the car moves more than its own width between two analysed frames: its boxes never
overlap, ByteTrack never confirms the track, and the sign is lost even though the detector saw it
clearly (ru_02: a keep-right sign at 0.95 confidence in 9 frames in a row, ByteTrack returned none).

Signs move smoothly on screen — away from the vanishing point, faster and bigger as we approach.
So every track predicts where its sign will be in the next frame from its recent motion (velocity
of the centre and growth rate of the size), and a new box joins the track whose prediction it is
closest to, measured in sign sizes (30 px is far for a 20 px sign but close for a 120 px one).
Pairs are linked closest first (each track and each box at most once per frame), so two signs on one
pole keep their own tracks. A track ends when its sign has not been seen for `track.max_gap_s`.

Thresholds are in `config/settings.yaml` under `track`. ByteTrack stays available for comparison
(`detect.tracker: bytetrack.yaml`; `eval/compare_trackers.py`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Box = tuple[float, float, float, float]


def _centre_size(box: Box) -> tuple[float, float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2, math.sqrt(max(1.0, (x2 - x1) * (y2 - y1)))


@dataclass
class _Track:
    id: int
    t: float          # last time seen (s)
    cx: float         # last centre (px)
    cy: float
    size: float       # sqrt(box area) (px)
    label: str        # last model class
    vx: float = 0.0   # centre velocity (px/s)
    vy: float = 0.0
    growth: float = 0.0  # d(log size)/dt (1/s)
    n: int = 1        # boxes so far

    def predict(self, t: float) -> tuple[float, float, float]:
        dt = t - self.t
        return self.cx + self.vx * dt, self.cy + self.vy * dt, self.size * math.exp(self.growth * dt)

    def add(self, t: float, cx: float, cy: float, size: float, label: str, smoothing: float) -> None:
        dt = max(1e-3, t - self.t)
        vx, vy, growth = (cx - self.cx) / dt, (cy - self.cy) / dt, math.log(size / self.size) / dt
        a = 1.0 if self.n == 1 else smoothing  # the first motion estimate is taken as it is
        self.vx, self.vy = a * vx + (1 - a) * self.vx, a * vy + (1 - a) * self.vy
        self.growth = a * growth + (1 - a) * self.growth
        self.t, self.cx, self.cy, self.size, self.label, self.n = t, cx, cy, size, label, self.n + 1


class SignTracker:
    """Call `update` once per analysed frame, in time order; `reset` before every new video."""

    def __init__(self, cfg: dict):
        self.max_gap_s = cfg["max_gap_s"]
        self.gate = cfg["gate"]
        self.gate_new = cfg["gate_new"]
        self.max_size_ratio = cfg["max_size_ratio"]
        self.class_change_cost = cfg["class_change_cost"]
        self.smoothing = cfg["velocity_smoothing"]
        self.reset()

    def reset(self) -> None:
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, t: float, boxes: list[Box], labels: list[str]) -> list[int]:
        """Track id for each box of one frame."""
        self._tracks = [tr for tr in self._tracks if t - tr.t <= self.max_gap_s]
        dets = [_centre_size(b) for b in boxes]
        ids = [0] * len(dets)
        pairs = []  # (cost, track index, box index) for every pair inside the gates
        for i, tr in enumerate(self._tracks):
            px, py, ps = tr.predict(t)
            gate = self.gate if tr.n > 1 else self.gate_new  # a one-box track's motion is still unknown
            for j, (cx, cy, s) in enumerate(dets):
                ratio = s / ps
                dist = math.hypot(cx - px, cy - py) / max(ps, s)
                if dist <= gate and 1 / self.max_size_ratio <= ratio <= self.max_size_ratio:
                    pairs.append((dist + abs(math.log(ratio)) + (self.class_change_cost if labels[j] != tr.label else 0.0), i, j))
        used = set()
        for _, i, j in sorted(pairs):
            if i not in used and not ids[j]:
                used.add(i)
                self._tracks[i].add(t, *dets[j], labels[j], self.smoothing)
                ids[j] = self._tracks[i].id
        for j, det in enumerate(dets):
            if not ids[j]:
                self._tracks.append(_Track(self._next_id, t, *det, labels[j]))
                ids[j] = self._next_id
                self._next_id += 1
        return ids
