"""Track voting: many per-frame detections of one physical sign -> one decision (CLAUDE.md §10.4).

A sign is usually seen in 10-60 consecutive frames, and a single frame can be wrong (a "60" read
as "80", a reflection mistaken for a sign). ByteTrack gives each physical sign a track id; for each
track we pick the class with the highest *sum* of confidences and keep the sign only if it was seen
in enough frames with enough mean confidence. Tracks of the same sign split by an id switch are
merged first (same class, close in time and position).

`vote.single_frame_baseline: true` skips voting (every confident detection counts on its own), so
eval/ can measure what voting adds.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean

from pipeline.types import Detection


def _area(d: Detection) -> float:
    x1, y1, x2, y2 = d.bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _centre(d: Detection) -> tuple[float, float]:
    x1, y1, x2, y2 = d.bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


@dataclass
class VotedSign:
    """One physical sign after voting, with all its detections (needed by relevance and render)."""

    sign_code: str
    value: int | None
    detections: list[Detection]                       # every detection of the (merged) track, by time
    winning: list[Detection] = field(default_factory=list)  # detections of the winning class

    @property
    def conf(self) -> float:
        return mean(d.conf for d in self.winning)

    @property
    def n_frames(self) -> int:
        return len({d.frame_idx for d in self.winning})

    @property
    def t_first(self) -> float:
        return self.detections[0].t

    @property
    def t_last(self) -> float:
        return self.detections[-1].t

    @property
    def best(self) -> Detection:
        """Most legible view: the largest box of the winning class."""
        return max(self.winning, key=_area)


def _vote_track(dets: list[Detection]) -> VotedSign | None:
    mapped = [d for d in dets if d.sign_code is not None]
    if not mapped:
        return None
    scores: dict[tuple[str, int | None], float] = defaultdict(float)
    for d in mapped:
        scores[(d.sign_code, d.value)] += d.conf
    code, value = max(scores, key=scores.get)
    winning = [d for d in mapped if (d.sign_code, d.value) == (code, value)]
    return VotedSign(code, value, sorted(dets, key=lambda d: (d.t, d.frame_idx)), winning)


def _merge_id_switches(signs: list[VotedSign], window_s: float, max_dist_px: float) -> list[VotedSign]:
    """Merge consecutive tracks of the same sign split by a tracker id switch."""
    signs = sorted(signs, key=lambda s: s.t_first)
    merged: list[VotedSign] = []
    for s in signs:
        target = None
        for m in reversed(merged):
            if (m.sign_code, m.value) != (s.sign_code, s.value) or s.t_first - m.t_last > window_s:
                continue
            (ax, ay), (bx, by) = _centre(m.winning[-1]), _centre(s.winning[0])
            if ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 <= max_dist_px:
                target = m
                break
        if target is None:
            merged.append(s)
        else:
            target.detections = sorted(target.detections + s.detections, key=lambda d: (d.t, d.frame_idx))
            target.winning = sorted(target.winning + s.winning, key=lambda d: (d.t, d.frame_idx))
    return merged


def vote(detections: list[Detection], cfg: dict, frame_width: int) -> list[VotedSign]:
    """All signs kept after voting, ordered by the time they were passed."""
    v = cfg["vote"]
    if v.get("single_frame_baseline"):
        kept = [VotedSign(d.sign_code, d.value, [d], [d]) for d in detections
                if d.sign_code is not None and d.conf >= v["min_conf"]]
        return sorted(kept, key=lambda s: s.t_last)

    tracks: dict[int, list[Detection]] = defaultdict(list)
    for d in detections:
        if d.track_id is not None:
            tracks[d.track_id].append(d)
    candidates = [s for s in (_vote_track(dets) for dets in tracks.values()) if s is not None]
    merged = _merge_id_switches(candidates, v["dedupe_window_s"], v["dedupe_max_center_dist"] * frame_width)
    kept = [s for s in merged if s.n_frames >= v["min_frames"] and s.conf >= v["min_conf"]]
    return sorted(kept, key=lambda s: s.t_last)
