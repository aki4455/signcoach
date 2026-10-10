"""Relevance heuristics (CLAUDE.md §10.5): side, facing, approaching, conditional plates."""

from __future__ import annotations

from app.settings import get_settings
from pipeline.relevance import assess
from pipeline.signs import VotedSign
from pipeline.types import Detection

CFG = get_settings().cfg
W, H = 1920, 1080


def track(code, x, y, sizes, aspect=1.0):
    dets = [Detection(frame_idx=i, t=i / 15, track_id=1, model_class=code, sign_code=code, value=None, conf=0.8,
                      bbox=(x, y, x + s * aspect, y + s)) for i, s in enumerate(sizes)]
    return VotedSign(code, None, dets, dets)


GROWING = [20, 24, 28, 34, 40, 48, 58, 70, 84]


def test_growing_sign_on_the_right_is_relevant():
    e = assess(track("2.5", 1500, 300, GROWING), "s1", [], W, H, "octagon", CFG)
    assert e.side == "right" and e.relevant


def test_sign_drifting_right_while_growing_is_on_the_right():
    # Starts left of centre (camera turned slightly) but streams to the right as we approach.
    sign = track("3.24", 300, 600, GROWING)
    for i, d in enumerate(sign.winning):
        x = 300 + 40 * i
        d.bbox = (x, d.bbox[1], x + (d.bbox[2] - d.bbox[0]), d.bbox[3])
    e = assess(sign, "s1", [], W, H, "round", CFG)
    assert e.side == "right" and e.relevant


def test_sign_on_the_left_is_not_relevant():
    e = assess(track("2.5", 200, 300, GROWING), "s1", [], W, H, "octagon", CFG)
    assert e.side == "left" and not e.relevant and "left" in e.relevance_reason


def test_high_central_sign_is_overhead_and_relevant():
    e = assess(track("3.24", 940, 60, GROWING), "s1", [], W, H, "round", CFG)
    assert e.side == "overhead" and e.relevant


def test_narrow_round_sign_faces_a_side_road():
    e = assess(track("3.1", 1500, 300, GROWING, aspect=0.4), "s1", [], W, H, "round", CFG)
    assert not e.relevant and "side" in e.relevance_reason


def test_sign_that_does_not_grow_is_not_relevant():
    e = assess(track("3.1", 1500, 300, [40] * 9), "s1", [], W, H, "round", CFG)
    assert not e.relevant and "not getting closer" in e.relevance_reason


def test_plate_under_sign_makes_it_conditional():
    sign = track("3.27", 1500, 300, GROWING)
    plate_dets = [Detection(frame_idx=d.frame_idx, t=d.t, track_id=2, model_class="8.5.4", sign_code="8.5.4",
                            value=None, conf=0.8, bbox=(d.bbox[0], d.bbox[3] + 4, d.bbox[2], d.bbox[3] + 30))
                  for d in sign.winning]
    e = assess(sign, "s1", [VotedSign("8.5.4", None, plate_dets, plate_dets)], W, H, "round", CFG)
    assert e.conditional
