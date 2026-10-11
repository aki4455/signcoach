"""Track voting (CLAUDE.md §10.4): one decision per physical sign, id switches merged, noise dropped."""

from __future__ import annotations

from app.settings import get_settings
from pipeline.signs import vote
from pipeline.types import Detection

CFG = get_settings().cfg
W = 1920


def det(i, track, code, value=None, conf=0.8, x=1500.0, size=40.0, model_class=None):
    return Detection(frame_idx=i, t=i / 15, track_id=track, model_class=model_class or f"{code}:{value}",
                     sign_code=code, value=value, conf=conf, bbox=(x, 300.0, x + size, 300.0 + size))


def test_majority_class_wins_by_summed_confidence():
    dets = [det(i, 1, "3.24", 60) for i in range(6)] + [det(6, 1, "3.24", 80, conf=0.9)]
    [s] = vote(dets, CFG, W)
    assert (s.sign_code, s.value, s.n_frames) == ("3.24", 60, 6)


def test_near_views_decide_the_digits():
    # Far away (small) the digits flicker; close up (large) they are sharp: 50, not 60.
    far = [det(i, 1, "3.24", 60, conf=0.7, x=1500 + i, size=16) for i in range(8)]
    near = [det(8 + i, 1, "3.24", 50, conf=0.7, x=1520 + 10 * i, size=80) for i in range(4)]
    [s] = vote(far + near, CFG, W)
    assert s.value == 50
    [s] = vote(far + near, CFG | {"vote": CFG["vote"] | {"weight": "confidence"}}, W)
    assert s.value == 60                                               # plain confidence sum: the far views win


def test_short_or_unsure_tracks_are_dropped():
    short = [det(i, 1, "2.5") for i in range(2)]                      # 2 frames < min_frames
    unsure = [det(i, 2, "3.1", conf=0.3) for i in range(10)]          # mean conf < min_conf
    assert vote(short + unsure, CFG, W) == []


def test_unmapped_classes_are_ignored():
    dets = [det(i, 1, None, model_class="TrafficLights") for i in range(10)]
    assert vote(dets, CFG, W) == []


def test_id_switch_is_merged_into_one_sign():
    a = [det(i, 1, "3.24", 40, x=1500 + i) for i in range(5)]
    b = [det(i, 2, "3.24", 40, x=1500 + i) for i in range(6, 11)]     # same sign, new id after a gap
    [s] = vote(a + b, CFG, W)
    assert s.n_frames == 10


def test_two_signs_far_apart_stay_separate():
    left = [det(i, 1, "3.24", 40, x=200) for i in range(5)]
    right = [det(i, 2, "3.24", 40, x=1700) for i in range(5)]
    assert len(vote(left + right, CFG, W)) == 2


def test_single_frame_baseline_counts_every_confident_detection():
    cfg = CFG | {"vote": CFG["vote"] | {"single_frame_baseline": True}}
    dets = [det(i, 1, "2.5") for i in range(4)]
    assert len(vote(dets, cfg, W)) == 4
