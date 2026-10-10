"""SignTracker on synthetic boxes: fast signs whose boxes never overlap stay one track, two signs on
one pole keep their own tracks, a long gap starts a new track, reset restarts the ids."""

from app.settings import get_settings
from pipeline.tracking import SignTracker

CFG = get_settings().cfg["track"]
DT = 0.08  # 12.5 analysed frames per second


def box(cx, cy, size):
    return (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def test_fast_sign_without_overlap_is_one_track():
    # A sign streaming right and growing, moving more than its own width per frame (accelerating).
    tr, ids, boxes = SignTracker(CFG), [], []
    x, size, v = 700.0, 30.0, 25.0
    for k in range(9):
        boxes.append(box(x, 300 - k * 4, size))
        ids += tr.update(k * DT, [boxes[-1]], ["4.2.1"])
        x, size, v = x + v, size * 1.12, v * 1.25
    assert max(iou(a, b) for a, b in zip(boxes, boxes[1:])) < 0.3   # ByteTrack could not link these
    assert len(set(ids)) == 1


def test_two_signs_on_one_pole_keep_their_tracks():
    tr, top, bottom = SignTracker(CFG), [], []
    for k in range(8):
        x, size = 800 + 12 * k, 30 + 2 * k
        a, b = tr.update(k * DT, [box(x, 300 - size * 0.55, size), box(x, 300 + size * 0.55, size)], ["2.1", "4.1.4"])
        top.append(a); bottom.append(b)
    assert len(set(top)) == 1 and len(set(bottom)) == 1 and top[0] != bottom[0]


def test_box_far_from_the_prediction_starts_a_new_track():
    tr = SignTracker(CFG)
    a = [tr.update(k * DT, [box(800 + 10 * k, 300, 30)], ["3.24.6"])[0] for k in range(4)]
    b = tr.update(4 * DT, [box(200, 300, 30)], ["3.24.6"])[0]   # other side of the frame
    assert len(set(a)) == 1 and b != a[0]


def test_gap_longer_than_max_gap_ends_the_track_and_reset_restarts_ids():
    tr = SignTracker(CFG)
    a = tr.update(0.0, [box(800, 300, 30)], ["2.4"])[0]
    b = tr.update(0.08, [box(805, 300, 31)], ["2.4"])[0]
    c = tr.update(1.0, [box(810, 300, 32)], ["2.4"])[0]          # unseen for 0.92 s
    assert a == b and c != a
    tr.reset()
    assert tr.update(0.0, [box(800, 300, 30)], ["2.4"]) == [1]
