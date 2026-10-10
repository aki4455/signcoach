"""Ego-motion on synthetic video with a known answer (CLAUDE.md §10.6, M3 acceptance).

A textured road plane is rendered with the same pinhole camera model the estimator assumes, moving
at a known speed; distant "scenery" above the horizon is panned sideways to simulate turning. A static
on-screen text box (like a dashcam's speed display) sits on top of the road: it must be masked out,
or it would pull the estimate towards zero.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from app.settings import get_settings
from pipeline.egomotion import Camera, EgoMotion, focal_length, static_mask

CFG = get_settings().cfg
W, H, FPS, CAM_H, HFOV = 640, 360, 15.0, 1.3, 70.0
F = focal_length(W, HFOV)
HORIZON = 0.45 * H
TEX_M = 0.02  # texture: metres per texel

rng = np.random.default_rng(0)
ROAD = cv2.GaussianBlur((rng.random((512, 512)) * 255).astype(np.float32), (0, 0), 2)
ROAD = cv2.normalize(ROAD, None, 0, 255, cv2.NORM_MINMAX)
SKY = cv2.GaussianBlur((rng.random((int(HORIZON), 4096)) * 255).astype(np.float32), (0, 0), 1.5)
SKY = cv2.normalize(SKY, None, 0, 255, cv2.NORM_MINMAX)

# Ground coordinates of every pixel below the horizon (pinhole camera, CLAUDE.md §10.6).
u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
below = v > HORIZON + 1
Z = np.where(below, F * CAM_H / np.maximum(v - HORIZON, 1e-3), 0)
X = np.where(below, (u - W / 2) * Z / F, 0)


def render(distance_m: float, sky_shift_px: float, overlay: bool) -> np.ndarray:
    map_x = (X / TEX_M) % 512
    map_y = ((Z + distance_m) / TEX_M) % 512
    road = cv2.remap(ROAD, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    frame = np.where(below, road, 0).astype(np.float32)
    sky_x = ((np.arange(W) + sky_shift_px) % SKY.shape[1]).astype(int)
    frame[: int(HORIZON)] = SKY[:, sky_x]
    img = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if overlay:  # static dashcam text box over the road, bottom right
        cv2.rectangle(img, (440, 300), (630, 345), (20, 20, 20), -1)
        cv2.putText(img, "55 KM/H", (452, 333), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2)
    return img


def run(speed_kmh: float, yaw_dps: float = 0.0, seconds: float = 4.0, overlay: bool = True):
    n = int(seconds * FPS)
    step_m = speed_kmh / 3.6 / FPS
    shift_px = F * math.tan(math.radians(yaw_dps / FPS))  # sideways shift of distant scenery per frame
    frames = [render(i * step_m, i * shift_px, overlay) for i in range(n)]  # turning right: scenery moves left
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames[:: max(1, n // 30)]]
    ego = EgoMotion(Camera(W, H, F, W / 2, HORIZON, CAM_H), static_mask(grays, CFG["egomotion"]["static_std"]), FPS, CFG)
    for i, f in enumerate(frames):
        ego.update(f, i / FPS, [])
    return ego.finish()


@pytest.mark.parametrize("speed", [20.0, 50.0, 90.0])
def test_known_speed_is_recovered_despite_static_overlay(speed):
    motion, _ = run(speed)
    measured = [m.speed_kmh for m in motion[int(FPS):] if m.speed_kmh is not None]
    assert len(measured) > 0.8 * len(motion[int(FPS):])
    assert abs(np.median(measured) - speed) / speed < 0.08


def test_standing_still_is_a_stop():
    motion, _ = run(0.0, seconds=3.0)
    assert all(m.speed_kmh is not None and m.speed_kmh < 1.0 for m in motion[int(FPS):])
    assert all(m.stopped for m in motion[int(FPS):])


def test_stop_in_the_middle_of_a_drive():
    # 30 km/h for 3 s, standing for 3 s, 30 km/h again for 3 s.
    n, step_m = int(3 * FPS), 30 / 3.6 / FPS
    dist = [i * step_m for i in range(n)] + [n * step_m] * n + [n * step_m + i * step_m for i in range(n)]
    frames = [render(d, 0.0, True) for d in dist]
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames[::4]]
    ego = EgoMotion(Camera(W, H, F, W / 2, HORIZON, CAM_H), static_mask(grays, CFG["egomotion"]["static_std"]), FPS, CFG)
    for i, f in enumerate(frames):
        ego.update(f, i / FPS, [])
    motion, _ = ego.finish()
    middle = motion[n + int(FPS * 0.7): 2 * n - int(FPS * 0.7)]
    assert all(m.stopped for m in middle)
    assert not any(m.stopped for m in motion[int(FPS): n - int(FPS)])


def test_sustained_yaw_is_a_right_turn():
    motion, turns = run(25.0, yaw_dps=20.0, seconds=6.0)  # 20 deg/s for ~6 s = ~115 deg
    assert len(turns) == 1 and turns[0].direction == "right"
    assert 90 <= turns[0].angle_deg <= 140
    assert motion[-1].heading_deg > 90


def test_driving_straight_has_no_turns():
    _, turns = run(40.0, yaw_dps=0.0, seconds=5.0)
    assert turns == []
