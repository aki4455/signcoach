"""Our car's motion from the video alone: speed, heading/turns and stops (CLAUDE.md §10.6).

There is no GPS (the input is only the dashcam video), so everything comes from the pixels.

Speed — road-plane method
  1. Camera model: focal length f = (W/2) / tan(hfov/2), principal point at the image centre,
     camera height h above the road, horizon row y_h (from the vanishing point of lane lines; if
     none is found, camera.horizon_fallback_frac * H).
  2. Bird's-eye view (BEV): a road point Z metres ahead and X metres sideways appears in the image
     at u = cx + f*X/Z, v = y_h + f*h/Z. Sampling the image at those positions over a grid
     (bev.x_range_m x bev.z_range_m, bev.m_per_px) with cv2.remap gives the road seen from above.
  3. Between two analysed frames the road in the BEV slides by (distance driven) / m_per_px rows.
     cv2.phaseCorrelate measures that shift; speed = shift * m_per_px * fps * 3.6 * scale_k (km/h).
     The strength of the correlation peak is the sample's quality.
  4. Things that do not move with the road are hidden first: other vehicles and people (boxes from
     the COCO detector) and static overlays — dashboard, on-screen speed/date text, logos — found
     as pixels that hardly change over the whole video.
  5. Robust smoothing (rolling median over speed.smooth_s); low-quality samples become None
     ("not measured") — never a guess. Expected error: speed.err_model (a + b*v).
  The result scales with the assumed camera height and lens; for an unknown dashcam the report
  calls it an estimate unless the camera was calibrated (scripts/calibrate_speed.py).

Turns — distant scenery above the horizon shifts sideways when the car turns. The median sideways
  optical flow (Lucas–Kanade) gives the yaw rate; summing it gives the heading. A heading change of
  at least turn.min_angle_deg within turn.max_window_s is a turn (>= turn.uturn_angle_deg: U-turn).

Stops — speed under stop.speed_kmh for at least stop.min_duration_s. "Not moving" needs no scale,
  so it is the most reliable output of this method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pipeline.types import MotionSample, TurnEvent


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    f: float            # focal length in pixels
    cx: float
    horizon_y: float
    height_m: float


def focal_length(width: int, hfov_deg: float) -> float:
    return (width / 2) / math.tan(math.radians(hfov_deg) / 2)


def sample_grays(video: Path, n: int) -> list[np.ndarray]:
    """n grey frames spread evenly over the video (for the horizon and the static-overlay mask)."""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    grays = []
    for i in np.linspace(0, total - 1, num=min(n, total)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return grays


def _intersect(a, b):
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
    return px, py


def estimate_horizon(grays: list[np.ndarray], fallback_frac: float) -> tuple[float, str]:
    """Horizon row = median height of vanishing points of left- and right-leaning lane-like lines."""
    H, W = grays[0].shape
    ys = []
    for g in grays:
        top = int(H * 0.4)
        edges = cv2.Canny(cv2.GaussianBlur(g[top:], (5, 5), 0), 60, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=W // 25, maxLineGap=W // 60)
        if lines is None:
            continue
        left, right = [], []
        for x1, y1, x2, y2 in lines.reshape(-1, 4):  # (N,1,4) in OpenCV 4, (N,4) in OpenCV 5
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if 0.3 < abs(slope) < 3:  # lane lines converge: left ones rise to the right, right ones to the left
                (right if slope > 0 else left).append((x1, y1 + top, x2, y2 + top))
        for a in left[:25]:
            for b in right[:25]:
                p = _intersect(a, b)
                if p and 0.25 * H < p[1] < 0.7 * H and 0.2 * W < p[0] < 0.8 * W:
                    ys.append(p[1])
    if len(ys) >= 15:
        return float(np.median(ys)), "vanishing_point"
    return fallback_frac * H, "fallback"


def static_mask(grays: list[np.ndarray], std_threshold: float) -> np.ndarray:
    """True where the picture hardly changes over the whole video: dashboard, on-screen text, logos."""
    H, W = grays[0].shape
    w = 320
    h = max(1, round(w * H / W))
    stack = np.stack([cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA) for g in grays]).astype(np.float32)
    static = (stack.std(axis=0) < std_threshold).astype(np.uint8)
    static = cv2.dilate(static, np.ones((5, 5), np.uint8))  # cover text edges and dashboard borders
    return cv2.resize(static, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)


class EgoMotion:
    """Feed it every analysed frame (with the vehicle/person boxes of that frame), then call finish()."""

    YAW_WIDTH = 640  # optical flow for the yaw runs on a downscaled copy

    def __init__(self, cam: Camera, static: np.ndarray, fps: float, cfg: dict):
        self.cam, self.fps, self.cfg = cam, fps, cfg
        b = cfg["bev"]
        m = b["m_per_px"]
        xs = np.arange(b["x_range_m"][0], b["x_range_m"][1], m) + m / 2
        zs = np.arange(b["z_range_m"][1], b["z_range_m"][0], -m) - m / 2  # far rows on top
        X, Z = np.meshgrid(xs, zs)
        self.map_x = (cam.cx + cam.f * X / Z).astype(np.float32)
        self.map_y = (cam.horizon_y + cam.f * cam.height_m / Z).astype(np.float32)
        inside = ((self.map_x >= 0) & (self.map_x <= cam.width - 1) &
                  (self.map_y >= 0) & (self.map_y <= cam.height - 1))
        static_bev = cv2.remap(static.astype(np.uint8), self.map_x, self.map_y, cv2.INTER_NEAREST,
                               borderValue=1).astype(bool)
        # Overlays cover a small part of the road. If nearly all of it looks static, the car is simply
        # not moving in this video (parked) — then nothing is masked and the estimate reads ~0 km/h.
        if (static_bev & inside).sum() > cfg["egomotion"]["max_static_share"] * max(1, inside.sum()):
            static_bev[:] = False
        self.valid = inside & ~static_bev
        self.window = cv2.createHanningWindow((len(xs), len(zs)), cv2.CV_32F)
        self.m_per_px = m
        # Yaw: features above the horizon (distant scenery), not on static overlays.
        self.yh = max(1, round(self.YAW_WIDTH * cam.height / cam.width))
        yaw_mask = np.zeros((self.yh, self.YAW_WIDTH), np.uint8)
        yaw_mask[: int(cam.horizon_y / cam.height * self.yh * 0.95)] = 255
        small_static = cv2.resize(static.astype(np.uint8), (self.YAW_WIDTH, self.yh), interpolation=cv2.INTER_NEAREST)
        yaw_mask[small_static > 0] = 0
        self.yaw_mask = yaw_mask
        self.prev_bev = self.prev_small = self.background = None
        self.raw: list[tuple[float, float | None, float, float | None]] = []  # t, shift px, quality, yaw deg/frame

    def _bev(self, gray: np.ndarray, occluders: list[tuple]) -> np.ndarray | None:
        bev = cv2.remap(gray, self.map_x, self.map_y, cv2.INTER_LINEAR).astype(np.float32)
        valid = self.valid.copy()
        if occluders:
            occ = np.zeros(gray.shape, np.uint8)
            for x1, y1, x2, y2 in occluders:
                occ[max(0, int(y1)):int(y2) + 1, max(0, int(x1)):int(x2) + 1] = 1
            valid &= ~cv2.remap(occ, self.map_x, self.map_y, cv2.INTER_NEAREST).astype(bool)
        if valid.mean() < self.cfg["egomotion"]["min_road_coverage"]:
            return None
        values = bev[valid]
        if values.std() < self.cfg["egomotion"]["min_texture_std"]:   # dark or featureless road
            return None
        return np.where(valid, (bev - values.mean()) / values.std(), 0.0).astype(np.float32)

    def _yaw(self, gray: np.ndarray) -> float | None:
        small = cv2.resize(gray, (self.YAW_WIDTH, self.yh), interpolation=cv2.INTER_AREA)
        yaw = None
        if self.prev_small is not None:
            pts = cv2.goodFeaturesToTrack(self.prev_small, maxCorners=150, qualityLevel=0.01, minDistance=8,
                                          mask=self.yaw_mask)
            if pts is not None and len(pts) >= 10:
                nxt, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_small, small, pts, None, winSize=(21, 21),
                                                          maxLevel=3)
                ok = status.ravel() == 1
                if ok.sum() >= 8:
                    dx = float(np.median((nxt[ok] - pts[ok])[:, 0, 0])) * self.cam.width / self.YAW_WIDTH
                    yaw = -math.degrees(math.atan2(dx, self.cam.f))  # scene moving left = we turn right (+)
        self.prev_small = small
        return yaw

    def _forward_shift(self, prev: np.ndarray, cur: np.ndarray, allow_zero: bool = True) -> tuple[float | None, float]:
        """Phase correlation that does not fall for things moving *with* the camera.

        The correlation surface has a peak at zero shift for anything that does not move in the
        picture (windshield reflections, a car ahead at our speed) and a peak at the road's true
        forward shift. We take the best peak among forward shifts and prefer it when it is at least
        `forward_peak_ratio` as strong as the zero-shift peak; only when there is no such peak is the
        car standing still. Returns (shift in BEV rows, peak strength).
        """
        ego = self.cfg["egomotion"]
        a, b = np.fft.fft2(prev * self.window), np.fft.fft2(cur * self.window)
        cross = b * np.conj(a)
        surface = np.fft.fftshift(np.real(np.fft.ifft2(cross / (np.abs(cross) + 1e-9))))
        cy, cx = surface.shape[0] // 2, surface.shape[1] // 2
        # Peak strength = sum of the 3x3 neighbourhood (a sub-pixel peak spreads over its neighbours).
        zero_peak = float(surface[cy - 1:cy + 2, cx - 1:cx + 2].sum())
        lo = cy + max(1, int(np.ceil(ego["min_forward_shift_px"])))
        hi = min(surface.shape[0] - 1, cy + int(ego["max_kmh"] / 3.6 / self.fps / self.m_per_px) + 1)
        lat = int(ego["max_lateral_shift_px"])
        band = surface[lo:hi, cx - lat:cx + lat + 1]
        iy, ix = np.unravel_index(int(np.argmax(band)), band.shape)
        y, x = lo + iy, cx - lat + ix
        fwd_peak = float(surface[y - 1:y + 2, x - 1:x + 2].sum())
        if not allow_zero or fwd_peak >= ego["forward_peak_ratio"] * zero_peak:
            # sub-pixel position from a parabola through the peak and its two neighbours (rows)
            c0, c1, c2 = surface[y - 1, x], surface[y, x], surface[min(y + 1, surface.shape[0] - 1), x]
            denom = c0 - 2 * c1 + c2
            offset = 0.5 * (c0 - c2) / denom if abs(denom) > 1e-9 else 0.0
            return (y - cy) + float(np.clip(offset, -0.5, 0.5)), fwd_peak
        return 0.0, zero_peak

    def update(self, frame: np.ndarray, t: float, occluders: list[tuple]) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bev = self._bev(gray, occluders)
        shift, quality = None, 0.0
        ego = self.cfg["egomotion"]
        if bev is not None and self.prev_bev is not None:
            shift, quality = self._forward_shift(self.prev_bev, bev)
            if shift == 0.0 and self.background is not None:
                # "Nothing moved" may be things fixed to the camera (compression blocks, reflections,
                # a car ahead at our speed). Subtract the recent average picture so only the moving road
                # texture is left; if that shows a clear forward motion, trust it over the zero.
                hp_shift, hp_quality = self._forward_shift(self.prev_bev - self.background, bev - self.background,
                                                           allow_zero=False)
                if hp_quality >= self.cfg["speed"]["min_quality"]:
                    shift, quality = hp_shift, hp_quality
        if bev is not None:
            a = ego["background_alpha"]
            self.background = bev.copy() if self.background is None else (1 - a) * self.background + a * bev
        self.prev_bev = bev
        self.raw.append((t, shift, quality, self._yaw(gray)))

    def finish(self) -> tuple[list[MotionSample], list[TurnEvent]]:
        cfg = self.cfg
        ego, sp = cfg["egomotion"], cfg["speed"]
        n = len(self.raw)
        ts = np.array([r[0] for r in self.raw])
        kmh = np.full(n, np.nan)
        for i, (_, shift, q, _) in enumerate(self.raw):
            if shift is None or q < sp["min_quality"]:
                continue
            v = shift * self.m_per_px * self.fps * 3.6 * sp["scale_k"]
            if -ego["max_backward_kmh"] <= v <= ego["max_kmh"]:
                kmh[i] = max(0.0, v)
        # Rolling median over speed.smooth_s; a value needs at least half of its window measured.
        half = max(1, round(sp["smooth_s"] * self.fps / 2))
        smooth = np.full(n, np.nan)
        for i in range(n):
            win = kmh[max(0, i - half): i + half + 1]
            ok = win[~np.isnan(win)]
            if len(ok) >= max(1, len(win) // 2):
                smooth[i] = float(np.median(ok))

        yaw = np.array([np.nan if r[3] is None else r[3] for r in self.raw]) * self.fps  # deg/s
        yaw_s = np.full(n, np.nan)
        hy = max(1, round(0.25 * self.fps))
        for i in range(n):
            win = yaw[max(0, i - hy): i + hy + 1]
            ok = win[~np.isnan(win)]
            if len(ok):
                yaw_s[i] = float(np.median(ok))
        heading = np.cumsum(np.nan_to_num(yaw_s) / self.fps)

        stopped = self._stops(ts, smooth, cfg["stop"])
        err = sp["err_model"]
        samples = [MotionSample(
            t=round(float(ts[i]), 3), speed_kmh=None if np.isnan(smooth[i]) else round(float(smooth[i]), 1),
            speed_err_kmh=None if np.isnan(smooth[i]) else round(err["a"] + err["b"] * float(smooth[i]), 1),
            source="none" if np.isnan(smooth[i]) else "video", quality=round(float(self.raw[i][2]), 3),
            yaw_rate_dps=None if np.isnan(yaw_s[i]) else round(float(yaw_s[i]), 2),
            heading_deg=round(float(heading[i]), 1), stopped=bool(stopped[i])) for i in range(n)]
        return samples, detect_turns(ts, yaw_s, self.fps, cfg["turn"], ego["turn_rate_min_dps"])

    @staticmethod
    def _stops(ts: np.ndarray, kmh: np.ndarray, stop_cfg: dict) -> np.ndarray:
        below = ~np.isnan(kmh) & (kmh < stop_cfg["speed_kmh"])
        stopped = np.zeros(len(ts), bool)
        i = 0
        while i < len(ts):
            if below[i]:
                j = i
                while j + 1 < len(ts) and below[j + 1]:
                    j += 1
                if ts[j] - ts[i] >= stop_cfg["min_duration_s"]:
                    stopped[i:j + 1] = True
                i = j + 1
            else:
                i += 1
        return stopped


def detect_turns(ts: np.ndarray, yaw_dps: np.ndarray, fps: float, turn_cfg: dict, rate_min: float) -> list[TurnEvent]:
    """Runs of sustained yaw in one direction whose total angle is big enough to be a turn."""
    turning = np.nan_to_num(yaw_dps)
    sign = np.where(turning > rate_min, 1, np.where(turning < -rate_min, -1, 0))
    turns, i, n = [], 0, len(ts)
    while i < n:
        if sign[i] == 0:
            i += 1
            continue
        s, j, gap = sign[i], i, 0
        while j + 1 < n and (sign[j + 1] == s or (sign[j + 1] == 0 and gap < fps)):  # bridge pauses < 1 s
            gap = gap + 1 if sign[j + 1] == 0 else 0
            j += 1
        j -= gap
        angle = float(np.sum(turning[i:j + 1]) / fps)
        duration = ts[j] - ts[i]
        if abs(angle) >= turn_cfg["min_angle_deg"] and duration <= turn_cfg["max_window_s"]:
            direction = "uturn" if abs(angle) >= turn_cfg["uturn_angle_deg"] else ("right" if angle > 0 else "left")
            turns.append(TurnEvent(t_start=round(float(ts[i]), 2), t_end=round(float(ts[j]), 2),
                                   direction=direction, angle_deg=round(abs(angle), 1)))
        i = j + 1
    return turns
