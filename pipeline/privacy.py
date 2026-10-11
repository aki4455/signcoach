"""Blur faces and licence plates in every frame we publish (CLAUDE.md hard rule 4, §10.9).

Four sources of regions to hide, all applied to the rendered (720p) frame:
1. faces found by the YuNet face detector (OpenCV Zoo, MIT licence);
2. the head area of every person box from the COCO detector — YuNet misses small and side-on
   faces, a person box does not;
3. licence plates found by the plate detector (also those found one frame before or after, so a
   single missed frame does not flash a readable plate);
4. a fallback area over the lower part of every vehicle box, for plates the detector misses (on
   close cars seen from a dashcam the plate sits around the middle of the box, so the area is big).

Regions are pixelated strongly (unrecoverable) instead of a light blur.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pipeline.vehicles import ObjectBox

VEHICLES = {"car", "motorcycle", "bus", "truck"}


class FaceDetector:
    """YuNet face detector (runs on the CPU, ~37 ms per 720p frame on one core). An instance must not be
    shared by two threads (it keeps the input size), so parallel rendering uses one `copy()` per thread."""

    def __init__(self, model_path: Path, score_threshold: float):
        self.model_path, self.score_threshold = model_path, score_threshold
        self.model = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score_threshold, 0.3, 5000)

    def copy(self) -> "FaceDetector":
        return FaceDetector(self.model_path, self.score_threshold)

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        self.model.setInputSize((w, h))
        _, faces = self.model.detect(frame)
        if faces is None:
            return []
        return [(int(x), int(y), int(x + fw), int(y + fh)) for x, y, fw, fh in faces[:, :4]]


def is_own_hood(box: tuple[float, float, float, float], frame_w: int, frame_h: int, hood: dict) -> bool:
    """A 'vehicle' box that is wide, low and glued to the bottom edge is our own car's bonnet."""
    x1, y1, x2, y2 = box
    return (x2 - x1 >= hood["min_width"] * frame_w and y1 >= hood["min_top"] * frame_h
            and y2 >= hood["min_bottom"] * frame_h)


def regions_to_hide(objects: list[ObjectBox], scale: float, band: dict, frame_wh: tuple[int, int] | None = None,
                    hood: dict | None = None) -> list[tuple[int, int, int, int]]:
    """Head areas of people and plate bands of vehicles, scaled to the rendered frame.

    Our own bonnet (in view on many phone/action-camera videos) is detected as a car; its fallback plate
    band would pixelate the bottom of every frame, so it is skipped. Plates found by the plate detector
    are always hidden."""
    regions = []
    for o in objects:
        x1, y1, x2, y2 = (v * scale for v in o.box)
        w, h = x2 - x1, y2 - y1
        if o.kind in VEHICLES and hood and frame_wh and is_own_hood((x1, y1, x2, y2), *frame_wh, hood):
            continue
        if o.kind == "person":
            regions.append((x1 + 0.1 * w, y1, x2 - 0.1 * w, y1 + 0.28 * h))       # head and shoulders
        elif o.kind == "plate":
            regions.append((x1, y1, x2, y2))
        elif o.kind in VEHICLES:
            regions.append((x1 + band["x_from"] * w, y1 + band["y_from"] * h,
                            x1 + band["x_to"] * w, y1 + band["y_to"] * h))
    return [tuple(int(round(v)) for v in r) for r in regions]


def pixelate(frame: np.ndarray, regions: list[tuple[int, int, int, int]], pad: float = 0.15) -> None:
    """Pixelate each region in place (with a small margin)."""
    H, W = frame.shape[:2]
    for x1, y1, x2, y2 in regions:
        pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
        x1, y1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
        x2, y2 = min(W, int(x2 + pw)), min(H, int(y2 + ph))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = frame[y1:y2, x1:x2]
        block = max(2, min(roi.shape[0], roi.shape[1]) // 5)  # ~5 blocks across the short side: unreadable
        small = cv2.resize(roi, (max(1, roi.shape[1] // block), max(1, roi.shape[0] // block)),
                           interpolation=cv2.INTER_AREA)
        frame[y1:y2, x1:x2] = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)


def blur_frame(frame: np.ndarray, faces: FaceDetector | None, objects: list[ObjectBox], scale: float,
               cfg: dict) -> None:
    """Hide faces, heads and plates in a rendered frame (in place)."""
    regions = regions_to_hide(objects if cfg.get("blur_person_heads", True) else
                              [o for o in objects if o.kind != "person"], scale, cfg["plate_band"],
                              (frame.shape[1], frame.shape[0]), cfg.get("own_hood"))
    if faces is not None:
        regions += faces.detect(frame)
    pixelate(frame, regions)
