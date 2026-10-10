"""Blur faces and licence plates in every frame we publish (CLAUDE.md hard rule 4, §10.9).

Three sources of regions to hide, all applied to the rendered (720p) frame:
1. faces found by the YuNet face detector (OpenCV Zoo, MIT licence);
2. the head area of every person box from the COCO detector — YuNet misses small and side-on
   faces, a person box does not;
3. a band in the lower-middle part of every vehicle box, where the licence plate usually is
   (no plate detector with a clear licence was available, so this fallback band is used).

Regions are pixelated strongly (unrecoverable) instead of a light blur.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pipeline.vehicles import ObjectBox

VEHICLES = {"car", "motorcycle", "bus", "truck"}


class FaceDetector:
    """YuNet face detector (works on CPU, a few ms per 720p frame)."""

    def __init__(self, model_path: Path, score_threshold: float):
        self.model = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score_threshold, 0.3, 5000)

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        self.model.setInputSize((w, h))
        _, faces = self.model.detect(frame)
        if faces is None:
            return []
        return [(int(x), int(y), int(x + fw), int(y + fh)) for x, y, fw, fh in faces[:, :4]]


def regions_to_hide(objects: list[ObjectBox], scale: float, band: dict) -> list[tuple[int, int, int, int]]:
    """Head areas of people and plate bands of vehicles, scaled to the rendered frame."""
    regions = []
    for o in objects:
        x1, y1, x2, y2 = (v * scale for v in o.box)
        w, h = x2 - x1, y2 - y1
        if o.kind == "person":
            regions.append((x1 + 0.1 * w, y1, x2 - 0.1 * w, y1 + 0.28 * h))       # head and shoulders
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
                              [o for o in objects if o.kind != "person"], scale, cfg["plate_band"])
    if faces is not None:
        regions += faces.detect(frame)
    pixelate(frame, regions)
