"""Vehicles, people and licence plates (CLAUDE.md §6, §10.9).

- VehicleDetector: COCO-trained YOLO11n -> cars, buses, trucks, motorcycles and people. Used for
  privacy (plate area of vehicles, head area of people) and to mask moving objects out of the
  road-surface speed estimate.
- PlateDetector: a YOLO11s fine-tuned for licence plates (AGPL-3.0, see DATA_SOURCES.md) -> the
  plates themselves. Privacy uses both: the detector's plate boxes and a fallback area on every
  vehicle, because neither alone catches every plate (checked on our Moscow clips).
No tracking is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

COCO_KINDS = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass(frozen=True)
class ObjectBox:
    kind: str                                   # person | car | motorcycle | bus | truck
    box: tuple[float, float, float, float]      # x1, y1, x2, y2 in analysis-frame pixels
    conf: float


class VehicleDetector:
    def __init__(self, weights: Path, imgsz: int, conf: float, device: str, half: bool = True):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.kwargs = dict(imgsz=imgsz, conf=conf, device=device, **({"quantize": 16} if half and device != "cpu" else {}),
                           classes=list(COCO_KINDS), verbose=False)

    def detect(self, frame: np.ndarray) -> list[ObjectBox]:
        boxes = self.model.predict(frame, **self.kwargs)[0].boxes
        if boxes is None:
            return []
        return [ObjectBox(COCO_KINDS[c], tuple(round(v, 1) for v in xyxy), round(float(p), 3))
                for c, p, xyxy in zip(boxes.cls.int().tolist(), boxes.conf.tolist(), boxes.xyxy.tolist())]


class PlateDetector:
    def __init__(self, weights: Path, imgsz: int, conf: float, device: str, half: bool = True):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.kwargs = dict(imgsz=imgsz, conf=conf, device=device, verbose=False,
                           **({"quantize": 16} if half and device != "cpu" else {}))

    def detect(self, frame: np.ndarray) -> list[ObjectBox]:
        boxes = self.model.predict(frame, **self.kwargs)[0].boxes
        if boxes is None:
            return []
        return [ObjectBox("plate", tuple(round(v, 1) for v in xyxy), round(float(p), 3))
                for p, xyxy in zip(boxes.conf.tolist(), boxes.xyxy.tolist())]
