"""Vehicles and people from a COCO-trained YOLO11n (CLAUDE.md §6, §10.9).

Used for privacy (blur licence plates inside vehicle boxes, blur the head area of people) and,
in M3, to mask moving vehicles out of the road-surface speed estimate. No tracking is needed.
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
