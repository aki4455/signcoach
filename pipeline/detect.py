"""Road-sign detection and tracking behind one small interface (CLAUDE.md §10.3).

Why an adapter: the final detector (our fine-tuned YOLO11s) did not exist while the rest of the
pipeline was being built. Every other module only sees `Detection` objects from a `SignDetector`,
so swapping models is a config change (`detect.backend`, `detect.model`) plus a class map.

Implementations:
- `UltralyticsDetector`: runs a YOLO model with ByteTrack tracking frame by frame
  (`model.track(..., persist=True)`), then maps the model's class names to catalogue sign codes
  (and km/h values) with ``config/class_maps/<model>.yaml``.
- `ReplayDetector`: replays detections saved in a JSON file (``scripts/dump_detections.py``), for
  tests and for working on later stages without a GPU.

Classes missing from the class map keep `sign_code=None`; they are ignored by voting (debug only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml

from pipeline.types import Detection

CLASS_MAP_DIR = Path(__file__).resolve().parents[1] / "config" / "class_maps"


class SignDetector(Protocol):
    name: str

    def reset(self) -> None:
        """Forget all tracks (call before every new video)."""

    def detect(self, frame: np.ndarray, t: float, frame_idx: int) -> list[Detection]:
        """Detected and tracked signs in one frame."""


class ClassMap:
    """Model class name -> (sign code, km/h value)."""

    def __init__(self, path: Path):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.model = data["model"]
        self._map = {str(name): (str(v["code"]), v.get("value")) for name, v in data["classes"].items()}

    def lookup(self, model_class: str) -> tuple[str | None, int | None]:
        return self._map.get(model_class, (None, None))

    @property
    def codes(self) -> set[str]:
        return {code for code, _ in self._map.values()}

    @classmethod
    def for_model(cls, model_name: str) -> "ClassMap":
        return cls(CLASS_MAP_DIR / f"{model_name}.yaml")


class UltralyticsDetector:
    """YOLO + ByteTrack, one frame at a time."""

    def __init__(self, name: str, weights: Path, class_map: ClassMap, imgsz: int, conf: float,
                 device: str, half: bool, tracker: str = "bytetrack.yaml"):
        from ultralytics import YOLO

        self.name = name
        self.model = YOLO(str(weights))
        self.class_map = class_map
        self.kwargs = dict(imgsz=imgsz, conf=conf, device=device, **({"quantize": 16} if half and device != "cpu" else {}),
                           tracker=tracker, persist=True, verbose=False)

    def reset(self) -> None:
        # Trackers live on the predictor after the first call; resetting them (and the id counter)
        # starts the next video with fresh tracks and ids from 1.
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()
            if hasattr(tracker, "reset_id"):
                tracker.reset_id()

    def detect(self, frame: np.ndarray, t: float, frame_idx: int) -> list[Detection]:
        result = self.model.track(frame, **self.kwargs)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
        out = []
        for cls_id, conf, xyxy, track_id in zip(boxes.cls.int().tolist(), boxes.conf.tolist(),
                                                boxes.xyxy.tolist(), ids):
            model_class = self.model.names[cls_id]
            code, value = self.class_map.lookup(model_class)
            out.append(Detection(frame_idx=frame_idx, t=round(t, 4), track_id=track_id, model_class=model_class,
                                 sign_code=code, value=value, conf=round(float(conf), 4),
                                 bbox=tuple(round(v, 1) for v in xyxy)))
        return out


class ReplayDetector:
    """Returns detections recorded earlier (JSON written by scripts/dump_detections.py)."""

    def __init__(self, path: Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.name = f"replay:{data.get('model', '?')}"
        self.by_frame: dict[int, list[Detection]] = {}
        for d in data.get("detections", []):
            det = Detection.model_validate(d)
            self.by_frame.setdefault(det.frame_idx, []).append(det)

    def reset(self) -> None:
        pass

    def detect(self, frame: np.ndarray, t: float, frame_idx: int) -> list[Detection]:
        return self.by_frame.get(frame_idx, [])


def build_detector(cfg: dict, models_dir: Path, device: str, replay_file: Path | None = None) -> SignDetector:
    """Create the detector chosen in settings (`detect.backend`, `detect.model`)."""
    det = cfg["detect"]
    if det["backend"] == "replay" or replay_file is not None:
        return ReplayDetector(replay_file or Path(det["replay_file"]))
    name = det["model"]
    model_cfg = cfg["models"][name]
    return UltralyticsDetector(name=name, weights=models_dir / model_cfg["weights"],
                               class_map=ClassMap.for_model(name), imgsz=det["imgsz"], conf=det["conf"],
                               device=device, half=det.get("half", True), tracker=det["tracker"])
