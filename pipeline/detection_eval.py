"""Detection evaluation helpers shared by training/compare_models.py and eval/failure_gallery.py.

Everything is compared in catalogue-code space: model classes and ground-truth labels are both
translated with the class maps (config/class_maps/), speed limits with their value ("3.24/60"),
so models with different class lists can be scored on the same held-out frames.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from pipeline.detect import ClassMap

DATASET = Path("~/signcoach-data/datasets/dognellaf_yolo").expanduser()


def key(code: str | None, value: int | None) -> str | None:
    if code is None:
        return None
    return f"{code}/{value}" if value is not None else code


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def ground_truth(images: list[Path], names: list[str], cmap: ClassMap) -> dict[str, list]:
    """image -> [(key, box_xyxy_normalised)] for labelled signs that the catalogue knows."""
    gt = {}
    for img in images:
        lbl = DATASET / "labels" / "val" / f"{img.stem}.txt"
        boxes = []
        for line in lbl.read_text().splitlines():
            c, x, y, w, h = line.split()
            k = key(*cmap.lookup(names[int(c)]))
            if k:
                x, y, w, h = map(float, (x, y, w, h))
                boxes.append((k, (x - w / 2, y - h / 2, x + w / 2, y + h / 2)))
        gt[img.name] = boxes
    return gt


def predict(model_name: str, weights: Path, imgsz: int, images: list[Path], device: str) -> tuple[dict, float]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    cmap = ClassMap.for_model(model_name)
    preds, t0, n = {}, None, 0
    for i in range(0, len(images), 16):
        batch = [str(p) for p in images[i:i + 16]]
        if i == 16:
            t0 = time.perf_counter()  # skip the warm-up batch in the timing
        results = model.predict(batch, imgsz=imgsz, conf=0.01, device=device, verbose=False,
                                **({"quantize": 16} if device != "cpu" else {}))
        if t0 is not None:
            n += len(batch)
        for path, r in zip(batch, results):
            out = []
            for c, p, b in zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist(), r.boxes.xyxyn.tolist()):
                k = key(*cmap.lookup(model.names[c]))
                if k:
                    out.append((k, float(p), tuple(b)))
            preds[Path(path).name] = out
    ms = (time.perf_counter() - t0) * 1000 / n if t0 and n else float("nan")
    return preds, ms


def score(preds: dict, gt: dict, keys: set[str], op_conf: float) -> dict[str, dict]:
    """Per sign key: AP50 (all-point interpolation), precision and recall at op_conf."""
    out = {}
    for k in sorted(keys):
        n_gt = sum(1 for boxes in gt.values() for kk, _ in boxes if kk == k)
        if n_gt == 0:
            continue
        dets = sorted(((p, img, b) for img, ps in preds.items() for kk, p, b in ps if kk == k), key=lambda d: -d[0])
        used, tp = defaultdict(set), []
        for p, img, b in dets:
            cands = [(iou(b, g), j) for j, (kk, g) in enumerate(gt.get(img, [])) if kk == k and j not in used[img]]
            best = max(cands, default=(0.0, None))
            if best[0] >= 0.5:
                used[img].add(best[1])
                tp.append(1)
            else:
                tp.append(0)
        tp = np.array(tp)
        ctp = np.cumsum(tp)
        recall = ctp / n_gt
        precision = ctp / np.arange(1, len(tp) + 1) if len(tp) else np.array([])
        ap = 0.0
        if len(tp):
            mrec = np.concatenate([[0.0], recall, [1.0]])
            mpre = np.concatenate([[1.0], precision, [0.0]])
            mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
            idx = np.where(mrec[1:] != mrec[:-1])[0]
            ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
        at_op = [i for i, d in enumerate(dets) if d[0] >= op_conf]
        last = at_op[-1] if at_op else None
        out[k] = {"gt": n_gt, "ap50": round(ap, 3),
                  "precision": round(float(precision[last]), 3) if last is not None else 0.0,
                  "recall": round(float(recall[last]), 3) if last is not None else 0.0}
    return out


