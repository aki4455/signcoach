"""Compare sign detectors on the held-out validation videos (training track T6 / milestone M6).

The two models speak different languages: the placeholder knows 63 European classes, ours 89
Russian ones. To compare them fairly, every prediction AND every ground-truth label is translated
into the same catalogue sign codes with the class maps (config/class_maps/), speed limits including
their km/h value ("3.24/60"). Then, for each sign both models can output:

- AP50: average precision at IoU >= 0.5 (area under the precision-recall curve),
- precision and recall at the pipeline's operating confidence (detect.conf),
plus inference speed (ms per image) on the current device.

The images are the validation split written by training/prepare_dognellaf.py: 3,027 frames from 119
source videos that the fine-tuned model never saw (split by video, not by frame); the placeholder
never saw Russian signs at all. Writes eval/results/model_comparison.{json,md}.

    uv run python training/compare_models.py \
        --model streetsignsense_y12s:1280 \
        --model dognellaf_y11s:1024:~/signcoach-data/train_runs/<run>/weights/best.pt \
        --model dognellaf_y11s:1280:~/signcoach-data/train_runs/<run>/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from app.settings import REPO_DIR, get_settings
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="name:imgsz[:weights]")
    ap.add_argument("--limit", type=int, help="only the first N validation images (quick check)")
    args = ap.parse_args()

    s = get_settings()
    names = [l.strip() for l in (Path("~/signcoach-data/datasets/dognellaf/classes.txt").expanduser()
                                 .read_text(encoding="utf-8").splitlines()) if l.strip()]
    images = sorted((DATASET / "images" / "val").iterdir())[: args.limit]
    gt = ground_truth(images, names, ClassMap.for_model("dognellaf_y11s"))
    gt_keys = {k for boxes in gt.values() for k, _ in boxes}

    runs = {}
    for spec in args.model:
        name, imgsz, *w = spec.split(":", 2)
        weights = Path(w[0]).expanduser() if w else s.models_dir / s.cfg["models"][name]["weights"]
        preds, ms = predict(name, weights, int(imgsz), images, s.device)
        label = f"{name}@{imgsz}"
        model_keys = {key(c, v) for c, v in ClassMap.for_model(name)._map.values()}
        runs[label] = {"weights": str(weights), "ms_per_image": round(ms, 1), "keys": model_keys,
                       "per_sign": score(preds, gt, gt_keys & model_keys, s.cfg["detect"]["conf"])}
        print(f"{label}: {ms:.1f} ms/image, {len(runs[label]['per_sign'])} signs scored")

    common = set.intersection(*(set(r["per_sign"]) for r in runs.values()))
    for r in runs.values():
        vals = [r["per_sign"][k]["ap50"] for k in common]
        r["mean_ap50_common"] = round(float(np.mean(vals)), 3) if vals else None
        r["keys"] = sorted(r["keys"])
    result = {"images": len(images), "source": "validation split (held-out videos)", "common_signs": sorted(common),
              "runs": runs}
    out = REPO_DIR / "eval" / "results"
    (out / "model_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    labels = list(runs)
    md = ["# Sign detector comparison", "", f"Generated by `training/compare_models.py` on {len(images)} frames from the "
          "held-out validation videos (split by video). AP50 = average precision at IoU 0.5; P/R at the pipeline's "
          f"confidence threshold ({s.cfg['detect']['conf']}). Only signs the catalogue uses are scored.", "",
          "| Model | ms/image | mean AP50 (signs both models know) |", "|---|---:|---:|"]
    md += [f"| {l} | {runs[l]['ms_per_image']} | {runs[l]['mean_ap50_common']} |" for l in labels]
    md += ["", "| Sign | GT | " + " | ".join(f"{l} AP50 / P / R" for l in labels) + " |",
           "|---|---:|" + "---:|" * len(labels)]
    for k in sorted({k for r in runs.values() for k in r["per_sign"]}):
        n = next(r["per_sign"][k]["gt"] for r in runs.values() if k in r["per_sign"])
        cells = [(f"{r['per_sign'][k]['ap50']:.2f} / {r['per_sign'][k]['precision']:.2f} / {r['per_sign'][k]['recall']:.2f}"
                  if k in r["per_sign"] else "—") for r in runs.values()]
        md.append(f"| {k} | {n} | " + " | ".join(cells) + " |")
    (out / "model_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out / 'model_comparison.md'}")


if __name__ == "__main__":
    main()
