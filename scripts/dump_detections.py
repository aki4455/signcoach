"""Run a sign detector over a video and save its detections as JSON (CLAUDE.md §10.3).

The JSON is what `ReplayDetector` reads, so later stages (voting, relevance, rules, render) can be
developed and tested without re-running the detector or using the GPU. It also prints a short
summary (detections per class, confidence) to compare models on the same clip.

    uv run python scripts/dump_detections.py --video clip.mp4 --model streetsignsense_y12s --out dets.json
    uv run python scripts/dump_detections.py --video clip.mp4 --model dognellaf_y11s \
        --weights ~/signcoach-data/train_runs/<run>/weights/best.pt --imgsz 1024 --out dets.json

Frames are taken from the video as-is at --fps (no normalization), so frame indices match the
pipeline only when the input is the normalized video.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from app.settings import get_settings
from pipeline import frames, ingest
from pipeline.detect import ClassMap, UltralyticsDetector


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--model", required=True, help="model name in settings.yaml / class map name")
    ap.add_argument("--weights", type=Path, help="override the weights file (e.g. a training checkpoint)")
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--fps", type=float)
    ap.add_argument("--device")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    s = get_settings()
    cfg = s.cfg
    det_cfg = cfg["detect"]
    weights = args.weights or s.models_dir / cfg["models"][args.model]["weights"]
    detector = UltralyticsDetector(args.model, weights.expanduser(), ClassMap.for_model(args.model),
                                   args.imgsz or det_cfg["imgsz"], det_cfg["conf"], args.device or s.device,
                                   det_cfg.get("half", True), det_cfg["tracker"])
    info = ingest.probe(args.video)
    fps = min(args.fps or cfg["analysis_fps"], info.fps)
    detections = []
    for idx, t, frame in frames.iter_frames(args.video, info.width, info.height, fps):
        detections += detector.detect(frame, t, idx)
    args.out.write_text(json.dumps({"model": args.model, "weights": str(weights), "fps": fps,
                                    "detections": [d.model_dump(mode="json") for d in detections]}), encoding="utf-8")

    per_class = Counter(d.model_class for d in detections)
    confs = defaultdict(list)
    for d in detections:
        confs[d.model_class].append(d.conf)
    print(f"{len(detections)} detections, {len({d.track_id for d in detections})} tracks -> {args.out}")
    for name, n in per_class.most_common(15):
        code, value = ClassMap.for_model(args.model).lookup(name)
        print(f"  {name:28s} -> {code or '-':7s} {value or '':>4}  n={n:4d}  max conf={max(confs[name]):.2f}")


if __name__ == "__main__":
    main()
