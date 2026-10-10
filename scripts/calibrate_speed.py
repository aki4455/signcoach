"""Calibrate the video speed estimate on a clip whose true speed is known (CLAUDE.md §15).

Ground truth is a CSV written by a person reading the dashcam's on-screen GPS speed every ~5 s
(``t_seconds,speed_kmh``; see eval/ground_truth/README.md). Use only the CALIBRATION clip here —
never the clips used to report accuracy (eval/eval_speed.py), or the numbers would be flattering.

It fits
- ``speed.scale_k`` = median of true / estimated speed (moments with true speed >= 10 km/h), which
  absorbs the unknown camera height and lens of that camera, and
- ``speed.err_model`` a + b*v from the 68th percentile of the remaining error in speed bands,
  which the rules engine uses to tell "speeding" from "possible speeding".
Results go to eval/results/speed_calibration.json; with --write also into config/settings.yaml.

    uv run python scripts/calibrate_speed.py --video ~/signcoach-data/datasets/own/ru_01_moscow_kremlin_speedOSD.mp4 \
        --truth eval/ground_truth/speed_ru_01.csv --write
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from app.settings import CONFIG_DIR, REPO_DIR, get_settings
from pipeline.motion_eval import motion_for_video, paired, read_truth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--hfov", type=float)
    ap.add_argument("--height", type=float)
    ap.add_argument("--write", action="store_true", help="store scale_k and err_model in config/settings.yaml")
    args = ap.parse_args()

    s = get_settings()
    cfg = s.cfg | {"speed": s.cfg["speed"] | {"scale_k": 1.0}}  # fit on the raw estimate
    motion, _, info = motion_for_video(args.video.expanduser(), cfg, s.models_dir, s.device, args.hfov, args.height)
    pairs = paired(motion, read_truth(args.truth))
    usable = [(v, e) for _, v, e in pairs if v >= 10 and e > 1]
    if len(usable) < 5:
        raise SystemExit(f"only {len(usable)} usable moments — not enough to calibrate")
    k = float(np.median([v / e for v, e in usable]))
    errors = [(v, abs(v - k * e)) for _, v, e in pairs]
    bands = [(lo, hi) for lo, hi in ((0, 20), (20, 40), (40, 60), (60, 200)) if sum(lo <= v < hi for v, _ in errors) >= 3]
    xs = [np.mean([v for v, _ in errors if lo <= v < hi]) for lo, hi in bands]
    ys = [np.percentile([err for v, err in errors if lo <= v < hi], 68) for lo, hi in bands]
    b, a = (np.polyfit(xs, ys, 1) if len(bands) >= 2 else (0.0, float(np.mean(ys))))
    a, b = round(max(1.0, float(a)), 1), round(max(0.0, float(b)), 3)

    result = {"video": args.video.name, "truth": str(args.truth), "moments": len(pairs), "used_for_scale": len(usable),
              "scale_k": round(k, 3), "err_model": {"a": a, "b": b}, "camera": info,
              "bands": [{"band_kmh": list(bd), "p68_error_kmh": round(float(y), 1)} for bd, y in zip(bands, ys)]}
    out = REPO_DIR / "eval" / "results" / "speed_calibration.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

    if args.write:
        settings = CONFIG_DIR / "settings.yaml"
        text = settings.read_text(encoding="utf-8")
        text = re.sub(r"(\n  scale_k: )[0-9.]+", rf"\g<1>{k:.3f}", text, count=1)
        text = re.sub(r"(\n  err_model: )\{[^}]*\}", rf"\g<1>{{a: {a}, b: {b}}}", text, count=1)
        text = re.sub(r"(\n  calibrated: )\w+", r"\g<1>true", text, count=1)
        settings.write_text(text, encoding="utf-8")
        print(f"updated {settings.relative_to(REPO_DIR)}")


if __name__ == "__main__":
    main()
