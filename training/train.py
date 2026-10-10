"""Train the road-sign detector on the prepared Dognellaf dataset (training track T4 and T5).

Settings follow CLAUDE.md §14.2:
- YOLO11s starting from COCO weights, image size 1024 (signs are small; 640 loses distant ones).
- fliplr=0.0 — hard rule 7: mirroring turns "no right turn" into "no left turn" and mirrors digits.
- degrees=0.0 (signs are upright), AMP, cosine learning rate, mosaic switched off for the last
  epochs, and a checkpoint every epoch (save_period=1), so the best weights so far (best.pt)
  are usable at any moment if training has to be stopped before the deadline.

T4 smoke test: training/smoke_test.sh (1 epoch on ~5% of the data).
T5 full run:   training/start_full_training.sh (detached in tmux, logs in $SIGNCOACH_DATA/train_logs).
"""

from __future__ import annotations

import argparse
from pathlib import Path

DATA = Path("~/signcoach-data").expanduser()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="data.yaml written by training/prepare_dognellaf.py")
    ap.add_argument("--model", default=str(DATA / "models" / "yolo11s.pt"), help="starting weights")
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--close-mosaic", type=int, default=3, help="last N epochs without mosaic")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--project", default=str(DATA / "train_runs"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO, settings

    settings.update({"sync": False})  # no usage analytics sent to Ultralytics
    YOLO(args.model).train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers,
        amp=True, cos_lr=True, close_mosaic=args.close_mosaic,
        fliplr=0.0, degrees=0.0,  # never mirror road signs (CLAUDE.md hard rule 7)
        save_period=1, patience=args.patience, project=args.project, name=args.name, exist_ok=False,
        device=args.device, deterministic=False, seed=0, plots=True,
    )


if __name__ == "__main__":
    main()
