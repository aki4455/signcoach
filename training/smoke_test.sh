#!/usr/bin/env bash
# T4 smoke test (CLAUDE.md §14.2): YOLO11s, imgsz 1024, 1 epoch on ~5% of the data.
# Pass = no errors, loss going down, mAP > 0. The epoch time is used to size the full run (T5).
# Extra arguments are passed to train.py (e.g. --batch 8).
set -euo pipefail
DATA="${SIGNCOACH_DATA:-$HOME/signcoach-data}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$DATA/train_runs" "$DATA/train_logs"
cd "$DATA/train_runs"  # Ultralytics drops helper downloads (AMP-check weights) into the working directory
uv run --project "$REPO" python -I "$REPO/training/train.py" \
  --data "$DATA/datasets/dognellaf_yolo/data_smoke.yaml" \
  --epochs 1 --close-mosaic 0 --patience 0 --name "smoke_$(date +%H%M)" "$@" \
  2>&1 | tee "$DATA/train_logs/smoke_$(date +%H%M).log"
