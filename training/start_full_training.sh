#!/usr/bin/env bash
# T5 full training run (CLAUDE.md §14.2), started detached in tmux so it survives this terminal
# and the Claude Code session. Watch it with:  tmux attach -t signcoach-train   (detach: Ctrl-b d)
# Progress per epoch:  $SIGNCOACH_DATA/train_runs/<name>/results.csv
#
# Epoch count: the smoke test measured ~9.4 min of training + ~0.9 min of subset validation per
# epoch on the RTX 3070 Ti, so 40 epochs end around 6-7 hours after the start.
# Usage: training/start_full_training.sh [EPOCHS] [extra train.py args...]
set -euo pipefail
DATA="${SIGNCOACH_DATA:-$HOME/signcoach-data}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
EPOCHS="${1:-40}"; shift || true
NAME="dognellaf_y11s_1024_$(date +%m%d_%H%M)"
LOG="$DATA/train_logs/$NAME.log"
mkdir -p "$DATA/train_runs" "$DATA/train_logs"
if tmux has-session -t signcoach-train 2>/dev/null; then
  echo "A training session is already running (tmux attach -t signcoach-train)." >&2
  exit 1
fi
tmux new-session -d -s signcoach-train -c "$DATA/train_runs" \
  "uv run --project '$REPO' python -I '$REPO/training/train.py' \
     --data '$DATA/datasets/dognellaf_yolo/data_train.yaml' --epochs $EPOCHS --name '$NAME' $* \
     2>&1 | tee '$LOG'; echo 'training finished'; sleep infinity"
echo "started: run=$DATA/train_runs/$NAME  log=$LOG"
