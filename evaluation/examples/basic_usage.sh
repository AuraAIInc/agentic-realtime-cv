#!/usr/bin/env bash
# Basic evaluation runs. Execute from the repo root.
set -euo pipefail

RUN=output/tracking_runs/view3_20261028-1633_1640_replay_db_20261104-110807-376138

# 1) Single run, everything auto-resolved (hour, GT, cost).
#    Report -> output/evaluation_runs/<run>_eval.md
#    (+ <run>_metrics.json and <run>_{original,revised,validated}.png).
.venv/bin/python evaluation/evaluate.py "$RUN"

# 2) Explicit hour + single IoU threshold, scratch output dir.
.venv/bin/python evaluation/evaluate.py "$RUN" \
  --hour image_server/retrieved_images/2026-10-28-16 \
  --gt-source labels --iou 0.5 --out /tmp/eval_out

# 3) Score against SAM labels as GT instead of labels/.
.venv/bin/python evaluation/evaluate.py "$RUN" --gt-source muse --out /tmp/eval_out
