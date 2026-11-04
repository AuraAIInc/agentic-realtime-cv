# Evaluation: one tracking run vs ground truth (+ cost)

Scores a finished tracking run against YOLO ground-truth labels and
reports the raw → supervisor-validated improvement plus wall time and
LLM spend. Takes a single run directory; the hour resolves automatically.

## Stages

| Stage | Source | Meaning |
| --- | --- | --- |
| muse | `<hour>/muse_labels/` (if present) | SAM 3.1 offline labels vs GT |
| spark | `<hour>/spark_labels/` (if present) | Spark 1.3 offline labels vs GT |
| raw | `stream.jsonl` | Raw detector output (e.g. DB replay) |
| revised | `revised.jsonl` | TrackCorrector output, minus tombstones and interpolated guesses |
| validated | `validated.jsonl` | Revised rows with supervisor verdicts, minus interpolated guesses |

## Ground truth

`<hour>/labels/` by default (`--gt-source {labels|muse|spark}` switches).
YOLO class ids map to cat ids through the hour's `classes_map.json`.
The hour resolves from the run: `--hour` first, then a labels-mode run's
`hour_dir` arg, else the run window's hour stamp under `--data-root`
(default: this repo's `image_server/retrieved_images`).

## Method

- Frames join on timestamp between the run's `images.jsonl` and GT label
  stems (microseconds normalized). Only run frames covered by GT score.
- Per frame, predictions match greedily in descending confidence order to
  the highest-IoU unmatched GT box (IoU >= threshold).
- Metrics: class-aware P/R/F1 (right class + overlap), class-agnostic
  P/R/F1 (localization only), per-class P/R/F1, mean matched IoU, and
  class accuracy on spatially matched pairs. Computed at IoU 0.5 and 0.75.
- Cost: wall time from `run.json` (`elapsed_s`); supervisor $/tokens/
  images/tool calls summed from `validations.jsonl` agent records.
  Offline label sets carry no pipeline cost.

## Usage

```bash
# Single run (absolute path fine; hour + GT auto-resolve)
.venv/bin/python evaluation/evaluate.py \
  output/tracking_runs/<run_name>

# Explicit hour / GT source / thresholds
.venv/bin/python evaluation/evaluate.py <run_dir> \
  --hour image_server/retrieved_images/2026-10-28-16 \
  --gt-source labels --iou 0.5 --out /tmp/eval_out
```

Outputs into `--out` (default: the repo's `output/` dir): `<run>_eval.md`
(headline first, then per-IoU details and cost) plus `<run>_metrics.json`
(machine-readable, same numbers).

```bash
.venv/bin/python -m pytest evaluation/test_metrics.py -q
```

## Files

- `metrics.py` — IoU, greedy frame matching, metric aggregation (no deps).
- `evaluate.py` — CLI: loads one run + GT, joins frames, writes report.
- `test_metrics.py` — unit tests for matching edge cases and CLI helpers.
- `reports/` — local generated reports (the default target is e2e `output/`).
