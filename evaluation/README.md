# Evaluation: raw vs revised vs validated detection accuracy

Compares the three detection stages of a tracking run against YOLO
ground-truth labels, frame by frame:

| Stage | Source file | Meaning |
| --- | --- | --- |
| raw | `stream.jsonl` | Raw detector output replayed from the DB |
| revised | `revised.jsonl` | TrackCorrector output: relabeled, interpolated, grouped into periods |
| validated | `validated.jsonl` | Revised rows with LLM validation verdicts applied |

## Ground truth

`image_server/retrieved_images/<hour>/` holds three label sets.
`labels/` is the raw detector export itself (marked "possibly wrong" in
`image_server/grounding_truth/README.md`), so scoring raw against it would
be circular. The grounding-truth labelers exist precisely for evaluation:

- `muse_labels/` (default): SAM 3.1 region proposals classified by Muse
  Spark — highest quality, full coverage of `2026-10-28-16`.
- `spark_labels/`: single-pass Muse Spark labels (cheaper, partial coverage).

Use `--gt-source {muse|spark|labels}` to switch. YOLO class ids map to cat
ids through the hour's `classes_map.json` (`order` array).

## Method

- Frames are joined on `create_time` between the run's `images.jsonl` and
  the GT label stems. Only frames with GT are scored; the report states the
  coverage.
- Per frame, predictions are matched greedily in descending confidence
  order to the highest-IoU unmatched GT box (IoU >= threshold).
- Metrics: class-aware P/R/F1 (right class + overlap), class-agnostic
  P/R/F1 (localization only), per-class P/R/F1, mean matched IoU, and
  class accuracy on spatially matched pairs. Computed at IoU 0.5 and 0.75.

## Usage

```bash
# Single run (from repo root)
.venv/bin/python evaluation/evaluate.py \
  --run output/tracking_runs/<run_name> \
  --hour image_server/retrieved_images/2026-10-28-16

# Multiple runs add a cross-run validated-stage comparison
.venv/bin/python evaluation/evaluate.py --run <run1> --run <run2> \
  --hour image_server/retrieved_images/2026-10-28-16 \
  --gt-source muse --iou 0.5 --iou 0.75 --out evaluation/reports
```

Outputs: `evaluation/reports/ACCURACY_REPORT.md` plus one
`<run>_metrics.json` per run (machine-readable, same numbers).

```bash
.venv/bin/python -m pytest evaluation/test_metrics.py -q
```

## Files

- `metrics.py` — IoU, greedy frame matching, metric aggregation (no deps).
- `evaluate.py` — CLI: loads runs + GT, joins frames, writes report.
- `test_metrics.py` — unit tests for matching edge cases.
- `reports/` — generated accuracy report + per-run metrics JSON.
