# Evaluation pipeline: one tracking run vs ground truth (+ cost)

`run_demo.py` at the repo root runs this on its demo run automatically (ground truth from
`demo/data/2026-10-28-16`, with and without the audit).

Scores a finished tracking run — offline label sets plus the
raw → revised → supervisor-validated detection stages — against YOLO
ground-truth labels, and reports the improvement with wall time and LLM
spend. Input is a single tracking-run directory; everything else
(hour, GT, cost records) resolves automatically from run artifacts.

## Stages

| Stage | Source | Meaning |
| --- | --- | --- |
| MUSE SAM 3.1 | `<hour>/muse_labels/` (if present) | MUSE SAM 3.1 offline labels vs GT |
| MUSE SPARK 1.3 | `<hour>/spark_labels/` (if present) | MUSE SPARK 1.3 offline labels vs GT |
| raw | `stream.jsonl` | Raw detector output (e.g. DB replay) |
| revised | `revised.jsonl` | TrackCorrector output, minus tombstones and interpolated guesses |
| validated | `validated.jsonl` | Revised rows with supervisor verdicts, minus interpolated guesses |

Offline label sets carry no pipeline cost; they measure input quality.
Tombstones (`deleted: true`) and tracker interpolations (`orig_cat_id:
null`) never count as predictions. Raw stream rows have no `orig_cat_id`
key, so all of them count.

## Ground truth

`<hour>/labels/` by default. `--gt-source {labels|"MUSE SAM 3.1"|"MUSE SPARK 1.3"}` switches
the label directory. YOLO class ids map to cat ids through the hour's
`classes_map.json` (`order` array); image size is read from the first
`images/*.jpg` (fallback 800x600).

The hour resolves in this order: `--hour`, then a labels-mode run's
`hour_dir` run.json arg (resolved against the run's own repo when
relative), else the run window's `YYYY-MM-DD-HH` stamp under
`--data-root` (default: this repo's
`image_server/retrieved_images` tree).

## Method

- Frames join on timestamp between the run's `images.jsonl` and GT label
  stems (fractional seconds normalized to microseconds). Only run frames
  covered by GT are scored; the report states the coverage.
- Per frame, predictions match greedily in descending confidence order
  to the highest-IoU unmatched GT box (IoU >= threshold).
- Metrics per stage and IoU threshold: class-aware P/R/F1 (right class +
  box overlap), class-agnostic P/R/F1 (localization only), per-class
  P/R/F1, mean matched IoU, and class accuracy on spatially matched
  pairs. Default thresholds: 0.5 and 0.75.
- Deltas: revised − raw, validated − revised, validated − raw.
- Cost: wall time from `run.json` (`elapsed_s`); supervisor dollars,
  tokens, images, tool calls, and per-verdict counts summed from
  `validations.jsonl` agent records. Missing file = zeros, never a crash.

## Usage

```bash
# From the repo root. Positional run dir (absolute path fine).
.venv/bin/python evaluation/evaluate.py \
  output/tracking_runs/<run_name>

# Explicit hour, GT source, thresholds, output dir
.venv/bin/python evaluation/evaluate.py <run_dir> \
  --hour image_server/retrieved_images/2026-10-28-16 \
  --gt-source labels --iou 0.5 --out /tmp/eval_out

# Full option list
.venv/bin/python evaluation/evaluate.py --help
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `run_dir` | (required) | Tracking run directory |
| `--hour` | auto | `retrieved_images/<hour>` dir with GT labels |
| `--data-root` | `<repo>/image_server/retrieved_images` | Hour lookup tree for replay/live runs |
| `--gt-source` | `labels` | `labels` \| `"MUSE SAM 3.1"` \| `"MUSE SPARK 1.3"` |
| `--iou` | 0.5, 0.75 | Repeatable IoU thresholds |
| `--out` | `<repo>/output/evaluation_runs/` | Report directory |

Outputs into `--out`: `<run>_eval.md` (headline first, then detection
charts, per-IoU details, and cost) plus `<run>_metrics.json`
(machine-readable, same numbers). The run's `original.png` /
`revised.png` / `validated.png` strip charts are copied alongside as
`<run>_<chart>.png` when present, and `<run>_gt.png` is rendered from
the ground-truth labels over the evaluated frames (same strip style,
run escalations overlaid) — all embedded in the report's
`Detection charts` section. The script also prints the report to stdout.

### Programmatic use

```python
import sys
sys.path.insert(0, "evaluation")
from evaluate import evaluate_run, render_report

res = evaluate_run(
    "output/tracking_runs/<run_name>",
    "image_server/retrieved_images/2026-10-28-16",
    gt_source="labels",
    iou_thrs=[0.5],
)
print(render_report(res, [0.5]))
print("validated F1:", res["stages"]["validated"]["metrics"]["iou_0.5"]["f1"])
print("spend: $", res["cost"]["est_cost_usd"])
```

See [examples/](examples/) for runnable versions (shell + Python).

### Documented GT corrections (`--audit`)

`--audit evaluation/gt_audit.yaml` applies per-hour corrections to the
ground truth in memory (label files are never modified): a validated
time window, dropping boxes on known decoy zones (`zones.yaml`
`kind: decoy`), and relabel spans where the frames show the other cat.
Each entry is justified in the yaml and in
`documents/e2e-identity-fix-20261104.md`. Report both the as-given and
audited numbers.

## Reading the report

- **Headline**: the one-line story — class-aware F1 and identity
  accuracy (fraction of spatially matched pairs with the right class)
  from raw detector to supervisor-validated output, plus wall time and
  supervisor spend.
- **Identity repair surfaces as recall/precision gain**: relabeling a
  wrong-class box to the right class converts an FP+FN pair into a TP,
  so a pure relabeling verdict lifts class-aware R/P without adding
  boxes. Localization (loc) columns stay flat in that case — that is
  how you tell relabeling apart from new detections.
- **A revised dip is normal**: the tracker associates by continuity and
  can drag one label across a visit (identity drops raw → revised);
  the supervisor exists to repair it (identity rises revised →
  validated). Judge the pipeline by the raw → validated row.
- **MUSE SAM 3.1 rows with mean IoU 1.000** mean that hour's `muse_labels`
  reuse the detector boxes verbatim and only re-classify/filter; its
  F1 then measures classification + coverage, not localization.

## Troubleshooting

- `hour dir not found`: the window-derived hour is missing under
  `--data-root` — pass `--hour` explicitly.
- `KeyError`/empty report on `run.json`: the run dir must contain
  `run.json` + `images.jsonl`; stage files may be absent (scored as —).
- `0 evaluated frames`: no timestamp overlap between run frames and GT
  (wrong hour or GT source) — check the `Frames:` line first.
- Missing `validations.jsonl` (runs without `--validate`): cost block
  reports zeros; accuracy stages still score.

```bash
.venv/bin/python -m pytest evaluation/test_metrics.py -q
```

## Files

- `metrics.py` — IoU, greedy frame matching, metric aggregation (no deps).
- `evaluate.py` — CLI: loads one run + GT, joins frames, writes report.
- `test_metrics.py` — unit tests for matching edge cases and CLI helpers.
- `examples/` — runnable shell + programmatic examples.
- `reports/` — historical local reports (the default target is `<repo>/output/evaluation_runs/`).
