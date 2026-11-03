# SAM grounding truth

Label images under `image_server/retrieved_images/<hour>/` with Muse SAM 3.1
to build grounding-truth detections for training and evaluation.

## How it works

Two stages per image (this division was verified by live probing: SAM grounds
short concepts reliably but returns nothing for descriptive prose, so prose
belongs to the VLM, not to SAM):

1. **Propose (SAM 3.1).** One request per grounding concept (`animal`,
   `a cat`; bare `cat`/`cats`/`kitten` do not ground). The union is
   deduplicated at IoU 0.5. SAM has no boxes-only mode — every record
   pairs a box with a mask — and billing is per image either way, so we
   request the compact `one_bit` mask encoding and keep only the boxes.
2. **Classify (Muse Spark).** One call per image that has proposed boxes.
   The prompt carries both cats' visual descriptions from
   `agent/agent_data/config/cats.yaml` plus the existing `labels/` YOLO
   boxes as **reference only** (marked as possibly wrong; the image is the
   evidence). Each box gets a class id, or -1 when it shows no cat or the
   model is unsure — rejected boxes are never written.

Accepted boxes are written as YOLO lines in `muse_labels/` next to
`images/` (same stem, same class ids as `labels/`). An empty `.txt` means
"processed, nothing survived" (no proposals, or all rejected).
- Requests run on a thread pool (`--workers`, default 10).

## Prerequisites

1. Retrieved hour folders exist under `image_server/retrieved_images/`
   (see `image_server/fetch_hour_images.py` to download them).
2. `MODEL_API_KEY` is set in the repo-root `.env` (not needed for
   `--dry-run`).
3. Python >= 3.10 with `pyyaml` installed. No other third-party packages
   are required (the script calls the Meta Model API over stdlib HTTPS).

All commands below run from the repo root.

## Example commands

### 1. Preview a range for free (always start here)

Resolves the files, prints the **full prompts of the first requests**
(SAM concepts plus an illustrated Spark prompt), shows the planned
request count and cost estimate, and writes a session report —
without calling the API or writing any labels:

```bash
python image_server/grounding_truth/label_with_sam.py \
  --start 2026-10-28-16-33 --end 2026-10-28-16-40 --dry-run
```

Expected output (abridged):

```text
1319 images in range 2026-10-28-16-33 .. 2026-10-28-16-41
========================================================================
FIRST SAM PROMPT (2026-10-28-16/5502_2026-10-28-16-33-00-304643.jpg (query: Qb, class 0)):
========================================================================
Segment every instance of: Qb: Short-haired white cat with ...
Appearance: A compact, cobby short-haired cat ...
Look for: Cool white to light silver colour, ...; ...; ...
This query is for one identity only: Qb (label id 0). ...
Reference detections from an older detector: none - it found nothing ...
Image size: 800x600 pixels.
========================================================================
DRY RUN: 1319 images, 2638 SAM requests planned (~$6.5950) plus one Spark
call per image with proposed boxes (~$0.005 each ...) ...
Done: 0 images processed, ... estimated cost $0.0000.
Session report: .../grounding_truth/reports/session-....md (+ .json)
```

### 2. Label a few images for real (smoke test, ~$0.03)

```bash
python image_server/grounding_truth/label_with_sam.py \
  --start 2026-10-28-16-33 --end 2026-10-28-16-33 --limit 5
```

This labels the first 5 images of minute 16:33 (10 SAM requests + up to 5
Spark calls). Check the results before scaling up:

```bash
ls image_server/retrieved_images/2026-10-28-16/muse_labels/ | head
cat image_server/retrieved_images/2026-10-28-16/muse_labels/5502_*.txt
```

### 3. Label a minute range (the demo window)

```bash
python image_server/grounding_truth/label_with_sam.py \
  --start 2026-10-28-16-33 --end 2026-10-28-16-40
```

Labels all 1,319 images in 16:33–16:40 (end minute fully included):
~2,638 SAM requests plus up to 1,319 Spark calls, about $13 total.
Progress is logged every 25 images.

### 4. Label a whole hour

```bash
python image_server/grounding_truth/label_with_sam.py 2026-10-28-16
```

Equivalent to `--start 2026-10-28-16-00 --end 2026-10-28-16-59`.
A full hour (~4,472 images) costs roughly $45 — run the `--dry-run`
first to see the exact SAM figure (Spark cost depends on how many
images get proposals).

### 5. Re-run / resume

Already-labeled images are skipped automatically, so an interrupted run can
simply be restarted with the same command. To force relabeling:

```bash
python image_server/grounding_truth/label_with_sam.py \
  --start 2026-10-28-16-33 --end 2026-10-28-16-40 --overwrite
```

### 6. Tune parallelism

```bash
python image_server/grounding_truth/label_with_sam.py \
  --start 2026-10-28-16-33 --end 2026-10-28-16-40 --workers 20
```

Default is 10. SAM 3.1 allows 3,000 requests/minute, so 10–20 workers are
comfortably within limits on a normal connection.

### 7. View the results in labelImg

The script copies `classes.txt` into each `muse_labels/` folder it writes,
so the output opens directly in labelImg
([install](https://pypi.org/project/labelImg/)):

1. Start labelImg and switch to YOLO mode (click the "PascalVOC" button
   until it reads "YOLO").
2. Open `retrieved_images/2026-10-28-16/images/` as the image directory.
3. When asked for the annotation directory, point it at
   `retrieved_images/2026-10-28-16/muse_labels/`.

To compare against the old detector, point the annotation directory at
`labels/` instead.

## All options

| Flag | Default | Meaning |
| --- | --- | --- |
| `hour` (positional) | — | Whole hour `YYYY-MM-DD-HH` (omit when `--start`/`--end` are given) |
| `--start` / `--end` | hour edges | Minute bounds `YYYY-MM-DD-HH-MM`; end minute fully included; may span hours |
| `--workers` | 10 | Parallel SAM requests |
| `--limit N` | all | Label only the first N images in range (debug / smoke test) |
| `--dry-run` | off | No API calls, no labels; prints first prompt + cost estimate, still writes a report |
| `--overwrite` | off | Relabel images that already have a `muse_labels/` file (default skips them) |
| `--retrieved` | `image_server/retrieved_images` | Retrieved-images root |
| `--cats` | `agent/agent_data/config/cats.yaml` | Cat profiles supplying the per-class descriptions |
| `--models` | `agent/supervisor_agent/models.json` | Supplies the SAM model name and API base URL |
| `--report-dir` | `grounding_truth/reports` | Where session reports are written |

## Output layout

```text
retrieved_images/2026-10-28-16/
  images/          # input frames (untouched)
  labels/          # old detector YOLO labels (untouched; reference only)
  muse_labels/     # NEW: SAM grounding truth, same stems + class ids
    5502_....txt   # one YOLO line per box: <class_id> xc yc w h
    ...
    classes.txt    # copy of the hour's classes (for labelImg)

grounding_truth/reports/
  session-YYYYMMDD-HHMMSS.md    # totals, cost, queries, first prompt, per-hour table, failures
  session-YYYYMMDD-HHMMSS.json  # same data + per-image records (machine-readable)
```

## Cost

SAM 3.1 is billed per image, not per token: **$2.50 per 1,000 images**;
Muse Spark is billed per token: **$1.25/$4.25 per 1M input/output tokens**
(Standard mode, verified 2026-11-03 at
<https://dev.meta.ai/docs/pricing-rate-limits>).
A typical image with cats costs 2 SAM requests ($0.005) + 1 Spark call
(~$0.005, ~800 in / ~1,000 out tokens):

| Job | Images | SAM + Spark calls | Approx. cost |
| --- | ---: | ---: | ---: |
| Smoke test (`--limit 5`) | 5 | 10 + ≤5 | ~$0.05 |
| One minute (~165 images) | ~165 | ~330 + ≤165 | ~$1.65 |
| Demo window (16:33–16:40) | 1,319 | 2,638 + ≤1,319 | ~$13 |
| Full hour | ~4,472 | ~8,944 + ≤4,472 | ~$45 |

Every session report records the exact call counts and estimated cost.

## Tests

```bash
python -m pytest image_server/grounding_truth/test_label_with_sam.py -q
```

Fully offline: SAM output parsing (against the documented grammar), prompt
construction, time-range discovery, cost math, and report rendering.
