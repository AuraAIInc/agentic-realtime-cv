# demo

Everything [`run_demo.py`](../run_demo.py) needs to run the whole pipeline from this repo
alone, with no camera, MySQL, image server or tracking server.

## The window

Camera view 3, **2026-10-28 16:33:00 – 16:38:59** (`--start 2026-10-28-16-33 --end
2026-10-28-16-38`; the end minute is included). Both cats are in view: Qb (cat 25) and Qbo
(cat 26), two white cats the small on-camera YOLO model often confuses. The ground truth has
988 Qb and 902 Qbo boxes; the detector finds only 686 boxes in total and is weakest on Qbo
(most frames with Qbo have no correct Qbo box). Replayed through the correctors, the window raises a dozen escalations of seven kinds
(T2, T4, T6, P2, P5, P6, P8).

## `data/2026-10-28-16/`

The folder is named after the hour because the evaluator keys ground truth by hour; it holds
only the six minutes above.

| Path | What it is | Count |
|---|---|---|
| `raw_detections.jsonl` | The camera's YOLO output, one camera-format message per frame (empty frames kept): `{"create_time", "view_id", "detections": [{"cat_id", "confidence", "box"}]}` | 989 frames, 686 boxes |
| `images/` | The camera frames, `<idx>_<YYYY-MM-DD-HH-MM-SS-ffffff>.jpg`, 800×600 | 989 |
| `labels/` | Ground truth, YOLO format (`<class> cx cy w h`), hand-validated | 989 |
| `muse_labels/`, `spark_labels/` | Offline label sets from Muse SAM 3.1 and Muse Spark 1.3, scored as reference rows | 988, 911 |
| `classes_map.json`, `classes.txt` | YOLO class index → cat id (`0` → 25 Qb, `1` → 26 Qbo) | |
| `manifest.json` | Window, counts and sources | |

Corrections to the ground truth that were checked by eye are listed in
[`evaluation/gt_audit.yaml`](../evaluation/gt_audit.yaml); the demo scores both the labels
as given and the audited version.

## How `run_demo.py` uses it

1. Symlinks `images/` into the image server's `YYYY-MM-DD/HH/MM/<stamp>.jpg` layout under the
   run's output folder and starts `image_server/mcp_image_server` on a free `127.0.0.1` port
   (local files only, random token).
2. Writes an isolated env file and points every child process at it through
   `AGENT_ENV_FILE`, so the repo-root `.env` (shared MySQL, LAN servers, MySQL invoke dump)
   is never used. Only `MODEL_API_KEY` is taken from your environment or `.env`.
3. Runs `python -m agent.main --replay raw_detections.jsonl --validate --memory` with only the
   images MCP server plus the in-process local tools (`local__get_frame` reads `images/`
   directly), and a fresh memory store under the output folder.
4. Scores the run with `evaluation/evaluate.py --hour demo/data/2026-10-28-16` (as given and
   with `--audit`), then writes `DEMO_REPORT.md`.

## Rebuilding the data (maintainers)

```bash
python image_server/fetch_hour_images.py 2026-10-28-16     # frames + labels into image_server/retrieved_images/
python demo/prepare_demo_data.py                            # raw detections from MySQL (read-only) + copy the window
```

`prepare_demo_data.py` needs the `MYSQL_*` settings in `.env`; `--start`, `--end`,
`--view-id` and `--hour-dir` pick another window (it must stay within one hour). If you
change the window, update `START`/`END`/`DATA` at the top of `run_demo.py`.

The frames come from the project authors' home camera (view 3).
