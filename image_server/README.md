# image_server

Two things live here:

- `fetch_hour_images.py` — download one hour of camera images and export a YOLO dataset with pre-labels (this README).
- `mcp_image_server/` — the image server it downloads from; see [mcp_image_server/README.md](mcp_image_server/README.md) and `run_mcp_server.sh`.

## Prerequisites

- Python >= 3.10. Image download is stdlib-only; labels need `pymysql` and the chart needs `matplotlib`/`seaborn`/`pandas` (all in the repo `.venv`).
- The repo-root `.env` (always that file, whatever the working directory) with `IMAGE_MCP_URL` + `MCP_AUTH_TOKEN` (or `IMAGE_MCP_TOKEN`), and for labels/chart the `MYSQL_*` settings — see `.env.example`.

## Output layout

`python image_server/fetch_hour_images.py 2025-05-05-19` writes to `image_server/retrieved_images/2025-05-05-19/`:

```
2025-05-05-19/
  classes.txt  classes_map.json  data.yaml  detections.png
  images/0000_2025-05-05-19-00-00-123456.jpg ...
  labels/0000_2025-05-05-19-00-00-123456.txt ... (+ classes.txt copy)
```

- The `0000_` index prefix keeps name sort in temporal order (grows to 5 digits past 9999 frames; old files are renamed automatically).
- `labels/` holds one YOLO `.txt` per image that has detections (`<class> cx cy w h`, normalized). Images without detections get no file.
- `classes.txt` line order is the class id; `classes_map.json` pins cat_id → class id so reruns never reorder classes.
- `detections.png` is the hour's detection chart (`plot_detection_details`). It never fails the export — missing deps or an empty hour just print a warning.

## Examples

Fetch one hour into the default folder (labels from `revised_detection` + chart):

```bash
python image_server/fetch_hour_images.py 2025-05-05-19
```

Custom output folder, faster download:

```bash
python image_server/fetch_hour_images.py 2025-05-05-19 --out ~/Downloads/cats-0505-19 --workers 16
```

Pre-labels from raw model output instead of tracker-revised rows, one view only:

```bash
python image_server/fetch_hour_images.py 2025-05-05-19 --labels detection --view-id 3
```

Cleaner pre-labels: drop tracker-interpolated boxes and low-confidence rows:

```bash
python image_server/fetch_hour_images.py 2025-05-05-19 --skip-interpolated --min-conf 0.6
```

Images only, no database access (skips labels and chart):

```bash
python image_server/fetch_hour_images.py 2025-05-05-19 --labels none --no-viz
```

Quick trial of 20 images (same names/indices as the full run, so it resumes cleanly):

```bash
python image_server/fetch_hour_images.py 2025-05-05-19 --limit 20 --out /tmp/yolo_trial
```

Fetch only minutes 33–59 of an hour (same indices as the full run, so it merges cleanly into the hour folder):

```bash
python image_server/fetch_hour_images.py --start 2026-10-28-16-33 --end 2026-10-28-16-59
```

Regenerate labels after changing filters (by default existing label files are kept, so labelImg edits survive reruns):

```bash
python image_server/fetch_hour_images.py 2025-05-05-19 --min-conf 0.6 --overwrite-labels
```

A rerun after an interruption just resumes: downloaded images are skipped and legacy flat files are moved into `images/` with their indexed name.

## Reviewing labels in labelImg

1. `pip install labelImg`, then run `labelImg`.
2. Toggle the `PascalVOC` button to `YOLO` mode.
3. Open `.../<hour>/images/`; when asked for the annotation directory, choose `.../<hour>/labels/`.
4. Correct boxes / classes and save — each `.txt` keeps the YOLO format, and a plain rerun of the fetch script will not clobber your edits.

## Training note

`data.yaml` already points at the dataset (`train: images`, `names: {...}`), so an Ultralytics-style trainer can consume the hour folder directly once the labels are reviewed.
