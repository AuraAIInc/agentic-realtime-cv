# Track / Period Corrector Visualization

Replays historical detections from MySQL through `TrackCorrector` + `PeriodCorrector`
and renders one frame per image, plus an MP4.

Read-only: DB writes from the correctors are stubbed out in the script.

For how the two correctors work, see [HOW_IT_WORKS.md](HOW_IT_WORKS.md).

## Run

From this folder, existing_work_before_hackathon/ (needs the usual `.env` with `MYSQL_*`):

```bash
python run_tracking_viz.py \
    --view-id 1 --cat-idx 0 \
    --start "2024-12-05 21:00" --end "2024-12-05 21:59"
```

This is the default test window from `notebooks/tracker_validation_test.ipynb`
(~5.8k frames, ~15 min, ~150 MB MP4). Data from that period has `view_id = NULL`
and is picked up under view 1, whose cats are `0,1,2`.

For a quick check, try a 3-minute window instead:

```bash
python run_tracking_viz.py \
    --view-id 1 --cat-idx 1 \
    --start "2025-12-18 15:03" --end "2025-12-18 15:06"
```

| Flag | Meaning |
| --- | --- |
| `--view-id` | camera view (`view` table) |
| `--cat-idx` | which cat to plot: index into the view's `cat_ids` (the script prints which indices have detections) |
| `--start` / `--end` | window, `YYYY-MM-DD HH:MM`; keep it to a few minutes (~1 frame/image) |
| `--out` | output dir, default `output` |
| `--fps` / `--no-video` | MP4 frame rate / skip the MP4 |

Output: `output/tracking_process/*.jpg` (chronological by filename) and `output/tracking_process.mp4`.

## Reading a frame

Each frame is a ±90 s window around the current image. Top band = x range (`x1..x2 / 640`),
bottom band = y range (`y / 480 + 1`, mostly clipped by `ylim(0, 1)`).

- **Shaded area**: the 15 s buffer still being corrected
- **Tan bars**: buffer contents
- **Cat-colored bars**: revised (committed) detections
- **Blue bars**: upcoming raw detections for this cat (grey = other cats)
- **Green / purple thick lines**: current base / target
