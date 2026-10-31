# Tracking module (`agent.tracking`)

Offline + live cat tracking: frame-level ID correction (`TrackCorrector`),
period-level identity votes (`PeriodCorrector`), JSONL run store, timeline
charts, and per-frame process visualization. Runnable via `agent/main.py`:

```bash
.venv/bin/python -m agent.main --help
```

## Layout

- `track_corrector.py` — frame-level correction; `process_images(...)` yields
  `(is_append, revised)` rows (legacy protocol, MySQL replaced by a `Sink`).
- `period_corrector.py` — period votes; `process_detection(...)` /
  `unprocess_detection(...)` / `finish_up(...)`.
- `sinks.py` — `Sink` protocol + `InMemorySink` (tests / viz).
- `store.py` — `RunStore` (run dir + append-only JSONL logs + `run.json`) and
  `FileSink` (corrector rows -> canonical row-dict JSON; formats documented in
  its module docstring).
- `cat_config.py` — `resolve_cat_ids(view_id)` from `cats.yaml` tracking ids.
- `charts.py` — `plot_detection_charts(...)` writes `original.png` +
  `revised.png` strip charts (one lane per cat, red escalation marks).
- `process_viz.py` — `render_process_frame(...)` per-frame JPGs +
  `stitch_mp4(...)` (needs `cv2`; frames are kept when it is missing).

## Usage examples (run from the repo root)

Live ingest from the camera (stops at `--end`, or Ctrl-C):

```bash
.venv/bin/python -m agent.main --view-id 3 \
  --start 2026-10-27-14-00 --end 2026-10-27-14-05 \
  --live --port 8000
```

Replay a recorded camera stream through the same ingest loop:

```bash
.venv/bin/python -m agent.main --view-id 3 \
  --start 2026-10-27-14-00 --end 2026-10-27-14-05 \
  --replay /tmp/stream.jsonl --plot
```

Replay with per-frame process visualization + MP4 for the first cat:

```bash
.venv/bin/python -m agent.main --view-id 3 \
  --start 2026-10-27-14-00 --end 2026-10-27-14-05 \
  --replay /tmp/stream.jsonl --process-viz --cat-idx 0 --fps 5
```

(Add `--no-video` to keep the `process_frames/*.jpg` without stitching
`process.mp4`, `--no-plot` to skip charts, `--min-conf 0.3` to keep weaker
detections.)

## Notes

- Window: `--start`/`--end` are minute-precision (`YYYY-MM-DD-HH-MM`); a
  message at time `t` is processed iff `start <= t < end + 60s`, i.e. the
  `--end` minute is fully included.
- Camera message format per line: `{"detections": [{"cat_id", "box",
  "confidence", ...}], "is_grey", "create_time" ("%Y-%m-%d-%H-%M-%S-%f"),
  "model_id", "view_id", "device_id"}`. Only detections strictly above
  `--min-conf` with a cat id known for the view are ingested; empty frames
  still advance buffer timeouts. Runs are detections-only (no pixels).
- Live runs append one raw camera message line per received message to the
  run's `stream.jsonl` alongside the canonical `FileSink` rows, so a live
  run's stream stays replayable; `--replay` consumes lines with a
  `"detections"` key and skips the canonical rows (`run.json` `counts.stream`
  counts canonical rows only).
- End of stream keeps legacy semantics: `finish_up` coordinates the collected
  periods but does not flush the tracker's trailing buffer, so detections in
  the last ~15 s of a window may never commit to `revised.jsonl`.
