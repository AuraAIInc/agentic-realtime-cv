# tracking_mcp: MCP tools over the cat tracker

Gives the [MCP agent](../../agentic_realtime_cv_agent/agentic_realtime_cv_agent/README.md) the tracker's results as `tracking__*` tools:
when and where each cat was last seen, its current and past periods with their scores,
where it came from, and when it last moved.

It runs **inside `detection_ingest_server.py`**. That is the only process holding the
TrackCorrector / PeriodCorrector state that isn't in MySQL yet (see
[HOW_IT_WORKS.md](../HOW_IT_WORKS.md)):

| Data | Where it lives | Reaches MySQL |
|---|---|---|
| Raw YOLO rows (conf > 0.5), `images` rows | `detection`, `images` | right away |
| Last ~15 s per cat, base / target | `TrackCorrector.buffer` (memory) | only once committed |
| Committed revised rows (box, cat_id, period_id) | `revised_detection` | when committed |
| Period row | `period` (empty row allocated at period start) | start/end/cat/stats only once the group is coordinated |
| Open, or closed-but-not-coordinated, periods | `PeriodCorrector` (memory) | when every cat has left the screen, or at day end |

Replies mark this: `final=false` on a frame means still in the buffer and can be rewritten;
a period's `status` is `ongoing`, `ended_pending` (cat_id computed now from memory, can still
change) or `final`. Field names avoid tracker internals: `detector_cat_id` is the raw
label (`original_cat_id`), a period's `cat_id` is `coordinated_cat_id`, and
`evidence_by_cat` holds per-cat `detections` / `avg_conf` / `score`
(`detect_length` / `detect_conf` / score).

## Tools

| Tool | Source | What it answers |
|---|---|---|
| `list_views` | DB | views, cat id → name, frame size |
| `get_live_status` | memory | last frame received (all detections, also those under threshold); per cat: on screen, current period, latest detection |
| `get_last_seen` | memory → DB | newest sighting per cat: time, seconds_ago, box, conf, detector vs tracker label, period, image_name |
| `get_current_periods` | memory | ongoing / ended-pending periods with the cat they would get now; recently final periods |
| `get_periods` | DB | final periods with per-cat evidence |
| `get_period_detail` | DB + memory | one period: summary, detected vs interpolated frames, label agreement, first/last/best frame, came_from, path, motion segments |
| `get_arrival_path` | DB + memory | key frames (empty scene → entry → path → settled) to verify where a cat came from |
| `get_last_movement` | DB + memory | last moving segment, and the rest from its end to the last seen frame |
| `get_raw_detections` | DB | raw detector rows, for comparison with the corrected track |

Every frame carries `image_name` (pass it to `images__get_image`) and, for boxes, a
`crop_box` padded 1.5× for zooming.

### Moving vs resting

`motion.py` uses the box-center speed method from `get_active_duration`. Three changes
make it work for single moves rather than daily totals:

- **Window.** Speed is measured over a trailing `window_s` (2 s), not frame to frame. At
  ~3 fps, box jitter alone is ~3 px/s frame to frame, which made a sleeping cat "move" 85%
  of the time. Over 2 s it is ~0.6 px/s.
- **Net displacement.** A moving segment needs at least `min_distance_px` (20 px) between
  where it starts and where it ends. Grooming or the box resizing in place is ignored.
- **Merging.** Pauses of at most `merge_gap_s` are merged, and moves shorter than
  `min_moving_s` are dropped.

All of these are tool arguments, along with `speed_threshold` (3 px/s).

## Run

`detection_ingest_server.py` starts it by default. Set a token so it requires auth:

```bash
# .env on the websocket server host
TRACKING_MCP_AUTH_TOKEN=<random string>
TRACKING_MCP_PORT=10002          # default
# TRACKING_MCP_ENABLED=true      # default; =false never starts, =auto starts only when a token is set
# TRACKING_IMAGE_ROOT=/mnt/extreme/Cat_monitor_data  # only for get_arrival_path include_images=true

python detection_ingest_server.py
```

Agent side (`.env` where the agent runs):

```bash
TRACKING_MCP_URL=http://<websocket host>:10002/mcp
TRACKING_MCP_TOKEN=<same token>
```

If `mcp` / `uvicorn` aren't installed, the websocket server logs a warning and runs without it.

### Concurrency

Tools are `async`. The `mcp` package runs sync tools on a worker thread, which would race
with `process_image`. An async tool reads memory before its first `await`, on the same
event loop as `process_image`, so it never sees a half-processed frame. DB queries run in
`asyncio.to_thread` so they don't hold up frames. No tool calls a corrector method that
writes.

## Check without a camera

```bash
python -m tracking_mcp.test_motion        # pure functions
python tracking_mcp/replay_check.py \
    --view-id 1 --start "2025-12-18 15:03" --end "2025-12-18 15:05"
```

`replay_check.py` replays a window from MySQL through fresh correctors. Their DB writes
are stubbed, and fake period ids start at 10^12. It then calls every tool in-process. DB
reads are real.

## Files

| File | Responsibility |
|---|---|
| `server.py` | tool definitions, bearer auth, `start_in_loop()` |
| `live_state.py` | read-only snapshots of the corrector pools, last frame and finalized periods |
| `history.py` | MySQL reads (view-filtered so the `(view_id, create_time, …)` indexes are used) |
| `motion.py` | moving / resting segments, edges, key-frame picking (pure) |
| `serialize.py` | rows / numpy / datetimes → JSON, image names, time parsing |
| `config.py`, `timing.py` | env settings, per-call timing |
