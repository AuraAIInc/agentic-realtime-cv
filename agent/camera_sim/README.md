# camera_sim: camera_server simulation on recorded images

Replays a time range of images from `image_server/retrieved_images` through the
**real `camera_v3` pipeline** (model registry, inference, detection loop, track
history, WebSocket sender, MCP server), so the streamed output is identical to
the camera_server's — without the view-info API. All constants live in
[sim_config.yaml](sim_config.yaml).

## Setup (from the repo root)

The sim uses the project's `.venv`:

```bash
source .venv/bin/activate
pip install -r agent/camera_sim/requirements.txt
```

Then fill in the `<...>` identity values in `agent/camera_sim/sim_config.yaml`
(`view_id`, `model_id`, `user_id`, `user_name`, `cat_names`). Until the real
cat model is configured, the loop runs `yolov8s`, which is downloaded into
`agent/camera_sim/models/` on first use.

To use the real cat model (`best.pt`), either set in `sim_config.yaml`:

```yaml
model_s3_save_path: "<s3 key of best.pt>"
model_training_bucket_name: "<training bucket>"
```

(with the `S3_AWS_*` keys in `.env`, as camera_server uses), or place the file
at `agent/camera_sim/models/best.pt`, and set `loop_model: "default"`.

## Run (from `agent/`)

```bash
cd agent
python -m camera_sim --start 2026-10-26-02-00-00 --end 2026-10-26-02-05-00
```

or `./run_sim.sh` (same thing, from `agent/camera_sim/`). Useful flags:

| Flag | Meaning |
|---|---|
| `--start/--end` | hour dir, name prefix, or full image name (overrides config) |
| `--speed 0` | uncap the replay (default 1.0 = realtime pacing from file timestamps) |
| `--limit N` | stop after N frames |
| `--loop` | restart at the first image when the range is exhausted |
| `--model yolov8s` | model the always-on loop runs |
| `--ws-url ws://host:port` | where to stream (default `ws://127.0.0.1:8000`, the ingest server) |
| `--no-ws` | don't open a WebSocket connection |
| `--jsonl out.jsonl` | tee every streamed message to a file |
| `--mcp-port 8766` / `--no-mcp` | MCP control server |
| `--stay` | stay up after the range is exhausted (for MCP inspection) |

## Streamed format

Each message is produced by `camera_v3.DetectionLoop`, exactly as on the Pi:

```json
{
  "detections": [{"cat_id": -1, "box": [x1, y1, x2, y2], "confidence": 0.83, "track_id": 1, "label": "cat"}],
  "is_grey": 0,
  "create_time": "2026-10-26-02-00-01-584724",
  "model_id": ..., "view_id": ..., "device_id": "...",
  "source": "default", "settings": null
}
```

`create_time` is the original capture time parsed from the image file name.

## MCP server

Same tools as camera_server (`get_status`, `capture_frame`, `detect`,
`set_schedule`, ...), plus replay progress in `get_status`:

```bash
curl http://127.0.0.1:8766/mcp  # streamable HTTP, like camera_server's :8765
```

`capture_high_res` returns the replay frame at its native file resolution
(the files have a fixed size, like an RTSP stream).

## Layout

| File | Responsibility |
|---|---|
| `sim_config.yaml` | all constants (identity, replay range, geometry, WS, models, MCP) |
| `models.json` | models `detect` / `set_schedule` can use (same schema as camera_v3) |
| `models/` | downloaded weights live here (git-ignored) |
| `config.py` | YAML -> `SimConfig` (same attribute names camera_v3 reads) |
| `replay_source.py` | recorded images served through the `FrameSource` API |
| `weights.py` | hub-model download into `models/`; lazy S3 for `best.pt` |
| `main.py` | wiring: registry -> replay -> loop -> WS sender + MCP |
