# camera_v3: always-on detection + MCP control

`camera_main_v2.py` is untouched. v3 does the same thing by default: it reads the
camera, runs YOLO tracking and streams JSON to the Aura server over WebSocket. It
also runs an MCP server, so the backend supervisor (VLM) can inspect frames, zoom,
escalate to bigger models and temporarily change how the loop runs.

```
uv run python -m camera_v3   # from camera_server/ (run_detection_v3.sh runs it in the original venv)
```

## How the pieces fit

```
frame_source  ──newest full-res frame──►  detection_loop  ──JSON──►  ws_sender ──► Aura server
   (camera)                                  │    ▲
                                             │    │ one-shot jobs + overrides
                              frame_buffer ◄─┘    │
                              track_history ◄─┘  controller ◄── camera_service ◄── mcp_server ◄── backend
```

* The **detection loop is the only code that runs YOLO.** Tools that need a model
  queue a job on the controller; the loop runs it between frames (delaying the
  default stream by one inference). Tools that only read pixels (capture, zoom,
  look-back) never touch the loop.
* **Overrides always expire.** `set_schedule(..., duration_seconds=...)` changes fps / roi /
  imgsz / model for at most 24 h, then the loop returns to defaults by itself.
  With no MCP traffic, v3 behaves like v2.
* **Capture thread.** The camera is read continuously and only the newest frame is
  kept, so a slow schedule (1 frame / 30 s) still processes a current frame.
* **Zoom detail.** Set `capture_width` / `capture_height` in view-info (for example
  the sensor's maximum). The default loop still downsizes to `width` × `height`
  exactly as before; `capture_frame(box=...)`, `detect(roi=...)` and roi schedules crop from
  the full-resolution frame. If they're not set, capture size = display size.
* **High-res stills.** `capture_high_res` doesn't change the always-on stream: the
  capture thread closes the camera, reopens it at the requested size, drops a few
  warm-up frames, grabs one and reopens at the normal size. Detection pauses for that
  time (~2 s). Sizes come from `v4l2-ctl --list-formats-ext` (`sudo apt install
  v4l-utils`); unlisted sizes snap to the nearest listed one. RTSP streams can't
  change resolution, so it returns the stream's native frame instead.
* **Always YUYV.** The stream and every still are captured in YUYV (`PIXEL_FORMAT` in
  `frame_source.py`): this camera scales the full sensor at every YUYV size, while MJPG
  centre-crops below 1920x1080. MCP clients never see the format; every size shows the
  same full view. A `box` is still aligned to the still by feature matching (ORB +
  RANSAC, cached per size), falling back to plain scaling with a `warning` if that fails.

| File | Responsibility |
|---|---|
| `config.py` | Device serial, view-info fetch (retries), `Config` dataclass |
| `frame_source.py` | Capture thread, newest-frame handoff, camera reopen on failure |
| `frame_buffer.py` | Ring buffer of recent processed frames (JPEG) for look-back |
| `imaging.py` | Resize / crop / coordinate mapping / encode helpers |
| `models.py` | Model registry from `models.json` (S3 download, lazy load), result → JSON |
| `models.json` | The models detect / set_schedule can use, with descriptions for MCP clients |
| `inference.py` | Run a model on a frame or ROI, map boxes to display coords |
| `controller.py` | Overrides with TTL + one-shot job queue for the loop thread |
| `detection_loop.py` | The always-on loop |
| `track_history.py` | First/last sighting per track ID, identity-switch flag |
| `ws_sender.py` | WebSocket client with queue and reconnect (v2 logic) |
| `image_store.py` | S3 history uploads, local saving |
| `camera_service.py` | Tool implementations, independent of MCP |
| `mcp_server.py` | MCP tool definitions, bearer auth, HTTP server thread |
| `main.py` | Wiring and startup |

## MCP tools

Endpoint `http://<pi>:8765/mcp` (streamable HTTP). All boxes are `[x1, y1, x2, y2]`
in display pixels, the same space as the WebSocket detections. Images are always JPEG.

| Tool | Supervisor job | What it does |
|---|---|---|
| `get_status` | – | mode, settings, override time left, fps, last message, models, buffer, WebSocket |
| `capture_frame` | validate | newest (or buffered, `at=`) stream frame, optional crop around a box, resized to `max_side` |
| `get_camera_modes` | – | sizes `capture_high_res` supports (via `v4l2-ctl`), `max_size` |
| `capture_high_res` | zoom / validate | one full-view still at any supported size (listed in the tool description; unlisted sizes snap to the nearest), optional crop around a box; pauses the loop ~2 s |
| `list_models` | – | every model in the models file: description, `cat_classes`, classes once loaded, extra fields |
| `detect` | escalate | one pass with any registered model, optional ROI / imgsz / conf, on newest or buffered frame |
| `get_track_history` | validate | per-track first/last seen, cat_id counts, `identity_consistent`; `at=` limits it to tracks seen around that time |
| `get_track_frame` | zoom / validate | buffered frame (cropped) where a track first or last appeared |
| `list_buffered_frames` | – | timestamps available for `at=` |
| `set_schedule` | schedule | fps / roi / imgsz / model for `duration_seconds` |
| `reset_schedule` | schedule | back to defaults now |

## Configuration

Environment (`.env`):

```
MCP_AUTH_TOKEN=<long random string>   # strongly recommended; clients send Authorization: Bearer <token>
MCP_PORT=8765                         # optional
MCP_HOST=0.0.0.0                      # optional; 127.0.0.1 = local only
MCP_ENABLED=1                         # 0 turns the MCP server off
MODELS_CONFIG=/path/to/models.json    # optional; default camera_v3/models.json
```

Models come from `models.json`. Each entry is a model name with `weights` (local path
or an Ultralytics name such as `yolov8m.pt`), optional `s3_key` (downloaded from the
model training bucket), `cat_classes`, and `description`. Any other key is passed
through to MCP clients by `list_models`, so new per-model details only need a line in
the file. The `default` model falls back to view-info `model_s3_save_path` when it has
no `s3_key`, as in v2.

Optional view-info keys (all have defaults, so the current API response works as is):

```json
{
  "capture_width": 1920, "capture_height": 1080,
  "frame_buffer_size": 120,
  "frame_buffer_jpeg_quality": 85,
  "local_image_dir": "/home/muyang/Cat_monitor_data",
  "ws_port": 8000,
  "extra_models": {
    "ir": {"s3_key": "models/user123/ir_best.pt", "weights": "./models/ir.pt", "cat_classes": true,
           "description": "Night (IR) model for this user's cats"}
  }
}
```

`extra_models` uses the same keys as `models.json` and is applied on top of it (per
user models from the backend). `cat_classes: true` means the model's class indices
are this user's cat IDs; otherwise detections carry the model's own label and
`cat_id: -1`.

## WebSocket message changes

v2 fields are unchanged. Added: `source` (`"default"` or `"override"`), `settings`
(the override settings, otherwise null) and, per detection, `track_id` and `label`.

## Networking

The backend has to be able to reach the Pi on the MCP port. On a home network
behind NAT, put the Pi and the backend on Tailscale (or use a Cloudflare
Tunnel) and connect to `http://<tailscale-name>:8765/mcp`. Keep `MCP_AUTH_TOKEN`
set even on a private network.

## Install and run

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` +
`uv.lock` at the repo root). uv also installs the Python version in `.python-version`.

```
curl -LsSf https://astral.sh/uv/install.sh | sh   # once per machine (installs to ~/.local/bin)
uv sync                                           # create/update .venv from uv.lock
```

Run from the repo root with `uv run`, which uses the project's `.venv` (no
`source .venv/bin/activate` needed):

```
uv run python -m camera_v3            # v3: detection + MCP server
uv run python camera_main_v2.py       # v2
./run_detection_v3.sh                 # v3 in the background via nohup, logs to detect_log.log
./run_detection.sh                    # v2 in the background
```

**Hackathon-temporary:** on the Pi, `run_detection_v3.sh` runs the code in this
repo folder (`~/agentic-realtime-cv/camera_server`) but with the original server's
venv (`~/camera_server/.venv`, override with `ORIGINAL_VENV`), logging to
`detect_log.log` here. By hand that is:

```
source ~/camera_server/.venv/bin/activate
cd ~/agentic-realtime-cv/camera_server
nohup python -m camera_v3 >> detect_log.log 2>&1 &
tail -f detect_log.log
```

`uv run` syncs `.venv` with the lock before starting, so a fresh `git pull` that
changed dependencies is picked up automatically. The run scripts pass `--frozen`,
which installs exactly what `uv.lock` says and never rewrites the lock on the Pi.

Changing dependencies (on a dev machine, then commit `pyproject.toml` and `uv.lock`):

```
uv add <package>          # add a dependency
uv remove <package>       # remove one
uv lock --upgrade         # upgrade everything within the ranges in pyproject.toml
```
