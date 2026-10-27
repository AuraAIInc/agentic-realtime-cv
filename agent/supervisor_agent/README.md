# supervisor_agent: Muse Spark agent over the RCAT MCP tools

An LLM agent that answers questions about the cats by calling tools on three MCP servers:

| Prefix | Server | Repo | Tools |
|---|---|---|---|
| `camera__` | live camera on the Pi | `camera_server` (`python -m camera_v3`) | get_status, capture_frame, capture_roi, get_camera_modes, capture_high_res, detect, get_track_history, get_track_frame, list_buffered_frames, set_schedule, reset_schedule |
| `images__` | stored images on the SSD | `image-server` (`python -m mcp_image_server`) | image_exists, images_exist, get_image, list_images |
| `tracking__` | tracker / period results, live and saved | this repo, inside `existing_work_before_hackathon/detection_ingest_server.py` ([tracking_mcp](../../existing_work_before_hackathon/tracking_mcp/README.md)) | list_views, get_live_status, get_last_seen, get_current_periods, get_periods, get_period_detail, get_arrival_path, get_last_movement, get_raw_detections |

The model (Muse Spark `muse-spark-1.3` on the Meta Model API, via its Responses endpoint) picks the tools a query needs. The
hub runs them (several at once when the model asks for several), and the results go back to
the model. Images go back as real images, so the model can look at what it fetched. This
repeats until the model answers.

```
query ─► Muse Spark ─► function calls ─► MCPHub ─► camera / image MCP servers
               ▲                                         │
               └──── text + images + timing ◄────────────┘
```

## Setup

From the agent/ folder (Python ≥ 3.10; the repo venv is 3.12):

```bash
pip install -r requirements.txt
```

Add to `.env` (the full list for every component is in the repo's [`.env.example`](../../.env.example)):

```bash
MODEL_API_KEY=<Meta Model API key>
MCP_AUTH_TOKEN=<the single shared value for all three MCP servers>
CAMERA_MCP_URL=http://<pi>:8765/mcp
IMAGE_MCP_URL=http://10.0.0.63:10001/mcp
TRACKING_MCP_URL=http://<websocket server host>:10002/mcp
```

If a server is down, the agent warns and runs with the others.

Models are set per role in [models.json](models.json):

| Role | Model | Use |
|---|---|---|
| `agent` | `muse-spark-1.3` | Agent reasoning, vision, tool calls; escalation teacher (the only role this agent calls today) |
| `segmentation` | `sam-3.1` | Masks/boxes for training data |
| `voice` | `muse-voice-transcribe-1.0` | Voice questions (if we build voice) |
| `creative` | `muse-image-1.0` + `muse-wonder-1.0` | Creative output, e.g. a daily "cat diary" comic |
| `open_weight` | `meta-models/muse-spark-open-1026` (Baseten) | Local or partner-hosted model; Muse Glimmer to self-host |
| `coding` | Muse Code | Coding (a product, not an API model) |

The file also holds the provider `base_url` / `api_key_env`, `reasoning_effort`, and the
per-1M-token prices used for the cost estimate. `muse-spark-1.3-contributor` is the cheap option.

## Run

```bash
# one query
python -m supervisor_agent "Is any cat on camera right now? Show me."

# interactive (follow-up questions keep context)
python -m supervisor_agent

# what the agent can call
python -m supervisor_agent --list-tools
```

Example queries:

- "Is a cat visible right now? Zoom in on it and tell me which cat it is."
- "Check whether 2026-10-24-20-15-03-123456.jpg exists, and how big it is."
- "List the images from 2026-10-24 20:15 and show me the first one."
- "Track 3's identity looks inconsistent. Show me where it first appeared and re-check it with yolov8s."
- "The cat is asleep; drop the camera to one frame every 30 s for an hour."
- "When was each cat last seen? Show me the frame."
- "How long has the current period lasted, and what is its score?"
- "Where did Qbo come from? Show me."
- "When did Yuki last move, and how long has it been resting since?"

Live output shows every tool call with its timing:

```
  → images__image_exists({"image_name": "2026-10-24-20-15-02-002000.jpg"})
  ← images__image_exists: ok (round trip 10.56 ms, server 3.599 ms)
  → images__get_image({"image_name": "...", "max_side": 256})
  ← images__get_image: ok (round trip 168.94 ms, server 163.156 ms, 1 image(s) saved)

Yes: Mochi is on camera, and the stored image exists.

[muse-spark-1.3 | turns 3 | tool calls 5 | LLM 0.214 s | tools 0.181 s | total 0.395 s | tokens in 300 / out 60 / total 360 | est. cost $0.0012]
```

- **round trip**: measured by the agent (network + server).
- **server**: the `timing.elapsed_ms` the image server reports in each response.

Each run writes `output/agent_runs/<timestamp>/run.json` (query, answer, every tool call
with arguments and timings, token usage) plus every image the tools returned. `output/` is
gitignored. Use `--no-save` to skip this, or `--json` to also print the trace.

### Invoke dump to MySQL

With `AGENT_INVOKE_DUMP=true` in `.env`, every `MCPAgent.run` (chat query or pipeline task) is
also written to the MySQL database in the `MYSQL_*` settings (the ingest server's database;
needs `pymysql`). The two tables are created on first use, and columns added later are added to
existing tables:

| Table | One row per | Columns |
|---|---|---|
| `agent_invoke` | agent run | `invoke_id`, `invoke_time`, `model`, `tag` (task name), `input_query`, `agent_response`, `input_tokens` / `output_tokens` / `cached_tokens` / `total_tokens`, `est_cost_usd` (total, from the token counts and the prices below), `price_input_per_1m` / `price_cached_per_1m` / `price_output_per_1m` (USD per 1M tokens from [models.json](models.json); NULL when the model has no pricing entry), `images_sent` (images in the query + tool results sent to the LLM), `tool_count`, `tool_names` (JSON list), `tools_ms`, `llm_ms`, `total_ms` (input to answer), `turns`, `stopped_reason`, `reasoning_effort`, `previous_response_id`, `last_response_id`, `run_dir` |
| `tool_invoke` | tool call | `invoke_id`, `agent_invoke_id` (→ `agent_invoke.invoke_id`), `invoke_time`, `turn`, `call_id`, `tool_name`, `tool_input` (arguments JSON), `tool_output` (text; images replaced by a size placeholder), `images_returned`, `response_ms` (round trip), `server_elapsed_ms`, `server_steps_ms`, `is_error`, `error` |

The agent only puts a snapshot on a queue at the end of a run; a background thread builds the
rows and inserts them, so the dump adds no time to a response. If the database is unreachable,
the record is dropped with a warning and the agent carries on. Runs that raise (e.g. the model
API fails) are not recorded.

## Options

| Flag | Meaning |
|---|---|
| `--model-config file.json` | use another model/pricing file instead of `models.json` |
| `--model` | Meta Model API model id (overrides `roles.agent.model` in `models.json`) |
| `--servers file.json` | use `[{"name", "url", "token"}]` instead of the env vars |
| `--max-steps N` | max model turns per query (default 12) |
| `--image-detail low/high/auto` | how much detail images are sent with (`low` is cheaper) |
| `-v` | log connections |

Set `"reasoning_effort": "low" | "medium" | "high"` in `models.json` for reasoning models.

## Tools and skills

The agent sees 38 tools. `camera__*`, `images__*` and `tracking__*` live on the MCP servers: names, descriptions and schemas below are exactly what those servers reported on 2026-10-27 (they can change server-side; `python -m supervisor_agent --list-tools` shows the current list). `local__*` run in-process and are defined in [`local_tools/server.py`](local_tools/server.py); `skill__*` load a skill's instructions when invoked (see [`local_tools/skills.py`](local_tools/skills.py) and [`skills/`](skills/)).

### Camera (`camera__*`, 11 tools)

Live camera on the Pi (`camera_server`).

#### `camera__get_status`

Current mode (default/override), effective settings and time left on any override, loop and camera FPS, last detection message, available models, cat names, frame buffer span, and WebSocket state.

<details>
<summary>Input schema</summary>

```json
{
  "properties": {},
  "type": "object",
  "title": "get_statusArguments"
}
```
</details>

#### `camera__capture_frame`

Grab the newest frame of the always-on stream (or the buffered frame nearest `at`). box crops it (display coords, padded by `padding` x box size); for a sharper zoom use capture_high_res with the same box. The image is shrunk so its longest side is <= max_side (null = no resize). Instant; does not interrupt the detection loop.

<details>
<summary>Input schema</summary>

```json
{
  "properties": {
    "box": {
      "anyOf": [
        {
          "items": {
            "type": "integer"
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Box"
    },
    "padding": {
      "default": 0.15,
      "title": "Padding",
      "type": "number"
    },
    "max_side": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": 1280,
      "title": "Max Side"
    },
    "at": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "At"
    }
  },
  "type": "object",
  "title": "capture_frameArguments"
}
```
</details>

#### `camera__get_camera_modes`

Resolutions capture_high_res supports (sizes, largest first; max_size = largest) and the current always-on capture_size.

<details>
<summary>Input schema</summary>

```json
{
  "properties": {},
  "type": "object",
  "title": "get_camera_modesArguments"
}
```
</details>

#### `camera__capture_high_res`

Take a still at a different resolution than the always-on stream. Supported sizes: 1920x1080, 1280x1024, 1280x720, 1024x768, 800x600, 640x480, 320x240. width/height omitted = the largest size. A size the camera doesn't list, or only width / only height, snaps to the nearest supported size (see used_size / size_choice in the reply). Every size shows the full camera view, like the always-on stream; only resolution (and aspect ratio) differ. A box (display coords, padded by `padding` x box size) is cropped from the still for zoom detail. Use this rather than capture_frame to zoom: the crop comes from the larger still, so it has more detail. max_side shrinks only the returned image (null = full resolution). Pauses the detection loop ~3 s while the camera switches resolution and back.

<details>
<summary>Input schema</summary>

```json
{
  "properties": {
    "width": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Width"
    },
    "height": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Height"
    },
    "box": {
      "anyOf": [
        {
          "items": {
            "type": "integer"
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Box"
    },
    "padding": {
      "default": 0.0,
      "title": "Padding",
      "type": "number"
    },
    "max_side": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Max Side"
    }
  },
  "type": "object",
  "title": "capture_high_resArguments"
}
```
</details>

#### `camera__list_models`

Every model detect / set_schedule can use, from the camera's models file: description, cat_classes (true = classes are this user's cats, detections carry cat_id), whether it is loaded, its class names once loaded, and any other details the file gives.

<details>
<summary>Input schema</summary>

```json
{
  "properties": {},
  "type": "object",
  "title": "list_modelsArguments"
}
```
</details>

#### `camera__detect`

Escalate: run one detection pass with `model` (one of: default, yolov8s; list_models describes them) on the newest frame or the buffered frame nearest `at`, optionally only inside `roi` (full-res crop) and at input size `imgsz` [w, h]. Does not touch the tracker. Delays the default loop by one inference.

<details>
<summary>Input schema</summary>

```json
{
  "properties": {
    "model": {
      "default": "default",
      "title": "Model",
      "type": "string"
    },
    "at": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "At"
    },
    "roi": {
      "anyOf": [
        {
          "items": {
            "type": "integer"
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Roi"
    },
    "imgsz": {
      "anyOf": [
        {
          "items": {
            "type": "integer"
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Imgsz"
    },
    "conf": {
      "anyOf": [
        {
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Conf"
    }
  },
  "type": "object",
  "title": "detectArguments"
}
```
</details>

#### `camera__get_track_history`

Tracks seen by the default loop: first/last seen time and box, hits, max confidence, and cat_id counts. identity_consistent=false means the track changed cat_id at least once (a likely ID switch worth validating). Track IDs restart (new epoch) whenever the loop's model/roi/imgsz changes. `at` returns only tracks seen within window_seconds of that time (from every epoch). Short-term only: kept in the camera's memory (last 500 tracks, lost on restart). For older or complete history use the backend, which records every track from the WebSocket stream.

<details>
<summary>Input schema</summary>

```json
{
  "properties": {
    "limit": {
      "default": 50,
      "title": "Limit",
      "type": "integer"
    },
    "all_epochs": {
      "default": false,
      "title": "All Epochs",
      "type": "boolean"
    },
    "at": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "At"
    },
    "window_seconds": {
      "default": 2.0,
      "title": "Window Seconds",
      "type": "number"
    }
  },
  "type": "object",
  "title": "get_track_historyArguments"
}
```
</details>

#### `camera__get_track_frame`

Return the buffered frame where a track first (or last) appeared, cropped around its box by default, for re-checking how a track started. Recent tracks only: the frame must still be in the look-back buffer (see list_buffered_frames for the time span it covers, often under a minute); older frames fail with an error.

<details>
<summary>Input schema</summary>

```json
{
  "properties": {
    "track_id": {
      "title": "Track Id",
      "type": "integer"
    },
    "which": {
      "default": "first",
      "enum": [
        "first",
        "last"
      ],
      "title": "Which",
      "type": "string"
    },
    "epoch": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Epoch"
    },
    "crop": {
      "default": true,
      "title": "Crop",
      "type": "boolean"
    },
    "padding": {
      "default": 0.3,
      "title": "Padding",
      "type": "number"
    },
    "max_side": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": 1024,
      "title": "Max Side"
    }
  },
  "required": [
    "track_id"
  ],
  "type": "object",
  "title": "get_track_frameArguments"
}
```
</details>

#### `camera__list_buffered_frames`

Timestamps of the most recent processed frames kept in the camera's look-back buffer (full capture resolution, a fixed number of frames, lost on restart). `at` in capture_frame / detect and get_track_frame can only reach frames between oldest and newest; at full frame rate that is often under a minute.

<details>
<summary>Input schema</summary>

```json
{
  "properties": {
    "limit": {
      "default": 50,
      "title": "Limit",
      "type": "integer"
    }
  },
  "type": "object",
  "title": "list_buffered_framesArguments"
}
```
</details>

#### `camera__set_schedule`

Temporarily change the default loop for duration_seconds (max 24 h), then it reverts. fps: frames per second (e.g. 1, or 0.0333 for one per 30 s). roi: only run the loop inside this box (display coords), e.g. the sleeping spot. imgsz: model input [w, h] (multiples of 32). model: run the loop with another model. Calling again while an override is active replaces it and restarts the timer. Fields you omit keep their current override value with merge=true (e.g. set fps, then add a roi without losing the fps), or go back to defaults with merge=false.

<details>
<summary>Input schema</summary>

```json
{
  "properties": {
    "duration_seconds": {
      "default": 300,
      "title": "Duration Seconds",
      "type": "number"
    },
    "fps": {
      "anyOf": [
        {
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Fps"
    },
    "roi": {
      "anyOf": [
        {
          "items": {
            "type": "integer"
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Roi"
    },
    "imgsz": {
      "anyOf": [
        {
          "items": {
            "type": "integer"
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Imgsz"
    },
    "model": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Model"
    },
    "merge": {
      "default": true,
      "title": "Merge",
      "type": "boolean"
    },
    "reason": {
      "default": "",
      "title": "Reason",
      "type": "string"
    }
  },
  "type": "object",
  "title": "set_scheduleArguments"
}
```
</details>

#### `camera__reset_schedule`

Cancel any override and return the loop to default settings now.

<details>
<summary>Input schema</summary>

```json
{
  "properties": {},
  "type": "object",
  "title": "reset_scheduleArguments"
}
```
</details>

### Stored images (`images__*`, 4 tools)

Stored images on the SSD host (`image_server`).

#### `images__image_exists`

Fast check whether an image is on the SSD. image_name is the capture timestamp YYYY-MM-DD-HH-MM-SS-ffffff.jpg (24-hour clock, 6-digit microseconds), e.g. 2026-10-24-20-15-03-123456.jpg. Returns exists and, if it does, size_bytes, captured_at, modified_at, relative_path, and (include_dimensions=true) width, height and channels read from the JPEG header without decoding. One stat() (+ a few hundred bytes read for dimensions); no image data is returned.

<details>
<summary>Input schema</summary>

```json
{
  "properties": {
    "image_name": {
      "title": "Image Name",
      "type": "string"
    },
    "include_dimensions": {
      "default": true,
      "title": "Include Dimensions",
      "type": "boolean"
    }
  },
  "required": [
    "image_name"
  ],
  "type": "object",
  "title": "image_existsArguments"
}
```
</details>

#### `images__images_exist`

Batch version of image_exists for up to 1000 names in one call, each named YYYY-MM-DD-HH-MM-SS-ffffff.jpg (e.g. 2026-10-24-20-15-03-123456.jpg). Invalid names are reported per item instead of failing the whole call.

<details>
<summary>Input schema</summary>

```json
{
  "properties": {
    "image_names": {
      "items": {
        "type": "string"
      },
      "title": "Image Names",
      "type": "array"
    },
    "include_dimensions": {
      "default": false,
      "title": "Include Dimensions",
      "type": "boolean"
    }
  },
  "required": [
    "image_names"
  ],
  "type": "object",
  "title": "images_existArguments"
}
```
</details>

#### `images__get_image`

Return a JPEG image named YYYY-MM-DD-HH-MM-SS-ffffff.jpg (e.g. 2026-10-24-20-15-03-123456.jpg). By default the file bytes are sent as stored (no decode, fastest). Optional: box=[x1, y1, x2, y2] crops in source pixels, max_side shrinks so the longest side is <= max_side; either one re-encodes the result as JPEG. If the image does not exist, returns exists=false and no image.

<details>
<summary>Input schema</summary>

```json
{
  "properties": {
    "image_name": {
      "title": "Image Name",
      "type": "string"
    },
    "max_side": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Max Side"
    },
    "box": {
      "anyOf": [
        {
          "items": {
            "type": "integer"
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Box"
    }
  },
  "required": [
    "image_name"
  ],
  "type": "object",
  "title": "get_imageArguments"
}
```
</details>

#### `images__list_images`

List image names sorted by time. prefix is the leading part of an image name, cut at a day, hour or minute boundary (zero-padded, 24-hour): YYYY-MM-DD (e.g. 2026-10-24), YYYY-MM-DD-HH (e.g. 2026-10-24-20) or YYYY-MM-DD-HH-MM (e.g. 2026-10-24-20-15). Seconds or partial fields (2026-10, 2026-10-24-2) are rejected. Use a minute or hour prefix for speed; a whole day can be tens of thousands of files.

<details>
<summary>Input schema</summary>

```json
{
  "properties": {
    "prefix": {
      "title": "Prefix",
      "type": "string"
    },
    "limit": {
      "default": 500,
      "title": "Limit",
      "type": "integer"
    }
  },
  "required": [
    "prefix"
  ],
  "type": "object",
  "title": "list_imagesArguments"
}
```
</details>

### Tracking (`tracking__*`, 9 tools)

Tracker and period results (`existing_work_before_hackathon/tracking_mcp`).

#### `tracking__list_views`

List the camera views: view_id, name, frame size and the cats in each view (cat_id -> name). live_views are the views this server is receiving frames for now. Call this first to turn a cat's name into its cat_id.

<details>
<summary>Input schema</summary>

```json
{
  "properties": {},
  "type": "object",
  "title": "list_viewsArguments"
}
```
</details>

#### `tracking__get_live_status`

What the camera sees right now. Per view: the last frame received (time, seconds_ago, image_name) and every detection on it, including ones below min_confidence that the tracker ignores. Per cat: name, on_screen, current_period_id (the ongoing visit, null if not on screen), last_period_id, and latest_detection (newest frame where the detector found it; final=false). view_id limits the reply to one view.

<details>
<summary>Input schema</summary>

```json
{
  "properties": {
    "view_id": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "View Id"
    }
  },
  "type": "object",
  "title": "get_live_statusArguments"
}
```
</details>

#### `tracking__get_last_seen`

When and where each cat (or only cat_id) was last seen: time, seconds_ago, box, conf, cat_id (tracked) and detector_cat_id, period_id and image_name. source is live (last few minutes, may still change if final=false), saved (from the database; period_cat_id / period_final tell the visit's cat and whether it is final) or detection_only (the detector saw it but it never became a track). Searches back lookback_days; last_seen is null if the cat was not seen in that time.

<details>
<summary>Input schema</summary>

```json
{
  "properties": {
    "cat_id": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Cat Id"
    },
    "view_id": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "View Id"
    },
    "lookback_days": {
      "default": 7,
      "title": "Lookback Days",
      "type": "number"
    }
  },
  "type": "object",
  "title": "get_last_seenArguments"
}
```
</details>

#### `tracking__get_current_periods`

Visits happening now or that just ended, before they are saved. Use it for "what is on camera now / in the last few minutes"; use get_periods for anything older. not_final: visits with status ongoing or ended_pending, each with start, end, duration_s, the cat_id it would get if decided now (can still change), evidence_by_cat, frame count and first / last frame. recently_final: the newest visits (up to final_limit) that became final since this server started.

<details>
<summary>Input schema</summary>

```json
{
  "properties": {
    "view_id": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "View Id"
    },
    "final_limit": {
      "default": 10,
      "title": "Final Limit",
      "type": "integer"
    }
  },
  "type": "object",
  "title": "get_current_periodsArguments"
}
```
</details>

#### `tracking__get_periods`

Search saved, final visits, newest first. Use it for history ("when was Mochi here yesterday"); visits still ongoing or just ended are not here yet, use get_current_periods for those. Filters: cat_id, view_id, time_from / time_to ('YYYY-MM-DD HH:MM[:SS]', any overlap counts), min_duration_s. Each visit: period_id, start, end, duration_s, cat_id and evidence_by_cat.

<details>
<summary>Input schema</summary>

```json
{
  "properties": {
    "cat_id": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Cat Id"
    },
    "view_id": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "View Id"
    },
    "time_from": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Time From"
    },
    "time_to": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Time To"
    },
    "limit": {
      "default": 20,
      "title": "Limit",
      "type": "integer"
    },
    "min_duration_s": {
      "default": 0,
      "title": "Min Duration S",
      "type": "number"
    }
  },
  "type": "object",
  "title": "get_periodsArguments"
}
```
</details>

#### `tracking__get_period_detail`

Look inside one visit by period_id (from get_current_periods, get_periods or get_last_seen). Works for ongoing, ended_pending and final visits. Returns:

- period: status, start, end, cat_id, evidence_by_cat.
- frames: how many frames have a box, how many were detected vs interpolated, how many are not final, and how often the detector's label matched the visit's cat_id (a low detector_agrees_fraction means the cat_id is less certain).
- first_frame, last_frame, best_frame (highest conf), each with image_name and crop_box to view it, and came_from / last_edge: the image border where the cat entered and where it was last.
- path: up to max_points box centers over time.
- motion: moving and resting segments. The cat counts as moving when its box center travels faster than speed_threshold px/s averaged over window_s; pauses up to merge_gap_s are joined; moves shorter than min_moving_s or covering less than min_distance_px are ignored.

<details>
<summary>Input schema</summary>

```json
{
  "properties": {
    "period_id": {
      "title": "Period Id",
      "type": "integer"
    },
    "max_points": {
      "default": 60,
      "title": "Max Points",
      "type": "integer"
    },
    "speed_threshold": {
      "default": 3.0,
      "title": "Speed Threshold",
      "type": "number"
    },
    "min_moving_s": {
      "default": 2.0,
      "title": "Min Moving S",
      "type": "number"
    },
    "merge_gap_s": {
      "default": 3.0,
      "title": "Merge Gap S",
      "type": "number"
    },
    "window_s": {
      "default": 2.0,
      "title": "Window S",
      "type": "number"
    },
    "min_distance_px": {
      "default": 20.0,
      "title": "Min Distance Px",
      "type": "number"
    }
  },
  "required": [
    "period_id"
  ],
  "type": "object",
  "title": "get_period_detailArguments"
}
```
</details>

#### `tracking__get_arrival_path`

Key frames showing where a cat came from and how it got to where it is, so you can check the track by eye. Give period_id, or cat_id to use its newest visit. key_frames are in time order, each with a role:

- empty_before: the last frame with no cat, to show the empty scene.
- entry: the earliest box, also searching detections up to pre_roll_s before the visit started.
- first_tracked: the first frame of the visit itself.
- path: frames spread evenly by distance travelled (n_frames in total).
- settled: where the cat first rested for settle_s or more; latest if it never did.

Each has image_name, box, crop_box, conf, cat_id, detector_cat_id and edge. Fetch them with images__get_image in one turn and check the same cat moves along the path. include_images=true attaches the frames with the box drawn (max_side px), if this server can read the image folder.

<details>
<summary>Input schema</summary>

```json
{
  "properties": {
    "period_id": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Period Id"
    },
    "cat_id": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Cat Id"
    },
    "view_id": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "View Id"
    },
    "n_frames": {
      "default": 5,
      "title": "N Frames",
      "type": "integer"
    },
    "pre_roll_s": {
      "default": 10,
      "title": "Pre Roll S",
      "type": "number"
    },
    "settle_s": {
      "default": 10,
      "title": "Settle S",
      "type": "number"
    },
    "include_images": {
      "default": false,
      "title": "Include Images",
      "type": "boolean"
    },
    "max_side": {
      "default": 384,
      "title": "Max Side",
      "type": "integer"
    }
  },
  "type": "object",
  "title": "get_arrival_pathArguments"
}
```
</details>

#### `tracking__get_last_movement`

When cat_id last moved and how long it has stayed still since. Checks the cat's visits newest first (up to max_periods, within lookback_hours) and returns last_seen, last_movement (start, end, duration, distance, and image names of its first and last frame) and resting_since_movement (from the end of that move to the last seen frame). The cat counts as moving when its box center travels faster than speed_threshold px/s averaged over window_s (this hides box jitter); pauses up to merge_gap_s are joined; moves shorter than min_moving_s or ending less than min_distance_px from where they started are ignored.

<details>
<summary>Input schema</summary>

```json
{
  "properties": {
    "cat_id": {
      "title": "Cat Id",
      "type": "integer"
    },
    "view_id": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "View Id"
    },
    "speed_threshold": {
      "default": 3.0,
      "title": "Speed Threshold",
      "type": "number"
    },
    "min_moving_s": {
      "default": 2.0,
      "title": "Min Moving S",
      "type": "number"
    },
    "merge_gap_s": {
      "default": 3.0,
      "title": "Merge Gap S",
      "type": "number"
    },
    "window_s": {
      "default": 2.0,
      "title": "Window S",
      "type": "number"
    },
    "min_distance_px": {
      "default": 20.0,
      "title": "Min Distance Px",
      "type": "number"
    },
    "lookback_hours": {
      "default": 24,
      "title": "Lookback Hours",
      "type": "number"
    },
    "max_periods": {
      "default": 20,
      "title": "Max Periods",
      "type": "integer"
    }
  },
  "required": [
    "cat_id"
  ],
  "type": "object",
  "title": "get_last_movementArguments"
}
```
</details>

#### `tracking__get_raw_detections`

Detector output before tracking (conf above 0.5) between time_from and time_to ('YYYY-MM-DD HH:MM[:SS]'), oldest first: time, cat_id, box, conf. Use it to compare what the detector saw on each frame with the tracked result.

<details>
<summary>Input schema</summary>

```json
{
  "properties": {
    "time_from": {
      "title": "Time From",
      "type": "string"
    },
    "time_to": {
      "title": "Time To",
      "type": "string"
    },
    "cat_id": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Cat Id"
    },
    "view_id": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "View Id"
    },
    "limit": {
      "default": 500,
      "title": "Limit",
      "type": "integer"
    }
  },
  "required": [
    "time_from",
    "time_to"
  ],
  "type": "object",
  "title": "get_raw_detectionsArguments"
}
```
</details>

### Local (`local__*`, 10 tools)

In-process tools defined in [`local_tools/server.py`](local_tools/server.py); they run here, not over MCP.

#### `local__get_cat_profiles`

Written profiles of the household cats: name, tracker ids per view, summary, low-res and camera cues, how to tell them apart. detail='full' adds the long description and high-res cues (eyes, nose). cat_ids may be profile ids or tracker ids (e.g. 25/26 on view 3).

<details>
<summary>Input schema</summary>

```json
{
  "type": "object",
  "properties": {
    "cat_ids": {
      "type": "array",
      "items": {
        "type": "integer"
      }
    },
    "view_id": {
      "type": "integer"
    },
    "detail": {
      "type": "string",
      "enum": [
        "low_res",
        "full"
      ],
      "default": "low_res"
    }
  }
}
```
</details>

#### `local__find_references`

Rank this cat's reference images WITHOUT returning any image: owner photos (source=profile) and confirmed camera crops (source=scene). Scored by same view, nearby box position and size, same pose and lighting, and photo quality. Fetch only the top one or two with get_reference_image.

<details>
<summary>Input schema</summary>

```json
{
  "type": "object",
  "required": [
    "cat_id"
  ],
  "properties": {
    "cat_id": {
      "type": "integer"
    },
    "view_id": {
      "type": "integer"
    },
    "box": {
      "type": "array",
      "items": {
        "type": "number"
      },
      "minItems": 4,
      "maxItems": 4,
      "description": "[x1, y1, x2, y2] in the view's pixels"
    },
    "pose": {
      "type": "string",
      "description": "sitting, lying, walking, crouching, sleeping, standing, playing"
    },
    "lighting": {
      "type": "string",
      "enum": [
        "day",
        "low_light",
        "night_ir"
      ]
    },
    "source": {
      "type": "string",
      "enum": [
        "profile",
        "scene"
      ]
    },
    "k": {
      "type": "integer",
      "default": 2
    }
  }
}
```
</details>

#### `local__get_reference_image`

Return one reference image (already cropped to the cat) by the `file` find_references gave. Counts against the image budget.

*Returns images (counts against the image budget).*

<details>
<summary>Input schema</summary>

```json
{
  "type": "object",
  "required": [
    "file"
  ],
  "properties": {
    "file": {
      "type": "string"
    },
    "max_side": {
      "type": "integer",
      "default": 384
    }
  }
}
```
</details>

#### `local__make_contact_sheet`

Build ONE numbered grid image from several crops, so several frames cost one image call. Each item is a stored frame {image_name, box} (cropped around the box with padding) or a reference {reference_file}. Tiles are numbered 1..n in the order given.

*Returns images (counts against the image budget).*

<details>
<summary>Input schema</summary>

```json
{
  "type": "object",
  "required": [
    "items"
  ],
  "properties": {
    "items": {
      "type": "array",
      "maxItems": 9,
      "items": {
        "type": "object",
        "properties": {
          "image_name": {
            "type": "string"
          },
          "box": {
            "type": "array",
            "items": {
              "type": "number"
            },
            "minItems": 4,
            "maxItems": 4,
            "description": "[x1, y1, x2, y2] in the view's pixels"
          },
          "reference_file": {
            "type": "string"
          },
          "caption": {
            "type": "string"
          }
        }
      }
    },
    "tile": {
      "type": "integer",
      "default": 256
    },
    "pad": {
      "type": "number",
      "default": 0.3
    }
  }
}
```
</details>

#### `local__analyze_trajectory`

Physics check of one visit with no images: rebuilds the visit's own boxes from raw detections and flags teleports, mid-frame entry/exit, long gaps, size jumps, detector identity flips, another cat close by, and a box that never moves. Also returns numbered keyframes that test each doubt.

<details>
<summary>Input schema</summary>

```json
{
  "type": "object",
  "required": [
    "period_id"
  ],
  "properties": {
    "period_id": {
      "type": "integer"
    },
    "pre_roll_s": {
      "type": "number",
      "default": 5
    }
  }
}
```
</details>

#### `local__get_label_set`

The label set: activity labels (pick exactly one) and context labels (any number), with descriptions, what evidence each needs, and code rules. Includes the version number.

<details>
<summary>Input schema</summary>

```json
{
  "type": "object",
  "properties": {}
}
```
</details>

#### `local__add_label`

Add a label to the label set. With confirm=false (default) it only validates and returns the draft, problems and similar existing labels: show that to the user. Call again with confirm=true only after the user says yes.

<details>
<summary>Input schema</summary>

```json
{
  "type": "object",
  "required": [
    "name",
    "kind",
    "description"
  ],
  "properties": {
    "name": {
      "type": "string",
      "description": "lower_snake_case, e.g. in_tent"
    },
    "kind": {
      "type": "string",
      "enum": [
        "activity",
        "context"
      ]
    },
    "description": {
      "type": "string"
    },
    "rule": {
      "type": "object",
      "description": "context labels only, e.g. {zone: sofa, min_overlap: 0.5}"
    },
    "needs": {
      "type": "array",
      "items": {},
      "description": "e.g. [image], [image, motion], [{zone: tent}]"
    },
    "confirm": {
      "type": "boolean",
      "default": false
    }
  }
}
```
</details>

#### `local__get_zones`

Named zones of a view (places, occluders, entries) with boxes or polygons, and the view size.

<details>
<summary>Input schema</summary>

```json
{
  "type": "object",
  "required": [
    "view_id"
  ],
  "properties": {
    "view_id": {
      "type": "integer"
    }
  }
}
```
</details>

#### `local__zones_for_box`

Which zones a box overlaps, and by what fraction of the box's area. No image.

<details>
<summary>Input schema</summary>

```json
{
  "type": "object",
  "required": [
    "view_id",
    "box"
  ],
  "properties": {
    "view_id": {
      "type": "integer"
    },
    "box": {
      "type": "array",
      "items": {
        "type": "number"
      },
      "minItems": 4,
      "maxItems": 4,
      "description": "[x1, y1, x2, y2] in the view's pixels"
    }
  }
}
```
</details>

#### `local__save_zone`

Define or replace a named zone on a view. confirm=false returns the draft for the user to check; confirm=true writes it (only after the user agrees).

<details>
<summary>Input schema</summary>

```json
{
  "type": "object",
  "required": [
    "view_id",
    "name",
    "kind"
  ],
  "properties": {
    "view_id": {
      "type": "integer"
    },
    "name": {
      "type": "string"
    },
    "kind": {
      "type": "string",
      "enum": [
        "place",
        "occluder",
        "entry"
      ]
    },
    "box": {
      "type": "array",
      "items": {
        "type": "number"
      },
      "minItems": 4,
      "maxItems": 4,
      "description": "[x1, y1, x2, y2] in the view's pixels"
    },
    "polygon": {
      "type": "array",
      "items": {
        "type": "array",
        "items": {
          "type": "number"
        }
      }
    },
    "confirm": {
      "type": "boolean",
      "default": false
    }
  }
}
```
</details>

### Skills (`skill__*`, 4 tools)

One tool per folder in [`skills/`](skills/): invoking it loads that skill's full instructions. Generated automatically; see [`local_tools/skills.py`](local_tools/skills.py).

#### `skill__assign-activity-label`

Skill 'assign-activity-label': Give one cat crop exactly one activity label and any matching context labels from labels.yaml, or propose a new label when nothing fits. Use for assign_label tasks. Invoke it (no arguments) to load the full skill instructions, then follow them step by step.

<details>
<summary>Input schema</summary>

```json
{
  "type": "object",
  "properties": {}
}
```
</details>

#### `skill__check-trajectory`

Skill 'check-trajectory': Check that one visit (period) is a physically plausible path of one animal, looking only at the few frames around what the code flagged. Use for check_trajectory tasks. Invoke it (no arguments) to load the full skill instructions, then follow them step by step.

<details>
<summary>Input schema</summary>

```json
{
  "type": "object",
  "properties": {}
}
```
</details>

#### `skill__manage-labels`

Skill 'manage-labels': Turn a user's plain-language request ("label the cat when it sits in the tent") into a new label, and a new zone if it is place-based, confirming with the user before saving. Use for label commands. Invoke it (no arguments) to load the full skill instructions, then follow them step by step.

<details>
<summary>Input schema</summary>

```json
{
  "type": "object",
  "properties": {}
}
```
</details>

#### `skill__validate-cat-identity`

Skill 'validate-cat-identity': Decide whether a detection is a cat and which household cat it is, climbing a cheapest-evidence-first ladder. Use for validate_detection tasks from the CV pipeline. Invoke it (no arguments) to load the full skill instructions, then follow them step by step.

<details>
<summary>Input schema</summary>

```json
{
  "type": "object",
  "properties": {}
}
```
</details>
[2 groups, 14 tools]

## Files

| File | Responsibility |
|---|---|
| `config.py` | env / JSON settings for model and servers |
| `mcp_hub.py` | connect to several MCP servers, namespace and route tool calls |
| `tool_adapter.py` | MCP tool schema → Responses API function tool; MCP result → function output (text + images), timing extraction |
| `prompts.py` | system instructions (plus each server's own instructions) |
| `agent.py` | the tool-calling loop, run trace |
| `invoke_log.py` | background-thread dump of each run and tool call to MySQL (`agent_invoke`, `tool_invoke`) |
| `cli.py` | one-shot and interactive command line |

## Using it from code

```python
from supervisor_agent.agent import MCPAgent
from supervisor_agent.config import AgentConfig
from supervisor_agent.mcp_hub import MCPHub

cfg = AgentConfig.from_env()
async with MCPHub(cfg.servers) as hub:
    result = await MCPAgent(cfg, hub).run("Is a cat on camera?")
    print(result.answer, [(s.tool, s.server_elapsed_ms) for s in result.tool_steps])
```


## Pipeline tasks (validate, trajectory, labels)

Besides free-form questions, the agent runs three tasks the CV pipeline can hand it, each as
one skill with a strict JSON answer. Design: "RCAT Agent Tasks" doc. Code: [tasks.py](tasks.py),
[local_tools/](local_tools/), [skills/](skills/); data: [../agent_data/](../agent_data/README.md).

| Task | Payload | Code does first (no images) | Model only when |
|---|---|---|---|
| `validate_detection` | `image_name, view_id, box, detector{cat_id, candidates}, period_id?` | visit summary, other cats seen at the same moment (exclusion), visit-consensus shortcut | the visit is not solid; gets 1 crop + the candidates' written profiles, can escalate to references |
| `check_trajectory` | `period_id` | `analyze_trajectory`: teleport, mid-frame entry/exit, long gap, size jump, identity flip, other cat close, static box; keyframes | something is flagged; looks at ONE contact sheet of keyframes |
| `assign_label` | `image_name, view_id, box, period_id?` | motion state, zone overlaps, other cat nearby → candidate activities, rule-based context labels | rules leave more than one answer; unknown activity is filed under `agent_data/label_suggestions/<name>/` |
| `label_visit` | `period_id` | one `assign_label` per motion segment, not per frame | as above |

```bash
python -m supervisor_agent --task check_trajectory --payload '{"period_id": 8812}'
python -m supervisor_agent --task validate_detection --payload @detection.json
python -m supervisor_agent --labels   # "label the cat when it sits in the tent"
```

- **Cheapest evidence first.** Every task has an image budget in
  `agent_data/config/agent.yaml`; image-returning calls past it are refused, so the model
  decides or answers `send_to_human`. Parallel calls reserve budget before they run.
- **Only the task's tools** are sent to the model (the list in agent.yaml), which keeps
  prompts small; the skill text is the system prompt, so it caches well.
- **`local__*` tools** run in-process on the hub next to the MCP servers: cat profiles,
  reference ranking (`find_references` returns no pixels), cat-only reference crops,
  contact sheets (n crops = 1 image), trajectory physics, label set, zones.
- **Cat ids.** The pipeline uses tracker ids (view 3: Qb = 25, Qbo = 26); `cats.yaml` maps
  them to profiles via `tracking_ids`.
- **Logs.** Every result goes to `agent_data/logs/<task>.jsonl` with images used, tool calls,
  tokens and the label-set version; `send_to_human` also goes to `review_queue.jsonl`.
  Splits and relabels are logged only, never written to the tracking database.
- **Labels.** New labels need an explicit yes (`add_label` with `confirm=true`). Saving
  rewrites labels.yaml: the header comment is kept, inline comments are not.

Tests (offline: fake servers and a scripted model):

```bash
python -m pytest supervisor_agent/tests -q
```
