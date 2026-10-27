# agentic_realtime_cv_agent: Muse Spark agent over the RCAT MCP tools

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

From the agentic_realtime_cv_agent/ folder (Python ≥ 3.10; the repo venv is 3.12):

```bash
pip install -r requirements.txt
```

Add to `.env` (the full list for every component is in the repo's [`.env.example`](../../.env.example)):

```bash
MODEL_API_KEY=<Meta Model API key>
CAMERA_MCP_URL=http://<pi>:8765/mcp
CAMERA_MCP_TOKEN=<camera MCP_AUTH_TOKEN>
IMAGE_MCP_URL=http://10.0.0.63:10001/mcp
IMAGE_MCP_TOKEN=<image server IMAGE_MCP_AUTH_TOKEN>
TRACKING_MCP_URL=http://<websocket server host>:10002/mcp
TRACKING_MCP_TOKEN=<websocket server TRACKING_MCP_AUTH_TOKEN>
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
python -m agentic_realtime_cv_agent "Is any cat on camera right now? Show me."

# interactive (follow-up questions keep context)
python -m agentic_realtime_cv_agent

# what the agent can call
python -m agentic_realtime_cv_agent --list-tools
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

## Files

| File | Responsibility |
|---|---|
| `config.py` | env / JSON settings for model and servers |
| `mcp_hub.py` | connect to several MCP servers, namespace and route tool calls |
| `tool_adapter.py` | MCP tool schema → Responses API function tool; MCP result → function output (text + images), timing extraction |
| `prompts.py` | system instructions (plus each server's own instructions) |
| `agent.py` | the tool-calling loop, run trace |
| `cli.py` | one-shot and interactive command line |

## Using it from code

```python
from agentic_realtime_cv_agent.agent import MCPAgent
from agentic_realtime_cv_agent.config import AgentConfig
from agentic_realtime_cv_agent.mcp_hub import MCPHub

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
python -m agentic_realtime_cv_agent --task check_trajectory --payload '{"period_id": 8812}'
python -m agentic_realtime_cv_agent --task validate_detection --payload @detection.json
python -m agentic_realtime_cv_agent --labels   # "label the cat when it sits in the tent"
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
python -m pytest agentic_realtime_cv_agent/tests -q
```
