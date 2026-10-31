# agentic-realtime-cv
An AI supervisor that checks a tiny edge detector only when needed, and teaches it from what it verifies.

A small YOLO model runs all the time on a Raspberry Pi camera and watches our cats. A tracker
cleans up its detections. When something needs a closer look, an agent on Meta's
**Muse Spark** checks the evidence through MCP tools: live frames, stored images and the
tracker's state. It confirms which cat it is, checks the path, and labels the activity.
What the agent verifies becomes training data for the edge model.

```
 Raspberry Pi                         ingest host                         anywhere
┌────────────────────┐  detections  ┌──────────────────────────────┐
│ camera_server      │ ──WebSocket─►│ detection_ingest_server.py   │
│  camera_v3 (YOLO)  │    :8000     │  TrackCorrector              │
│  MCP :8765         │              │  PeriodCorrector ──► MySQL   │
└─────────▲──────────┘              │  tracking_mcp MCP :10002     │
          │                         └──────────────▲───────────────┘
          │      ┌───────────────────────────┐     │
          │      │ image_server              │     │
          │      │  mcp_image_server :10001  │     │
          │      │  (stored images on SSD)   │     │
          │      └─────────────▲─────────────┘     │
          │                    │                   │
          └──────── MCP ───────┴─────── MCP ───────┤
                                                   │
                               ┌───────────────────┴───────────────┐
                               │ agent/supervisor_agent            │
                               │  Muse Spark (Meta Model API)      │
                               │  + local tools over agent_data/   │
                               └───────────────────────────────────┘
```

## Repo layout

| Folder | What it is | Built (hackathon starts Oct 27) |
|---|---|---|
| [`existing_work_before_hackathon/`](existing_work_before_hackathon/) | `detection_ingest_server.py` (WebSocket ingest; was `websocket_server.py`), the TrackCorrector / PeriodCorrector (`model/`), the tracking MCP server (`tracking_mcp/`) and the tracker visualization (`run_tracking_viz.py`) | Correctors and ingest 2025; tracking MCP and viz Oct 23-25 |
| [`camera_server/`](camera_server/) | `camera_v3`: always-on YOLO detection on the Pi, streaming to the ingest server, plus an MCP control server (capture, zoom, high-res stills, detect, schedules) | camera_v3 + MCP added Oct 24-25 |
| [`image_server/`](image_server/) | `mcp_image_server`: MCP server over the stored camera images | Added Oct 25 |
| [`agent/`](agent/) | The agent (`supervisor_agent` package, was `mcp_agent`): a tool-calling loop over the three MCP servers, pipeline tasks (validate identity, check trajectory, assign labels), skills and `agent_data/` (cat profiles, reference images, labels, zones); design notes in `memory/`, `escalation/`, `generalization/`; the tracking pipeline (`main.py` entry, `tracking/` correctors, `escalation/` trigger vocabulary) migrated from `existing_work_before_hackathon/model/` | Built Oct 23-25, ported to Meta models, restructured Oct 26; tracking migrated Oct 30 |
| [`documents/`](documents/) | How the correctors work, the tracker visualization, the hackathon guide and the project plan | |

Each folder is run from inside itself (for example `cd agent`).

## Models (LLM)

All LLM use goes through Meta models. They're set per role in
[`agent/supervisor_agent/models.json`](agent/supervisor_agent/models.json),
together with the provider endpoint and token prices.

| Role | Model | Use | Wired in code |
|---|---|---|---|
| `agent` | Muse Spark (`muse-spark-1.3`) | Agent reasoning, vision, tool calls; escalation teacher | Yes |
| `segmentation` | SAM 3.1 (`sam-3.1`) | Masks/boxes for training data | Not yet |
| `voice` | Muse Voice Transcribe (`muse-voice-transcribe-1.0`) | Voice questions (if we build voice) | Not yet |
| `creative` | Muse Image (`muse-image-1.0`) + Muse Wonder (`muse-wonder-1.0`) | Creative output, e.g. a daily "cat diary" comic | Not yet |
| `open_weight` | Muse Spark Open (`meta-models/muse-spark-open-1026`, Baseten) / Muse Glimmer | Local or partner-hosted model (privacy) | Not yet |
| `coding` | Muse Code | Coding (a product, not an API model) | n/a |

The agent calls the Meta Model API (`https://api.meta.ai/v1`) through its OpenAI-compatible
Responses endpoint, using `MODEL_API_KEY`. The `openai` Python package is only the HTTP
client; no OpenAI models are used. The camera's edge detector is YOLO (Ultralytics), set in
`camera_server/camera_v3/models.json`.

## Setup

Copy [`.env.example`](.env.example) to `.env` and fill it in. It lists every variable the
code reads, grouped by component. One `.env` at the repo root works for everything on the
same machine. The camera (Pi) and the image server (SSD host) each need their own `.env`
with just their section.

Python ≥ 3.10 everywhere (the MCP SDK needs it).

## Run

| Component | Install | Run | Port |
|---|---|---|---|
| Ingest server + tracking MCP | `pip install -r existing_work_before_hackathon/requirements.txt` | `cd existing_work_before_hackathon && python detection_ingest_server.py` (or `run_ingest_server.sh`) | WebSocket 8000, MCP 10002 |
| Camera (on the Pi) | `cd camera_server && uv sync` | `cd camera_server && uv run python -m camera_v3` (or `run_detection_v3.sh`, which uses `~/camera_server/.venv`) | MCP 8765 |
| Image server | `pip install -r image_server/requirements-mcp.txt` | `cd image_server && python -m mcp_image_server` (or `run_mcp_server.sh`, which uses `~/camera_server/.venv`) | MCP 10001 |
| Agent | `pip install -r agent/requirements.txt` | `cd agent && python -m supervisor_agent "Is any cat on camera right now?"` | — |
| Tracking (live) | `pip install -r agent/requirements.txt` | `python -m agent.main --view-id 3 --start 2026-10-27-14-00 --end 2026-10-27-14-05 --live` (from repo root; camera pushes to :8000) | WebSocket 8000 |
| Tracking (replay) | same | `python -m agent.main --view-id 3 --start 2026-10-27-14-00 --end 2026-10-27-14-05 --replay agent/output/tracking_runs/<run>/stream.jsonl` (add `--process-viz` for per-frame frames + MP4) | — |

Agent modes:

```bash
cd agent
python -m supervisor_agent  # interactive chat
python -m supervisor_agent --list-tools  # what the agent can call
python -m supervisor_agent --task check_trajectory --payload '{"period_id": 8812}'
python -m supervisor_agent --labels  # add labels / zones by chatting
```

If an MCP server is down, the agent warns and carries on with the others.

Download one hour of images from the image server (into `image_server/retrieved_images/`,
git-ignored; see the [image server README](image_server/mcp_image_server/README.md#download-an-hour-of-images)):

```bash
python image_server/fetch_hour_images.py 2025-05-05-19
```

## Tests

```bash
python -m pytest agent/escalation/tests agent/tracking/tests -q     # tracking + escalation, no MySQL needed
cd agent && python -m pytest supervisor_agent/tests -q              # fake model client, no API key needed
cd existing_work_before_hackathon && python -m tracking_mcp.test_motion  # tracker motion helpers
```

## More docs

- Agent: [`agent/README.md`](agent/README.md) ([`agent/supervisor_agent/README.md`](agent/supervisor_agent/README.md))
- Tracking pipeline: [`agent/tracking/README.md`](agent/tracking/README.md); escalation triggers: [`agent/escalation/README.md`](agent/escalation/README.md)
- Camera: [`camera_server/camera_v3/README.md`](camera_server/camera_v3/README.md)
- Image server: [`image_server/mcp_image_server/README.md`](image_server/mcp_image_server/README.md)
- Tracking MCP: [`existing_work_before_hackathon/tracking_mcp/README.md`](existing_work_before_hackathon/tracking_mcp/README.md)
- How the correctors work: [`documents/HOW_IT_WORKS.md`](documents/HOW_IT_WORKS.md)
- Tracker visualization: [`documents/TRACKING_VIZ.md`](documents/TRACKING_VIZ.md)
- Hackathon guide and project plan: [`documents/HACKATHON_GUIDE.md`](documents/HACKATHON_GUIDE.md), [`documents/PROJECT_PLAN.md`](documents/PROJECT_PLAN.md) ([中文](documents/PROJECT_PLAN.zh.md))
