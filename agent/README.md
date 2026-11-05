# agent

The hackathon pipeline: YOLO detections in, validated and memory-backed cat visits out.
`main.py` runs the two tracking correctors over a live or replayed detection stream; every
uncertain decision becomes an escalation that the supervisor agent (Muse Spark) reviews with
image tools; verdicts are applied to the rows; memory keeps what was seen. The full flow,
stage by stage, is in [`documents/end-to-end-flow.md`](../documents/end-to-end-flow.md).

Run everything in this package from the repo root (`python -m agent.main`,
`python -m pytest agent/...`). The quickest way to see it all work is
[`run_demo.py`](../run_demo.py) at the repo root.

## Modules

| Path | What it is |
|---|---|
| [`main.py`](main.py) | Entry point (`TrackingRun`): ingest → TrackCorrector → PeriodCorrector → escalations → validation worker → memory → charts, `report.md`, `run.json` |
| [`tracking/`](tracking/README.md) | `TrackCorrector` + `PeriodCorrector` (MySQL-free), run store, decoy zones, read-only MySQL source, charts, process viz, session report |
| [`escalation/`](escalation/README.md) | Escalation vocabulary (`T1`–`T7`, `P1`–`P8`, `H1`), thresholds, per-code reviewer prompts and verdict schemas |
| [`validation/`](validation/README.md) | Background validation worker (merged batches → supervisor agent → verdicts), verdict applier (`validated.jsonl` / `validated.db`), period activity labeler |
| [`memory/`](memory/README.md) | Short-term day memory and long-term spatial-temporal memory (plain-language YAML/Markdown) |
| [`supervisor_agent/`](supervisor_agent/README.md) | The Muse Spark tool-calling agent over the MCP servers + local tools; also a chat CLI and pipeline tasks |
| [`agent_data/`](agent_data/README.md) | Cat profiles, reference images, activity labels, zones (incl. decoys) |
| [`camera_sim/`](camera_sim/README.md) | Simulated camera: replays archived frames through YOLO and pushes the camera's message format, for `--live` runs without the Pi |
| [`replay_labels.py`](replay_labels.py) | Run the pipeline over a YOLO label folder instead of detector output (used to make ground-truth exports) |
| [`tests/`](tests/) | `TrackingRun` wiring, heartbeat, `.env` loading |

Outputs go to the repo-root `output/` (git-ignored): `tracking_runs/<run>/` per run,
`agent_runs/<stamp>-validation/trace.md` per supervisor call, `memory_runs/` per memory LLM
call. Memory stores live in `agent/memory/data/` (git-ignored; `MEMORY_STORE_DIR` overrides).

## Running

```bash
pip install -r requirements.txt   # from the repo root

# Full loop from a recorded camera stream: correctors + supervisor verdicts + memory + charts
python -m agent.main --view-id 3 --start 2026-10-28-16-33 --end 2026-10-28-16-38 \
  --replay demo/data/2026-10-28-16/raw_detections.jsonl --validate --memory

# Tracking + escalation only (no LLM calls)
python -m agent.main --view-id 3 --start 2026-10-28-16-33 --end 2026-10-28-16-38 \
  --replay demo/data/2026-10-28-16/raw_detections.jsonl

# Straight from MySQL (read-only), or live from the camera / camera_sim on :8000
python -m agent.main --view-id 3 --start 2026-10-28-16-33 --end 2026-10-28-16-40 --replay-db --validate --memory
python -m agent.main --view-id 3 --start 2026-10-28-16-33 --end 2026-10-28-16-59 --live --port 8000

# Per-frame process visualization + MP4 for the second cat
python -m agent.main --view-id 3 --start 2026-10-28-16-33 --end 2026-10-28-16-38 \
  --replay demo/data/2026-10-28-16/raw_detections.jsonl --process-viz --cat-idx 1
```

| Flag | Default | Meaning |
|---|---|---|
| `--live` / `--replay PATH` / `--replay-db` | (one required) | detection source |
| `--start`, `--end` | required | minute precision `YYYY-MM-DD-HH-MM`; window is `start <= t < end + 60s` |
| `--validate` | off | send every escalation to the supervisor agent and write validated outputs |
| `--memory` | off | short-term + long-term memory (falls back to a stub LLM without `MODEL_API_KEY`) |
| `--bootstrap-frame PATH` | none | one frame for the memory's first view description |
| `--plot` / `--report` | on | `original.png` / `revised.png` / `validated.png` charts; `report.md` session summary |
| `--process-viz`, `--cat-idx`, `--fps`, `--no-video` | off | per-frame tracker visualization |
| `--min-conf` | 0.5 | keep detections strictly above this confidence |
| `--heartbeat-seconds`, `--heartbeat-at-end` | 300 s, on | `H1` quiet-window spot checks |
| `--out` | `output/tracking_runs` | run base directory |

The supervisor uses the MCP servers named in `.env` (`IMAGE_MCP_URL`, `TRACKING_MCP_URL`,
`CAMERA_MCP_URL`; an empty value leaves one out) plus the in-process `local__*` tools; if a
server is down it carries on with the rest. Edge cases of each stage:
[`tracking/README.md`](tracking/README.md).

## Tests

```bash
python -m pytest agent/tests agent/escalation/tests agent/tracking/tests \
  agent/memory/tests agent/validation/tests -q        # no MySQL, no API key
(cd agent && python -m pytest supervisor_agent/tests camera_sim/tests -q)   # fake model client
```
