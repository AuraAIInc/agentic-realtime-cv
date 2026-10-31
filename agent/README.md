# agent

Tracking pipeline plus the supervisor agent. `main.py` runs the tracking
correctors over the live camera stream (or a replay); `supervisor_agent/`
is the tool-calling agent that inspects tracks, frames and images.

Run everything in this package from the repo root (`python -m agent.main`,
`python -m pytest agent/...`).

## Modules

| Path | What it is |
|---|---|
| [`main.py`](main.py) | Entry point: ingest camera detections, run both correctors, write charts + run dir |
| [`tracking/`](tracking/README.md) | `TrackCorrector` + `PeriodCorrector` (MySQL-free), run store, charts, process viz |
| [`escalation/`](escalation/README.md) | Escalation trigger vocabulary (`T1`–`T7` / `P1`–`P7`) and thresholds |
| [`supervisor_agent/`](supervisor_agent/README.md) | The Muse Spark agent (chat, tasks, skills, local tools) |
| [`agent_data/`](agent_data/README.md) | Cat profiles, reference images, labels, zones |
| [`output/`](output/) | Run outputs: `tracking_runs/` (tracker) and `agent_runs/` (agent) |

## Tracking runs

```bash
pip install -r agent/requirements.txt

# Live: serve the ingest endpoint the camera pushes to (stops at --end)
python -m agent.main --view-id 3 \
  --start 2026-10-27-14-00 --end 2026-10-27-14-05 \
  --live --port 8000

# Replay: re-run a recorded stream through the same pipeline
python -m agent.main --view-id 3 \
  --start 2026-10-27-14-00 --end 2026-10-27-14-05 \
  --replay agent/output/tracking_runs/<run>/stream.jsonl --plot

# Replay with per-frame tracking-process visualization + MP4
python -m agent.main --view-id 3 \
  --start 2026-10-27-14-00 --end 2026-10-27-14-05 \
  --replay agent/output/tracking_runs/<run>/stream.jsonl \
  --process-viz --cat-idx 0
```

`--start`/`--end` are minute-precision (`YYYY-MM-DD-HH-MM`); the window is
`start <= t < end + 60s`, so the `--end` minute is fully included. Every run
writes `agent/output/tracking_runs/<run>/` with `stream.jsonl`,
`revised.jsonl`, `periods.jsonl`, `escalations.jsonl`, `original.png` /
`revised.png` (unless `--no-plot`), and `run.json`. Details and edge cases:
[`tracking/README.md`](tracking/README.md).

## Tests

```bash
python -m pytest agent/escalation/tests agent/tracking/tests -q  # no MySQL, no API key
python -m pytest agent/supervisor_agent/tests -q                  # fake model client
```
