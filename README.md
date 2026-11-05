# agentic-realtime-cv

**An AI supervisor that checks a tiny edge detector only when it is unsure, and remembers what
it verified.**

A small YOLO model runs all the time on a Raspberry Pi camera and watches two look-alike
white cats. Two tracking correctors clean up its detections frame by frame and visit by
visit. Whenever a corrector has to guess (a jump, a near-tied vote, a weak relabel), it raises
an *escalation*. A supervisor agent on Meta's **Muse Spark** reviews each escalation with
image tools (stored frames, crops, reference photos of each cat), returns a verdict, and the
verdict is applied to the tracked rows. Short-term and long-term memory keep what happened
in plain language, and every run is scored against hand-validated ground truth.

Built for the Meta Global AI Developer Hackathon 2026 (Oct 27 – Nov 4). Code that existed
before the event is kept apart in [`existing_work_before_hackathon/`](existing_work_before_hackathon/).

## How it works

```
 camera (YOLO, ~2.7 fps)
        │ detections: {create_time, detections:[{cat_id, box, confidence}]}
        ▼
 ┌─────────────────┐  frame-level fixes   ┌──────────────────┐  visit-level votes
 │ TrackCorrector  │ ───────────────────► │ PeriodCorrector  │ ──► periods (visits)
 │ relabel,        │  revised rows        │ open/close visits│
 │ interpolate,    │                      │ score each cat   │
 │ split on jumps  │                      │                  │
 └───────┬─────────┘                      └────────┬─────────┘
         │ T1–T7 escalations                       │ P1–P8 escalations (+ H1 heartbeat)
         └──────────────────┬──────────────────────┘
                            ▼
              ValidationWorker (background thread; close cases merge into one call)
                            │ prompt + JSON schema + memory brief
                            ▼
              Supervisor agent — Muse Spark ── tools: images__get_image, local__get_frame (crops),
                            │                          local__find_references, local__get_track_box, …
                            ▼ verdict + rationale per escalation
              apply verdicts ──► validated.jsonl / validated.db
                            │
       ┌────────────────────┴──────────────────────┐
       ▼                                           ▼
 Short-term day memory                    Long-term spatial-temporal memory
 (every visit, episode, verdict)          (only high-confidence, LLM-labeled activity)
                            │
                            ▼
              evaluation vs ground truth  ──►  report
```

Stage-by-stage detail with code pointers: [`documents/end-to-end-flow.md`](documents/end-to-end-flow.md).

## Run the demo

The demo replays one real window — camera view 3, **2026-10-28 16:33–16:38** — through the
entire pipeline, using only data committed to this repo ([`demo/`](demo/README.md): 989
frames, the detector's 686 raw boxes, ground-truth labels). It needs no camera, database or
servers: it starts its own image MCP server on localhost.

```bash
python3 -m venv .venv && source .venv/bin/activate     # Python >= 3.10
pip install -r requirements.txt
echo "MODEL_API_KEY=<your Meta Model API key>" >> .env  # or export MODEL_API_KEY=...

python run_demo.py              # full loop: ~11 min, about $0.35 of Muse Spark calls
python run_demo.py --offline    # no key needed: tracking, escalation, memory (stub), evaluation; ~10 s
```

A full run on 2026-11-04 raised 12 escalations (T2, T4, T6, P2, P5, P6, P8), resolved all
of them in 4 merged supervisor calls (43 images, $0.33, 645 s), and against ground truth
(IoU ≥ 0.5) moved the output from the raw detector to the validated rows as follows:

| Stage | Precision | Recall | F1 | Identity accuracy |
|---|---|---|---|---|
| Raw YOLO | 0.905 | 0.329 | 0.482 | 0.907 |
| After TrackCorrector | 0.877 | 0.298 | 0.445 | 0.878 |
| After supervisor verdicts | **0.988** | **0.335** | **0.501** | **0.989** |

Recall stays low because the detector has no box for the cat in most frames and the
evaluation does not count the tracker's interpolated boxes; what the supervisor fixes is
precision and identity. LLM
answers vary, so expect the numbers to move a little between runs.

What it runs, in order:

| # | Stage | What happens |
|---|---|---|
| 1 | Image MCP server | `image_server/mcp_image_server` over the demo frames, `127.0.0.1`, random token |
| 2 | TrackCorrector | Frame-level identity fixes, interpolation, jump splits; raises `T*` escalations |
| 3 | PeriodCorrector | Groups rows into visits, votes each visit's cat; raises `P*` escalations |
| 4 | Supervisor agent | Muse Spark reviews every escalation (merged when close in time), verdicts applied → `validated.jsonl` |
| 5 | Memory | Short-term day log + long-term spatial-temporal memory, fresh store per run; period activity labeled by a vision call |
| 6 | Evaluation | Raw → revised → validated vs ground truth (P/R/F1, identity accuracy), as given and with the documented GT audit |
| 7 | Final report | `output/demo/<timestamp>/DEMO_REPORT.md` |

Output (`output/demo/<timestamp>/`, git-ignored):

```
DEMO_REPORT.md            the final report: periods, escalations + verdicts + rationales, spend,
                          memory excerpts, evaluation tables, timelines
tracking_runs/<run>/      stream / revised / periods / escalations / validations / validated JSONL,
                          validated.db, original|revised|validated.png, report.md, run.json
evaluation/as_given/      <run>_eval.md + metrics.json + charts (incl. ground-truth timeline)
evaluation/audited/       the same with evaluation/gt_audit.yaml applied
memory/                   view_3_short_term_2026-10-28.md/.json, view_3_long_term.yaml
logs/                     pipeline, evaluation and image-server logs
```

Supervisor traces (every prompt, tool call, image and token) land in `output/agent_runs/`,
memory LLM traces in `output/memory_runs/`; the report links to them. Options:
`--no-memory`, `--memory-dir DIR` (keep memory across runs, so the second run recalls the
first), `--process-viz` (per-frame tracker video), `--report-only OUT_DIR` (rebuild the
report). The demo reads only `MODEL_API_KEY` from your `.env`; everything else it needs is in
an isolated config it writes per run.

## Repo layout

| Folder | What it is |
|---|---|
| [`run_demo.py`](run_demo.py) | The end-to-end demo described above |
| [`demo/`](demo/README.md) | Demo data for the 16:33–16:38 window and the script that rebuilt it |
| [`agent/`](agent/README.md) | The hackathon pipeline: `main.py` (tracking run), `tracking/` (correctors), `escalation/` (codes, prompts, verdict schemas), `validation/` (worker, batcher, verdict applier, activity labeler), `memory/` (short-/long-term), `supervisor_agent/` (Muse Spark agent + CLI), `agent_data/` (cat profiles, references, labels, zones), `camera_sim/` (simulated camera) |
| [`evaluation/`](evaluation/README.md) | Scores a run against ground truth (`evaluate.py`), GT audit, error attribution, offline re-apply |
| [`image_server/`](image_server/README.md) | `mcp_image_server` (MCP over stored frames, SSD + S3), `fetch_hour_images.py` (hour → YOLO dataset), `grounding_truth/` (SAM 3.1 / Spark labeling for ground truth) |
| [`camera_server/`](camera_server/camera_v3/README.md) | `camera_v3` on the Pi: always-on YOLO, WebSocket stream, MCP control server (capture, zoom, high-res, detect, schedules) |
| [`existing_work_before_hackathon/`](existing_work_before_hackathon/README.md) | Pre-hackathon ingest server, MySQL-backed correctors, tracking MCP server |
| [`documents/`](documents/README.md) | Design notes, experiment reports, hackathon report and demo video material |
| `output/` | Every run's artifacts (git-ignored) |

## Models

All LLM use goes through Meta models, set per role in
[`agent/supervisor_agent/models.json`](agent/supervisor_agent/models.json) (provider
endpoint, models, token prices). The agent calls the Meta Model API
(`https://api.meta.ai/v1`, OpenAI-compatible Responses endpoint) with `MODEL_API_KEY`; the
`openai` package is only the HTTP client.

| Role | Model | Used for |
|---|---|---|
| `agent` | Muse Spark (`muse-spark-1.3`) | Supervisor: escalation review, vision, tool calls; chat CLI |
| `memory` | Muse Spark | Period activity labels, memory summaries |
| `segmentation` | SAM 3.1 | Ground-truth labeling (`image_server/grounding_truth/`) |

The `voice`, `creative` and `open_weight` roles in `models.json` are configured but not used
by any code yet.

The camera's edge detector is YOLO (Ultralytics), set in `camera_server/camera_v3/models.json`.

## Running the live system

For the deployed setup each component runs on its own machine and reads its own `.env`
(copy [`.env.example`](.env.example); it lists every variable by component).

```
 Raspberry Pi                         ingest host                         anywhere
┌────────────────────┐  detections  ┌──────────────────────────────┐
│ camera_server      │ ──WebSocket─►│ detection_ingest_server.py   │  (or agent.main --live)
│  camera_v3 (YOLO)  │    :8000     │  correctors ──► MySQL        │
│  MCP :8765         │              │  tracking_mcp MCP :10002     │
└─────────▲──────────┘              └──────────────▲───────────────┘
          │      ┌───────────────────────────┐     │
          │      │ image_server :10001 (MCP) │     │
          │      └─────────────▲─────────────┘     │
          └──────── MCP ───────┴─────── MCP ───────┤
                               ┌───────────────────┴───────────────┐
                               │ agent (supervisor_agent, Muse Spark)│
                               └───────────────────────────────────┘
```

| Component | Install | Run | Port |
|---|---|---|---|
| Camera (Pi) | `cd camera_server && uv sync` | `uv run python -m camera_v3` | MCP 8765 |
| Image server | `pip install -r image_server/requirements-mcp.txt` | `cd image_server && python -m mcp_image_server` | MCP 10001 |
| Ingest server + tracking MCP | `pip install -r existing_work_before_hackathon/requirements.txt` | `cd existing_work_before_hackathon && python detection_ingest_server.py` | WS 8000, MCP 10002 |
| Pipeline, live | `pip install -r requirements.txt` | `python -m agent.main --view-id 3 --start <min> --end <min> --live --validate --memory` | WS 8000 |
| Pipeline, from MySQL | same | `python -m agent.main --view-id 3 --start <min> --end <min> --replay-db --validate --memory` | — |
| Simulated camera | `pip install -r agent/camera_sim/requirements.txt` | `cd agent && python -m camera_sim --start 2026-10-28-16-33-00 --end 2026-10-28-16-59-00` | pushes to :8000 |
| Agent chat | same | `cd agent && python -m supervisor_agent "Is any cat on camera right now?"` | — |
| Evaluate a run | same | `python evaluation/evaluate.py output/tracking_runs/<run> --audit evaluation/gt_audit.yaml` | — |

Times are minute precision (`YYYY-MM-DD-HH-MM`) and the `--end` minute is included.
If an MCP server is down, the agent warns and carries on with the others.

## Tests

No MySQL, network or API key needed (fake model clients):

```bash
python -m pytest agent/tests agent/escalation/tests agent/tracking/tests \
  agent/memory/tests agent/validation/tests evaluation/test_metrics.py -q
(cd agent && python -m pytest supervisor_agent/tests camera_sim/tests -q)
(cd image_server && python -m pytest mcp_image_server grounding_truth test_fetch_hour_images.py -q)
(cd existing_work_before_hackathon && python -m pytest model/test_escalation.py tracking_mcp -q)
```

## More docs

- Pipeline: [`documents/end-to-end-flow.md`](documents/end-to-end-flow.md), correctors:
  [`documents/how-track-correctors-work.md`](documents/how-track-correctors-work.md),
  escalation codes: [`agent/escalation/README.md`](agent/escalation/README.md)
- Memory: [`agent/memory/README.md`](agent/memory/README.md),
  [`documents/spatial-temporal-memory-design.md`](documents/spatial-temporal-memory-design.md)
- Results: [`documents/final-model-performance-20261104.md`](documents/final-model-performance-20261104.md),
  hackathon report: [`documents/report.md`](documents/report.md)
- Everything else: [`documents/README.md`](documents/README.md)
