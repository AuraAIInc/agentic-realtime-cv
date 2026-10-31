# Spatial-temporal long-term memory

Plain-language memory for the tracking pipeline: what each part of the camera
view looks like, which cat is usually where, and what the daily routine is.
The store is a human-readable YAML file; the direct consumer of `invoke` is
the `agent/escalation` module. Full design:
`documents/SPATIAL_TEMPORAL_MEMORY_DESIGN.md`.

## Layout

- `memory_config.yaml` / `memory_config.py` — checked-in defaults + `MEMORY_*`
  env overrides (same pattern as `escalation_config.py`).
- `models.py` — `MemoryEvent`, `MemoryQuery`, `MemoryContext`, `MemoryHit`.
- `spatial.py` — grid math (point/box -> cell, parent/children, neighbors).
- `temporal.py` — bucket math (time -> slot/hour/day/week ids, history windows).
- `store.py` — YAML load/save (atomic tmp+rename), sparse units, retention.
- `worker.py` — `MemoryWorker` thread + FIFO job queue, promote coalescing,
  `drain()` / `shutdown()`.
- `llm_client.py` — Meta Model API wrapper + `StubLLMClient` for offline tests.
- `prompts.py` — store-update, promote-summary, environment-bootstrap prompts.
- `long_term.py` — `LongTermMemory`: store/invoke/promote/reset/tick/
  semantic_search.
- `short_term.py` — interface + in-memory stub (full build is a later phase).
- `data/` — durable runtime YAML per view (`view_<id>_long_term.yaml`,
  git-ignored). Never under `agent/output/` (temp-only).

## Usage

```python
from agent.memory import LongTermMemory, MemoryConfig, MemoryEvent, MemoryQuery

mem = LongTermMemory(config=MemoryConfig.from_yaml())
mem.store(MemoryEvent(time=now, view_id=3, box=(500, 300, 600, 400),
                      cat_id=25, period_id=12, kind="period",
                      text="Qb naps on the dark sofa."))
ctx = mem.invoke(MemoryQuery(time=now, box=(500, 300, 600, 400)))
print(ctx.brief)          # one-line summary of what was found
print(ctx.spatial)        # nearby cells: environment / background / contrast
print(ctx.temporal)       # same-slot history + hour / day / week routine
mem.shutdown()            # drain the queue, then stop the worker
```

Writes enqueue and return immediately; reads are synchronous:

| Method | Behavior |
|---|---|
| `store(event, image=None)` | Enqueue a `store` job; stamps `last_store` at arrival. First `image` auto-bootstraps vision ahead of itself. |
| `bootstrap(image)` | One vision call filling L2 + L1 environment/background/contrast; no-op once bootstrapped. |
| `invoke(query)` | Sync locked read: neighbor cells + same-slot history + hour/day/week. |
| `invoke_for_escalation(esc, box=None)` | Sync read tuned to the escalation code family (see below). |
| `semantic_search(text, top_k=5)` | Keyword search over stored units (embeddings are a later phase). |
| `tick(now)` | Enqueue `promote` after `idle_seconds` without a store; cheap, call per message. |
| `promote(force=False)` | Bottom-up compression over dirty subtrees (`force`: whole tree). |
| `drain()` / `shutdown()` | Block until the queue is empty; `shutdown` then stops the thread. |
| `reset(keep_backup=True)` | Drain, clear tree + YAML (timestamped `.bak`), keep the worker running. |

## Config

Defaults live in `memory_config.yaml`:

- Spatial: L0 `8x6` cells (100 px), L1 `4x3` zones (2x2 cells each), L2 full view.
- Temporal: `slot_5min` (events) -> `hour` (activity mix) -> `day` (routine) ->
  `week` (habits/change). Retention 7 / 30 / 180 days; week kept forever.
- Retrieval: neighbor radius 1, 7-day same-slot history, semantic top-k 5.
- Promotion: idle 300 s, `min_pattern_support` 3, `stable_weight_cap` 50,
  `flip_threshold` 3, `recent_kept` 5.
- Worker: warn past depth 100, promote coalescing on.
- LLM: `roles.memory` from `supervisor_agent/models.json` (falls back to
  `roles.agent`, then `muse-spark-1.3`), max 8 units per call.

Every knob has a `MEMORY_*` override, e.g. `MEMORY_IDLE_SECONDS=60`,
`MEMORY_MIN_PATTERN_SUPPORT=5`, `MEMORY_STORE_DIR=/var/lib/memory`.

Promotion keeps dominant patterns only: a pattern needs `min_pattern_support`
observations to enter stable text, and stable text flips only after
`flip_threshold` consecutive contradicting batches (judged by the LLM,
counted programmatically). One-offs stay in the bounded `recent` buffer.

## Worker lifecycle

`MemoryWorker` owns one daemon thread and a FIFO queue (`store`, `promote`,
`bootstrap`). The worker thread is the only writer of the tree and the YAML;
reads take an `RLock`, so `invoke` never waits on LLM jobs. Rules:

- Consecutive `promote` jobs coalesce into one; `bootstrap` jumps ahead of
  queued stores.
- A failing job is logged and dropped; an LLM error degrades that unit to
  pending-observation mode (raw text kept in `recent`). The worker never dies
  and never raises into the ingest loop.
- `TrackingRun` calls `tick()` per message, `store()` on period close and on
  every emitted escalation, and `shutdown()` in `finish()`, so nothing is lost.
  Run with `--no-memory` to skip memory entirely (no `agent.memory` import),
  or `--bootstrap-frame frame.jpg` to seed vision descriptions from one image.
- Tests use `StubLLMClient` (deterministic, offline) + `drain()` between
  phases; `shutdown()` in a fixture teardown.

## Escalation queries

`invoke_for_escalation(esc)` maps the code to a retrieval family:

| Codes | Family | Context returned |
|---|---|---|
| T1, T2, T5 | identity | Hit cell + neighbors: environment, background, contrast |
| T4, T6 | motion | Wider neighbor radius + current slot/hour |
| T3, T7 | continuity | Current slot + hour continuity |
| P1, P2, P5 | routine | Day/hour routine + same-slot history |
| P3 | conflict | Same-slot history (which cat is usually here now) |
| P4 | occlusion | Environment + occluder/contrast notes |
| P6, P7 | merge | Gap-span buckets + place routine |

The returned `brief` is prefixed with `[<code> <family>]`. Escalations carry
no boxes, so period events contribute the spatial anchor (median box of the
period's detections, best-effort).

## Tests

```sh
.venv/bin/python -m pytest agent/memory/tests -q
```

Offline (stubbed LLM, drained worker): grid math, bucketing, YAML round-trip,
neighbor/history windows, keyword ranking, worker ordering + coalescing +
drain/shutdown, invoke-during-store smoke, compression and weighting rules,
bootstrap markers, reset, escalation families, and the `test_replay.py`
integration (synthetic fixture + one real run read-only through
store -> tick -> promote -> invoke).
