# End-to-End Flow: detection → tracking → escalation → validation → memory

How one `agent.main` run turns raw YOLO detections into validated, memory-backed
cat-visit records. Code pointers are relative to the repo root.

```
detections ──► ingest ──► TrackCorrector ──► PeriodCorrector ──► charts/run.json
(live/sim/        (window,        (frame scope)      (period scope)
 file/db)         min-conf)            │                    │
                                       │ escalations        │ escalations
                                       ▼                    ▼
                              ValidationWorker queue ──► supervisor agent ──► verdicts
                              (background thread)      (images + tracking      (validations.jsonl,
                                                        tools, trace.md)       validated.jsonl/.db)
                                       │                    ▲
                                       ▼                    │ past context
                              LongTermMemory ◄──────────────┘
                       (combined + labeled period activity)
```

## 0. Detection sources (4 modes, one message shape)

Every mode feeds the same `TrackingRun.process_message()`
([agent/main.py:534](../agent/main.py)). A message is
`{view_id, create_time, detections: [{cat_id, box, confidence}], ...}` with
`create_time` in image-name format (`%Y-%m-%d-%H-%M-%S-%f`).

| Mode | Driver | Source |
| ---- | ------ | ------ |
| `--live` | `run_live` ([main.py:797](../agent/main.py)) | websocket server on `:8000`; the camera pushes until `--end` or Ctrl-C. Raw messages are also recorded to `stream.jsonl`. |
| `--replay PATH` | `run_replay` ([main.py:816](../agent/main.py)) | a recorded camera-format `stream.jsonl` |
| `--replay-db` | `run_replay_db` ([main.py:842](../agent/main.py)) | MySQL, **read-only** single `SELECT` from `images`/`detection` ([db_stream.py](../agent/tracking/db_stream.py)). No intermediate file; the shared DB is never written. |
| camera_sim | `agent/camera_sim` | simulated camera pushing the same message shape over websocket (see below) |

`camera_sim` ([agent/camera_sim/main.py](../agent/camera_sim/main.py)) replays
archived frames through the same `CameraService` interface as the real Pi
camera, so `--live` runs can be tested without hardware:
`python -m camera_sim --start 2026-10-28-16-33-00 --end 2026-10-28-16-59-00`
on one side, `agent.main --live` on the other.

## 1. Ingest: window filter + confidence gate

`process_message()` per message ([main.py:534](../agent/main.py)):

1. Shape check (`is_camera_message`), view match, timestamp parse.
2. Window filter: `start <= t < end + 60s` (the whole `--end` minute counts).
3. `build_img_detected()` keeps detections **strictly above** `--min-conf`
   (default 0.5) whose cat id is known for this view (from `cats.yaml`
   `tracking_ids`); unknowns/low-conf/malformed boxes are counted, not kept.
4. Empty frames are ingested too — they advance buffer timeouts and close
   periods.

## 2. TrackCorrector: frame scope (tracks)

`TrackingRun._ingest()` ([main.py:572](../agent/main.py)) calls
`TrackCorrector.process_images()` ([agent/tracking/track_corrector.py](../agent/tracking/track_corrector.py)),
which per frame: relabels detections by spatial continuity, interpolates
across gaps, commits revised rows (or tombstones for retractions) through the
`FileSink` into `stream.jsonl` / `revised.jsonl`.

When a frame-level choice is uncertain it emits a **T-code escalation**
(full mechanics: [how-track-correctors-work.md](how-track-correctors-work.md)):

| Code | Name | Meaning |
| ---- | ---- | ------- |
| T1 | multi_match_relabel | detection fits 2+ cats' histories; first match won |
| T2 | weak_rule_relabel | relabeled by center distance alone, no IoU support |
| T3 | duplicate_tie | duplicate boxes killed with near-tied confidence |
| T4 | jump_reset | new box missed the anchor (base); buffer wiped, visit restarts |
| T5 | target_fallback | no overlap with base and nothing confident; guessed |
| T6 | teleport_split | committed box jumped > 300 px; period split |
| T7 | long_synthesis | committed a long run of interpolated-only frames |

## 3. PeriodCorrector: period scope (visits)

Each committed revised row flows into `PeriodCorrector.process_detection()`
([agent/tracking/period_corrector.py](../agent/tracking/period_corrector.py)),
which opens/closes **periods** (continuous visits), scores candidate cats per
period (avg certainty, original-frame count, area, bonuses — x0.2 penalty
under 10 original frames), and awards each period to a winner. Verdicts land
in `periods.jsonl`; every close also logs a one-line `period closed:` summary.

Shaky verdicts emit **P-code escalations**:

| Code | Name | Meaning |
| ---- | ---- | ------- |
| P1 | close_vote | top-two cat scores near-tied |
| P2 | weak_winner | winner barely above min score, or short (x0.2) period |
| P3 | conflict_reassign | top cat taken by an overlapping period; got 2nd choice |
| P4 | long_unknown | assigned no cat but the period is long |
| P5 | low_quality | winner's mean conf low or mostly missing frames |
| P6 | merge_candidate | period may connect to the same cat's previous period |
| P7 | static_link | both ends static and overlapping; sleeping cat split by a long gap |

## 4. Escalation: numbering, prompts, output schemas

Both correctors share one run-wide counter, so names number continuously
across scopes: `E1-T7, E2-T6, E3-P5, ...` (`E<seq>-<code>`).
`emit()` ([agent/escalation/escalation.py](../agent/escalation/escalation.py))
appends to the corrector's list and logs three records: the escalation JSON,
the reviewer prompt, and the expected output schema.

After every frame (and after `finish_up`), `TrackingRun`
drains new items via `_EscCollector` (survives the correctors' list rebinds),
appends each to `escalations.jsonl`, stores it to memory, and — only with
`--validate` — enqueues it to the validation worker
([main.py:608](../agent/main.py)).

Each code has its own prompt builder (`build_<code>_prompt` in
[agent/escalation/prompting.py](../agent/escalation/prompting.py)):
plain-language task description, full context (original detection, proposed
correction, uncertain point, alternatives), resources (period ids, key frames
~1/min with boxes and pick reasons), and the involved cats' appearance
profiles. The small on-camera YOLO is explicitly framed as a hint, not
evidence. Worked examples for every code:
[escalation-prompt-examples-20261101.md](escalation-prompt-examples-20261101.md).

Each code also has a slim pydantic review model (`<Code>Review` in
[agent/escalation/review_schema.py](../agent/escalation/review_schema.py)):
one `verdict` enum plus at most one qualifier field (e.g. `cut_time`,
`chosen_cat_id`), keeping LLM generation short. Verdicts:

- T1: confirm_assigned / reassign_candidate / reject_box
- T2: confirm / keep_original / reassign_other / drop
- T3: confirm_survivor / keep_killed / keep_both
- T4: confirm_reset / link_continuation / drop_box_as_outlier
- T5: commit_target / wait_for_stronger / close_visit
- T6: confirm_split / merge_visits / drop_box
- T7: keep_synthesis / cut_at_gap_start / shorten_gap
- P1: confirm_winner / award_runner_up / split_visit
- P2: keep / drop / merge_neighbor
- P3: confirm_reassign / restore_wanted / mark_unknown
- P4: assign_cat / split_fragments / confirm_unknown
- P5: keep / drop / trim_core / reassign
- P6, P7: merge / keep_separate

## 5. Validation: queue → supervisor agent → verdicts

`ValidationWorker` ([agent/validation/worker.py](../agent/validation/worker.py),
`--validate`) is a background thread consuming the escalation queue in
**merge batches**: escalations within a 1-minute window merge into one joint
supervisor-agent call sharing evidence across cases (see
[escalation-merge-design.md](escalation-merge-design.md)); lone escalations
validate alone.

Per batch (`_validate_batch`) or single (`_validate_one`):

1. Optional memory context: `memory.invoke_for_escalation()` returns a brief
   (≤600 chars) appended to the prompt as possibly-stale past context.
2. `build_validation_query()` (or `build_merged_query()` for a batch) =
   full escalation prompt(s) + memory brief(s) + "return ONLY a JSON object"
   instruction; the review JSON schema is passed as the LLM's
   structured-output format.
3. `MCPAgent.run()` with tools restricted to `images__get_image`,
   `images__list_images`, `tracking__get_period_detail`, tagged `validation`.
   The agent inspects key frames and period rows, then answers with the review
   JSON. Every run dumps `output/agent_runs/<stamp>-validation/trace.md`
   (message table, system prompt, LLM I/O, tool I/O, tokens/cost/time).
4. `parse_review()` validates the answer against the pydantic model;
   failures become `{"verdict": "error"}` — never an exception. Every review
   carries a short `rationale` for the decision, logged and kept with the op.
5. The verdict (+ rationale) is attached to the short-term day's episode, and
   the op `{escalation, review}` is journaled to `validations.jsonl`.
   Verdicts never store to long-term memory directly — they reach durable
   memory through the labeled period activity they confirm or correct.

If the supervisor backends are unreachable, the worker logs once and every
queued escalation validates to `error` (tracked, not lost).

## 6. Apply: verdicts → validated_detection

At `finish()`, `validator.stop()` drains the queue and `materialize()`
([agent/validation/apply.py](../agent/validation/apply.py)) applies ops to the
revised rows:

- Frame ops match rows by exact timestamp (+ cat); period ops match by
  `period_id`. Untouched rows pass through unchanged.
- Executable verdicts rewrite rows (relabel, drop, cut/trim spans, reassign
  winners); each touched row records `validated_by` (escalation name) and
  `verdict`.
- Verdicts that cannot be executed precisely (splits, merges into an unnamed
  neighbor, free-text-span trims) **keep the rows and record a note** instead
  of guessing.

Outputs: `validated.jsonl` + `validated.db` (per-run SQLite,
`validated_detection` table). The shared MySQL stays read-only; when a MySQL
copy is wanted, `MYSQL_VALIDATED_DDL` in `apply.py` is the table to create.

## 7. Memory: observe everything, promote in background

Memory (`--memory`) runs on its own worker thread; every hook is best-effort
and never breaks the run. Design: [spatial-temporal-memory-design.md](spatial-temporal-memory-design.md).

**Long-term** (`LongTermMemory`) keeps only high-confidence knowledge, and
it attaches to **period activity, never escalations**: closed periods of the
same cat with gaps under `activity_combine_gap_minutes` (default 5.0,
`agent/memory/memory_config.yaml`) merge into one `ActivityRecord`; each
member period's key frames are judged by one vision LLM call
(`PeriodLabeler`, own background thread with an images-MCP connection)
against the activity set in `agent/agent_data/config/labels.yaml`, with the
tracked box per frame disambiguating multi-cat views. A record stores only
when the detector and every member label are all strong (min confidence ≥
0.7) — anything weaker stays in short-term day context. Every record carries
its original image names, the detection snapshot (box, cat, activity,
periods), and plain-language text (which cat, what span, which area box,
what activity, looks, background); every zone describes cat appearance,
activities, behavior, and background view. `tick(now)` per frame; idle ticks
and explicit `promote()` roll stats bottom-up and rewrite parent summaries
via the LLM (StubLLM offline fallback when no model key).

**Short-term** (`DailyShortTermMemory`) remembers every period activity across
one day (all confidences): each **combined** activity record (member periods,
span, area box, LLM-judged activity, looks, background) with detections + key
frames, plus tracker episodes and their verdicts (+ rationales), in a
plain-language `view_<id>_short_term_<day>.md` re-rendered on every change.
New sessions open a new section; midnight (event time) merges the day into
long-term memory (curated summaries promoting into the top spatial levels and
the day bucket); session end merges the session the same way, marked so
nothing merges twice.

The loop closes through activity: tracking → escalation → verdict (corrects
the validated rows) → labeled period activity → memory → future validation
context.

## 8. Run artifacts (`output/tracking_runs/<run>/`)

`stream.jsonl` (canonical detections; + raw messages in `--live`),
`revised.jsonl` (committed rows + tombstones), `periods.jsonl`,
`images.jsonl`, `escalations.jsonl`, `validations.jsonl` / `validated.jsonl` /
`validated.db` (with `--validate`), `original.png` / `revised.png` timeline
charts (with `--plot`, escalation names under red lines), `process_frames/` +
`process.mp4` (with `--process-viz`), and `run.json` (args, window, counts
incl. `escalations_by_code` / `validations_by_verdict`).

## 9. Commands (copy-paste, from repo root)

```bash
# Full loop on the test window: replay DB + validate + memory + charts
python -m agent.main --view-id 3 \
  --start 2026-10-28-16-33 --end 2026-10-28-16-40 \
  --replay-db --validate --memory --plot

# Tracking + escalation only (no agent, no memory)
python -m agent.main --view-id 3 \
  --start 2026-10-28-16-33 --end 2026-10-28-16-40 \
  --replay-db --plot

# With per-frame process visualization + MP4 for the 2nd cat
python -m agent.main --view-id 3 \
  --start 2026-10-28-16-33 --end 2026-10-28-16-40 \
  --replay-db --process-viz --cat-idx 1 --plot

# Live ingest from camera_sim (two terminals)
python -m camera_sim --start 2026-10-28-16-33-00 --end 2026-10-28-16-59-00
python -m agent.main --view-id 3 \
  --start 2026-10-28-16-33 --end 2026-10-28-16-59 \
  --live --validate --memory --plot
```

Every entry point loads the repo-root `.env` first. Log format everywhere:
`timestamp - [LEVEL] file:lineno: message`.

## 10. `main.py` code map (where each stage lives)

`main.py` is a thin orchestrator: `main()` parses args and builds one
`TrackingRun`, a mode driver feeds it messages, and `TrackingRun` runs the
whole pipeline per message.

**Ingest — `run_replay_db()` → `process_message()`**

- [main.py:842](../agent/main.py) `run_replay_db()` connects to MySQL
  (read-only via `agent/tracking/db_stream.py`) and streams camera-format
  messages for the view in the window `start <= t < end+60s`, calling
  `run.process_message(msg)` per message. (The other two drivers: `run_live()`
  serves a websocket at :8000 for camera pushes; `run_replay()` reads a
  `stream.jsonl` file.)
- [main.py:534](../agent/main.py) `process_message()` filters (valid message?
  right view? parseable time? in window?), drops detections at/below
  `--min-conf` (default 0.5) via `build_img_detected()`, then calls `_ingest()`.

**Corrector — `_ingest()`**

- [main.py:572](../agent/main.py) `_ingest()` runs the two-stage loop:
  `TrackCorrector.process_images()` (frame scope: relabeling, interpolation,
  jump splits) yields committed revised rows, each of which goes into
  `PeriodCorrector.process_detection()` (period scope: open/close visits,
  winner votes). A `FileSink` writes `stream.jsonl` / `revised.jsonl` /
  `periods.jsonl` / `images.jsonl` as rows flow through. Both correctors are
  constructed in [main.py:396](../agent/main.py), sharing one escalation
  counter so names (`E1-T7`, `E2-T6`, …) number continuously across frame and
  period scope.

**Escalation — inside the correctors + `_persist_new_escalations()`**

- The correctors themselves raise escalations (T1–T7/P1–P7) by calling
  `emit()`, which appends to the corrector's list and logs the record JSON +
  reviewer prompt + output schema.
- [main.py:608](../agent/main.py) `_persist_new_escalations()`, called after
  every frame and after `finish_up`, drains new items via `_EscCollector`
  (which survives the correctors' list rebinds), appends each to
  `escalations.jsonl`, feeds it to memory, and — only with `--validate` —
  enqueues it to the validation worker.

**Supervisor agent — `ValidationWorker` (background thread)**

- Built and started in [main.py:429](../agent/main.py) when `--validate` is
  passed. It consumes the escalation queue one at a time, sends each full
  escalation prompt to the supervisor `MCPAgent` (images + tracking tools),
  parses the verdict into ops, and journals them via `store.append_validation`
  → `validations.jsonl`.
- In [main.py:646](../agent/main.py), `validator.stop()` drains the queue,
  then `materialize()` applies the ops to the revised rows and writes
  `validated.jsonl` + `validated.db` (per-run SQLite — the shared MySQL is
  never written).

**Memory — `LongTermMemory` hooks**

- Initialized in [main.py](../agent/main.py) `TrackingRun.__init__` when
  `--memory` is passed (falls back to an offline stub if the Meta model key
  is missing; init failure just disables memory), together with the
  `DailyShortTermMemory`, the `PeriodActivityAggregator` (gap knob
  `activity_combine_gap_minutes`), and the `PeriodLabeler` background thread.
- Per frame: `_memory_tick` advances time; each closed period row feeds the
  aggregator via `_memory_store_period_row`, which submits members to the
  labeler; labels come back through `_on_period_label`, and completed
  `ActivityRecord`s land in both memories via `_store_activity_record`.
  Each escalation is recorded as a short-term episode via
  `_memory_store_escalation`. All hooks are best-effort — they log and
  continue on error. The validator also receives the memory handle (briefs)
  and the short-term handle (verdicts). Memory shuts down at the end of
  `finish()`.

**Finalize — `finish()`**

`finish()` flushes both correctors (`finish_up`), drains the labeler and
flushes leftover activity records (`_finish_activity_labels`), stops the
validator and materializes validated outputs, draws
`original.png`/`revised.png`/`validated.png` (with `--plot`), then writes
`run.json` with counts including `escalations_by_code` and
`validations_by_verdict`.

Data flow for a full-loop run: MySQL → `process_message` → TrackCorrector →
PeriodCorrector → escalations → (async) supervisor agent → validated outputs,
with memory observing combined period activity + short-term episodes on the
side.
