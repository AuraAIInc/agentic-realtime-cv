# Spatial-Temporal Memory: Long-Term (Design)

## Goal

A spatial-temporal long-term memory in `agent/memory/` whose store is a
plain-language YAML file, with a hierarchical spatial tree (2D grid, N=3 levels
by default) and a hierarchical temporal tree (1D time buckets, 4 levels),
plus `store` / `invoke` / `promote` / `reset` and a semantic search augment.
All store/promote LLM work runs on a dedicated worker thread fed by a FIFO job
queue: callers enqueue and continue, the worker executes jobs one by one.
Promotion compresses each parent to dominant patterns only, weighting old stable
patterns over fresh input so the latest observation cannot overfit memory. The
store bootstraps from the first received image. The direct consumer of `invoke`
is the validation worker's escalation briefs. Plain-text token-overlap search
for v1 (no LLM-generated keywords anywhere); true embedding search is a
deferred future phase.

Trust policy (built): long-term memory keeps **only high-confidence
knowledge** — original high-confidence detections and high-confidence
supervisor-validation conclusions. Escalation occurrences and anything
untrusted belong to short-term day context, never durable knowledge. Every
record carries its original image names, the detection snapshot, and
plain-language text; every spatial zone describes cat appearance, activities,
behavior, and background view. Short-term memory is designed and built in
[short-term-memory-design.md](short-term-memory-design.md).

## Success Criteria

- `agent/memory/` holds config, store, spatial, temporal, retrieval, promotion,
  worker, and LLM-client code with tests; `agent/memory/README.md` documents usage.
- `memory_config.yaml` defines the spatial/temporal structure with working defaults.
- Long-term store persists as human-readable YAML; top spatial level describes the
  whole view; each temporal level captures its own aspect (events, mix, routine, habits).
  Every zone carries appearance/behavior descriptions and bounded evidence
  (original image names + detection snapshots); every hit surfaces them.
- `store` and `promote` use the Muse LLM API via the worker queue; the ingest
  loop never blocks on an LLM call; `invoke` does structural nearest-unit
  retrieval; `semantic_search` supports natural-language queries.
- Idle promotion fires after 5 min without `store`; `reset` clears the store.
- Promotion output keeps dominant patterns only; one-off observations never
  rewrite a stable summary; a single contradicting input cannot flip an
  established routine (repeated contradiction, default 3+, can).
- First-image bootstrap fills L2 + L1 environment/background/contrast before any
  observation arrives; L0 cells inherit marked-unobserved text until refined.
- `invoke_for_escalation(esc)` answers every escalation code family (T1-T7, P1-P7)
  with relevant spatial + temporal context.
- All unit tests pass offline (stubbed LLM, drained worker); one replay
  integration test runs store -> promote -> invoke over a real tracking run.

## Background (grounded facts)

- `agent/memory/` holds the built store: config, tree math, YAML store, worker,
  LLM client, prompts, long-term + short-term memory, and tests.
- `agent/escalation/escalation.py` defines `Escalation(time, scope, code, reason,
  end_time, cat_id, period_id, metrics)` and `emit()`; codes T1-T7 (frame) and
  P1-P7 (period) are the query vocabulary memory must serve.
- `TrackingRun.process_message` is a synchronous per-message loop (`agent/main.py`),
  so any inline LLM call would stall ingest: store/promote must run off-thread.
- LLM access pattern: `supervisor_agent/agent.py` builds `AsyncOpenAI(base_url,
  api_key)` from `supervisor_agent/models.json` (`roles.agent.model =
  muse-spark-1.3`, Meta Model API); installed `openai` is 3.39.0.
- `models.json` lists no embeddings endpoint (responses / images / asr only), so
  v1 semantic search scores token overlap over each unit's own text plus a
  phrase bonus; embeddings wait for a confirmed endpoint.
- View 3 is 800x600 (`agent_data/config/zones.yaml`); `PeriodCorrector` already
  uses a 50x50 px grid precedent (`period_area_count`).
- `output/` (repo root) is git-ignored temp data (`.gitignore`); tracking runs persist
  there via `RunStore`. The memory store is durable data, so it lives in the
  module at `agent/memory/data/` (new dir, added to `.gitignore` as runtime data).
- The tracking pipeline is detections-only (no pixels in `agent/main.py`), so the
  bootstrap image must be fetched explicitly (image server / camera_sim replay /
  `--bootstrap-frame` file) and passed to `bootstrap()`.

## Design

### File layout (new, all under `agent/memory/`)

- `memory_config.yaml` - checked-in defaults (structure, retrieval, promotion).
- `memory_config.py` - `MemoryConfig.from_yaml()` + `MEMORY_*` env overrides
  (same pattern as `escalation_config.py`).
- `models.py` - `MemoryEvent`, `MemoryQuery`, `MemoryContext`, `MemoryHit` dataclasses.
- `spatial.py` - grid math (point/box -> cell, parent/children, neighbors).
- `temporal.py` - bucket math (time -> slot/hour/day/week ids, history windows).
- `store.py` - YAML load/save (atomic tmp+rename), sparse units, retention.
- `worker.py` - `MemoryWorker` thread + FIFO job queue, promote coalescing,
  `drain()`/`shutdown()`.
- `llm_client.py` - Meta Model API wrapper + `StubLLMClient` for tests.
- `prompts.py` - store-update, promote-summary, environment-bootstrap prompts.
- `long_term.py` - `LongTermMemory`: store/invoke/promote/reset/tick/semantic_search.
- `short_term.py` - `DailyShortTermMemory`: day log + day-end merge (see the
  short-term design doc).
- `tests/` - offline unit tests + replay integration test.

### Memory config defaults (`memory_config.yaml`)

```yaml
version: 1
view_id: 3
frame_size: [800, 600]
spatial_levels:
  - {name: cells, grid: [8, 6]}      # L0: 100px cells, 48 total
  - {name: zones, grid: [4, 3]}       # L1: each covers 2x2 L0 cells
  - {name: full_view, grid: [1, 1]}   # L2: whole camera view
temporal_levels:
  - {name: slot_5min, minutes: 5}
  - {name: hour, minutes: 60}
  - {name: day, minutes: 1440}
  - {name: week, minutes: 10080}       # month deferred to keep v1 simple
retention_days: {slot_5min: 7, hour: 30, day: 180, week: 0}  # 0 = keep
retrieval: {spatial_chebyshev_radius: 1, temporal_history_days: 7, semantic_top_k: 5}
promotion: {idle_seconds: 300, min_pattern_support: 3, stable_weight_cap: 50,
            flip_threshold: 3, recent_kept: 5}
worker: {queue_warn_depth: 100, promote_coalesce: true}
store_dir: "agent/memory/data"          # per-view file view_<id>_long_term.yaml (durable; git-ignored)
llm: {model_role: memory, max_units_per_call: 8}
quality: {min_store_confidence: 0.7, short_term_keyframes: 3}  # trust gate + short-term key frames
```

L0 at 100px (2x the 50px `period_area_count` precedent) distinguishes sofa from
food bowl while keeping LLM summarization affordable. The 6-hour level from the
request is dropped for v1: hour -> day covers it, and fewer levels mean fewer
LLM calls per promotion.

### Store format (plain-language YAML, sparse)

Only non-empty units are stored. Durable runtime file: `agent/memory/data/
view_<id>_long_term.yaml` (repo-root `output/` is temp-only). Each unit splits slow-moving stable text from a
bounded `recent` buffer plus programmatic counters; the counters (not prose) are
the source of truth for dominance and weighting.

```yaml
version: 1
view_id: 3
bootstrapped: true
spatial:
  L0:
    - id: r3c5
      bounds: [500, 300, 600, 400]
      environment: "Part of the dark fabric sofa, left cushion."
      background: "Dark grey fabric, fairly uniform."
      cat_contrast: "High contrast: a white cat stands out clearly here."
      activity: "Qb naps here most afternoons; Qbo rarely visits."
      appearance: "Qb: short-haired white cat with faint silver shading."
      behavior: "Qb settles on the left cushion and sleeps for hours."
      weight: 24                 # effective observations, capped at stable_weight_cap
      contradictions: 0          # consecutive contradicting batches vs stable text
      recent: ["2026-10-31T14:00 Qb settled on the left cushion."]
      images: ["2026-10-31-14-00-00-000000.jpg"]   # originals, newest 5
      detections: [{box: [500, 300, 600, 400], cat_id: 25, conf: 0.9}]  # newest 3
      stats: {visits: 24, by_cat: {qb: 21, qbo: 3}}
      updated: "2026-10-31T14:05:00"
      sources: ["period 12", "period 15"]
  L1: [...]   # same fields, summarized over its 2x2 children
  L2: [...]   # one unit: full-view understanding
temporal:
  slot_5min:
    - id: "2026-10-31T14:00"
      summary: "Qb walked from the doorway to the sofa and settled."
      images: [...]              # same bounded evidence as spatial units
      detections: [...]
  hour: [...]  # activity mix: who was active/resting, where
  day: [...]   # routine: meals, naps, play bouts, anything unusual
  week: [...]  # habits and drift: stable routines, new spots, changes
```

Per-level temporal aspects are fixed: slot = events, hour = activity mix,
day = routine, week = habits/change. Higher levels are LLM summaries of their
children plus rolled-up counts (computed programmatically, e.g. visit counts).

### Worker thread + queue

- `MemoryWorker` owns a `queue.Queue` and one daemon thread. Job kinds: `store`,
  `promote`, `bootstrap`. FIFO execution, one job at a time; consecutive
  `promote` jobs coalesce into one when `promote_coalesce` is true; `bootstrap`
  jumps ahead of queued stores.
- The worker thread is the only writer of the in-memory tree and the YAML file.
  `invoke`/`semantic_search` are synchronous reads under an `RLock`, so callers
  (escalation) get answers without waiting for LLM jobs.
- `store()`/`promote()`/`bootstrap()` enqueue and return a receipt immediately;
  `last_store` (idle detection) is stamped at enqueue time, i.e. observation
  arrival, not LLM completion. `tick(now)` stays cheap: it only checks the idle
  clock and enqueues a promote.
- Failure isolation: a job exception is logged and the job is dropped; an LLM
  error degrades that unit to pending-observation mode (raw text kept in
  `recent`); the worker never dies and never raises into the ingest loop.
- Queue depth is small in practice (stores fire per period close / escalation,
  not per frame). Past `queue_warn_depth` the worker logs a warning; v1 does not
  drop stores.
- Lifecycle: `drain(timeout)` blocks until the queue is empty (tests, run
  finish); `shutdown()` drains then stops the thread. `TrackingRun.finish()`
  calls `shutdown()` so nothing is lost at exit. `reset()` drains, clears the
  tree and the YAML (optional timestamped `.bak`), resets the bootstrap flag,
  and keeps the worker running.

### API surface

```python
@dataclass
class MemoryEvent:
    time: datetime; end_time: datetime | None; view_id: int
    box: tuple[float, float, float, float] | None  # view pixels
    cat_id: int | None; period_id: int | None
    kind: str            # "period" | "escalation" | "label" | "summary"
    text: str            # plain-language observation
    code: str | None     # T1..T7 / P1..P7 when from escalation
    confidence: float | None     # 0..1; None = untrusted (trust gate)
    image_names: list[str]       # original image names (evidence)
    detection: dict              # {box, cat_id, conf, ...} snapshot
    appearance: str              # profile appearance line for the event's cat

@dataclass
class MemoryQuery:
    time: datetime; box: tuple | None; cat_id: int | None
    spatial_radius: int = 1     # Chebyshev neighbor rings around the hit cell
    history_days: int = 7       # same-slot history across days
    text: str | None = None     # when set, blend in semantic_search hits

class LongTermMemory:
    def store(self, event: MemoryEvent, image: bytes | None = None) -> JobReceipt
    def bootstrap(self, image: bytes) -> JobReceipt  # once per view; no-op after
    def invoke(self, query: MemoryQuery) -> MemoryContext          # sync, locked read
    def invoke_for_escalation(self, esc: Escalation, box=None) -> MemoryContext
    def semantic_search(self, text: str, top_k: int = 5) -> list[MemoryHit]
    def tick(self, now: datetime) -> JobReceipt | None  # enqueue promote when idle
    def promote(self, force: bool = False) -> JobReceipt
    def drain(self, timeout: float | None = None) -> None
    def shutdown(self) -> None
    def reset(self, keep_backup: bool = True) -> None
```

### Promotion: compression to dominant patterns, anti-overfitting weights

- Bottom-up over dirty subtrees only (L0 -> L1 -> L2; slot -> hour -> day ->
  week). Stats roll up programmatically first (`visits`, `by_cat`); the LLM then
  rewrites the parent's stable text from children stable texts + stats.
- Compression rule: only patterns with support >= `min_pattern_support`
  (default 3 observations) may enter stable text. One-offs stay in the bounded
  `recent` buffer (default last 5) and age out. Stable text has a word cap per
  level (L0 ~60, L1 ~100, L2 ~150; temporal slot ~40, hour ~80, day ~120,
  week ~150); the prompt must drop the lowest-support detail first.
- Weighting rule: each unit's stable text carries `weight` = min(total
  observations, `stable_weight_cap`). The store-update LLM returns
  `contradicts_stable: bool` per merge; the worker increments `contradictions`
  on true and resets it on agreement. Stable text is rewritten only when
  `contradictions >= flip_threshold` (default 3, across distinct slots); until
  then the new input lives in `recent` and the prompt is told the established
  pattern outranks it. Conflict judgement is the LLM's job; counting, caps, and
  thresholds are programmatic, so behavior is testable with the stub.
- Persist after each promotion (atomic tmp+rename); clear dirty flags only for
  units actually summarized.
- LLM observability (built): every real store/promote/bootstrap LLM call
  dumps one markdown trace (`output/memory_runs/<ts>-<seq>-<kind>-<label>.md`,
  `MEMORY_TRACE_DIR` override) with the full system prompt, input, raw
  output, parsed answer, and timing/token/cost stats — same layout as the
  supervisor agent traces, single-turn scale. Trace writes are best-effort
  and never break memory; the offline stub writes no traces.
- Trust gate (`_store_qualifies`, built): `store()` keeps only `period` /
  `label` events at or above `min_store_confidence` (default 0.7) plus
  `summary` events (curated day merges, always qualify). Escalation
  occurrences, error verdicts (confidence 0.0), and untrusted (`None`) events
  are skipped with a log line — they belong to short-term day context.
- Curated rollup (built): `summary`-sourced units always qualify for text
  rollup regardless of visit support, and a summarized parent inherits its
  children's `sources`, so day narratives roll all the way up instead of
  stopping at stats-only parents. Spatial summaries return
  `{activity, appearance, behavior}`; appearance/behavior come only
  from children's lines, never invented.

### Bootstrap from the first image

- One vision call on the first full-view frame returns strict JSON: the L2
  full-view description plus per-L1-zone `environment`/`background`/
  `cat_contrast`. L0 cells start empty but inherit their L1 text at read time
  with an `unobserved: true` marker until real observations refine them. One
  call total keeps bootstrap cheap; per-cell contrast sharpens later from crops
  attached to `store(image=...)`.
- Trigger order: (1) explicit `bootstrap()` from run setup with a frame fetched
  from the image server / camera_sim replay / `--bootstrap-frame` file;
  (2) fallback: the first `store(image=...)` auto-enqueues a bootstrap ahead of
  itself when `bootstrapped` is false. `store()` without an image on an
  unbootstrapped tree works in text-only mode with vision fields marked unknown.

### Escalation -> invoke mapping

| Codes | Question memory answers | Units returned |
|---|---|---|
| T1, T2, T5 (identity / weak relabel / fallback) | Is the cat hard to see here? | Hit cell + neighbors: environment, background, contrast |
| T4, T6 (jump / teleport) | Was this cat recently in the target cell? | Cells along the path + current slot/hour |
| T3, T7 (duplicate tie / long synthesis) | Is the track continuous? | Current slot + hour continuity |
| P1, P2, P5 (close / weak / low-quality vote) | What is this cat's routine? | Day/hour routine + cells visited |
| P3 (conflict reassign) | Which cat is usually here now? | Same-slot history for both cats |
| P4 (long unknown) | Is this an occluded / low-contrast spot? | Environment + occluder/contrast notes |
| P6, P7 (merge / static link) | Same place, usual nap? | Gap-span summary + place routine |

### LLM usage

- Add `roles.memory` to `supervisor_agent/models.json` (default `muse-spark-1.3`);
  reuse `base_url` / `api_key_env`. The worker uses a sync client (callers and
  the worker are sync). `StubLLMClient` implements the same interface for
  offline tests.
- Four prompts: store-update (merge observation into `recent`, report
  `contradicts_stable`), promote-summary (compress children into parent
  stable text under the word cap, dominant patterns only), bootstrap-vision
  (full-view + per-zone environment/background/contrast JSON), plus shared
  plain-language style instructions.
- Schema stability: the LLM returns strict JSON with fixed fields; `store.py`
  renders it into YAML. Plain language is in the values, not the keys.
- Cost guards: at most `max_units_per_call` units per LLM call, dirty subtrees
  only, promote coalescing, and a per-promotion call budget with graceful
  deferral.

### Short-term memory (built)

`DailyShortTermMemory` (`agent/memory/short_term.py`) remembers every period
activity across one day — all confidences — plus tracker episodes and their
validation verdicts, in a plain-language `view_<id>_short_term_<day>.md`
re-rendered on every change. New sessions open a new section; midnight (event
time) merges the day into long-term memory as curated `summary` events that
promote into the top spatial levels and the day bucket; session end merges the
same way, marked so nothing merges twice. Full design:
[short-term-memory-design.md](short-term-memory-design.md).

### Future phases

- Escalation read-through: `invoke` merges live short-term entries with
  long-term hits for validation briefs (currently briefs read long-term only).
- Embedding-based semantic search once a Meta embeddings endpoint is confirmed
  (new `roles.embedding`; vectors stored beside the YAML, never in it); month
  temporal level if week summaries grow unwieldy.

## Approach

Build bottom-up in four phases: (1) config + tree + YAML store + reset + worker
skeleton, no LLM; (2) text-path store/invoke through the worker + text
search + compression data structures; (3) LLM-backed store/promote with
weighting prompts + bootstrap vision + escalation-facing invoke + idle tick;
(4) TrackingRun hooks + replay integration + docs. Short-term memory follows as
phase 5, embeddings as phase 6.

## Steps

1. Config, tree math, YAML store, reset, worker skeleton.
   Surfaces: `memory_config.yaml/.py`, `models.py`, `spatial.py`, `temporal.py`,
   `store.py`, `worker.py`, `reset()`, plus an `agent/memory/data/` entry in
   `.gitignore` (durable runtime data). Grid parent/child/neighbor math; time
   bucketing; atomic YAML round-trip; retention pruning; FIFO worker with
   receipts, coalesced promotes, `drain()`/`shutdown()`, reader lock.
2. Text-path store/invoke through the worker, semantic search, stub short-term.
   Surfaces: `long_term.py` (text merge without LLM), `short_term.py` interface,
   `MemoryContext` brief rendering, text scorer, unit counters (`weight`,
   `contradictions`, `recent`, `stats`). Neighbor-radius, history-window, and
   dominance-support behavior pinned by tests with `StubLLMClient`.
3. LLM-backed store/promote, weighting, bootstrap, escalation adapter, tick.
   Surfaces: `llm_client.py`, `prompts.py`, `roles.memory` in `models.json`,
   dirty-subtree promote with compression caps, `contradicts_stable` counting vs
   `flip_threshold`, `bootstrap()` vision job + auto-bootstrap fallback,
   `invoke_for_escalation` + code-family mapping, idle `tick`. LLM errors
   degrade to pending-observation mode, never raise.
4. Integration, replay test, docs.
   Surfaces: `TrackingRun` hooks (trusted `store` on period close, short-term
   record on period close + `emit`, `tick` per message, short-term merge +
   `shutdown` in `finish()`, optional `--bootstrap-frame`), replay integration
   test over an existing `output/tracking_runs/` run,
   `agent/memory/README.md` rewrite.

## Validation Plan

- `pytest agent/memory/tests -q` from the repo root, offline: grid mapping
  (corners, edges, parent coverage), bucket boundaries (slot/hour/day/week),
  YAML round-trip + retention, neighbor-radius counts, history-window contents,
  text ranking on a fixture tree, worker ordering + promote coalescing +
  `drain`/`shutdown` flush, concurrent invoke-during-store smoke test,
  compression (dominant pattern kept, one-off dropped, word caps held),
  weighting (one contradiction keeps routine, three flip it), bootstrap marked
  unobserved until refined, reset with/without backup.
- Replay integration: feed `periods.jsonl`/`escalations.jsonl` from an existing
  tracking run through store -> tick -> promote -> invoke; assert YAML exists,
  L2/L1/day units are non-empty, dominant-pattern compression holds, and
  `invoke_for_escalation` returns context for at least one T-code and one P-code.
  Highest-risk check: LLM-backed promote produces schema-valid YAML (run once
  live, then pin with the stub).
- Manual: read the generated YAML for plain-language quality; run a 2-min live
  replay and confirm the ingest loop never waits on the worker (LLM only on
  worker jobs with budgets); confirm a mid-run kill loses at most unpromoted
  `recent` entries (last persisted YAML stays valid).
- Regression: `pytest agent/escalation/tests agent/tracking/tests -q` stays green.

## Risks / Open Questions

- Threading bugs (lost updates, torn reads): single-writer worker + `RLock`
  readers + atomic YAML writes contain them; the smoke test hammers
  invoke-during-store every run.
- Queue backup under LLM outage: stores are low-rate (period/escalation events,
  not frames), jobs degrade to text-only mode, and depth warnings surface stalls.
- YAML growth over weeks: sparse storage + retention pruning bound it; week level
  is kept forever but is one short summary per week.
- Bootstrap image availability: the pipeline is detections-only, so run setup
  must fetch one frame explicitly; until then text-only mode marks vision fields
  unknown rather than guessing.
- `openai==3.39.0` API parity for the responses-style calls `agent.py` uses:
  verify at build time and match the exact working call shape.
- No embeddings endpoint in `models.json` today: token-overlap text search is
  the v1 approach (no stored keywords); embeddings are a future phase, only
  if an endpoint is confirmed.
- Open question: none blocking. Store sources are high-confidence period
  verdicts + decisive supervisor-validation labels + curated day summaries;
  escalation occurrences stay in short-term day context by design.
