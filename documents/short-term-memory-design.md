# Short-Term Memory: Daily Activity Log (Design + Built Behavior)

Daily full-fidelity memory for the tracking pipeline: **every** period
activity across one day (all confidences), as combined activity records —
which cat, from what time to what time, at which area (box), doing what
LLM-judged activity, what the cat looks like, what the background looks
like — plus tracker episodes and their validation verdicts (+ rationales),
in a plain-language Markdown file re-rendered on every change. At day end
the day merges into long-term memory as curated summaries.
Implementation: `agent/memory/short_term.py` (`DailyShortTermMemory`).

Companion doc: [spatial-temporal-memory-design.md](spatial-temporal-memory-design.md)
(long-term store, promotion, retrieval).

## Goal

Long-term memory keeps only high-confidence knowledge (trust gate), which
means most of what happened on a given day never lands there: low-confidence
periods, tracker uncertainty episodes, and the verdicts that resolved them.
Short-term memory keeps the full day so that:

- a human can read what happened on any day without opening JSONL files;
- validation briefs and debugging have same-day context (what else happened
  today, what was already judged);
- the day-end merge carries a curated day narrative — including low-confidence
  visits and episode resolutions — up into the long-term top spatial levels
  and the day bucket, where the trust gate would otherwise exclude them.

Non-goals: cross-day reasoning (long-term's job), per-frame storage (the
tracking run dir already has that), LLM calls on the ingest path (merge
reuses the long-term worker queue).

## Lifecycle

```
session start ──► record periods/episodes/verdicts ──► session end ──► merge
                        │ midnight (event time)                         │
                        ▼                                               ▼
              merge old day ──► new day                        long-term worker
```

- **Session start** (`start_session(now, run_name)`): opens a new `## Session`
  section in the day file. Live state resets (a session never inherits live
  entries); the day file accumulates sections. If event time already crossed
  midnight, the old day rolls over first (see below).
- **Ingest** (all best-effort, never raising into tracking):
  - `record_period(...)` on every **completed activity record** (one or more
    combined periods): span, cat, sighting count, mean confidence, area box,
    key-frame image names, plus member period ids, LLM-judged activity (+
    confidence), appearance, and background. Keyed by `(period_id, start)` —
    re-running the same window does not duplicate.
  - `record_episode(...)` per escalation occurrence (tracker uncertainty
    episodes live here, never in long-term memory). Keyed by escalation name.
  - `record_verdict(name, verdict, rationale=None)` attaches the validation
    verdict (+ short rationale) to its episode once judged.
- **Threading**: period records arrive from the labeler thread while episodes
  and verdicts arrive from the tracking/validation threads; every mutation
  and render holds an internal lock, and every record persists immediately.
- **Midnight rollover** (`rollover(now)`, checked on every tick): when event
  time crosses into a new day, the old day merges into long-term memory and a
  new day starts. Event time (data timestamps), not wall clock, so replays of
  historical days behave like live days.
- **Session end** (`merge_into(long_term)` from run shutdown): merges the
  session's entries the same way. Merged sessions are marked in the day
  state, so a later session or a midnight rollover never merges twice.
- After merging, the caller enqueues one `promote()`; shutdown drains the
  worker, so summaries roll into L1/L2/day/week before exit.

## Data model

```python
@dataclass
class PeriodEntry:
    period_id: int | None          # first member period (display + dedup key)
    cat_id: int | None; cat_name: str | None
    start: datetime; end: datetime  # combined span
    detections: int; mean_conf: float | None
    box: tuple | None              # area box (zone grouping at merge)
    keyframes: list[str]           # original image names across the span
    period_ids: list[int]          # member periods of this record
    activity: str | None           # labels.yaml activity judged from key frames
    activity_confidence: float | None
    appearance: str                # what the cat looks like in this record
    background: str                # what the background looks like

@dataclass
class EpisodeEntry:
    name: str                      # E<seq>-<code>
    code: str; time: datetime; reason: str
    cat_id: int | None; period_id: int | None
    image_names: list[str]
    verdict: str | None            # attached later by record_verdict
    rationale: str | None          # short reason for the verdict
```

Each entry renders one plain-language line, e.g.:

- `16:34–16:40: Qb (cat 25) at [339.0, 172.0, 396.0, 244.0] — 772 sightings, mean confidence 0.71. Activity: resting (confidence 0.90). Looks: fluffy white cat. Background: rug by the sofa. Key frames: …`
- `16:35:21 [E6-T2]: detection labeled cat 26 relabeled to cat 25 by center distance alone, no IoU support Validation verdict: keep_original. Coat matches cat 25.`

## Files

Per view/day under `MemoryConfig.store_dir` (default `agent/memory/data/`,
git-ignored runtime data):

- `view_<id>_short_term_<YYYY-MM-DD>.md` — the human artifact. Header (day,
  sessions, merged sessions) + one `## Session` section per run with
  `### Periods` and `### Episodes`. Re-rendered from state on every change.
- `view_<id>_short_term_<YYYY-MM-DD>.json` — machine state (sessions, entries,
  `merged_sessions`) the Markdown renders from. Unreadable/corrupt state logs
  a warning and starts fresh; it never blocks tracking.

## Day-end merge algorithm

`merge_into(long_term)` processes each unmerged session with entries:

1. **Full-day narrative** → one `MemoryEvent(kind="summary")`, `box=None`,
   `time` = day start. Text covers per-cat visit counts/spans/sightings and
   seen activities plus episode counts and resolutions. Temporal-only: lands
   in a slot, promotes into hour → day → week. Confidence = mean of the
   session's period confidences (honest aggregate, not inflated).
2. **Per-zone summaries** → one `kind="summary"` event per active L0 cell
   (periods grouped by `box_to_cell`; boxless periods are covered by the day
   narrative only). Each carries a representative box plus the zone's seen
   activities, so it lands in that cell and promotes into L1/L2 — the "top
   spatial levels" landing.
3. Sessions with no entries are marked merged without events. All merged
   session names persist in the day state.

`kind="summary"` events always pass the long-term trust gate (curated day
truth), and curated units always qualify for text rollup with provenance
inheritance (`sources`), so the narrative rolls all the way up instead of
stopping at stats-only parents (see the long-term doc's promotion section).

## API surface

```python
class DailyShortTermMemory:
    def __init__(self, config, view_id: int, long_term=None)
    def start_session(self, now: datetime, session_name: str) -> None
    def rollover(self, now: datetime) -> date | None   # merged day or None
    def record_period(self, *, period_id, cat_id, cat_name, start, end,
                      detections, mean_conf, box=None, keyframes=None,
                      period_ids=None, activity=None, activity_confidence=None,
                      appearance="", background="") -> PeriodEntry | None
    def record_episode(self, *, name, code, time, reason, cat_id=None,
                       period_id=None, image_names=None) -> EpisodeEntry | None
    def record_verdict(self, name: str, verdict: str, rationale=None) -> bool
    def merge_into(self, long_term) -> list  # store receipts
    def render(self) -> str                  # full day Markdown
```

`TrackingRun` owns one instance when `--memory` is passed (created beside
`LongTermMemory`, session = run dir name): completed activity records feed
it (via the labeler sink), escalations feed it per `emit`, `tick` checks
rollover, shutdown merges + promotes. `ValidationWorker` takes it as
`short_term=` and records verdicts (+ rationales).

## Failure modes

- Short-term failures (disk, corrupt state, merge errors) are caught at every
  call site and logged; tracking and validation never see them.
- A mid-run kill loses at most the entries since the last record call — every
  record persists immediately (atomic JSON write + Markdown rewrite).
- Re-running the same window appends a new session section; per-entry
  dedup keys prevent double-recording within a session, and merge markers
  prevent double-merging across sessions.

## Config knobs

- `quality.short_term_keyframes` (default 3, `MEMORY_SHORT_TERM_KEYFRAMES`):
  key-frame images per period entry (evenly spread, endpoints included).
- `activity_combine_gap_minutes` (default 5.0,
  `MEMORY_ACTIVITY_COMBINE_GAP_MINUTES`): same-cat periods with smaller gaps
  merge into one short-term entry (and one long-term record).
- Day file location: `MemoryConfig.short_term_path(day, view_id)`.

## Tests

`agent/memory/tests/test_short_term.py`: plain-language rendering + file
persistence, duplicate skip, session-section accumulation, rollover merge
into long-term day/L1 with merge markers, partial-session merge with
no-op re-merge, combined-record activity rendering, legacy state compat,
activity-carrying merge narratives. Merge/rollup paths run against
`StubLLMClient` offline.

## Future work

- Read-through `invoke`: blend live short-term entries with long-term hits
  for validation briefs (currently briefs read long-term only).
- Cross-day episode patterns ("this same T2 fired three days running") as a
  long-term week-level input.
- Key-frame thumbnails embedded in the day Markdown (currently names only;
  bytes stay on the image server).
