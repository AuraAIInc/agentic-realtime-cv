# Escalation queue merging: design

## 1. Goal

When a new escalation is temporally close to an escalation already waiting in
the validation queue (gap ≤ a small window, default 60 s), validate them
together in **one supervisor-agent call** instead of one call each. Fewer
calls, shared image fetches and shared context across related cases (e.g. the
E6/E7/E8 same-frame cluster), one verdict op per escalation out the other end.

Non-goals: changing what escalates (corrector thresholds untouched),
`escalations.jsonl` / `emit()` logging (untouched), apply-order semantics
(preserved), and any artificial delay in the default path (merging is
opportunistic — see §3).

## 2. Current path (what changes)

Producer: `TrackingRun._persist_new_escalations` (`agent/main.py:699`) calls
`validator.enqueue(esc)` once per escalation, after every frame and after
`finish_up`. Consumer: `ValidationWorker` (`agent/validation/worker.py`)
pulls one escalation at a time from a `queue.Queue`, builds one prompt
(`build_validation_query`), makes one agent call, parses one review
(`parse_review`), and journals one op (`{"escalation", "review"}`), which
`materialize` applies in journal order.

Only the queue and the single-item validate step change. Everything upstream
(emit, prompts, schemas) and downstream (journal shape, apply, memory verdict
events, short-term `record_verdict`) keeps working per escalation.

## 3. Merge point: a batching queue

Replace the raw `queue.Queue` with a small `EscalationBatcher`
(`agent/validation/batcher.py`):

- `put(esc)` (tracking thread): if the **tail waiting batch** is temporally
  close to `esc` (§4) and under the size cap, append `esc` to it; otherwise
  start a new tail batch. Log at most one line on merge
  (`merged E7-T2 into batch [E6-T2] (gap 0.0 s)`).
- `take()` (worker thread): block until a batch is waiting or the batcher is
  closed; return the head batch (`list[Escalation]`) or `None` when closed
  and drained.
- `close()` (`stop()`): stop accepting; the worker drains remaining batches,
  then `take()` returns `None`.

Merge candidates are **waiting** batches only. A batch the worker already
took is in flight and can never gain members — this is what makes the design
race-free without locks beyond one `threading.Condition`.

Default behavior is **opportunistic with zero added latency**: the worker
takes whatever is waiting the moment it is free. In replay mode the producer
dumps all escalations in ~1 s while one validation takes ~30–80 s, so the
queue is almost always full and merging captures nearly everything. In live
mode sparse arrivals are usually taken immediately and validate solo — which
is correct: holding them back would add delay for no guaranteed partner.
An optional linger for live mode is phase 2 (§9).

## 4. Closeness rule and caps

Each escalation has a span: `(time, end_time or time)` — a point for frame
scope, the flagged span for period scope. A batch has the min/max span of its
members. `esc` joins the tail batch when the gap between the two spans is
≤ `merge_window_s` (default 60.0; overlapping spans have gap 0 and always
join). Temporal only — no cat/code affinity rules in v1; same-frame
cross-cat pairs (E6/E7) *benefit* from joint review, and nothing observed so
far argues for exclusions.

Two caps bound prompt size:

- `merge_max_batch` (default 4): a full tail batch never gains members; the
  newcomer starts a new batch.
- Prompt budget: if the merged prompt would exceed `merge_max_chars`
  (default ~12_000), the newcomer starts a new batch. (Single prompts run
  ~1.5–2.5k chars, so the count cap normally binds first.)

### Worked example (session `002728`, arrival order, window 60 s, cap 4)

| Batch | Members | Why |
| ----- | ------- | --- |
| 1 | E1-T7, E2-T6, E3-P5, E4-T7 | chain 16:33:06–16:34:18; then full |
| 2 | E5-T2, E6-T2, E7-T2, E8-T4 | E5 16:34:36 opens it (batch 1 full); E6/E7/E8 same frame 16:35:21 join (gap 45 s); then full |
| 3 | E9-P6, E10-T7, E11-P2, E12-P2 | E9 span 16:34:42–16:35:21 overlaps batch span; E10/E12 join; E11 (16:34:41–16:34:42) joins |

12 calls → 3. Without the size cap, 60 s chaining would collapse all 12 into
one oversized call (span 16:33:06–16:35:41) — the cap is load-bearing, not
decorative.

## 5. Merged prompt format

One preamble + one section per member, each section byte-identical to today's
`build_prompt(esc)` output plus its own memory brief:

```text
You are reviewing 3 related escalation cases from the same camera window.
They may share frames and context — fetch each image once and use it for
every case it belongs to. Leverage evidence across the whole time window
for all of the cases, and resolve every case with the minimum total tool
calls and reasoning steps. Judge each case on its own evidence.

=== Case 1 of 3: E6-T2 ===
<E6 build_prompt output>
Past context from memory (may help, may be stale): <E6 brief>

=== Case 2 of 3: E7-T2 ===
...

Return ONLY a JSON object following the output schema — one review per case,
keyed by case name. No prose, no fences.
```

Single-member batches use today's exact prompt and schema (byte-identical
agent input), so merging changes nothing when nothing merges.

## 6. Merged output schema and parsing

Strict `json_schema` needs a fixed shape, but member codes have different
verdict enums — so the merged schema is a uniform envelope, and per-code
strictness moves to post-parse validation (the per-code pydantic models
already exist):

```json
{"reviews": [
  {"name": "E6-T2", "verdict": "<union of member codes' literals>",
   "chosen_cat_id": null, "cut_time": null},
  ...
]}
```

- `verdict` enum = union of the member codes' allowed literals (built
  dynamically per batch); `chosen_cat_id` / `cut_time` always present,
  null-allowed (strict mode requires all properties required).
- `text_format` name: `merged_review` (today: `{code}_review`).
- New `parse_merged_review(escalations, answer) -> dict[name, review]`:
  parse the envelope, match items to members by `name`, filter each item to
  its code model's fields (the envelope carries `chosen_cat_id`/`cut_time`
  for every item but each model accepts only its own subset and forbids
  extras), then validate against `review_model_for(code)`.

**Partial-failure isolation** (better than one shared fate): an item that is
missing, names an unknown case, or fails its code's model becomes an
`error` op for that escalation only; valid siblings still journal their
verdicts. Only a fully unparseable answer errors the whole batch — the same
as today.

## 7. Ops, journal, memory, apply

The worker still emits **one op per escalation**, in `seq` order within the
batch, each journaled individually — so `materialize` sees the same op
stream shape as today and apply-order semantics (E6→E7→E9→E12) are preserved.
Per member, unchanged: `escalation <name> verdict: <verdict>` log line,
`escalation <name> rationale: <rationale>` log line, short-term
`record_verdict` (verdict + rationale). Verdicts never reach long-term
memory directly — it learns from labeled period activity instead.
`validations_by_verdict` counts and `validations.jsonl` are unaffected in
shape.

## 8. Logging and traces

- One line per merge at `put` time (§3); one line per batch at `take` time
  (`validating batch [E6-T2, E7-T2, E8-T4] (3 cases, span 16:35:21–16:35:21)`).
- The existing `validation agent full prompt for …` log carries the merged
  prompt (log code becomes the batch names, e.g. `E6-T2+E7-T2+E8-T4`).
- One agent-run dir per batch (tag `validation`); the trace embeds the
  merged prompt, so member mapping is self-describing. Session reports map
  dirs to member lists instead of single escalations.

## 9. Config and rollout

`ValidationWorker(..., merge_window_s=60.0, merge_max_batch=4,
merge_max_chars=12_000)`. No CLI flags in v1 — constructor params only;
add `--no-merge` / `--merge-window` later if live tuning needs them.

- **Phase 1 (this design): opportunistic merging, linger 0.** All replay
  gains, zero latency risk, single-batch behavior byte-identical.
- **Phase 2 (optional): `merge_linger_s`.** `take()` holds a non-full head
  batch up to `linger` seconds past its first item so late arrivals can
  join. Bounds added live-mode delay; validation is not on the tracking
  critical path (ops apply at `finish()`), so a 10–15 s linger only delays
  memory verdict freshness and short-term episode updates. Implement only if
  live traces show solo validations that a short wait would have batched.

## 10. Tests

- `agent/validation/tests/test_batcher.py`: gap join vs split (incl.
  period-span overlap, exact-boundary), size-cap split, char-budget split,
  in-flight isolation (take then put → new batch), close/drain/`take→None`,
  producer/consumer thread smoke test.
- `parse_merged_review` tests: happy path (2 codes), unknown-name item,
  missing member, per-item schema failure with sibling success, fully bad
  JSON → all error.
- Worker test with a callable backend: one batch in, N ops out in seq
  order, journal called N times, memory/short-term hooks called per member.
- Replay determinism: solo batches produce byte-identical prompts/schemas
  to today (golden test on one E-prompt).

## 11. Expected payoff and risks

Payoff (to verify by rerunning the 16:33–16:40 window): 12 → ~3 agent calls;
the E6/E7/E8 cluster fetches the 16:35:21 frame once instead of 3×; one
system prompt and one minute-listing per batch instead of per case. Rough
direction: fewer turns and meaningfully fewer tokens per escalation than the
current ~47k. Joint review should also *improve* same-frame verdict
coherence (the agent sees both sides of the E6/E7 swap at once).

Risks: one hard case delays its batch siblings' verdicts (bounded by batch
cap; siblings' rows still apply in the same `materialize`); bigger prompts
risk cross-case confusion (mitigated by `=== Case ===` separators and
name-keyed output — watch the first merged traces for leakage); a batch-wide
API 400 now drops up to 4 verdicts instead of 1 (this strengthens the case
for the already-planned error-retry from the `012059` report §11.4).

## 12. Open questions

1. `merge_max_batch`: is 4 the right default, or should the char budget
   alone decide? (Lean: keep both; count cap is the predictable one.)
2. Should phase 2 linger exist at all, or is live-mode solo validation
   acceptable long-term?
3. Future affinity rules (same-cat preference, never-merge pairs)? None
   proposed until a merged trace shows a real confusion.
