# Agent & Tool Invoke Report (2026-10-27 → 2026-11-03)

Source: `aura_cat.agent_invoke` (124 rows) and `aura_cat.tool_invoke`
(1,058 rows), fetched 2026-11-03. Every agent run links to its local trace
(`run_dir` set on all 124); every tool row joins cleanly to a parent run
(0 orphans).

## Headline numbers

| Metric | Value |
|---|---|
| Agent runs | 124 (all `muse-spark-1.3`, all `stopped_reason=answered`) |
| Tool calls | 1,058 across 26 distinct tools |
| Total cost | $9.22 ($6.56 validation, $2.66 untagged) |
| Tokens in / out / cached | 9,746,914 / 625,428 / 5,108,850 |
| Images fetched | 589 sent, 589 returned |
| Tool error rate | 104 / 1,058 (9.8%), touching 47 runs (38%) |
| Median run | 4 turns, 8 tools, 60 s wall (57 s LLM, 2.6 s tools) |
| Runs per day | 10-27: 1, 10-31: 13, 11-01: 29, 11-02: 42, 11-03: 39 |

Pricing on every run: $1.25 / $4.25 / $0.15 per 1M input / output /
cached tokens. Cached tokens (5.1M, 52% of inputs) are doing real work.

## Agent invokes

Tags split 93 `validation` vs 31 untagged. The two cohorts look alike in
shape (median 8 tools/run both) but differ in cost drivers: validation
runs fetch 512 of the 589 images and account for 71% of spend; untagged
runs skew to more turns (median 5 vs 4) with fewer images (77).

- Turns distribution: 1×1, 2×15, 3×38, 4×35, 5×21, 6×6, 7×6, 8×1, 9×1.
- Tools/run: min 0, median 8, mean 8.5, max 25.
- One zero-tool run (10-31 17:44, untagged, 1 turn, answered from prompt alone).
- Costliest run: 11-03 22:57 validation, $0.31 (7 turns, 23 tools,
  287K input tokens). Next two are untagged 11-01 runs ($0.23, $0.19).
- First tool is usually evidence-gathering: `tracking__get_period_detail`
  opens 63 runs, `images__get_image` 26, `images__list_images` 7.

## Tool invokes

`images__get_image` dominates (670 calls, 63%), then
`tracking__get_period_detail` (124, 12%) and `images__list_images`
(60, 6%). Full per-tool table (n, error %, median round-trip ms):

| Tool | n | err% | med ms |
|---|---|---|---|
| images__get_image | 670 | 14.2 | 687 |
| tracking__get_period_detail | 124 | 0.0 | 2,796 |
| images__list_images | 60 | 0.0 | 405 |
| camera__capture_frame | 27 | 7.4 | 7 |
| camera__get_status | 20 | 5.0 | 5 |
| tracking__list_views | 19 | 0.0 | 59 |
| camera__detect | 18 | 5.6 | 281 |
| tracking__get_live_status | 15 | 0.0 | 54 |
| local__get_cat_profiles | 15 | 0.0 | 13 |
| tracking__get_current_periods | 13 | 0.0 | 66 |
| local__find_references | 12 | 0.0 | 12 |
| local__get_reference_image | 11 | 0.0 | 4 |
| camera__get_track_history | 9 | 0.0 | 7 |
| local__analyze_trajectory | 9 | 0.0 | 824 |
| camera__get_track_frame | 6 | 50.0 | 9 |
| camera__capture_high_res | 5 | 20.0 | 5 |
| camera__get_camera_modes | 4 | 25.0 | 9 |
| tracking__get_arrival_path | 4 | 0.0 | 7,252 |
| tracking__get_periods | 4 | 0.0 | 184 |
| skill__check-live-detections | 3 | 0.0 | 0 |
| local__make_contact_sheet | 3 | 0.0 | 902 |
| tracking__get_last_seen | 2 | 0.0 | 64 |
| camera__list_buffered_frames | 2 | 0.0 | 10 |
| camera__list_models | 1 | 0.0 | 11 |
| local__get_zones | 1 | 0.0 | 3 |
| skill__validate-cat-identity | 1 | 0.0 | 2 |

Error breakdown (104 total):

- 95× `ModuleNotFoundError: No module named 'PIL'` — all on
  `images__get_image`. Decisive split: **91 of 95 errors carried a
  `box` crop param, and 0 of 575 successful fetches did.** Every crop
  failed; every plain fetch succeeded. The image server cannot
  transform without Pillow.
- 6× `MCPError: Session not found` — camera tools, transient session loss.
- 3× `RuntimeError: frame … has already left the buffer` —
  `camera__get_track_frame` racing the rolling buffer.

Slowest tools by median: `tracking__get_arrival_path` (7.3 s, n=4),
`tracking__get_period_detail` (2.8 s, n=124). Everything local
(`local__*`, `skill__*`) answers in under a second.

## Skill usage

Skills execute as `skill__*` tools — 4 calls total, all on 11-01, all in
untagged runs, all successful, all sub-2 ms (instruction payloads, no
images):

- `skill__check-live-detections` ×3 (13:22, 13:49, 14:10) — opened 2 of
  its 3 runs as the first tool call.
- `skill__validate-cat-identity` ×1 (00:35).

Zero skill calls in the 93 validation-tagged runs: the validation agent
currently works from its prompt + image/tracking tools only.

## Data-quality notes

- The `tool_invoke.error` column is blank on 98 of 104 error rows; the
  real message sits in element [1] of the `tool_output` JSON pair. Any
  error dashboard must parse `tool_output`, or the writer should populate
  `error`.
- `reasoning_effort` is NULL on all 124 runs; `stopped_reason` is
  `answered` everywhere — no timeout/cancel signal to analyze.

## Recommendations

1. Install Pillow on the image server (or stop requesting `box` crops):
   it is 91% of all tool errors and forced whole batches onto full-frame
   retries.
2. Populate `tool_invoke.error` at write time so failures are queryable
   without JSON parsing.
3. Consider skill adoption in validation runs: `validate-cat-identity`
   exists but the validation cohort never calls it.
4. `tracking__get_period_detail` (2.8 s median × 124 calls) and
   `tracking__get_arrival_path` (7.3 s) dominate tool wall-time; cache or
   narrow their windows if validation latency matters.
