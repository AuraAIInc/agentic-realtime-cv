# Review: agent run 20261101-003501 + tools, prompts and skills

Trace: `agent/output/agent_runs/20261101-003501-252989/trace.md`
Query: *"check the live camera view and find any cat in the view. compare with the detection from the model to validate the detection accuracy"*

Scope: the trace, the supervisor agent (`agent/supervisor_agent/*`), local tools, the 4 skills, and the
camera / image / tracking MCP servers. Nothing is implemented here; this is findings + suggested fixes only.

---

## 0. Summary

| Metric | Value | Comment |
|---|---|---|
| Turns / tool calls | 9 / 22 | The question needs ~3 turns / ~6 calls |
| Wall time | 152.4 s | **LLM 142.8 s (94 %)**, tools 9.6 s |
| Input tokens | 158 k (72 k cached) | ~8 k/turn is just tool schemas (≈45 % of all input) |
| Output tokens | 13.9 k | Final answer alone ≈1.1 k tokens, mostly raw data dumps |
| Images | 10 calls, 4 failed | 4× `ModuleNotFoundError: No module named 'PIL'` |
| Answer correctness | **Wrong on two key points** (see §1) | |

The three biggest levers, in order:

1. **Correctness:** the answer validated a false positive (a ball tracked as "Qbo-2" for 26 min) and
   very likely a mislabel (both visible cats labelled 26). The agent mixed a *replayed* camera (Oct-28)
   with *live* tracking data (Nov-01) for a different moment.
2. **Turns, not tools, cost the time.** Each LLM turn took 11–25 s. Cutting 9 turns to 3 saves ~100 s.
   That comes from a task-specific skill, fewer exposed tools, and tools that return "frame +
   detections for the same instant" in one call.
3. **Tokens:** 38 tool schemas (~8 k tokens) are resent every turn; several tool replies are mostly
   noise (60 identical path points, 50 buffer timestamps, S3 bucket/key metadata).

---

## 1. Correctness problems found in this run

### 1.1 Period 103063 ("Qbo-2", 26 min, 100 % detector agreement) is a ball, not a cat  — HIGH
- `step11_get_image_1.jpg` (07:35:12, the period's `last_frame`) shows **no cat**. The box
  `[351,47,407,104]` sits exactly on the pink/blue exercise ball by the back wall.
- `camera__detect model=yolov8s` on the replay frame labels the same spot `sports ball [353,52,405,103]`.
- The period's motion: `resting_s 1618.65 / duration 1621.95`, `net_displacement_px 0.7` over 26 min:
  the box never moves. The entry frame (07:09:19) has a real cat next to the ball; the tracker then
  shrinks onto the ball and keeps it.
- The agent concluded "Tracking 103063 Qbo-2: Positionally accurate … consensus very strong".

Why it slipped through:
- Crops failed (PIL), and the full frame at `max_side` default was not inspected closely.
- The prompt and skills treat *detector agreement* as identity evidence, but nothing says that a
  fully static box with 100 % agreement is the classic **decoy** pattern (the `check-trajectory` skill
  has a `static` flag for exactly this, but it was not used in chat).

**Same bug exists in code (not just the model):** `tasks.py::_task_validate_detection` rung 1
auto-accepts a *final* period with ≥20 detections and ≥80 % agreement, no model call. Period 103060
(06:44–07:08, 1415 s, 3374 detections of 26) is probably the same ball and would be auto-accepted.
Fix: in the consensus shortcut, also require "not static" (e.g. `moving_s > 0` or net displacement
> N px, or `duration_s < trajectory.static_min_s`), otherwise go to the model / `check_trajectory`.

### 1.2 Both visible cats labelled 26 (Qbo-2) is almost certainly one mislabel — MEDIUM
- `step05_capture_frame_1.jpg`: one cat on the rug (dark ears/face/tail tip → Qbo cues) and one white
  cat in the green bed (smooth white → Qb cues).
- Track 1 (bed cat): `cat_id_counts {26: 268}`, `max_conf 0.598`, last loop conf 0.37.
  Track 11 (rug cat): `{26: 151, 25: 16}`.
- With two cats in the house and two cats on screen, two simultaneous tracks with the same cat_id
  means at least one is wrong (the "exclusion" rule from `validate-cat-identity` rung 1). The agent
  said "Default cat model: ACCURATE … Both tracks dominantly 26" and never cropped the bed cat
  (its `get_track_frame` failed because the frame left the buffer).

### 1.3 Mixed timelines: replay camera vs live tracker — HIGH (design)
- `camera__get_status`: `sim: true`, replay of 2026-10-28 16:33, `websocket.connected: false`.
- `tracking__*`: live data from 2026-11-01 07:35 for the same `device_id`.
- The two sources describe different moments, but 5 of the 9 turns were spent on the Nov-01 period
  (current_periods, period_detail, arrival_path, 5 image fetches) to "validate" a camera view from
  Oct-28. The agent only noticed in the final note.
- Fix (tool side): when `sim` is true, have `get_status` return a top-level
  `"warning": "replay mode: timestamps are replay times; tracking__ live data is NOT from this stream"`.
  Prompt side: one rule: "If camera get_status.sim is true, don't cross-reference tracking__ live data
  with camera frames."

### 1.4 "empty_before" frame is not empty — LOW
- `get_arrival_path` key frame `empty_before` (07:09:18.73) already shows the cat; the agent noticed.
- `history.last_empty_frame` picks the newest frame with `detected = 0`, i.e. "no detection above
  threshold", which includes frames where the cat is present but undetected. 0.35 s before entry is
  almost never truly empty.
- Fix: require a minimum gap before entry (e.g. ≥ 2–3 s with no raw detections at all near the entry
  box), or rename to `last_frame_without_detection` and say so in the description.

---

## 2. Bugs

| # | Where | Bug | Effect in this trace / risk |
|---|---|---|---|
| B1 | image server env | Pillow not installed in the running image-server venv (it is in `requirements-mcp.txt`) | 4 failed `get_image` calls, ~1.9 s and 1 whole turn wasted; agent fell back to full 800×600 frames |
| B2 | `image_store.transform` | `from PIL import Image` is lazy, so the error is found only *after* the S3 download (200–330 ms each) | Fail fast: import at startup and log/disable `box`/`max_side`, or return the original bytes with a `"warning"` instead of an error |
| B3 | `get_image` error text | `ModuleNotFoundError` is not actionable; the model retried 3 more crops in turn 7 | Error should say "crop/resize unavailable on this server; call without box/max_side, or use local__make_contact_sheet (crops locally)". `make_contact_sheet` would have worked here |
| B4 | `agent.py::_budget_refusal` | Image budget is reserved *before* the call and never refunded on error | In tasks (budget 2–3), a PIL / not-found error eats the budget → forced `send_to_human` |
| B5 | `agent.py` `IMAGE_TOOLS` + `prompts.py` | `camera__capture_roi` is referenced but no such tool exists on the camera server | Prompt advertises a tool the model can't call |
| B6 | `prompts.py` | "always give a **ttl_s**" but `set_schedule`'s parameter is `duration_seconds` | Model may send an unknown arg |
| B7 | `agent.py` `self._opts` | Per-run options stored on the shared agent instance | Two concurrent `run()` calls on one `MCPAgent` (e.g. a future web server or parallel tasks) would read each other's budgets / tool lists. Pass `opts` through `_execute` instead |
| B8 | `tasks.py::_ask` retry | Retry reuses `opts` incl. `instructions`, so `_fold_instructions` **re-sends the whole system prompt + skill** into a chained conversation that already has it | Duplicate ~1–2 k tokens per retry. Retry should pass `instructions=""` |
| B9 | `tasks.py::_merge_runs` | Doesn't merge `tools_ms`, `images_used`, `high_res_used`, `stopped_reason` | Under-reported metrics in task logs |
| B10 | `camera__get_track_frame` | `get_track_history` returns tracks whose frames already left the 43 s buffer | Turn 8 call failed. Add `first_frame_buffered` / `last_frame_buffered` booleans to track history |
| B11 | tracking `get_arrival_path` timing | `pick` step 7.19 s server-side, although the same `compute_motion` over the same rows took 64 ms in `get_period_detail` in the same turn | CPU work runs on the ingest server's shared event loop; with concurrent requests + `asyncio.to_thread` DB calls it starved. Move `load_period`+`compute_motion` into one `to_thread` call, cache `load_period` per period for a few seconds (both tools loaded the same 4 k rows in parallel), and profile |
| B12 | timestamps | Tracking `now` is naive UTC (`07:35`), the run started at local `00:35`, camera replay times are naive local | Model can't reason about "now" reliably. Emit ISO with offset (`Z` / `-07:00`) everywhere |
| B13 | `local__get_cat_profiles` | Enum allows `low_res`/`full`, `Store.profiles` also accepts `camera`/`high_res`; returned `cat_id` is the profile id (0/2) even when `view_id=3` is given | Model has to map 0/2 ↔ 25/26 itself. When `view_id` is given, return that view's tracker id as `cat_id` (keep `profile_id` separately) |
| B14 | `skills.py` docstring | Says skills are exposed as `local__<name>`, they are `skill__<name>` | Doc only |
| B15 | `image_exists` description | "Fast check whether an image is on the SSD" — it also checks S3 | Doc only |

---

## 3. Inefficient tool design (tokens + time)

### 3.1 38 tool schemas on every turn (~8 k tokens/turn)
Turn 1 input = 9,722 tokens; system prompt ≈1,622, query ≈33 → **≈8 k tokens of tool schemas**,
resent on all 9 turns (≈72 k of 158 k input tokens). Options, cheapest first:
- **Drop write / rare tools from default chat:** `set_schedule`, `reset_schedule`, `save_zone`,
  `add_label`, `skill__manage-labels`, `list_models`, `get_camera_modes`, `get_raw_detections`.
  Expose them only when the query asks (simple keyword router, or a `skill__*` that unlocks them).
- **Hide pipeline-only skills in chat:** `validate-cat-identity`, `check-trajectory`,
  `assign-activity-label` are written for `tasks.py` inputs (see §4.1).
- **Merge overlapping tools:** `image_exists` + `images_exist` → one batch tool;
  `get_live_status` + `get_last_seen` + `get_current_periods` overlap heavily;
  `get_camera_modes` can be a field of `get_status`.
- Trim long descriptions (`capture_high_res` ≈700 chars; `get_period_detail` documents 6 tuning
  params the model never needs — hide `speed_threshold`, `min_moving_s`, `merge_gap_s`, `window_s`,
  `min_distance_px` from the schema and keep server defaults).

### 3.2 No way to see "a frame + the model's detections on that same frame"
This was the user's actual question and it took 6 calls that still never lined up in time:
`get_status.last_message` (16:33:50), `capture_frame` (16:34:15), `detect yolov8s` (16:34:28),
`detect default` (16:34:48), `track_frame` (16:35:06). Every one is a different frame.
- Add `draw_detections: bool` (and `detections` in the JSON) to `capture_frame`: return the frame
  with the loop's boxes for that frame drawn + labelled. One call answers "what does the model see vs
  what is there". The camera already has the boxes per buffered frame.
- Or: `detect(at=<captured_at>, return_image=true)` to run and draw on the exact same frame.

### 3.3 Verbose replies
| Tool | Tokens | Noise |
|---|---|---|
| `tracking__get_period_detail` | 1,884 | `path`: 60 points, 58 of them `x≈379, y≈75`. Dedupe consecutive points within a few px, or omit `path` when `motion.segments` already describes it. `params` block repeats defaults. `period` duplicates `first_frame`/`last_frame` with the top-level ones |
| `tracking__get_current_periods` | 1,148 | 10 `recently_final` periods with full `evidence_by_cat` incl. zero rows (`"25": {0, 0.0, 0.0}`). Default `final_limit` 3, drop zero-evidence entries |
| `camera__list_buffered_frames` | 386 | 50 raw timestamps; the model only needs `oldest/newest/count`. Default `limit=0` |
| `images__get_image` (success) | 173 | `s3_bucket`, `s3_key`, `relative_path`, `modified_at`, `mime_type`, `returned_mime_type`, `size_bytes` — none useful to the model. Keep `image_name`, `source`, `width/height`, `timing.elapsed_ms` |
| `camera__get_status` | 344 | `control.defaults`, `websocket.queue`, `camera_read_ms`, `frame_buffer.bytes`… Fine for debugging, too much as the first call. A `verbose=false` default would help |
| every tool result | ~15 | The agent prepends `{"tool": ..., "round_trip_ms": ...}` to every result. The model already knows the tool name; round-trip is only needed for the trace (it is already in `ToolStep`) |

### 3.4 `camera__detect` "bigger model" is a generic COCO model
The prompt says "detect to re-check with a bigger model". `yolov8s` is COCO (bicycle, sports ball,
dog…), `cat_id: -1`. It cannot validate cat identity and called the rug cat "dog 0.39". The model
spent a call and part of the answer on it. Either describe it honestly in the prompt / tool
description ("generic COCO classes, only useful for is-it-a-cat-or-an-object; no cat identity"), or
filter its output to animal classes + flag `cat_classes: false` in the reply.

### 3.5 Image size defaults
- `images__get_image` default = full stored size (800×600 here). With PIL fixed, a server-side
  default `max_side` (e.g. 640) or agent-side default saves image tokens on every full-frame look.
- `capture_frame` default `max_side=1280` is larger than the stream (800×600) — harmless but a smaller
  default for "overview" looks (e.g. 640) is cheaper; crops already get detail from `box`.
- `image_detail="auto"` for everything; consider `"low"` for overview frames and `"high"` only for crops.

---

## 4. Skills

### 4.1 Chat loaded a pipeline skill that doesn't fit (cost: 1 turn, 14 s, ~850 tokens)
In turn 2 the model called `skill__validate-cat-identity`. That skill is written for the
`validate_detection` **task**: it refers to `context`, `cats`, `crop` inputs that don't exist in chat,
says "Return only the JSON object the schema asks for", and its ladder is about one stored detection.
The model then had to improvise. Its description even says "Use for validate_detection tasks from the
CV pipeline" — but it is still exposed in chat because `SkillTools` registers every SKILL.md.

Suggestions:
- Add frontmatter like `surface: task | chat | both` and only register `chat`/`both` skills as
  `skill__*` tools. `tasks.py` already loads task skills directly as instructions.
- Write a chat skill for this exact recurring request, e.g. **`check-live-detections`**:
  1. `camera__get_status` (note `sim`, `capture_size`, loop model). If `sim`, don't use tracking live data.
  2. `camera__capture_frame(draw_detections=true)` — one frame with the loop's boxes (§3.2).
  3. If any box looks wrong or a cat has no box: one `local__make_contact_sheet` / crop per box
     (all in one turn), plus `local__get_cat_profiles(view_id)`.
  4. Rules: two simultaneous boxes with the same cat_id ⇒ at least one is wrong; a box that never moves
     for minutes ⇒ check for a decoy; COCO models are not identity checks.
  5. Answer as a small table: box → what is there → model label → verdict (TP / FP / FN / wrong ID).
  Expected: 3 turns, ~4–6 tool calls, ~30–40 s.

### 4.2 Skill payload format
- `SkillTools.call` returns `{"skill": ..., "instructions": "<full file incl. frontmatter>"}` as JSON,
  so the markdown arrives with `\n` escapes and the frontmatter repeated. Return plain text and strip
  the frontmatter (as `tasks.load_skill` already does). Saves ~10–15 % of the skill's tokens and reads better.
- `tasks.load_skill` and `skills.parse` both parse frontmatter separately; use one helper.

### 4.3 Skill content
- `validate-cat-identity` rung 1 "visit consensus … accept the visit's cat": add "unless the visit is
  static (decoy check)" — see §1.1. Same for the code shortcut.
- `validate-cat-identity` hard-codes "on view 3: Qb = 25, Qbo = 26" and the tail cue. Fine for now, but
  it duplicates `cats.yaml`; the `cats` input already carries cues. Keep the ids in one place.
- `check-trajectory` and `validate-cat-identity` both restate the Qb/Qbo quick cues — fine as a short
  reminder, but if cues change, three places need edits (skills + `cats.yaml`).
- `manage-labels` step 3 asks for `camera__capture_frame(max_side=800)` "to see the room": a stored
  frame or zone overlay image would avoid touching the live camera; minor.
- `assign-activity-label` is good: code pre-filters candidates, one image. No change needed.

---

## 5. Prompt redundancy / clarity (`prompts.py` + server notes)

- **Tool-group section repeats tool descriptions.** The model already gets every tool's description;
  the 4 group paragraphs (~450 tokens) restate them. Keep 1 line per group (what the server is for)
  and the cross-tool rules only.
- **Server notes repeat the base guidelines:**
  - images notes: "Use image_exists / images_exist first … then get_image" = base guideline 1.
  - camera notes: capture_frame vs capture_high_res = base guideline 2. Box coordinates stated 3 times
    (base, camera notes, tracking notes).
  - tracking notes: `image_name` → `images__get_image(box=crop_box)` = base guideline 6.
  - `[local server notes]` and `[skill server notes]` add almost nothing (~30 tokens).
- **Contradictory / stale:** `capture_roi` (doesn't exist), `ttl_s` (should be `duration_seconds`),
  "past frames are stored on an SSD" (also S3), "detect to re-check with a bigger model" (it's COCO).
- **"Prefer cheap calls first (status, existence checks)"** encourages a broad fan-out on turn 1
  (status + live_status + list_views) even when only the camera matters. Better: "Start from the
  source the question names (live camera ⇒ camera__*, history ⇒ tracking__/images__)".
- **Output length:** "mention which tools you used and the server-side elapsed_ms" produced a long
  tool list in the final answer; the trace already records timings. Make it opt-in ("only if the
  user asks about speed") and add "Answer in ≤ ~200 words unless asked for detail; don't paste raw
  JSON". The final answer was 1.1 k tokens of mostly copied tool output; the last turn took 24.7 s.
- **Missing rules that would have saved this run** (one line each):
  - "If camera get_status.sim is true, camera frames and tracking live data are different moments."
  - "A box that stays still for many minutes with high agreement may be an object (decoy); look at it."
  - "Two cats on screen with the same cat_id ⇒ one label is wrong."
  - "If a tool error says a feature is unavailable, don't retry the same arguments."
  - "Fetch all images you need in one turn" (generalise the existing arrival-path rule).

---

## 6. Agent loop / platform

- **Turn latency dominates.** Per-turn LLM time 11–25 s (turn 1: 2 s). Options:
  set `reasoning_effort: low` for chat (models.json currently `null`), fewer turns via skills (§4.1),
  and shorter final answers (§5).
- **Prompt caching is inconsistent.** Cached tokens per turn: 0, 113, 113, 113, 12 k, 113, 11 k, 24 k,
  25 k. Turns 2/3/4/6 barely hit the cache despite `previous_response_id` chaining. Worth a probe
  (similar to `documents/model-api-chaining-probe.md`): does the cache miss after turns that add
  images? Do changing `tools` order / content break it? (Tool order is dict order, stable, but verify.)
  If every turn hit like turns 8–9, input cost would drop ~40 %.
- **Parallel calls in one turn compete on the same server** (B11). The tracking server got
  `get_period_detail` and `get_arrival_path` for the same period at once and both loaded the same
  ~4.3 k rows (DB 418 ms + 458 ms). A short-TTL cache for `load_period` helps both.
- **Trace token column is chars/4 and excludes images and tool schemas**, so the TOC suggests the
  system prompt is the biggest cost; actually tool schemas and images are. Suggest adding per-turn
  "Δinput tokens" from real usage, plus a one-time "tool schema tokens" line.
- `to_function_tool` truncates descriptions at 1,024 chars silently; log a warning when it happens.

---

## 7. What an efficient run of this query looks like

| Turn | Calls | Notes |
|---|---|---|
| 1 | `camera__get_status`, `camera__capture_frame(draw_detections=true, max_side=800)` | sim noticed; frame + loop boxes for the same instant |
| 2 | `local__make_contact_sheet` (crop of each box + any unboxed cat), `local__get_cat_profiles(view_id=3)` | one image for all crops |
| 3 | answer | table of boxes vs reality, short |

≈3 LLM turns (~35–45 s), ~4 calls, 2 images, ~35–40 k input tokens (vs 152 s / 22 calls / 158 k).
And it would have flagged the two-cats-both-26 issue directly.

---

## 8. Suggested priority

1. Install Pillow in the image-server env + fail-fast / actionable error (B1–B3). *Minutes.*
2. Fix the static-decoy hole in the validate consensus shortcut and skill (§1.1). *Correctness.*
3. `sim` warning in `get_status` + one prompt rule (§1.3).
4. Hide task-only skills and write-tools from chat; add `check-live-detections` skill (§4.1, §3.1).
5. `capture_frame(draw_detections=true)` (§3.2).
6. Trim verbose replies (§3.3) and the prompt (§5); fix stale names (B5, B6).
7. Budget refund on error (B4), per-run options (B7), retry instructions (B8).
8. Investigate arrival-path latency (B11) and prompt-cache misses (§6).
