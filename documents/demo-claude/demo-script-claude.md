# Demo Video Production Script (3:00)

Production version of the 3-minute demo narration. For every beat it says what goes on
screen, which format to use, the exact asset file, where that asset's data came from, and how
to animate it. Every asset listed here is already in this folder (`assets/`), except the items
under [To record or source](#to-record-or-source).

- Narration: **341 words, ~156 s of speech** at ~2.5 words/s (normal pace), plus ~24 s of
  visual holds = **3:00**.
- Canvas: **1920×1080, 30 fps**, dark theme throughout.
- Numbers: all from the merged-batch replay run `231613` (view 3, 2026-10-28 16:33–16:41,
  1,319 frames) unless a beat says otherwise. Every slide's footer names its source.
- The narration-only text for voice recording is `assets/text/narration.txt`. Burned-in or
  sidecar captions are in `assets/text/captions.srt`.
- The earlier version of this script is kept as `demo-script-claude.original.md`.
  [Changes from the original](#changes-from-the-original-script) lists what changed and why.

---

## Format legend: when to use what

| Format | Use it when… | Beats |
| --- | --- | --- |
| **Photo / real frame** | The claim is about something the camera actually saw. Always a real image from the image server or the cat reference set, never a mock-up. | 1, 2, 3, 11, 13 |
| **Video clip (generated)** | Something changes over time: a number growing, a zoom revealing a detail, the tracker running. | 2, 4, 7 |
| **Screen recording (you record)** | Proving the agent is real: the trace with its tool calls and image picks. | 10 |
| **Section title** | The story turns from problem to solution. One per video; more than that slows a 3-minute cut. | 6 |
| **Blocks / cards** | 2–4 parallel ideas a viewer should read as a set (questions, roadmap items, domains). | 3, 18, 19, 20 |
| **Table** | Two things compared on the same attributes. | 5 |
| **Equation / hero number** | A single computed number is the point. | 4 |
| **Chart** | Quantities compared or placed in time: bars for before/after cost, a timeline for when doubts fired. | 8, 12, 17 |
| **Flow chart** | There is a decision branch (the "Unsure?" diamond) and a loop. | 9 |
| **Diagram (non-flow)** | Structure without a decision: grouping (12 doubts → 3 calls), hierarchy (memory pyramid), taxonomy grid. | 8b, 12, 16 |
| **Code / data card** | The literal artifact matters: the real prompt, the real row before and after. | 10 (fallback), 14 |
| **Document card** | Output meant to be read as prose (the diary). | 15 |
| **KPI tiles** | 3–5 headline numbers at once. | 17 |
| **B-roll (you source)** | Domains we did not build for, so we have no real footage. | 19 |

---

## Visual system

| Role | Color | Used for |
| --- | --- | --- |
| Surface | `#1a1a19` | All backgrounds |
| Qb | `#3987e5` blue | Every Qb box, label, bar |
| Qbo | `#d95926` orange | Every Qbo box, label, bar (including the ball decoy, since the tracker called it Qbo) |
| Edge / Pi | `#199e70` green | "Cheap eyes" |
| Agent / Muse Spark | `#9085e9` violet | Agent verdicts, "corrected", the merged bars |
| Doubt | `#fab219` yellow | Escalations, always paired with a code label (T2, P5…) |
| Text | `#f2f1ed` / `#c3c2b7` / `#8e8d84` | Primary / secondary / muted |

The four identity colors pass a colorblind-safety check on the dark surface (worst adjacent
CVD ΔE 9.4, all ≥3:1 contrast). Type is the system sans (SF Pro) for text and SF Mono for
codes and data. Keep the editor's lower-thirds and titles in the same type.

**Identity rule:** blue always means Qb and orange always means Qbo, on every slide and clip.
Never recolor them for emphasis.

---

## Shot list

### Beat 1 · Meet the cats · 0:00–0:05 (5 s)

> **VO:** "These are Qb and Qbo. My camera can't tell them apart."

- **Format:** Photo pair (hook).
- **Asset:** `assets/slides/s01_meet_qb_qbo.png`
- **Sources:** `assets/reference/qb_standing_front.jpg`, `assets/reference/qbo_on_back_face.jpg`
  (from `agent/agent_data/references/*/profile/`, the same photos the agent uses for identity).
  The shipping label beside Qb is blurred because it showed a home address.
- **On screen:** names and the one visual cue that tells each cat apart (thin upright tail vs
  dark plume tail). Beat 11 pays this off.
- **Motion:** slow 3% push-in on both panels; names fade up at 0:01.

### Beat 2 · The ball · 0:05–0:11 (6 s)

> **VO:** "Last week, it decided this ball was Qbo, for twenty-six minutes."

- **Format:** Video push-in, then a hero-number card.
- **Assets:** 0:05–0:09 `assets/video/v02_ball_push_in.mp4` (5 s, trim the tail);
  0:09–0:11 `assets/slides/s02_ball_decoy.png`
- **Source:** real frame `2026-11-01-07-35-00-150851.jpg` (image server, S3), box
  `[351,47,407,104]` and "26 min / 0.7 px / 100% agreement" from
  `documents/agent-trace-review-20261101-003501.md` §1.1.
- **Why:** the joke lands only if the viewer sees it's obviously a ball, so the zoom ends
  tight on it.
- **Motion:** cut on the word "ball". Hold s02 under the start of beat 3 if the VO runs long.

### Beat 3 · What I want to know · 0:11–0:22 (11 s)

> **VO:** "I work all day. I just want to know simple things: did my cats eat, sleep, play? Cats hide illness, so small changes matter."

- **Format:** Blocks over a dimmed real frame.
- **Asset:** `assets/slides/s03_simple_questions.png`
- **Source:** background `2026-11-03-16-30-00-139271.jpg` (empty room in daylight, one cat
  asleep in the green bed). Claim: AAFP senior care brochure (report §3.1).
- **Motion:** the three cards pop in on "eat", "sleep", "play"; the yellow note slides in on
  "Cats hide illness".

### Beat 4 · The cost of sending everything · 0:22–0:34 (12 s)

> **VO:** "Why not send every frame to an AI? One camera makes over a hundred and seventy thousand frames a day. That's about sixty-three thousand dollars a year."

- **Format:** Animated counter → equation hero numbers.
- **Assets:** 0:22–0:28 `assets/video/v01_cost_counter.mp4` (counts 0 → 172,800 and
  $0 → $63,072); 0:28–0:34 `assets/slides/s04_cost_math.png` (the full equation, plus the
  "but it makes mistakes" line that bridges to beat 5).
- **Source:** report §10.1: 2 fps × 86,400 s = 172,800; $0.001/frame ≈ 800 input tokens at
  $1.25/M.
- **Motion:** time the counter so it lands on 172,800 as the VO says "a day".

### Beat 5 · Two halves · 0:34–0:41 (7 s)

> **VO:** "A tiny model on a Raspberry Pi is almost free, but it makes mistakes."

- **Format:** Table (two things compared on the same attributes).
- **Asset:** `assets/slides/s05_two_halves.png`
- **Source:** report §4 (Pi 5 latency from Ultralytics, power from Sixfab, review time from
  our traces).
- **Motion:** reveal the "Fast eyes" column on "tiny model", then the "Smart brain" column.
  Rows stay static.

### Beat 6 · Section title · 0:41–0:44 (3 s)

> **VO:** "So we built both."

- **Format:** Section title, the only one in the video.
- **Asset:** `assets/slides/s06_title_built_both.png`
- **Motion:** line 1 then line 2, 0.4 s apart. Short musical lift here.

### Beat 7 · The Pi watches, and says why · 0:44–0:55 (11 s)

> **VO:** "The Pi watches every frame. When it's unsure, it says exactly why."

- **Format:** Video (real tracker replay) → timeline chart.
- **Assets:** 0:44–0:50 `assets/video/v03_replay_tracker.mp4`. Use 0:00–0:06 of the clip,
  where the E5 doubt banner pops at 16:34:36. Then 0:50–0:55
  `assets/slides/s08_timeline.png`.
- **Source:** v03 = real frames `image_server/retrieved_images/2026-10-28-16/images/`, boxes from
  `revised.jsonl` (the Pi tracker's own labels), doubts from `escalations.jsonl`, verdicts from
  `validations.jsonl`. s08 = the same three files, full 8-minute window.
- **Why the timeline:** it shows rarity, 12 yellow flags across 1,319 frames, better than
  any sentence can.
- **Honesty note (keep it on screen):** the Qbo lane annotation says most of that lane is
  really Qb asleep in the bed. The tracker mislabeled it and no doubt fired. This is the open
  problem from report §12. Don't crop it out.
- **Motion:** on s08, draw the lanes first, then drop the 12 markers left to right in ~1.5 s.
  Purple = corrected by the agent, gray = upheld.

### Beat 8 · Fourteen kinds of doubt · 0:55–1:03 (8 s)

> **VO:** "It knows fourteen kinds of doubt, like: I relabeled this cat by distance alone."

- **Format:** Taxonomy diagram (grid).
- **Asset:** `assets/slides/s09_doubt_taxonomy.png`
- **Source:** `agent/escalation/review_schema.py`, report §7.2. Badges = codes that fired in
  this window (T2 ×3, T7 ×3, P2 ×2, T4, T6, P5, P6).
- **Motion:** grid fades in dimmed; T2 lights up on "distance alone".
- **Note:** the code now also has P8 and H1 (added after the measured runs). The footnote
  mentions them; the VO keeps "fourteen" because that's what was measured.

### Beat 9 · The loop · 1:03–1:08 (5 s)

> **VO:** "That doubt becomes a plain-English question for Meta's Muse Spark."

- **Format:** Flow chart, because it has a decision ("Unsure?") and a feedback loop.
- **Asset:** `assets/slides/s07_loop_diagram.png`
- **Motion:** build in reading order: Pi → Correctors → Unsure? → (no) Tracked rows. Then
  (yes) Plain-English question → Muse Spark agent → verdict back up, and finally the blue
  memory loop. If time is tight, show the full chart and pulse only the "yes" path.

### Beat 10 · The agent works · 1:08–1:15 (7 s)

> **VO:** "The agent pulls the frames it needs, compares the cats, and answers with a reason."

- **Format:** Screen recording (preferred), with the code card as fallback.
- **Asset (record):** **REC-1**, see [To record](#to-record-or-source). Scroll
  `output/agent_runs/20261102-231717-320413-validation/trace.md` in a Markdown preview:
  the case prompt → `images__get_image` calls with thumbnails → the JSON verdict.
- **Fallback:** `assets/slides/s10_question_card.png`, the real E5-T2 prompt and T2 verdict
  schema, condensed.
- **Motion:** 2× speed scroll. Hold 1 s on the final `{"verdict": "keep_original", …}`.

### Beat 11 · Case E5: overturned · 1:15–1:24 (9 s)

> **VO:** "Watch. Cool white, thin upright tail. This is Qb, not Qbo. Overturned."

- **Format:** Real frame with boxes + reference photos + quote + stamp.
- **Asset:** `assets/slides/s11_case_e5.png`
- **Source:** frame `2026-10-28-16-34-36-154433.jpg` (`assets/frames/…_E5.jpg`), box
  `[296,369,371,531]`, verdict `keep_original`, rationale from trace `20261102-231717`
  (shortened: "cool white, smooth sharp outline, thin upright tail = Qb 25, not … Qbo 26").
- **Motion:** first show only the orange "tracker: Qbo" box (0.8 s). Then the reference photos
  slide in and the quote types on. On "Overturned", the blue "agent: Qb" label snaps under the
  box and the stamp drops in. If you animate, build it from `scripts/html/s11_case_e5.html`
  (one HTML layer per element).

### Beat 12 · Merged review · 1:24–1:32 (8 s)

> **VO:** "Related doubts get reviewed together, fetching each image once. Twelve questions, three calls."

- **Format:** Grouping diagram (left) + bar chart, solo vs merged (right).
- **Asset:** `assets/slides/s12_batching.png`
- **Source:** report §9.2. Run `012059` (one call per doubt) vs `231613` (merged): calls
  12 → 3, tokens 517.6k → 229.6k, cost $0.60 → $0.26, time 9.4 → 3.8 min.
- **Motion:** the 12 chips fly into 3 call boxes on "together", then the four violet bars
  shrink from full width to their merged length on "three calls".

### Beat 13 · Case E10: the imagined cat · 1:32–1:46 (14 s)

> **VO:** "Here, the tracker kept painting Qbo onto Qb's spot after Qbo had walked away. The agent saw no Qbo there, and cut the guesses."

- **Format:** Real frame with boxes + quote + stamp.
- **Asset:** `assets/slides/s13_case_e10.png`
- **Source:** frame `2026-10-28-16-35-36-588746.jpg`, guessed box `[178,465,266,585]` held
  for 5.2 s, verdict `cut_at_gap_start`, 14 interpolated rows removed (16:35:36.9–16:35:41.8 in
  `revised.jsonl`, absent from `validated.jsonl`). The "Qbo is here" box over the cat house is
  hand-placed for viewers; the agent's rationale names the "cat in left house".
- **Motion:** dashed box pulses twice, then the "Qbo is here" box draws on, then CUT.

### Beat 14 · Verdicts become data · 1:46–1:51 (5 s)

> **VO:** "Every verdict rewrites the record, and every change is traceable."

- **Format:** Data card, before and after.
- **Asset:** `assets/slides/s14_row_diff.png`
- **Source:** the real E5 row at 16:34:36.154 from `revised.jsonl` and `validated.jsonl`,
  run `231613`.
- **Motion:** left card first; the right card slides over it with the three violet lines
  highlighting in sequence.

### Beat 15 · Memory as a diary · 1:51–2:02 (11 s)

> **VO:** "Then memory. Not boxes and timestamps, but a diary: Qb slept in the green bed, four thirty-four to four forty."

- **Format:** Document card (paper style, serif), on purpose the one light element in the
  video.
- **Asset:** `assets/slides/s15_day_log.png`
- **Source:** day entry from `agent/memory/data/view_3_short_term_2026-10-28.md` (latest
  session: sleeping, 0.92, 772 sightings). The zone line is verbatim from
  `agent/memory/data/view_3_long_term.yaml`.
- **Motion:** cross-dissolve from s14 (rows → prose is the point). Typewriter on the first
  sentence, timed to the VO.

### Beat 16 · Learning normal · 2:02–2:09 (7 s)

> **VO:** "Over weeks, it learns what normal looks like for each cat, in each spot."

- **Format:** Hierarchy diagram (two pyramids).
- **Asset:** `assets/slides/s16_memory_pyramid.png`
- **Source:** report §7.7 (48 cells → 12 zones → view; 5-min → hour → day → week, with
  retention; trust gate 0.7; 3 sightings; 3 contradictions).
- **Motion:** pyramids build bottom-up ("each spot" = left, "over weeks" = right); the three
  rule pills fade in last.

### Beat 17 · Results · 2:09–2:29 (20 s)

> **VO:** "The numbers: twelve doubts in eight minutes, less than one percent of frames. Twenty-six cents to review them, about a fifth of the cost of sending every frame to the AI. And only those frames ever leave my home."

- **Format:** KPI tiles + horizontal bar chart.
- **Asset:** `assets/slides/s17_results.png`
- **Source:** report §1 and §10.2. All-frames baseline 1,319 × $0.001 = $1.32; solo $0.60 (45%);
  merged $0.26 (19%). Memory adds ~$0.14 (in the footer, not hidden).
- **Motion:** tiles count up one by one, each on its phrase: "twelve doubts" → "twenty-six
  cents" → "leave my home". Bars grow on "a fifth of the cost".

### Beat 18 · Roadmap · 2:29–2:37 (8 s)

> **VO:** "Next: catch more errors, run full days, and teach the Pi from what the agent learns."

- **Format:** Three blocks.
- **Asset:** `assets/slides/s18_roadmap.png`
- **Source:** report §16, priorities 1, 2, 6. Card 01 is the fix for the honesty note in
  beat 7.
- **Motion:** cards rise in, one per clause.

### Beat 19 · Beyond cats · 2:37–2:48 (11 s)

> **VO:** "Cats are just the start. Any camera watching for rare moments, like an elderly parent, wildlife or a factory line, has the same gap."

- **Format:** Blocks with B-roll inside each card.
- **Asset:** `assets/slides/s19_beyond_cats.png` (the hatched areas are placeholders).
  Replace each with B-roll **19a / 19b / 19c**, see [To source](#to-record-or-source).
- **Source:** report §14.
- **Motion:** each card's B-roll plays as its domain is named.

### Beat 20 · Close · 2:48–3:00 (12 s)

> **VO:** "Our answer: cheap eyes on every frame, a smart brain only when it matters, and a memory that keeps growing. Built this week, on Meta's Muse Spark."

- **Format:** End card (three-part tagline + credits).
- **Asset:** `assets/slides/s20_end_card.png`
- **Motion:** the three columns appear one per clause; the credit line fades in on "Built
  this week". Hold 2 s of silence before the cut to black.

---

## Asset index

| File | Kind | Used in |
| --- | --- | --- |
| `assets/slides/s01_meet_qb_qbo.png` | photo pair | beat 1 |
| `assets/slides/s02_ball_decoy.png` | frame + hero numbers | beat 2 |
| `assets/slides/s03_simple_questions.png` | blocks over frame | beat 3 |
| `assets/slides/s04_cost_math.png` | equation | beat 4 |
| `assets/slides/s05_two_halves.png` | table | beat 5 |
| `assets/slides/s06_title_built_both.png` | section title | beat 6 |
| `assets/slides/s07_loop_diagram.png` | flow chart | beat 9 |
| `assets/slides/s08_timeline.png` | timeline chart | beat 7 |
| `assets/slides/s09_doubt_taxonomy.png` | taxonomy grid | beat 8 |
| `assets/slides/s10_question_card.png` | code card | beat 10 (fallback) |
| `assets/slides/s11_case_e5.png` | case slide | beat 11 |
| `assets/slides/s12_batching.png` | diagram + bars | beat 12 |
| `assets/slides/s13_case_e10.png` | case slide | beat 13 |
| `assets/slides/s14_row_diff.png` | data card | beat 14 |
| `assets/slides/s15_day_log.png` | document card | beat 15 |
| `assets/slides/s16_memory_pyramid.png` | hierarchy diagram | beat 16 |
| `assets/slides/s17_results.png` | KPI tiles + bars | beat 17 |
| `assets/slides/s18_roadmap.png` | blocks | beat 18 |
| `assets/slides/s19_beyond_cats.png` | blocks (B-roll slots) | beat 19 |
| `assets/slides/s20_end_card.png` | end card | beat 20 |
| `assets/video/v01_cost_counter.mp4` | 6 s counter | beat 4 |
| `assets/video/v02_ball_push_in.mp4` | 5 s push-in | beat 2 |
| `assets/video/v03_replay_tracker.mp4` | 27 s replay, 16:34:25–16:35:45 at ~3× | beat 7 (and spare B-roll for beats 11–13) |
| `assets/frames/*.jpg` | raw real frames (800×600) | sources for s02, s03, s11, s13 |
| `assets/reference/*.jpg` | cat profile photos (Qb label blurred) | s01, s11 |
| `assets/text/narration.txt` | VO script | voice recording |
| `assets/text/captions.srt` | captions | edit |

Slides are 1920×1080 PNG. Each also exists as editable HTML in `scripts/html/`, so an
editor can screen-capture it with CSS animation or re-layer it.

---

## To record or source

| ID | What | How | Beat |
| --- | --- | --- | --- |
| **VO** | Voice-over, 341 words | Read `assets/text/narration.txt` at a relaxed pace. Each line is one beat; leave ~1 s between lines. | all |
| **REC-1** | Agent trace screen recording, ~15 s raw | Open `output/agent_runs/20261102-231717-320413-validation/trace.md` in a Markdown preview (dark theme, 125% zoom). Scroll from "Case 1 of 4: E5-T2" through the `images__get_image` thumbnails to the final JSON. Record at 1920×1080. | 10 |
| REC-2 (optional) | Live chat with the agent | `cd agent && python -m supervisor_agent "Is any cat on camera right now?"` in a large-font terminal. Use only if beat 10 needs variety. | 10 alt |
| **19a** | Elderly-parent B-roll | Stock: hands with a kettle or armchair, **no faces** (consent theme) | 19 |
| **19b** | Wildlife B-roll | Stock: trail-cam footage, night IR | 19 |
| **19c** | Factory B-roll | Stock: conveyor close-up | 19 |
| Music | Low, steady bed; one lift at beat 6 | Royalty-free | all |

---

## Changes from the original script

1. **Beat 13 narration was factually wrong, and is fixed.** The original said the agent
   "caught the tracker inventing a cat on an empty floor." Drawing the real guessed box
   `[178,465,266,585]` on frame 16:35:36 shows it sits on **Qb in the green bed**, not on
   empty floor; Qbo had gone into the cat house. The agent's cut was correct (no Qbo at that
   spot), but its own rationale's "empty wood floor" wording is wrong. The new line describes
   what the frame shows. **The same "empty floor" wording appears in the report (§1, §9.2,
   §9.3) and should be corrected there too.**
2. **The beat 11 quote matches the trace.** "Thin upright tail, cool white fur" became "Cool
   white, thin upright tail", which is closer to the agent's real rationale and matches the
   slide.
3. **Pacing.** The original crammed ~50 words into its last 15 s (3.5 words/s). The new
   timecodes are computed from word counts at ~2.5 words/s, so beats 19–20 get 23 s.
4. **Section split.** "It knows fourteen kinds of doubt" now has its own beat (8) and a slide,
   because it's the core idea and deserves a visual.
5. Ending: "Cats are just the start… same gap." and "Our answer…" are now two beats with
   separate visuals.

## Open questions for you

- **"Fourteen" vs the current code.** `review_schema.py` now has P8 and H1 too (16 codes,
  15 doubt kinds plus a heartbeat). The measured runs used 14, so the VO says fourteen and
  s09 footnotes the new two. Change both if you'd rather cite the current code.
- **The open problem shown in beat 7.** s08 says most of the Qbo lane is really Qb, which no
  doubt caught. That's honest and sets up the roadmap, but it's a viewer-visible weakness. If
  you'd rather not show it, crop s08 to 16:33–16:36 or remove the annotation in
  `scripts/build_slides.py` (`s08_timeline`). The report already discloses it (§12).
- `documents/demo-muse/assets/` has an older asset set. Its `chart_cost.png` labels the
  8-minute all-frames cost as **$25.61**, which is the SAM labeling cost (§9.6). The correct
  all-frames LLM cost is **$1.32**. Don't mix the two sets.

## Rebuild

```bash
# slides (headless Chrome renders scripts/html/*.html -> assets/slides/*.png)
.venv/bin/python documents/demo-claude/scripts/build_slides.py          # all
.venv/bin/python documents/demo-claude/scripts/build_slides.py s11 s13  # some
# clips (PIL frames + ffmpeg)
.venv/bin/python documents/demo-claude/scripts/build_videos.py          # v01 v02 v03
# narration + captions from this script's timecodes
.venv/bin/python documents/demo-claude/scripts/build_text.py
```
