# Demo Video Production Script (3:00)

Production version of the 3-minute demo narration. For every beat it says what goes on
screen, which format to use, the exact asset file, where that asset's data came from, and how
to animate it. Every asset listed here is already in this folder (`assets/`), except the items
under [To record or source](#to-record-or-source).

- Narration: the final cut is **voice first**: `assets/text/voiceover_v2.txt` is read as one
  continuous take (Kokoro neural TTS, voice `af_heart` blended 20% with `af_nicole`, soft and friendly, 572 words) and the
  video is timed to it (`assets/audio/voiceover_timings.json`), 175.1 s in total. The beat
  timecodes below are the planning grid; the voice track's timings are what the video uses.
- Canvas: **1920×1080, 30 fps**, dark theme throughout.
- Numbers: all from replay run `20261104-140502` (view 3, 2026-10-28 16:33–16:41, 1,319
  frames, current code, merged review) unless a beat says otherwise. Every slide's footer names its source.
- The narration-only text for voice recording is `assets/text/narration.txt`. Burned-in or
  sidecar captions are in `assets/text/captions.srt`.
- The earlier version of this script is kept as `demo-script.original.md`.
  [Changes from the original](#changes-from-the-original-script) lists what changed and why.

## Positioning and scope

Say this the same way everywhere (VO, slides, Q&A):

- **Name and promise:** *Agentic Realtime Computer Vision*, a system meant to run 24 hours a
  day, all year round. The AI gives the camera a brain; the camera gives the AI eyes.
- **Where it goes next (mention briefly, beat 22 only):** with Meta's **Muse Glimmer** model
  deployed locally, the same loop gives a personal agent eyes on the physical things you care
  about, with no cloud in the loop.

- **What we built:** a high-performance workflow for **compute-limited environments**, such as
  local deployment on a Raspberry Pi or private deployment where video shouldn't leave the
  site. The detector stays small enough to run on that hardware, and only ~1% of frames (the
  doubts) are ever sent to the agent.
- **Hardware fact:** a Raspberry Pi 5 CPU runs only **YOLO nano, at ~3 frames/s** (the camera
  captures 2 fps); larger models need more compute or a GPU. This project therefore uses YOLO
  nano on the Pi 5 CPU as its base model. Shown on s04 (bottom panel) and s05.
- **What we admit:** a larger, retrained computer vision model would detect better. Training
  one needs a big labeled dataset, GPU training and hardware this setting doesn't have, so it
  is **out of this project's scope**.
- **The project's scope:** take a small model with limited detection accuracy as a given, and
  make its output trustworthy with AI: the tracker names its own doubts, an agent reviews only
  those, and memory learns what's normal over time.
- **Key numbers to repeat** (from `documents/final-model-performance-20261104.md`, run
  140502, audited ground truth): **−93% false alarms** (107 → 8), **99% of what the camera saw
  gets the right cat** (was 93%), **$0.39** for the 8-minute window (≈63× cheaper than AI on
  every frame), **~1% of frames** raise a doubt. Hardest 2 minutes: precision 0.53 → 0.97.
  Recall is capped by the small detector (it boxes ~40% of cats at all), which is the stated
  scope, not a hidden weakness.
- Beats 5 and 6 carry this framing (slides `s05_two_halves`, `s05b_scope`, `s06_title_built_both`).
  Beat 23's "Teach the Pi" is retraining the **same small** edge model from verdicts, which
  stays consistent with the scope.

---

## Format legend: when to use what

| Format | Use it when… | Beats |
| --- | --- | --- |
| **Photo / real frame** | The claim is about something the camera actually saw. Always a real image from the image server or the cat reference set, never a mock-up. | 1, 2, 3, 17, 18, 19 |
| **Video clip (generated)** | Something changes over time: a number growing, a zoom revealing a detail, the tracker running. | 3, 4, 9 |
| **Screen recording (you record)** | Proving the agent is real: the trace with its tool calls and image picks. | 12 |
| **Section title** | The story turns from problem to solution. One per video; more than that slows a 3-minute cut. | 6 |
| **Blocks / cards** | 2–4 parallel ideas a viewer should read as a set (questions, scope, roadmap items, domains). | 2, 6, 7, 22, 23 |
| **Table** | Two things compared on the same attributes. | 5 |
| **Equation / hero number** | A single computed number is the point. | 4 |
| **Chart** | Quantities compared or placed in time: bars for before/after cost, timelines for when doubts fired and for before/after review. | 9, 13, 20, 21 |
| **Flow chart** | There is a decision branch (the "Unsure?" diamond) and a loop. | 8 |
| **Diagram (non-flow)** | Structure without a decision: the bridge, grouping (12 doubts → 5 calls), nested memory levels, taxonomy grid. | 7, 10, 13, 16 |
| **Code / data card** | The literal artifact matters: the real prompt, the real row before and after. | 11, 14 |
| **Document card** | Output meant to be read as prose (the diary). | 15 |
| **KPI tiles** | 3–5 headline numbers at once. | 21 |
| **Photos of other domains** | Domains we did not build for (credited Commons photos). | 22 |

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

#### Act 1 · My pain point

### Beat 1 · Meet the cats · 0:00–0:05 (5 s)

> **VO:** "Hi! Meet Qb and Qbo, my two cats. They look so alike that my camera can't tell them apart."

- **Format:** Photo pair (hook).
- **Asset:** `assets/slides/s01_meet_qb_qbo.png`
- **Sources:** `assets/reference/qb_standing_front.jpg`, `assets/reference/qbo_on_back_face.jpg`
  (from `agent/agent_data/references/*/profile/`, the same photos the agent uses for identity).
  The shipping label beside Qb is blurred because it showed a home address.
- **On screen:** names and the one visual cue that tells each cat apart (thin upright tail vs
  dark plume tail). Beat 17 pays this off.
- **Motion:** slow 3% push-in on both panels; names fade up at 0:01.

### Beat 2 · What I want to know · 0:05–0:15 (10 s)

> **VO:** "I'm at work all day. I just want to know: did they eat, sleep, play? Cats hide illness, so small changes matter."

- **Format:** Blocks over a dimmed real frame.
- **Asset:** `assets/slides/s03_simple_questions.png`
- **Source:** background `2026-11-03-16-30-00-139271.jpg` (empty room in daylight, one cat
  asleep in the green bed). Claim: AAFP senior care brochure (report §3.1).
- **Motion:** the three cards pop in on "eat", "sleep", "play"; the yellow note slides in on
  "Cats hide illness".

#### Act 2 · The technical barrier

### Beat 3 · The ball · 0:15–0:20 (5 s)

> **VO:** "But here's the catch: last week, my camera decided this exercise ball was Qbo. For twenty-six minutes!"

- **Format:** Video push-in, then a hero-number card.
- **Assets:** 0:15–0:18 `assets/video/v02_ball_push_in.mp4` (5 s clip, trim the tail);
  0:18–0:20 `assets/slides/s02_ball_decoy.png`
- **Source:** real frame `2026-11-01-07-35-00-150851.jpg` (image server, S3), box
  `[351,47,407,104]` and "26 min / 0.7 px / 100% agreement" from
  `documents/agent-trace-review-20261101-003501.md` §1.1.
- **Why:** the joke lands only if the viewer sees it's obviously a ball, so the zoom ends
  tight on it.
- **Motion:** cut on the word "ball". Hold s02 under the start of beat 2 if the VO runs long.

### Beat 4 · The cost of sending everything · 0:20–0:31 (11 s)

> **VO:** "Why not send every frame to a big AI? We tried: eight minutes of video took an hour and twenty-five dollars, and still missed a cat."

- **Format:** Animated counter → KPI tiles + comparison table.
- **Assets:** 0:20–0:25 `assets/video/v01_cost_counter.mp4` (counts 0 → 172,800 frames a
  day and $0 → $1,179,446 a year at the measured $0.0187/frame); 0:25–0:31
  `assets/slides/s04_cost_math.png` (the problem only: measured time, cost and accuracy of AI
  on every frame, and the ≈ $1.2M a year line). Our own result is held back for beat 21.
- **Source:** `documents/labeling-sam-vs-spark-20261103.md`: every frame of the same window
  (1,318 frames) labeled by AI. SAM 3.1 → Spark: 66 min, $24.65 ($0.0187/frame), F1 0.794,
  Qbo recall 0.380. Spark 1.3 alone: 99 min, ~$25.61, F1 0.612, Qbo recall 0.310. Both at
  Standard pricing ($1.25/$4.25 per 1M tokens); the Contributor tier is excluded because it
  grants training rights over the footage.
  Year: 172,800 × 365 × $0.0187 ≈ $1.18M.
- **On-device limit (bottom panel):** "And a bigger model on the device? A Raspberry Pi 5 CPU
  runs only YOLO nano, at ~3 frames/s. Larger models need more compute or a GPU. So this
  project uses YOLO nano on the Pi 5 CPU as its base model." Not in the VO; it stays on
  screen for viewers who pause.
- **Why the change:** the old $0.001/frame was an input-only estimate. These numbers are
  measured on the same 8 minutes the rest of the video uses, and they add time and accuracy,
  not just money.
- **Motion:** counter lands on 172,800 as the VO says "every frame". On s04 the three tiles
  pop in on "over an hour", "twenty-five dollars", "still missed a cat"; table bars grow last.

### Beat 5 · Two halves · 0:31–0:41 (10 s)

> **VO:** "A bigger model on the Pi? It only runs YOLO nano, at three frames a second: almost free, but it makes mistakes."

- **Format:** Table (two things compared on the same attributes), green = strength, red =
  weakness, ending on the "but it makes mistakes" line.
- **Asset:** `assets/slides/s05_two_halves.png`
- **Order:** s04 shows the problem (AI on every frame, measured); s05 compares the two halves
  with the **same** numbers (8–12× slower than real time, ≈ $1.2M a year, every frame leaves
  home, still misses the sleeping cat); s05b/s06 then say "so we built both". Our result
  appears only on s17, so the video doesn't give away its ending.
- **Source:** Pi 5 CPU runs YOLO nano at ~3 frames/s (our measurement), power (Sixfab); AI column from beat 4's labeling
  shootout. "Memory" row sets up beats 15–16.
- **Motion:** hold on the empty table during "small, local hardware… private". Reveal the
  "Fast eyes" column on "tiny model", then the "Smart brain" column; the closing line lands
  on "but it makes mistakes". Rows stay static.

#### Act 3 · What we built

### Beat 6 · Scope, then section title · 0:41–0:52 (11 s)

> **VO:** "A bigger detector is out of scope. We keep the small model, and make it smarter with AI. So, we built both."

- **Format:** Blocks (out of scope vs this project) → section title, the only one in the video.
- **Assets:** 0:41–0:49 `assets/slides/s05b_scope.png`; 0:49–0:52
  `assets/slides/s06_title_built_both.png`.
- **Why say it:** judges will ask "why not just train a better detector?" Answering first, in
  one sentence, frames everything after it as the scope we chose, not a gap we missed.
- **Motion:** on s05b, the four setting pills pop in left to right under the headline; the
  dashed "Out of scope" card fades in on "bigger, retrained model", the violet "This project"
  card on "keep the small model". Cut to s06 on "So we built both": line 1 then line 2, 0.4 s
  apart. Short musical lift here.

### Beat 7 · What we built: the bridge · 0:52–1:00 (8 s)

> **VO:** "This is Agentic Realtime Computer Vision: the AI gives the camera a brain, and the camera gives the AI eyes."

- **Format:** Title card with the system in one picture (camera → this project, the bridge →
  your private agent) and a four-number proof strip that previews the results.
- **Asset:** `assets/slides/s06b_bridge.png`
- **Why here:** it answers "So we built both" with one picture of what we built (camera →
  this project, the bridge → your private agent), and the proof strip previews the results
  before the architecture beat explains how.
- **Motion:** name and the brain / eyes line first; the three boxes land on "brain" / "eyes";
  arrows draw; the proof strip fades in last and holds 1 s.

#### Act 4 · Architecture

### Beat 8 · The architecture · 1:00–1:10 (10 s)

> **VO:** "Here's how it works. The Pi watches every frame. When it's unsure, our bridge asks Meta's Muse Spark, and remembers what's normal."

- **Format:** Flow chart with three highlighted ideas (shadow boxes): 1 escalation, 2 agent
  tools + skills, 3 spatial-temporal memory. The next three sections walk through them in
  that order.
- **Asset:** `assets/slides/s07_loop_diagram.png`
- **Layout:** the three new ideas are tinted shadow boxes behind their steps, each with
  three bullets inside: **1 Escalation** (Unsure? + Plain-English: 15 typed doubt codes, ~1%
  escalate, 12 doubts → 5 calls), **2 Agent tools + skills** (39 tools over camera / images /
  tracking MCP + local, 5 skills on demand, cheapest evidence first; from
  `agent/supervisor_agent/README.md` and `skills/`), **3 Spatial-temporal memory**. Connectors
  have rounded corners.
- **Motion:** build in reading order: Pi → Correctors → Unsure? → (no) Tracked rows. Then
  (yes) Plain-English question → Muse Spark agent → verdict back up, and finally the blue
  memory loop. Each shadow box fades in with its step, bullets one by one. If time is tight,
  show the full chart and pulse only the "yes" path.

#### Act 5a · How it works: escalation

### Beat 9 · The Pi watches, and says why · 1:10–1:15 (5 s)

> **VO:** "And doubts are rare: twelve in eight minutes."

- **Format:** Video (real tracker replay) → timeline chart.
- **Assets:** 1:10–1:12 `assets/video/v03_replay_tracker.mp4`. Use 0:03–0:05 of the clip,
  where the E3-T2 doubt banner pops at 16:34:36. Then 1:12–1:15
  `assets/slides/s08_timeline.png`.
- **Source:** v03 = real frames `image_server/retrieved_images/2026-10-28-16/images/`, first
  10 s only (16:34:25–16:34:54, the period being analyzed). The walking cat's boxes come from
  `revised.jsonl` (the Pi tracker's own labels, mistakes included); the cat asleep in the bed
  is labeled "Qb · asleep" on every frame from `validated.jsonl`, dashed where interpolated
  (the bridge disclosed on s17a), and the panel caption says so; doubts from `escalations.jsonl`, verdicts from
  `validations.jsonl`, all from run 140502 (v03 rebuilt from it). s08 = the same three files,
  full 8-minute window.
- **s08 layout:** kicker asks "How often does the cheap tracker need help?", headline answers
  it. Lanes use `visualization.py`'s axis (a frame's detection is a vertical tick from box x1
  to x2, left edge at the bottom); panel **A** shows one real frame (16:34:42) turning into one
  tick. Doubts are yellow markers on the time axis, grouped above into the agent calls that
  reviewed them, and the last five doubt-free minutes are bracketed.
- **Why the timeline:** it shows rarity, 12 yellow flags across 1,319 frames, better than
  any sentence can.
- **s08 lanes = the raw detector** (`stream.jsonl`, the Pi's own labels before any review),
  so the viewer sees *when* the Pi escalates: labels flip exactly where the yellow doubts
  cluster (16:33–16:36). The reviewed lanes are the evaluation slide (beat 20, `s17a`).
- **Motion:** on s08, show panel A first (frame → ruler → "turn 90°"), then draw the lanes,
  drop the 12 markers left to right in ~1.5 s, then the call pills, then the "0 doubts"
  bracket. Purple = corrected by the agent, gray = upheld.

### Beat 10 · Fifteen kinds of doubt · 1:15–1:21 (6 s)

> **VO:** "Each doubt has a name. There are fifteen kinds, like: I relabeled this cat by distance alone."

- **Format:** Taxonomy diagram (grid).
- **Asset:** `assets/slides/s09_doubt_taxonomy.png`
- **Source:** `agent/escalation/review_schema.py`, report §7.2. Badges = codes that fired in
  this window (T2 ×3, T7 ×3, P2 ×2, T4, T6, P5, P6).
- **Motion:** grid fades in dimmed; T2 lights up on "distance alone".
- **Note:** the code now also has P8 and H1 (added after the measured runs). The footnote
  mentions them; the VO keeps "fourteen" because that's what was measured.

### Beat 11 · Doubt → question · 1:21–1:26 (5 s)

> **VO:** "It becomes a plain-English question, with evidence and a strict answer format."

- **Format:** Code card: the real question the agent received.
- **Asset:** `assets/slides/s10_question_card.png` (the real E5-T2 prompt from run 140502,
  including the memory brief it carried, and the T2 answer schema).
- **Motion:** type the header, then the Validate / Conflict / Look at / Memory lines; the
  Answer schema appears last.

#### Act 5b · How it works: the agent

### Beat 12 · The agent works · 1:26–1:32 (6 s)

> **VO:** "Then the agent pulls the frames it needs, zooms into each cat, and answers with a reason."

- **Format:** Screen recording (preferred), with the code card as fallback.
- **Asset (record):** **REC-1**, see [To record](#to-record-or-source). Scroll
  `output/agent_runs/20261104-140629-508165-validation/trace.md` in a Markdown preview:
  the case prompt → `images__get_image` calls with thumbnails → the JSON verdict.
- **Fallback:** `assets/slides/s10_question_card.png`, the real E5-T2 prompt from run 140502
  (including the memory brief it carried) and the T2 verdict schema, condensed.
- **Motion:** 2× speed scroll. Hold 1 s on the final `{"verdict": "keep_original", …}`.

### Beat 13 · Merged review · 1:32–1:37 (5 s)

> **VO:** "Related doubts share a call: twelve questions, just five calls."

- **Format:** Grouping diagram (left) + bar chart, solo vs merged (right).
- **Asset:** `assets/slides/s12_batching.png`
- **Source:** run `012059` (one call per doubt, earlier tracker) vs `140502` (merged, ≤4
  cases a call): calls 12 → 5, tokens 517.6k → 330.0k, cost $0.60 → $0.39, agent time 9.4 →
  8.7 min. The doubts differ between the two code versions, and the footer says so.
- **Motion:** the 12 chips fly into 5 call boxes on "together", then the four violet bars
  shrink from full width to their merged length on "five calls".

### Beat 14 · Verdicts become data · 1:37–1:42 (5 s)

> **VO:** "Every verdict rewrites the record, and every change is traceable."

- **Format:** Data card, before and after.
- **Asset:** `assets/slides/s14_row_diff.png`
- **Source:** the real E2 row at 16:33:06.797 from `revised.jsonl` and `validated.jsonl`,
  run 140502 (`cat_id` 26 → 25, `validated_by: E2-P5`, `verdict: reassign`).
- **Motion:** left card first; the right card slides over it with the three violet lines
  highlighting in sequence.

#### Act 5c · How it works: memory

### Beat 15 · Memory as a diary · 1:42–1:50 (8 s)

> **VO:** "Then there's memory. Not boxes and timestamps, but a diary: Qb slept in the green bed, from four thirty-four to four forty."

- **Format:** Document card (paper style, serif), on purpose the one light element in the
  video.
- **Asset:** `assets/slides/s15_day_log.png`
- **Source:** day entry from `agent/memory/data/view_3_short_term_2026-10-28.md` (latest
  session: sleeping, 0.92, 772 sightings). The zone line is verbatim from
  `agent/memory/data/view_3_long_term.yaml`.
- **Motion:** cross-dissolve from s14 (rows → prose is the point). Typewriter on the first
  sentence, timed to the VO.

### Beat 16 · Learning normal · 1:50–1:56 (6 s)

> **VO:** "Over weeks, it learns what normal looks like, for each cat, in each spot."

- **Format:** Diagram: three stacked 3D grids (where) + a zooming four-lane timeline (when),
  each level with one real memory snapshot card.
- **Asset:** `assets/slides/s16_memory_pyramid.png`
- **Source:** report §7.7 and `agent/memory/data/view_3_long_term.yaml` (view 3). Grids are
  the configured 8×6 cells (100 px) → 4×3 zones (200 px) → whole view; every block is a stored
  entry, block height = its weight. The highlighted chain cell `r5c2` → zone `r2c1` → view
  `r0c0` nests (the green bed). Timeline: week `2026-W44` → day Oct 28 → hour 16:00 → slot
  16:30, with retention (7 d / 30 d / 180 d / forever). Card quotes are the stored text,
  trimmed. Rule pills: trust gate 0.7, 3 sightings, 3 contradictions.
- **Motion:** "each spot": the cells grid draws first, blocks rise to their heights, then zones,
  then the whole-view slab; the dashed "rolls up" line climbs through the three highlighted
  blocks and each snapshot card slides in from its leader line. "Over weeks": the week lane
  appears, then each zoom wedge opens down to the next lane, with its card. Rule pills fade in
  last.

#### Act 6 · Evaluation: does it work?

### Beat 17 · Case E2: overturned · 1:56–2:02 (6 s)

> **VO:** "So, does it work? The tracker called the sleeping cat Qbo. The agent saw cool white fur and white paws: that's Qb. Overturned!"

- **Format:** Real frame with boxes + reference photos + quote + stamp.
- **Asset:** `assets/slides/s11_case_e5.png`
- **Source:** run 140502, E2-P5. Frame `2026-10-28-16-33-06-797209.jpg`
  (`assets/frames/…_E2.jpg`), the cat asleep in the bed, box `[209,505,311,599]`; the tracker
  called the whole visit Qbo, verdict `reassign` → Qb, rationale from agent run
  `20261104-140503` (shortened: "cool white, smooth sharp outline, white paws, no fluffy halo
  … label 26 is wrong").
- **Motion:** first show only the dashed orange "tracker: Qbo" box (0.8 s). Then the reference
  photos slide in and the quote types on. On "Overturned", the blue "agent: Qb" label snaps on
  and the stamp drops in. If you animate, build it from `scripts/html/s11_case_e5.html`
  (one HTML layer per element).

### Beat 18 · Case E4 + E5: the swapped cats · 2:02–2:09 (7 s)

> **VO:** "Here, the tracker swapped two cats walking close together. The agent checked both, and swapped them right back."

- **Format:** Real frame with boxes + quote + stamp.
- **Asset:** `assets/slides/s13_case_e10.png` (file name kept so links don't break)
- **Source:** run 140502, E4-T2 + E5-T2 at 16:35:21.976 (`assets/frames/…_E4E5.jpg`). The
  detector said rug cat = 26 (Qbo) `[145,357,234,462]` and bed cat = 25 (Qb)
  `[174,457,279,585]`; the tracker relabeled both by distance alone (152 px and 92 px, zero
  overlap). Both verdicts `keep_original`, in one call with E6-T4 and E7-P6 (agent run
  `20261104-140629`).
- **Motion:** both boxes appear with the tracker's (wrong) colours, then flip colours as the
  quote types on; SWAPPED BACK stamps in last.

### Beat 19 · The detector's own mistakes · 2:09–2:15 (6 s)

> **VO:** "When the detector itself is wrong, the cat gets relabeled, and that ball becomes a known decoy."

- **Format:** Two panels: zoomed real frame + reference photos (left), the ball frame (right).
- **Asset:** `assets/slides/s13b_detector_fixes.png`
- **Wrong cat:** run 140502, E10-P2, frame `2026-10-28-16-34-41-265746` (`…_E10P2.jpg`), box
  `[290,389,373,542]`. Detector said Qb on 2 frames; verdict `reassign` → Qbo (agent run
  `20261104-140809`). Checked by eye against `agent/agent_data/config/cats.yaml`: golden-brown
  face mask, neck ruff, brown paws, plume tail.
- **Not a cat:** the exercise ball from beat 3 (`…_ball_decoy.jpg`, box `[351,47,407,104]`).
  **This is not an agent verdict.** It was found in the 2026-11-01 trace review and saved as the
  `exercise_ball` decoy zone in `zones.yaml`; `agent/tracking/decoys.py` drops it at ingest
  (1,016 boxes on the 10-26 replay, 660 on 10-25; `documents/e2e-identity-fix-20261104.md`).
- **Not used:** the night cases (10-25 04:20, 10-26 02:05). The agent called a plume-tailed cat
  Qb there, and those hours' `labels/` make the same mistake, so GT can't vouch for them.
- **Motion:** left panel first (box, then refs, then the cue ticks, RELABELED), then the ball
  panel with REMOVED.

### Beat 20 · Before and after · 2:15–2:22 (7 s)

> **VO:** "Put together, the same eight minutes tell the true story: Qb slept the whole time, and Qbo walked in and out."

- **Format:** Two stacked timelines on one time axis (same lanes as beat 9): the raw detector
  vs. the output after agent review, with the escalation moments as yellow lines.
- **Asset:** `assets/slides/s17a_timeline_eval.png`
- **Source:** run 140502: raw = `stream.jsonl` (the detector's own labels); after review =
  `validated.jsonl`, drawn with `s08_rows()`: Qb's undetected sleep 16:34:02–16:35:20 is
  bridged by interpolation (faint ticks) and 3 bed-cat frames still labeled Qbo are drawn as
  Qb. The footer discloses the bridge; the metrics on the next slide never count it.
- **Motion:** raw chart first; the "after" chart wipes in under it left to right; the yellow
  lines pulse once on "same eight minutes".

### Beat 21 · Results · 2:22–2:36 (14 s)

> **VO:** "The numbers? One percent of frames call the AI, false alarms drop ninety-three percent, and ninety-nine percent of what the camera saw gets the right cat. All for thirty-nine cents!"

- **Format:** KPI tiles + before/after dumbbells + three supporting bullets.
- **Asset:** `assets/slides/s17_results.png`
- **Source:** `documents/final-model-performance-20261104.md` (run 140502 vs the audited
  ground truth, IoU 0.5; reproduce with `evaluation/final_performance.py`):
  - false positives 107 → 8 (−93%); precision 0.895 → 0.992; identity accuracy 0.934 → 0.993
  - "99% of what the camera saw": after review 0.391 recall of a 0.394 ceiling (the share of
    ground-truth cats the detector boxed at all); raw was 0.369 (93%)
  - $0.39 for 12 doubts in 5 calls vs $24.65 measured AI labeling of every frame (≈63×)
  - hardest 2 minutes = the largest-gain 2-minute window with ≥300 frames and ≥300 GT boxes
    (16:33–16:35, case E2): precision 0.529 → 0.971, false positives 65 → 4
  - 3 runs, 65 min: doubts 0.8–1.1% of frames; $3.77 of review vs ~$205 to label every frame
- **Don't say** "only those frames leave the home": the agent also fetches nearby frames for
  context (40 images for 12 doubts in this run, ~3% of frames). The slide says "~1% of frames
  raise a doubt: the only time AI is called", which is exact.
- **Motion:** tiles count up one by one on their phrase: "one percent" (tile 4 first) →
  "ninety-three percent" → "ninety-nine percent" → "thirty-nine cents". Dumbbells slide from
  green to violet last.

#### Act 7 · Beyond cats, and what's next

### Beat 22 · Beyond cats · 2:36–2:47 (11 s)

> **VO:** "And cats are just the start: lab mice, tiny drones, cars in fog, production lines. Any small model watching for rare moments has this gap."

- **Format:** Blocks with a real photo inside each card, plus a row of other domains.
- **Asset:** `assets/slides/s19_beyond_cats.png`; photos in `assets/broll/19a–19d_*.jpg`.
- **Story:** each domain maps to something the viewer already saw in the demo (the violet chip
  on each card). Lab mice = look-alike identity swaps (Qb vs Qbo). Drone / bird / ball = tiny
  objects and decoys (the ball). Cars in fog = tracks that break or swap (the swapped cats, case E4 + E5).
  Can line = a small model on every item, AI on only the odd few, all on site (~1% reviewed).
  "Also" row: pet cams (cats and dogs), robot eyes, sports balls, wildlife. No human
  monitoring, by design.
- **Source:** photos from Wikimedia Commons (CC BY-SA, credited on each photo and in
  `assets/broll/CREDITS.md`). The boxes are illustrative overlays, **not model output**, and
  the footer says so. Licence plate blurred in 19c.
- **Motion:** cards rise in left to right as each domain is named; on each card, the green
  boxes draw first, then the yellow doubt box pops; the "Also" row fades in on "same gap".

### Beat 23 · Roadmap: your agent gets eyes · 2:47–2:55 (8 s)

> **VO:** "Next: run Muse Glimmer locally, give your own agent these eyes, and let its verdicts teach the Pi."

- **Format:** Three blocks.
- **Asset:** `assets/slides/s18_roadmap.png`
- **Cards:** 01 Give your agent eyes (Muse Glimmer, locally; the agent asks the camera and
  hears when a routine changes) · 02 Teach the Pi (verdicts → labeled data → retrain the
  small model) · 03 Every camera, all year (full days, more views, then labs, lines, roads).
- **Motion:** cards rise in, one per clause.

### Beat 24 · Close · 2:55–3:00 (5 s)

> **VO:** "Agentic Realtime Computer Vision. Built this week, on Meta's Muse Spark. Thanks for watching!"

- **Format:** End card (name, the brain / eyes promise, credits).
- **Asset:** `assets/slides/s20_end_card.png`
- **Motion:** the promise fades in on "realtime computer vision", credits on "Built this
  week". Hold 1 s of silence before the cut to black.

---

## Asset index

| File | Kind | Used in |
| --- | --- | --- |
| `assets/slides/s06b_bridge.png` | the bridge + proof strip (what we built) | beat 7 |
| `assets/slides/s01_meet_qb_qbo.png` | photo pair | beat 1 |
| `assets/slides/s03_simple_questions.png` | blocks over frame | beat 2 |
| `assets/slides/s02_ball_decoy.png` | frame + hero numbers | beat 3 |
| `assets/slides/s04_cost_math.png` | KPI tiles + table (the problem) | beat 4 |
| `assets/slides/s05_two_halves.png` | table | beat 5 |
| `assets/slides/s05b_scope.png` | scope blocks | beat 6 |
| `assets/slides/s06_title_built_both.png` | section title | beat 6 |
| `assets/slides/s07_loop_diagram.png` | flow chart (architecture) | beat 8 |
| `assets/slides/s08_timeline.png` | timeline: raw detector + doubts | beat 9 |
| `assets/slides/s09_doubt_taxonomy.png` | taxonomy grid | beat 10 |
| `assets/slides/s10_question_card.png` | code card | beat 11 |
| `assets/slides/s10b_agent_work.png` | the real agent run: frames → crops → verdict (stands in for REC-1) | beat 12 |
| `assets/slides/s12_batching.png` | diagram + bars | beat 13 |
| `assets/slides/s14_row_diff.png` | data card | beat 14 |
| `assets/slides/s15_day_log.png` | document card | beat 15 |
| `assets/slides/s16_memory_pyramid.png` | 3D memory grids + zoom timeline | beat 16 |
| `assets/slides/s11_case_e5.png` | evaluation case E2 | beat 17 |
| `assets/slides/s13_case_e10.png` | evaluation case E4 + E5 | beat 18 |
| `assets/slides/s13b_detector_fixes.png` | evaluation: detector mistakes | beat 19 |
| `assets/slides/s17a_timeline_eval.png` | timelines: raw vs after review | beat 20 |
| `assets/slides/s17_results.png` | KPI tiles + dumbbells | beat 21 |
| `assets/slides/s19_beyond_cats.png` | photo cards (other domains) | beat 22 |
| `assets/slides/s18_roadmap.png` | blocks (personal agent first) | beat 23 |
| `assets/slides/s20_end_card.png` | end card | beat 24 |
| `../submission/agentic-realtime-cv-demo.mp4` | **the final cut**: 2:55, 1080p30, voice-over, captions track | all |
| `assets/video/v02_ball_push_in.mp4` | 5 s push-in | beat 3 |
| `assets/video/v01_cost_counter.mp4` | 6 s counter | beat 4 |
| `assets/video/v03_replay_tracker.mp4` | 10 s replay, 16:34:25–16:34:54 at ~3×, full frame | beat 9 |
| `assets/frames/*.jpg` | raw real frames (800×600) | sources for s02, s03, s11, s13, s13b |
| `assets/reference/*.jpg` | cat profile photos (Qb label blurred) | s01, s11 |
| `assets/text/narration.txt` | VO script | voice recording |
| `assets/text/captions.srt` | captions | edit |

Slides are 1920×1080 PNG. Each also exists as editable HTML in `scripts/html/`, so an
editor can screen-capture it with CSS animation or re-layer it.

---

## To record or source

| ID | What | How | Beat |
| --- | --- | --- | --- |
| **VO** | Voice-over, see narration.txt | Read `assets/text/narration.txt` at a relaxed pace. Each line is one beat; leave ~1 s between lines. | all |
| **REC-1** | Agent trace screen recording, ~15 s raw | Open `output/agent_runs/20261102-231717-320413-validation/trace.md` in a Markdown preview (dark theme, 125% zoom). Scroll from "Case 1 of 4: E5-T2" through the `images__get_image` thumbnails to the final JSON. Record at 1920×1080. | 10 |
| REC-2 (optional) | Live chat with the agent | `cd agent && python -m supervisor_agent "Is any cat on camera right now?"` in a large-font terminal. Use only if beat 12 needs variety. | 10 alt |
| 19a–19d (optional) | Moving B-roll to replace the four stills | Stills are done (`assets/broll/`). If you want motion, match each: mice in a home cage, a drone in open sky, traffic in fog, a can or bottle conveyor. Keep CC/royalty-free credits. | 19 |
| Music | Low, steady bed; one lift at beat 6 | Royalty-free | all |

---

## Changes from the original script

1. **Beat 15 narration was factually wrong, and is fixed.** The original said the agent
   "caught the tracker inventing a cat on an empty floor." Drawing the real guessed box
   `[178,465,266,585]` on frame 16:35:36 shows it sits on **Qb in the green bed**, not on
   empty floor; Qbo had gone into the cat house. The agent's cut was correct (no Qbo at that
   spot), but its own rationale's "empty wood floor" wording is wrong. The new line describes
   what the frame shows. **The same "empty floor" wording appears in the report (§1, §9.2,
   §9.3) and should be corrected there too.**
2. **The beat 13 quote matches the trace.** "Thin upright tail, cool white fur" became "Cool
   white, thin upright tail", which is closer to the agent's real rationale and matches the
   slide.
3. **Pacing.** The original crammed ~50 words into its last 15 s (3.5 words/s). The new
   timecodes are computed from word counts at ~2.5 words/s, so beats 19–20 get 23 s.
4. **Section split.** "It knows fourteen kinds of doubt" now has its own beat (8) and a slide,
   because it's the core idea and deserves a visual.
5. Ending: "Cats are just the start… same gap." and "Our answer…" are now two beats with
   separate visuals.
6. **Positioning and scope.** Beats 5–6 now say who this is for (compute-limited, local and
   private deployment) and admit that a bigger retrained detector would see better but is out
   of scope; the project optimizes detection with AI around a small model. A new scope slide
   (`s05b_scope`) carries it. The 11 s came out of visual holds in beats 7, 8, 11, 13, 15 and
   17; the total stays 3:00.
11. **Deck moved to run 140502** (more accurate; current code). Numbers: 12 doubts → 5 calls,
   $0.39, 7 of 12 corrected. New cases: E2 (bed cat overturned to Qb, beat 13) and E4 + E5
   (swapped cats swapped back, beat 15); s10 / s14 use the same run's prompt and row; s09
   adds P8 ("fifteen kinds of doubt"); v03 rebuilt. The end card (s20) now shows the camera,
   the bridge (this project) and your private agent.
15. **Title card moved after the scope beat.** The video now opens on the pain point (meet
   Qb and Qbo); the bridge card (now `s06b_bridge`) follows "So we built both" as the picture of what we
   built.
14. **Evaluation act.** The case studies (E2, E4 + E5, detector mistakes) now sit together
   after the technical sections, followed by the before/after timeline as the overall view
   and the results. The detector-mistakes beat (s13b) is now in the 3:00 cut. 24 beats.
13. **Demo reorganized into acts:** Open (bridge card as title) · pain point · technical
   barrier · what we built · architecture · how it works (escalation → agent → memory) ·
   evaluation (new before/after timeline s17a, then results) · beyond cats, roadmap
   (personal agent first), short close. s08 now shows the raw detector. 23 beats, 3:00.
7. **Beat 18 slide redesigned.** The two pyramids became stacked 3D grids and a zooming
   timeline, each level showing one real stored memory entry.
8. **Beat 5 uses measured numbers** from the labeling shootout (time, cost and accuracy of
   labeling every frame with AI) instead of the $0.001/frame estimate; v01 and s04 rebuilt.
10. **Project name and close.** Beat 20 now names the project (Agentic Realtime Computer
   Vision), says it runs all year, frames it as "AI gives the camera a brain, the camera gives
   AI eyes", and briefly mentions local Muse Glimmer for a personal agent. The 4 s came from
   holds in beats 11, 13 and 15.
12. **Results come from the final report.** Beat 19 / s17 now quote
   `documents/final-model-performance-20261104.md`: −93% false alarms, 99% of what the camera
   saw labeled right, $0.39 (63× cheaper), and the hardest-2-minutes window, chosen by a stated
   rule. The old "only those frames leave my home" line was dropped as inexact.
9. **Beat 20 domains replaced:** no human monitoring. Lab mice, drones / tiny objects, cars
   in fog and a can line, each tied to a demo moment, with credited Commons photos.

## Open questions for you

- **Slide structure.** Content slides now use: kicker = the question or problem, headline =
  the conclusion, details small underneath (`HEAD()` in `build_slides.py`).

- **The hackathon report still cites run 231613** (`documents/submission/agentic-realtime-cv-report.md`
  §1, §9, §10: 3 calls, $0.26, case E5 "overturned"). Run 140502's agent reached the opposite
  verdict on that 16:34:36 cat (Qbo), consistent with the diary. Update the report before
  anyone compares it with the video.
- **Two "every frame" baselines, on purpose.** Beat 5 uses the **measured labeling cost**
  ($24.65 for the 8-minute window, detect + identify every frame). Beat 19's bars use the
  **input-only floor** ($1.32 = 1,319 × $0.001, classify only), so "≈5× cheaper" there is the
  conservative claim; against the measured labeling cost ours is ~95× cheaper. Say so if asked,
  or switch s17 to the measured baseline. (`documents/demo-muse/assets/chart_cost.png` labels
  $25.61, which is the Spark-only labeling cost, not an LLM review cost. Don't mix the sets.)

## Rebuild

```bash
# slides (headless Chrome renders scripts/html/*.html -> assets/slides/*.png)
.venv/bin/python documents/demo/scripts/build_slides.py          # all
.venv/bin/python documents/demo/scripts/build_slides.py s11 s13  # some
# clips (PIL frames + ffmpeg)
.venv/bin/python documents/demo/scripts/build_videos.py          # v01 v02 v03
# narration + captions from this script's timecodes
.venv/bin/python documents/demo/scripts/build_text.py
# final video, voice first: 1) one continuous voice-over (Kokoro neural TTS, voice af_heart,
#    blended 20% with af_nicole, speed 1.2; script assets/text/voiceover_v2.txt) -> assets/audio/voiceover_v2.wav
.tts/bin/python documents/demo/scripts/build_voice.py          # needs: brew install espeak-ng
# 2) video timed to that track: still slides + 0.4 s cross-fades, clips inside their beats
.venv/bin/python documents/demo/scripts/build_final_video.py   # -> documents/submission/agentic-realtime-cv-demo.mp4
```
