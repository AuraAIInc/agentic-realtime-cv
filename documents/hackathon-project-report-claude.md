# Agentic Computer Vision for Real-Time Detection with Memory

**Author: Muyang Lu** <muyang0089@gmail.com> · **Hackathon project report** · Meta Global AI
Developer Hackathon 2026 · Indie Developer track · Problem statement: *Build for Your World*

Team: Muyang Lu, Terry Li <li.terry710@gmail.com> · Report date: 2026-11-04 · Code state:
commit `e0e8b35`; measurements come from replay runs between 2026-11-02 and 2026-11-04

> **One line:** a $100 Raspberry Pi camera that watches our two cats all day with a tiny
> detector, asks Meta's **Muse Spark** for help *only* when it is unsure, writes the verified
> answers back into its records, and keeps a plain-language memory of each cat's day.

---

## Contents

1. [TL;DR for judges](#1-tldr-for-judges)
2. [The problem we live with](#2-the-problem-we-live-with)
3. [Why it matters beyond one home](#3-why-it-matters-beyond-one-home)
4. [The core insight: fast eyes, slow brain, shared memory](#4-the-core-insight-fast-eyes-slow-brain-shared-memory)
5. [System architecture](#5-system-architecture)
6. [What we built during the hackathon](#6-what-we-built-during-the-hackathon)
7. [Module deep dives](#7-module-deep-dives)
8. [How we use Muse](#8-how-we-use-muse)
9. [Results: measured, not promised](#9-results-measured-not-promised)
10. [Cost analysis](#10-cost-analysis)
11. [Engineering quality](#11-engineering-quality)
12. [What we learned, including what went wrong](#12-what-we-learned-including-what-went-wrong)
13. [Competitive landscape](#13-competitive-landscape)
14. [Impact and generalization](#14-impact-and-generalization)
15. [Responsible design](#15-responsible-design)
16. [Roadmap](#16-roadmap)
17. [Demo script](#17-demo-script-3-minutes)
18. [How to run it](#18-how-to-run-it)
19. [Judging criteria map](#19-judging-criteria-map)
20. [References](#20-references)

---

## 1. TL;DR for judges

**The problem.** A home camera at 2 frames per second produces **172,800 frames a day**.
Sending every frame to a vision LLM would cost about **$63,072 a year** per camera at our
working rate of $0.001 a frame. A small on-device detector costs almost nothing to run, but
it misses sleeping cats, mixes up our two white cats and tracked an exercise ball as a cat for
26 minutes.

**What we built.** We built an agentic loop that joins the two:

1. A **YOLO detector on a Raspberry Pi** watches every frame.
2. **Correctors** turn boxes into tracks and visits, and they name their own doubt in
   **14 escalation scenarios** (T1–T7 for single frames, P1–P7 for whole visits).
3. Each doubt becomes a **plain-language question** with a strict output schema.
4. A **Muse Spark supervisor agent** investigates through **MCP tools**: stored images, tracker
   state, live camera and reference photos. It returns a verdict with a written rationale.
5. Verdicts **rewrite the tracked rows**, and every changed row records which escalation
   changed it.
6. A **spatial-temporal memory** turns trusted, LLM-labeled visits into a day log and a
   long-term picture of each cat ("Qb slept in the green bed 16:34–16:40, confidence 0.92").

**What we measured** on a real 8-minute replay with both cats on camera (1,319 frames):

| | One call per escalation (`012059`) | Merged batches (`231613`, new) |
| --- | --- | --- |
| Escalations raised | 12 (0.9% of frames) | 12 (0.9% of frames) |
| Agent calls | 12 | **3** |
| Verdicts returned | 11 (1 API error) | **12 of 12** |
| Tokens | 517,607 | **229,605 (−56%)** |
| Validation cost | $0.60 | **$0.26 (−57%)** |
| Validation time | ~9.4 min | **~3.8 min (−60%)** |
| Cost vs. sending all 1,319 frames to the LLM ($1.32) | 45% | **19%, about 5× cheaper** |
| Rows the agent corrected | 2 relabels | 2 relabels + **14 imagined rows removed** |
| Rows the agent confirmed | 167 | 172 |

**What we found along the way.** The agent overturned **every** nearness-only relabel it was
shown (3 of 3, in every run). It also upheld the tracker where the tracker was right: 1 of 1
teleport splits and 2 of 3 interpolated gaps. In the third gap it showed that the tracker had
"imagined" a cat on an empty floor. The same runs also exposed the biggest open problem: the
tracker relabeled **772 of 818** detections of one cat, and only 3 of those raised an
escalation. **The next win is better triggers, not a bigger model.** We report this
openly in §12.

**Why it matters.** 53 million US households have a cat ([APPA 2025](https://www.petfoodindustry.com/pet-food-market/market-trends-and-reports/news/15834866/report-cat-ownership-grows-to-53-million-us-households)),
and multi-cat homes are growing fastest. Cats hide illness, and owners recall their
routines worse than video records them. The same gap between cheap, constant attention and
expensive understanding shows up in elder care, wildlife cameras, labs, farms and robots.

---

## 2. The problem we live with

We live with two cats, **Qb** and **Qbo**. On a workday we see them for two or three hours.
The rest of their day is invisible, and we want to answer simple questions about each cat:

- When did he eat, and when did he use the litter box?
- When did he sleep, and for how long?
- When did he play, and for how long?
- How does today compare with his normal month or year?

We built a smart camera for this before the hackathon. It exposed three problems that no
off-the-shelf product solves together:

| Problem | What we saw in our own data |
| --- | --- |
| **General models miss our cats in our rooms** | In a recorded audit (Oct 31), a cat asleep in its green bed was plainly visible, yet the custom detector returned no detections and generic `yolov8s` found only a bicycle, balls and the sofa. In another run, an exercise ball was tracked as "Qbo" for 26 minutes with 100% detector agreement: its box moved 0.7 px in 26 minutes. |
| **24/7 monitoring has to run on cheap hardware** | Only a small model fits on a Pi at 2 FPS. YOLO26n on a Pi 5 CPU takes [67 ms per image](https://docs.ultralytics.com/guides/raspberry-pi), and a Pi 5 with an AI HAT averages [6–8 W](https://sixfab.com/blog/raspberry-pi-5-ai-hat-benchmark-deepx-vs-hailo/). |
| **Sending every frame to an LLM costs too much** | $172.80 a day per camera at $0.001 a frame. |

Two near-identical white cats make this harder than usual. Qb is a cool white short-hair with
a thin tail. Qbo is a cream long-hair with a dark plume tail. A small detector at 800×600
often cannot tell them apart. A vision LLM can, if someone tells it where to look.

---

## 3. Why it matters beyond one home

### 3.1 Cats hide problems, and owners miss slow changes

- The American Association of Feline Practitioners calls cats ["masters of hiding signs of
  disease and pain"](https://catvets.com/wp-content/uploads/2024/01/AAFPSeniorCareBrochureBW.pdf),
  and it treats changes in routine as a possible sign of pain.
- Owners' recall differs from video: guardians reported **lower** urination frequency than
  video showed. Healthy cats made 1–7 litter box visits a day across 6,555 daily records from
  37 cats ([Shin et al., 2024](https://koreascience.kr/article/JAKO202414243207275.pdf)).
- Chronic kidney disease is common in older cats. In one screening of apparently healthy older
  cats, **15% of cats over 10** had stage 2+ CKD ([PubMed 24217707](https://pubmed.ncbi.nlm.nih.gov/24217707/)).
  Behavioral monitoring of 136 cats found patterns linked to CKD
  ([dvm360 on Purina/Ohio State](https://www.dvm360.com/view/smart-litter-box-monitoring-identifies-behavioral-patterns-associated-with-feline-ckd)).

**Our honest claim:** long-term tracking gives owners and vets better information. It
prompts a call to the vet; it is not a diagnosis. The hackathon rules ban AI medical advice,
and we agree with that rule.

### 3.2 The audience is large and growing

- **53 million US households (39%)** own a cat, up 5% year over year
  ([APPA via Petfood Industry](https://www.petfoodindustry.com/pet-food-market/market-trends-and-reports/news/15834866/report-cat-ownership-grows-to-53-million-us-households)).
- Households with **two cats grew 7%** in a year, and those with **three or more grew 45%
  since 2018** ([APPA 2025 Dog & Cat Report](https://americanpetproducts.org/news/the-american-pet-products-association-appa-releases-2025-dog-cat-report)).
  Multi-cat homes are exactly where "which cat was it?" matters.
- AI pet cameras are an active, growing category, estimated at $389M in 2024 with 13.4% annual
  growth ([Market.us](https://market.us/report/ai-pet-camera-market/)). Market estimates vary
  widely by definition, so we treat this as direction, not size.

### 3.3 Cats are mostly still, so few frames need expensive thought

A 28-cat accelerometer study found cats active about **3%** of the day and lying or sitting
about **74%** ([AVSAB summary of Smit et al., 2024](https://avsab.org/can-computers-tell-us-what-cats-do-all-day/)).
This is why a cascade works here: most frames are easy, and the hard moments are rare and
clustered.

---

## 4. The core insight: fast eyes, slow brain, shared memory

Machine vision today comes in two halves that fail in opposite ways:

| | Real-time detector (YOLO on a Pi) | Vision LLM (Muse Spark) |
| --- | --- | --- |
| Speed | ~67 ms/frame on a Pi 5 CPU | ~30–80 s per agentic review in our runs |
| Cost of running 24/7 | Fixed hardware, ~6–8 W | Pay per frame: $63,072 a year at our rate |
| Scope | Only its trained classes | Any question in plain language |
| Best at | *Where* things are, frame after frame | *What* things mean and why |
| How it fails | Misses what it never trained on | Can hallucinate; cannot watch everything |
| Memory | Seconds of tracking | Context window, not months of video |

The fast model sees the moment without understanding it. The smart model could understand
it but never sees it. Better models alone will not close this gap, because it comes from
**scale**: about 770 million surveillance cameras existed in 2019
([CNBC/IHS Markit](https://www.cnbc.com/2019/12/06/one-billion-surveillance-cameras-will-be-watching-globally-in-2021.html)).

Three things have to cross the gap:

- **Attention flows up.** The fast model finds the moments that deserve expensive thought,
  and it says *why* (our 14 escalation scenarios).
- **Knowledge flows down.** The agent's verdicts correct the record now. Next, they become
  training data for the edge model.
- **Memory spans both.** Plain-language history makes "unusual" mean unusual *for this cat,
  in this room*.

Prior work covers each piece on its own. Specialized cascades
([NoScope](https://web.cs.ucla.edu/~ravi/CS239_W20/papers/noscope.pdf)), online distillation
from a costly teacher ([Mullapudi et al.](https://arxiv.org/pdf/1812.02699)), cascaded VLM
anomaly detection ([Cerberus](https://arxiv.org/pdf/2510.16290)) and agentic memory for long
video ([Avas](https://arxiv.org/html/2505.00254v3)) all exist. Within our targeted review,
no project joins all three into one loop on a $100 device for long-term tracking of named
individuals.

---

## 5. System architecture

```
  Raspberry Pi (home)                   ingest / pipeline host                     cloud
 ┌───────────────────────┐  JSON over  ┌──────────────────────────────────────┐
 │ camera_v3             │  WebSocket  │ agent.main  (TrackingRun)            │
 │  always-on YOLO 2 fps ├────────────►│  ingest: window + min-conf gate      │
 │  frame buffer         │   :8000     │  TrackCorrector  ── T1..T7 ─┐        │
 │  MCP server :8765 ◄───┼──────┐      │  PeriodCorrector ── P1..P7 ─┤        │
 └───────────────────────┘      │      │                             ▼        │
 ┌───────────────────────┐      │      │  EscalationBatcher (≤60 s, ≤4 cases) │
 │ image_server (SSD+S3) │      │      │        │ background thread           │
 │  MCP :10001  ◄────────┼──────┤      │        ▼                             │   ┌──────────────────┐
 └───────────────────────┘      │      │  ValidationWorker ──────────────────────►│ Muse Spark 1.3   │
 ┌───────────────────────┐      │      │   prompt + memory brief + JSON schema│   │ Meta Model API   │
 │ tracking_mcp :10002 ◄─┼──────┤      │        │ verdict + rationale          │◄──┤ (Responses API)  │
 └───────────────────────┘      │      │        ▼                             │   └──────────────────┘
                                │      │  materialize → validated.jsonl/.db   │            ▲
                                │      │                                      │            │
                                │      │  Memory (own worker thread)          │            │
                                │      │   PeriodLabeler (vision, activity) ──┼────────────┤
                                │      │   short-term day log (.md)           │            │
                                │      │   long-term spatial-temporal YAML ───┼────────────┘
                                │      └──────────────────────────────────────┘   store / promote
                                │
                                └──── MCP tools ◄── supervisor agent (also usable as a chat CLI)
```

**Design rules that hold everywhere:**

- **Tracking never waits for an LLM.** Validation and memory each run on their own background
  thread, and every memory hook is best-effort.
- **One message shape.** Live camera, recorded `stream.jsonl`, read-only MySQL replay and
  `camera_sim` all feed the same `process_message()`, so any recorded window replays
  bit-for-bit.
- **Language out, structure back.** Every escalation leaves as prose and comes back as a
  schema-checked verdict that a handler applies to rows.
- **Everything is auditable.** Every agent run writes a `trace.md` with the system prompt,
  each tool call and result, the evidence images, tokens, cost and time. Every memory LLM call
  writes its own trace too.

---

## 6. What we built during the hackathon

The rules require a clear line between earlier work and event work. The building period
was **Oct 27 – Nov 4**. The repo has **49 commits** since Oct 27 (about 29,900 lines
added).

| Existed before Oct 27 (kept in a separate, clearly marked folder) | Built during the hackathon |
| --- | --- |
| WebSocket ingest server | **Agent ported from OpenAI to Meta Muse Spark** (Responses API), plus a workaround for the API's chained-image behavior (§8) |
| TrackCorrector and PeriodCorrector (2025) | **14 escalation scenarios** wired into both correctors (Oct 30) |
| First tracking MCP server and visualization (Oct 23–25) | **Per-scenario prompt builders and pydantic output schemas** (Nov 1) |
| First camera/image MCP servers and the OpenAI-based agent (Oct 24–25) | **Escalation → agent → verdict → validated rows** loop (Nov 1) |
| | **Batching queue** that merges close escalations into joint calls (Nov 2) |
| | **Spatial-temporal long-term memory** with trust gate and anti-overfitting promotion (Oct 31 – Nov 2) |
| | **Daily short-term memory** and **vision activity labeling** of visits (Nov 2) |
| | **camera_sim**: replays archived frames through the real camera interface (Oct 31) |
| | **High-res capture + S3 upload**; image server serving S3 for new captures (Oct 29–31) |
| | **Read-only MySQL replay**, in-pipeline `original/revised/validated` charts, process MP4 |
| | Two trace reviews that turned into fixes: slimmer tools, skill gating, chat budgets, raw-pixel grounding (Nov 1) |
| | YOLO dataset export with database pre-labels (Oct 31) |

The pre-hackathon tracking code was migrated into the new agent pipeline on Oct 31, so that
the escalation hooks could live inside it.

---

## 7. Module deep dives

### 7.1 An agent-steerable camera

A Pi cannot run a large model at full resolution all day, so the stream runs at 800×600 with
a small custom model (`best.pt`, whose classes are the cats themselves). `camera_v3` keeps
that always-on loop and adds an **MCP server beside it**, so the agent can look harder only
when needed:

| Tool | What it gives the agent |
| --- | --- |
| `get_status` | Current settings, frame rate, last detections, models |
| `capture_frame` | Newest or buffered frame, optional crop |
| `capture_high_res` | One sharper still (digital zoom via crop); pauses detection ~3 s |
| `detect` | One pass with any model, region, input size or threshold, without touching the tracker |
| `get_track_history` / `get_track_frame` | When each track was seen, identity flips, first/last frames |
| `list_buffered_frames` / `get_camera_modes` / `list_models` | What can still be revisited, and with what |
| `set_schedule` / `reset_schedule` | Temporary change to frame rate, region, input size or model |

Every override **expires within 24 hours**. With no MCP traffic, the camera behaves exactly as
before. The agent's full toolbox is **39 tools**: 24 MCP tools across three servers (camera,
stored images, tracking), plus 10 local helpers over a local knowledge folder (cat profiles, reference
photos, contact sheets, trajectory physics, labels, zones) and 5 skills.

### 7.2 The correctors name their own doubt

**TrackCorrector** keeps a ~15 s buffer, so later frames can fix earlier ones. It removes
duplicate boxes (IoU > 0.7) and keeps or relabels each box against each cat's history. It
then interpolates toward a trusted future box and commits older rows. A jump over 300 px
starts a new visit.
**PeriodCorrector** votes on each whole visit using certainty, frame count, detection rate and
spread. It enforces that **one cat cannot be in two overlapping visits**, which is how two
cats in one frame are resolved.

The hackathon addition is that both correctors now **report their uncertainty at the exact
decision point** where it arises:

| Frame scope | Trigger | Visit scope | Trigger |
| --- | --- | --- | --- |
| T1 multi-match relabel | Box fits 2+ cats' histories | P1 close vote | Top-two scores nearly tied |
| T2 weak-rule relabel | Relabeled by distance alone, no overlap | P2 weak winner | Barely above minimum, or short visit |
| T3 duplicate tie | Duplicates removed on near-tied confidence | P3 conflict reassign | Top cat taken by an overlapping visit |
| T4 jump reset | New box missed the anchor; buffer wiped | P4 long unknown | Long visit with no cat |
| T5 target fallback | Nothing confident; tracker guessed | P5 low quality | Low confidence or mostly guessed frames |
| T6 teleport split | Committed box jumped > 300 px | P6 merge candidate | May continue the cat's previous visit |
| T7 long synthesis | Long run of interpolated frames | P7 static link | Sleeping cat split by a long gap |

A low-confidence box does **not** trigger a review by itself. These rules fire on conflicts in
the tracker's *decisions*, which is what makes the escalation rate low (0.9% of frames).

### 7.3 From numbers to a question, and back

Raw tracker output (boxes, timestamps, scores) means nothing to an LLM. Each of the 14
scenarios therefore has its own **prompt builder**, which states the conflict, the question,
the alternatives, the key frames to inspect and each cat's written appearance. It also
frames the Pi's YOLO honestly, as *"a hint, not evidence"*. A real case:

```text
[E2-T6] Cat seemingly teleported; visit split.
Task: validate whether splitting visit 1 into visit 2 at 16:34:05 was right:
cat 26 jumped 362 px in one step, and the tracker deleted 10 guessed frames.
Validate: Is this one fast-moving cat (merge the visits back), or two different
cats or a false box (the split was right)?
Alternatives: merge visits 1 and 2 into one, or drop the jumped box.
```

Each scenario also has a **slim pydantic schema**: one `verdict` enum, at most one qualifier
(`chosen_cat_id` or `cut_time`) and a short `rationale`. Structured output keeps generation
short, and `parse_review()` turns any malformed answer into `{"verdict": "error"}` instead of
an exception.

### 7.4 The supervisor agent

The agent is a tool-calling loop over the three MCP servers and local helpers, running on
**Muse Spark 1.3**. Its system prompt sets working rules that come straight from our trace
reviews:

- Check cheap sources before pixels, and **stop once the evidence settles the case**.
- Say only what is actually visible, and never invent detections.
- Treat a box that stays still for many minutes as a **possible object (decoy)**.
- Two simultaneous boxes with the same cat ID mean one label is wrong.
- Use generic detectors to check for objects, never for cat identity.

For identity checks, the `validate-cat-identity` skill defines an **evidence ladder** with
image budgets (3 images and 1 high-res capture): metadata, then written profiles, then a
sharper view, then owner photos, then confirmed crops, and finally asking a human rather
than forcing a call. In the pipeline the agent gets a narrow set of three tools
(`images__get_image`, `images__list_images`, `tracking__get_period_detail`), which cuts the
tool-schema tokens sent on every turn.

### 7.5 A queue that merges related doubts

Escalations cluster in time. E6, E7 and E8, for example, all fired on the same frame. The
`EscalationBatcher` lets a new escalation join the waiting batch if it falls **within 60 s**,
up to **4 cases** and about **12,000 prompt characters**. The worker takes the next batch the
moment it is free, so merging **adds no latency**. One agent call then reviews all cases
together, fetching each frame once, and returns **one verdict per case** in a merged schema.
Parsing is **isolated per case**: one malformed item becomes an `error` for that case only,
and its siblings still apply. This happened in run `221606`, where E10 returned a verdict
from another scenario's list.

### 7.6 Verdicts become data

At the end of a run, `materialize()` applies verdicts to the revised rows. Frame verdicts
match rows by timestamp and cat, and period verdicts match by visit. `keep_original` restores
the detector's label, `cut_at_gap_start` removes guessed rows, `reject_box`/`drop` remove
boxes, and `confirm_*`/`keep` stamp rows as confirmed. **Every touched row records
`validated_by` and `verdict`**, so each correction can be audited back to a trace. Verdicts
that cannot be applied precisely (for example, a merge into an unnamed neighbor) keep the
rows and record a note instead of guessing. The output is written to `validated.jsonl` and a
per-run SQLite `validated.db`; the shared MySQL is never written.

### 7.7 Spatial-temporal memory

Detectors see frames and trackers see paths, but neither knows that *Qb usually naps in the
green bed in the afternoon*. Agent frameworks treat memory as conversation history, while
video needs memory organized by **where** and **when**. The design has these parts:

- **Visits, not frames.** Closed visits by the same cat less than 5 minutes apart merge into
  one activity record. A **Muse Spark vision call** (`PeriodLabeler`) labels the key frames
  with an activity from a configurable label set (sleeping, resting, sitting, walking, eating,
  grooming, playing, …), plus the cat's appearance and the background.
- **Two memories.** The *short-term* day log keeps everything, including weak visits,
  escalations, verdicts and rationales, as a readable Markdown file. The *long-term* store
  keeps only **trusted** knowledge: the detector and every activity label must reach 0.7.
  Escalations and raw verdicts never enter it directly.
- **Spatial pyramid.** 48 cells of 100 px roll up into 12 zones and then one whole-view
  summary. Each holds plain text about environment, contrast, activity, appearance and
  behavior, plus visit counts per cat.
- **Temporal pyramid.** 5-minute slots hold events (kept 7 days), hours hold the activity mix
  (30 days), days hold routine (180 days) and weeks hold habits (kept indefinitely).
- **Resistance to noise.** A pattern needs ≥3 observations to enter stable text. One-offs
  live in a 5-item recent buffer. Stable text flips only after **3 consecutive
  contradictions**, and word caps force each summary to keep only its strongest patterns. The
  LLM judges whether new evidence contradicts the stable text; *code* does the counting, so
  the rules are testable offline.
- **Targeted retrieval.** Each escalation family asks memory a different question. T1/T2/T5
  ask whether the cat is hard to see here, P3 asks which cat is usually here now, and P6/P7
  ask whether this is the usual nap spot. The answer is a brief of at most 600 characters,
  attached to the prompt as *"may help, may be stale"*. Memory informs the agent but never
  overrides the image.

Real output from the latest session's day log:

```text
16:34–16:40: Qb (cat 25) — 772 sightings, mean confidence 0.71.
Activity: sleeping (confidence 0.92). Looks: qb is a white fluffy cat curled up tight
in a small green bed with eyes closed and paws tucked in. Background: light brown wood
floor with a large white ridged rug and a light gray couch along the window wall.
```

---

## 8. How we use Muse

Muse is not one dependency in this project; it is the toolchain that built it and the
model family that runs it. We split work across Muse models by reasoning depth and
latency, and we build the system itself inside Muse Code.

### 8.1 Muse Code is our development environment

- **Project development.** The pipeline, correctors, supervisor agent, memory system and
  tests were implemented in Muse Code, driven by small, verified steps (each change
  landed with a focused failing test first, then the fix).
- **Brainstorming and design.** Architecture exploration happened conversationally with
  Muse before any code: the escalation taxonomy, the verdict contracts, and the
  spatial-temporal memory design all started as argued design notes.
- **Design docs and report writing.** Specs, runbooks, session reports and this report
  were drafted and revised in Muse Code alongside the code they describe, so
  documentation cannot drift far from the implementation.
- **Voice input.** Muse Code's voice function let us speak design and thoughts out
  loud and have them land as structured documentation — the fastest way we found to
  capture a half-formed idea before it evaporated.
- **Workflow skills.** Complex, multi-file work was executed as Muse workflow
  skills with focused subagents: e.g. the escalation system (trigger → prompt → strict
  schema → verdict applier, across six files) was implemented as one orchestrated task
  rather than a sequence of one-off prompts.

### 8.2 One family, four tiers of work

| Muse model | Role in this project | Why this tier |
| --- | --- | --- |
| **Muse Wonder** | High-level design: the escalation taxonomy (14 kinds of tracker doubt, their prompts and verdict contracts) and the spatial-temporal memory architecture | Deepest reasoning, used rarely and early, where a wrong design costs the most |
| **Muse Spark 1.3** | Main workload. Implementation in Muse Code (agentic tools, workflow skills, experiments, evaluation harnesses); at runtime it is the supervisor agent reviewing escalations, the activity labeler, and the memory store/promote/bootstrap engine | Strong multi-turn vision + tool calling at everyday cost; strict `json_schema` verdicts; cached-token pricing |
| **Muse SAM 3.1** | Offline ground-truth labeling: segmentation-first boxes for training/fine-tuning the edge detector and for the self-learning eval loop (§9.6) | Mask-true geometry no VLM box regression matches (F1 0.794 vs 0.612 head-to-head) |
| **Muse Glimmer** | Edge deployment: the track corrector and camera server run on the camera device; a small locally-deployed Muse model can be invoked on-device for 100% data privacy — footage never leaves home — with super-fast response for interactive checks | Latency and privacy are hard requirements at the edge; deep reasoning stays in the cloud tier |

Runtime usage in one table:

| Where | Meta model | What it does | Status |
| --- | --- | --- | --- |
| Supervisor agent (pipeline) | **Muse Spark 1.3** | Multi-turn, tool-calling **vision** agent: picks frames, compares cats, returns a schema-bound verdict with rationale | Built, measured |
| Joint validation | **Muse Spark 1.3** | Reviews up to 4 related cases in one call with a dynamic merged JSON schema | Built, measured |
| Activity labeler | **Muse Spark 1.3** (vision) | Labels each visit's key frames with activity, appearance and background | Built, measured |
| Memory store / promote / bootstrap | **Muse Spark 1.3** | Judges contradictions; compresses children into parent summaries under word caps | Built, measured |
| Interactive agent (CLI) | **Muse Spark 1.3** | Chat over live camera, history and tracker ("Is any cat on camera right now?"); custom labels by chat | Built |
| Ground-truth labeling | **SAM 3.1** (+ Spark classify) | Two-stage labels for edge retraining and self-learning eval (§9.6) | Built, measured |
| Edge inference (camera device) | **Muse Glimmer** | On-device correctors + camera server; local small-model invocation for 100% data privacy | Built, measured locally |
| Voice questions | Muse Voice Transcribe | "When did Qbo last eat?" by voice — same voice path we used to dictate design docs | Planned |
| Daily "cat diary" | Muse Image / Muse Wonder | Creative recap of each cat's day | Planned |
| Private, local reasoning | Muse Spark Open (`muse-spark-open-1026`) | Keep home footage on partner or own hardware | Planned |

**Capabilities we relied on:** vision inside tool outputs (images returned in
`function_call_output`), multi-turn chaining with `previous_response_id`, strict
`json_schema` structured output, including schemas built at runtime from the union of
scenario verdicts, and cached input pricing (28.5k of 58.8k input tokens were cached in one
merged call; 52% of all inputs across the fleet, §9.5).

**What we learned about the API** (from a controlled probe on Oct 31): chained turns that
carry images sometimes return `400 referenced response not found or expired`. We ran a
controlled probe that ruled out the root `instructions` parameter, image size and total
payload size, and concluded the failure is intermittent on the gateway side. We folded
instructions into turn 1 and recommend retrying with a fallback to unchained history. Other
findings: only `tool_choice: "auto"` is accepted; images must ride inside function outputs;
and a very small `max_output_tokens` returns reasoning only.

---

## 9. Results: measured, not promised

All runs replay the same real window from read-only MySQL:
**view 3, 2026-10-28 16:33–16:41, both cats on camera, 1,319 frames**. Tracking is
bit-deterministic (identical chart checksums across runs), so the runs differ only in the
agent stage.

### 9.1 Pipeline funnel

| Stage | In | Out |
| --- | --- | --- |
| Ingest + YOLO | 1,319 frames | 1,016 detections (818 Qb, 198 Qbo) |
| TrackCorrector | 1,016 detections | 1,265 connected rows, 8 frame escalations |
| PeriodCorrector | 1,265 rows | 4 visits, 4 visit escalations |
| Supervisor agent | 12 escalations | 12 verdicts with rationales (merged run) |
| Apply | 1,275 rows + 12 verdicts | 1,261 rows: 2 relabeled, 14 removed, 172 confirmed |
| Memory | 4 visits + verdicts | Day log with activity labels; long-term YAML |

Escalation mix: T7 ×3, T2 ×3, P2 ×2, T6, T4, P5, P6.

### 9.2 One call per case vs. merged batches

| Metric | `012059` (solo) | `221606` (merged) | `231613` (merged) |
| --- | --- | --- | --- |
| Agent calls | 12 | 3 | 3 |
| LLM turns | 45 | 9 | 10 |
| Images read | 55 | 17 | 19 |
| Tokens | 517,607¹ | 194,901 | 229,605 |
| Cost | $0.60¹ | $0.24 | $0.26 |
| Validation time (sum of calls) | ~9.4 min | ~3.5 min | ~3.8 min |
| Cost per escalation | ~$0.054 | ~$0.020 | ~$0.021 |
| Verdicts / errors | 11 / 1 (API 400) | 11 / 1 (schema) | **12 / 0** |

¹ Excludes the failed E9 call, which left no trace.

Merging cut tokens by **56–62%**, cost by **57–59%** and time by **60–63%**. It also made
the agent see related cases together. In `231613` it resolved the E5/E6/E7 same-frame identity
swap consistently and noticed that the E10 interpolation ran across an empty floor, a case it
had accepted when shown alone.

### 9.3 What the agent decided, and why (run `231613`)

| Case | Verdict | Agent's rationale (shortened) |
| --- | --- | --- |
| E1-T7 gap | keep_synthesis | White cat clearly asleep in the green bed at both ends; detector missed it, so keep the guesses |
| E2-T6 teleport | confirm_split | Both positions occupied at once, so a 362 px jump cannot be one cat |
| E3-P5 weak visit | keep | Only 44 of 153 frames detected, but images show continuous presence |
| E5-T2 relabel | **keep_original** | Cool white, smooth outline, thin upright tail = Qb, not Qbo; nearness stole the neighbor's label |
| E6-T2 relabel | **keep_original** | Warm cream with dark bushy plume tail = Qbo; relabel by 152 px nearness was wrong |
| E7-T2 relabel | **keep_original** | Bottom cat is cool white = Qb; the second cat is separately visible |
| E8-T4 reset | confirm_reset | Anchor spot empty; the new box is Qbo entering from the left |
| E9-P6 merge? | keep_separate | Visit ends and starts show different animals |
| E10-T7 gap | **cut_at_gap_start** | Guessed box sits on **empty wood floor** in both frames, so the track imagined a cat |
| E11/E12-P2 thin visits | keep | Real cats, clearly visible; the low score comes from the short-visit penalty |

Across runs, by scenario:

| Code | Tracker upheld | Tracker overturned | Reading |
| --- | --- | --- | --- |
| T2 nearness relabel | 0 / 3 | **3 / 3**, in every run | Distance-only relabeling is unreliable with two cats |
| T6 teleport split | 1 / 1 | 0 | The split rule works |
| T7 interpolation | 2–3 / 3 | 0–1 / 3 | Usually right; once it painted a cat on empty floor |
| P2/P5 weak visits | mostly upheld | — | Weak scores came from penalties, not false visits |

### 9.4 What memory learned

- **The trust gate held.** Visit 1 (confidence 0.54) was refused from long-term memory with a
  log line, while visits at 0.71 and 0.82 were stored. Error verdicts were refused too.
- **Activity labels landed.** Qb was labeled *sleeping, 0.92* for 16:34–16:40, with
  appearance and background descriptions in plain language.
- **Plain-language long-term text** rolled up from cells to zones, the whole view, hour, day
  and week. For example: *"Qb and qbo visit on October 28 with 17 visits in total. Most
  visits happen in the afternoon."*
- **Memory cost** for the latest session was about **$0.14** across 18 traced LLM calls
  (activity labels, stores, promotions).

### 9.5 Fleet-wide invoke statistics (124 agent runs, 1,058 tool calls)

The invoke tables (`agent_invoke` / `tool_invoke` in the project database) now hold every
agent run from Oct 27 to Nov 3, so the design above can be measured as a fleet, not a demo:

| Metric | Value |
| --- | --- |
| Agent runs (all Muse Spark 1.3, all answered) | 124 (93 validation, 31 untagged) |
| Tool calls across 26 distinct tools | 1,058 |
| Total cost | **$9.22** ($6.56 validation, $2.66 untagged) |
| Tokens in / out / cached | 9,746,914 / 625,428 / 5,108,850 (52% of inputs cached) |
| Median run | 4 turns, 8 tools, 60 s wall (57 s LLM, 2.6 s tools) |

`images__get_image` dominates (670 calls, 63%), followed by
`tracking__get_period_detail` (124) and `images__list_images` (60). Median latencies:
image fetch 0.7 s, period detail 2.8 s, everything local sub-second.

Tool errors are 104/1,058 (9.8%), touching 38% of runs — and 95 of them are one bug:
`ModuleNotFoundError: No module named 'PIL'` on the image server. The split is decisive:
**91 of 95 failures carried a crop box, and 0 of 575 successful fetches did** — every
crop failed, every plain fetch succeeded. The remaining errors are 6 transient MCP
session losses and 3 rolling-buffer races. The prompt now forbids crop retries and
ignores stale period detail, and transient API errors (429/5xx/timeouts) retry with
backoff in the labeling tools (a 108-image timeout wave was the costliest single failure).

Skill usage is thin: 4 `skill__*` calls total (`check-live-detections` ×3,
`validate-cat-identity` ×1), all in untagged runs, all successful, all sub-2 ms — and
zero in the 93 validation runs, an adoption gap we now track.

### 9.6 Ground truth, self-learning, and the label-replay loop

To fine-tune the edge detector and to evaluate anything honestly, we need ground truth
that is not the detector's own output. Three hourly label sets now exist, built two ways
and compared head-to-head on the same 1,318-frame window (2026-10-28 16:33–16:40):

| | SAM 3.1 pipeline | Spark 1.3 single-call |
| --- | --- | --- |
| Pipeline | SAM proposes regions → Spark classifies | One VLM call detects + identifies |
| Wall (10 workers) | 66 min (3.0 s/img) | 99 min (4.5 s/img, timeout-inflated) |
| Cost (Standard rates) | **$24.65 ($0.0187/img)** | ~$25.61 ($0.0195/img) |
| Coverage | 1318/1318 (100%) | 1210/1318 (108 timeouts, since fixed with retry) |
| F1 @ IoU 0.5 | **0.794** (P 0.936 / R 0.689) | 0.612 (P 0.656 / R 0.574) |
| Qb / Qbo recall | 0.977 / 0.380 | 0.821 / 0.310 |

SAM's errors are misses, not hallucinations (117 FP vs 766 FN); Spark's are both
(743 FP, 1,050 FN). Both models systematically lose Qbo — the sleeping long-haired cat —
which is now the top recall target. One caveat: the eval's ground truth is validated
tracking output that itself ingested SAM labels, so SAM's precision is partially
circular; its recall gap is not.

The loop closes with a labels-mode runner: any YOLO label folder can be replayed as
raw camera detections through the correctors and supervisor validation, and the
validated result is exported back as ground-truth YOLO files (real sightings only —
interpolated guesses never become ground truth). Three hours are through it:

| Window | Frames | Periods → validated | Escalations → verdict highlights |
| --- | --- | --- | --- |
| 10-28 16:33 (dense, both cats) | 1,318 | 2 → 2 | 1 (routine spot-check); exported 2,465 GT boxes |
| 10-26 02:00 (sparse, night) | 5,320 | 20 → 13 | 38 → merges, `different_cats` relabels, split confirms, drops |
| 10-25 04:19 (mixed) | 3,234 | 16 → merged/dropped | 43 → 4 split confirms, 2 merges, 6 drops |

The verdict set keeps learning from review: a confirmed T6 split now deletes its whole
interpolated bridge (the guesses belong to neither cat), a confirmed P6/P7 merge
re-interpolates the bridge so the visit reads as one track, P5 reviews always carry
both cats' profiles with exactly three actions (drop / confirm / relabel), and a P6
`different_cats` verdict applies the label revision to the new visit. On the 04:19 hour,
993 rows were honestly left unlabeled (`mark_unknown`) when image evidence was
unreachable — the loop records that gap instead of guessing.

---

## 10. Cost analysis

### 10.1 The bill for sending everything to the LLM

At $0.001 a frame (our working rate, about 800 input tokens at the
[reported $1.25 per million](https://tokencost.app/blog/meta-muse-spark-1-1-pricing)):

| Share of frames sent to the LLM | Frames a day | Cost a day | Cost a year |
| --- | --- | --- | --- |
| 100% | 172,800 | $172.80 | $63,072 |
| 5% | 8,640 | $8.64 | $3,154 |
| 1% | 1,728 | $1.73 | $631 |
| 0.1% | 173 | $0.17 | $63 |

### 10.2 What we measured

For the 8-minute window, sending all frames to the LLM costs **$1.32**. Our hybrid spent:

| Component | Cost | Share of the all-LLM bill |
| --- | --- | --- |
| Pi detection + correctors | ~$0 (fixed hardware, ~1 s replay) | 0% |
| Validation, merged (`231613`) | $0.26 | 19% |
| Memory (activity, store, promote) | ~$0.14 | 11% |
| **Total** | **~$0.40** | **~30%** |

Before merging, validation alone was 45% of the all-LLM bill. Merging moved the hybrid from
"saves half" to "saves 70–80%".

### 10.3 What a full day would cost

This window is the **worst case**: both cats moving, with 12 escalations in 8 minutes (about
90 an hour). Cost scales with escalations, not frames:

| Escalations a day | Validation at $0.021 each | Comment |
| --- | --- | --- |
| 1,555 (0.9% of every frame, all day) | $33 | Upper bound; assumes the busy rate never drops |
| 360 (busy rate for 4 active hours) | $7.6 | Cats active or moving about 3–15% of the day |
| 50 | $1.05 | Target once triggers are tuned and memory skips easy cases |

To be honest about this, savings over sending every frame are measured on one busy window,
not on a full day. A **full-day run is the next measurement**, and §16 lists the levers that
push the rate down further: memory-only decisions in shadow mode, and an edge model retrained
on verdicts.

**Privacy is a side effect of cost control.** Only escalated frames, about 1 in 100, ever
leave the home.

---

## 11. Engineering quality

| | |
| --- | --- |
| Code written for the new pipeline | Memory ~4.0k lines, supervisor agent ~3.1k, tracking ~2.1k, validation ~1.2k, escalation ~1.1k, camera simulator ~0.7k (Python, tests excluded) |
| Tests | **45 test files, 375 test functions**. Memory and agent tests run fully offline with a stub LLM and a fake model client. |
| Determinism | Tracking replays bit-identically. Single-case batches produce byte-identical prompts to the unmerged path. |
| Observability | One `trace.md` per agent run and per memory LLM call, with tokens, cost, time and evidence images. Each run writes `run.json` with counts by scenario and verdict. |
| Failure isolation | A dropped MCP server degrades with a warning. An API failure becomes one `error` verdict, never a crash. Memory failures never reach tracking. |
| Safety of shared data | Production MySQL is read-only. Validated rows go to a per-run SQLite file. |
| Self-review culture | During the event we wrote design documents for each module, two reviews of agent traces, an API probe report and per-session reports. Each one ended in concrete fixes. |

The trace reviews led to concrete changes. One chat query took 9 turns and 22 calls, with
94% of its 152 s spent in the LLM, and its answer validated a ball as a cat. The fixes were
fewer exposed tools, skill gating, default budgets, raw-pixel grounding and an explicit
decoy rule in the system prompt.

---

## 12. What we learned, including what went wrong

We would rather show judges what we found than hide it.

1. **Triggers matter more than the model.** The tracker relabeled **772 of 818** Qb detections
   as Qbo, and only **3** raised a T2 escalation. The agent overturned all 3, so the remaining
   ~770 errors are a trigger-recall problem. The latest day log shows a second gap: visit 1, a
   white cat asleep in the green bed, was labeled Qbo. The agent's own rationales for E5 and E7
   describe the bed cat as Qb. P5 asked "real visit or noise?" and never asked "which cat?".
   **Fix:** add a cross-check that escalates when a visit's identity disagrees with the
   agent's appearance verdicts at the same spot, and escalate *runs* of T2 relabels instead of
   single frames.
2. **Silent misses never escalate.** With no detection, no trigger fires. The sleeping cat in
   our false-negative audit (§2) needs a
   different mechanism: periodic sampling of frames with no detection in zones where memory
   says a cat usually rests.
3. **Decoys look like perfect evidence.** A static ball with 100% detector agreement for 26
   minutes is exactly what a confident shortcut would accept. A memory built on such labels
   would repeat the mistake with more confidence. Our identity skill now flags long static
   boxes for a visual check.
4. **Agent verdicts are not ground truth.** Two verdicts flipped between runs on identical
   inputs (both had no effect on rows). One API 400 dropped a merge with no retry. **Fix:** retry
   with an unchained fallback, and run self-consistency checks on cases that change rows.
5. **Apply needs hardening.** Two same-frame T2 verdicts can interfere with each other, and
   the shared `verdict` field keeps only the last writer.
6. **No accuracy numbers yet.** We have no hand-labeled ground truth, so we do not claim an
   overall accuracy gain. We claim what we can show: the agent found and explained specific
   tracker errors and upheld the tracker where it was right.

---

## 13. Competitive landscape

| | Pet cameras (Furbo, Petcube, …) | Smart litter boxes (Litter-Robot 5 Pro, Petlibro Luma, Petkit) | Cloud VLM on every frame | **This project** |
| --- | --- | --- | --- | --- |
| Watches | One room, motion/bark events | Only the litter box | Anything | Whole room, continuously |
| Tells cats apart | Generally not a core feature | Weight or face ID at the box ([Whisker](https://www.techedt.com/whisker-introduces-litter-robot-5-pro-with-ai-facial-recognition-for-cats); [Petlibro](https://ca.petlibro.com/blogs/all/how-does-the-new-luma-smart-litter-box-stack-up)) | Yes, at high cost | Yes, with an agent verifying doubtful cases |
| Knows its own doubt | No | No | No | **14 named scenarios** |
| Explains decisions | No | No | Sometimes | **Rationale per verdict, full trace** |
| Long-term memory of routine | Event clips | Per-visit logs | Context window only | **Spatial-temporal plain-language memory** |
| Running cost | Subscription | $899-class hardware + subscription | ~$63k a year per camera at our rate | Pi + about 1 in 100 frames to the LLM |
| Gets better at *your* home | No | No | No | Verdicts correct records now; retraining next |

Litter-box devices measure one important behavior very well, and they complement this
project rather than compete with it. This project covers the rest of the day: sleeping,
resting, playing, and which cat was where.

---

## 14. Impact and generalization

The pattern of a cheap camera that watches continuously, asks for help when unsure and
remembers what it learns fits any setting with **fixed cameras, rare or slow events, and
too much video for people to review**:

| Domain | Today | What this method adds |
| --- | --- | --- |
| Multi-pet homes and vet visits | Owners miss slow changes, and their recall differs from recordings | A daily log per animal to bring to the vet |
| Care at home | Falls caused an estimated [842,000 deaths in 2023](https://www.who.int/en/news-room/fact-sheets/detail/falls) | A consent-based routine-change alert, with minimal footage leaving the home |
| Animal research | [Home-cage video](https://www.frontiersin.org/journals/behavioral-neuroscience/articles/10.3389/fnbeh.2022.877323/pdf) suffers identity swaps and needs manual correction | Named animals, flagged doubt, written activity logs |
| Wildlife conservation | [Manual review took 6–7 months](https://news.wsu.edu/?p=260863) in one study | Keep automated sorting, and send only hard sightings to deeper review |
| Farms | [Visual scoring is slow and subjective](https://cris.unibo.it/retrieve/handle/11585/1046536/0198d1c4-e7c7-4dbe-b1d7-548678026540/animals-15-02508-v2.pdf) | Per-animal baselines with changes flagged |
| Factories | Defects [can be as rare as 0.1%](https://arxiv.org/pdf/2509.05796v1) | Fast checks on every part, LLM review of odd ones as training data |
| Robots | New situations break trained policies ([Sinha et al.](https://arxiv.org/abs/2407.08735)) | A model for knowing when to stop and think |

The code is built around a detector, a scenario vocabulary, prompts and memory, not around
cats. Replacing cats with people, cars or parts means adapting those four pieces, with new
accuracy and safety evidence for each domain.

**Path:** pets and homes now, where one person can check every answer. Labs, conservation
and farms next. Robotics and industry later, where reliability needs far more proof.

---

## 15. Responsible design

- **Privacy by architecture.** Frames stay on the home network. Only escalated frames (~1%)
  go to the model API. Memory is plain text that the owner can read and delete. Muse Spark
  Open is the path to keeping even those frames local.
- **No diagnosis.** The system says "Qb slept 40% less than his usual afternoon". It never
  says "Qb may be sick". A person and a vet decide.
- **Watching people is different.** Any use with people needs clear consent from the person
  watched. The [EU AI Act](https://presencis.com/regulations/eu-ai-act/article-5/) bans some
  uses outright, such as inferring emotions at work.
- **Humans in the loop.** The identity ladder ends with sending the case to a human rather
  than forcing a call.
- **Honest claims.** Shown on two cats in one room. Every new domain needs its own testing.

---

## 16. Roadmap

| Priority | Item | Why |
| --- | --- | --- |
| 1 | **Identity-consistency trigger**, plus escalating *runs* of T2 relabels | Closes the 772-row gap from §12 |
| 2 | **Full-day replay** with merging, and real cost per day | Replaces the projection in §10.3 with a measurement |
| 3 | **Hand-labeled clip** and a head-to-head with ByteTrack / BoT-SORT (IDF1, identity accuracy) | First real accuracy number |
| 4 | **Memory tool for the agent**: "When did Qbo eat today?" | Turns memory into the product |
| 5 | **Retry + unchained fallback** for API 400s; self-consistency on row-changing verdicts | Reliability |
| 6 | **Verdicts → YOLO dataset → retrain** (export with pre-labels already exists); measure escalation rate before and after | Closes the teacher–student loop |
| 7 | **Memory-only decisions in shadow mode**, then skip images only where memory proves reliable | Next cost cut |
| 8 | **Silent-miss sampler**: periodic checks of empty frames in usual rest zones | Catches sleeping cats |
| 9 | Adaptive camera policy (agent changes model, region or resolution on its own); night/IR model | Uses the camera tools that already exist |
| 10 | Voice questions (Muse Voice Transcribe), daily cat diary (Muse Image/Wonder), Muse Spark Open locally | Broader use of Meta's models; privacy |

---

## 17. Demo script (3 minutes)

About 325 spoken words, or roughly 1.8 words a second, which leaves room to pause on the
visuals. The first 7 seconds stack three surprises: two identical cats, a camera that confuses
them, and a ball mistaken for a cat.

| Time | On screen | Narration |
| --- | --- | --- |
| 0:00 | Close-up of two white cats, then the ball with a "Qbo" box on it | These are Qb and Qbo. My camera can't tell them apart. Last week, it decided this *ball* was Qbo, for twenty-six minutes. |
| 0:08 | Empty apartment, morning light | I work all day. I just want to know simple things: did my cats eat, sleep, play? Cats hide illness, so small changes matter. |
| 0:20 | Cost counter ticking up | Why not send every frame to an AI? One camera makes over a hundred and seventy thousand frames a day. That's about sixty-three thousand dollars a year. A tiny model on a Raspberry Pi is almost free, but it makes mistakes. |
| 0:40 | Timeline chart; red escalation lines appear | So we built both. The Pi watches every frame. When it's unsure, it says exactly why. It knows fourteen kinds of doubt, like "I relabeled this cat by distance alone." |
| 1:00 | Agent trace: the frames it picks, its reasoning | That doubt becomes a plain-English question for Meta's Muse Spark. The agent pulls the frames it needs, compares the cats, and answers with a reason. Watch: "Thin upright tail, cool white fur. This is Qb, not Qbo." Overturned. |
| 1:25 | Three cases in one trace; empty floor highlighted | Related doubts get reviewed together, fetching each image once. Twelve questions, three calls. Here it caught the tracker inventing a cat on an empty floor, and deleted it. |
| 1:45 | A corrected row, tagged with its verdict | Every verdict rewrites the record, and every change is traceable. |
| 1:55 | The day log scrolling | Then memory. Not boxes and timestamps, but a diary: "Qb slept in the green bed, four thirty-four to four forty." Over weeks, it learns what normal looks like for each cat, in each spot. |
| 2:15 | Results card | The numbers: twelve doubts in eight minutes, less than one percent of frames. Twenty-six cents to review them, about a fifth of the cost of sending every frame to the AI. And only those frames ever leave my home. |
| 2:35 | Roadmap card | Next: catch more errors, run full days, and teach the Pi from what the agent learns. |
| 2:45 | Montage: elderly parent, wildlife, factory line | Cats are just the start. Any camera watching for rare moments, like an elderly parent, wildlife or a factory line, has the same gap. Our answer: cheap eyes on every frame, a smart brain only when it matters, and a memory that keeps growing. Built this week, on Meta's Muse Spark. |

---

## 18. How to run it

```bash
# full loop on the measured window: replay + agent validation + memory + charts
python -m agent.main --view-id 3 \
  --start 2026-10-28-16-33 --end 2026-10-28-16-40 \
  --replay-db --validate --memory --plot

# live, without hardware: simulated camera in one terminal, pipeline in another
python -m camera_sim --start 2026-10-28-16-33-00 --end 2026-10-28-16-59-00
python -m agent.main --view-id 3 --start 2026-10-28-16-33 --end 2026-10-28-16-59 \
  --live --validate --memory --plot

# chat with the agent over camera, history and tracker
cd agent && python -m supervisor_agent "Is any cat on camera right now?"

# label ground truth with SAM 3.1 (+ Spark classification), then replay labels
# through the correctors + supervisor validation and export validated ground truth
python image_server/grounding_truth/label_with_sam.py --start 2026-10-28-16-33 --end 2026-10-28-16-40
python -m agent.replay_labels --view-id 3 \
  --hour-dir image_server/retrieved_images/2026-10-28-16 --labels muse_labels --validate

# fleet statistics (agent/tool invokes + skill usage)
# -> documents/agent-tool-invoke-report-20261103.md (regenerate from aura_cat)

# tests (offline)
python -m pytest -q
```

Each run writes its rows, verdicts, charts and counts, one agent trace with evidence images per
batch, and one trace per memory LLM call. A Meta Model API key is required for validation and
memory; without one, memory falls back to an offline stub.

---

## 19. Judging criteria map

| Criterion (weight) | Where this report answers it |
| --- | --- |
| **Impact (30%)** | §2 the problem, firsthand · §3 cat health evidence and 53M households · §14 generalization · §15 responsible design |
| **Use of Meta AI tooling (25%)** | §8: Muse Code as development environment (voice input, workflow skills); Wonder for high-level design, Spark 1.3 as multi-turn vision agent with MCP tools, runtime-built strict schemas, joint multi-case review, vision activity labeling and memory compression; SAM 3.1 for ground-truth labeling; Glimmer at the edge; API probe findings |
| **Demo (25%)** | §17 script built from real artifacts · §9 measured runs · deterministic replay anyone can rerun (§18) |
| **Technical depth (20%)** | §5–7 architecture · §11 tests, determinism, traces · §12 honest failure analysis |
| Bonus: Best Agentic Use-Case | Doubt-driven escalation, tool choice under image budgets, merged reasoning, verdicts that change data |

---

## 20. References

**Project measurements.** Session figures (runs `012059`, `221606` and `231613`, view 3, window
2026-10-28 16:33–16:41) come from the project's own replay runs and agent traces of 2026-11-02.

**Cat health, behavior and market**

- [AAFP senior cat care brochure](https://catvets.com/wp-content/uploads/2024/01/AAFPSeniorCareBrochureBW.pdf)
- [Shin et al., litter box use, J Vet Clin 2024](https://koreascience.kr/article/JAKO202414243207275.pdf)
- [dvm360: smart litter box monitoring and feline CKD](https://www.dvm360.com/view/smart-litter-box-monitoring-identifies-behavioral-patterns-associated-with-feline-ckd)
- [CKD prevalence in apparently healthy older cats (PubMed 24217707)](https://pubmed.ncbi.nlm.nih.gov/24217707/)
- [AVSAB summary of Smit et al., 2024 (cat activity budgets)](https://avsab.org/can-computers-tell-us-what-cats-do-all-day/)
- [Petfood Industry: cat ownership grows to 53 million US households](https://www.petfoodindustry.com/pet-food-market/market-trends-and-reports/news/15834866/report-cat-ownership-grows-to-53-million-us-households)
- [APPA 2025 Dog & Cat Report](https://americanpetproducts.org/news/the-american-pet-products-association-appa-releases-2025-dog-cat-report)
- [Market.us: AI pet camera market](https://market.us/report/ai-pet-camera-market/)
- [Whisker Litter-Robot 5 Pro with AI cat recognition](https://www.techedt.com/whisker-introduces-litter-robot-5-pro-with-ai-facial-recognition-for-cats)
- [Petlibro Luma comparison](https://ca.petlibro.com/blogs/all/how-does-the-new-luma-smart-litter-box-stack-up)

**Computer vision and systems**

- [Beery et al., Recognition in Terra Incognita](https://arxiv.org/pdf/1807.04975)
- [Koh et al., WILDS](https://arxiv.org/pdf/2012.07421)
- [Dodge and Karam, image quality and DNNs](https://arxiv.org/pdf/1604.04004.pdf)
- [Kang et al., NoScope](https://web.cs.ucla.edu/~ravi/CS239_W20/papers/noscope.pdf)
- [Mullapudi et al., Online Model Distillation](https://arxiv.org/pdf/1812.02699)
- [Bhardwaj et al., Ekya](https://arxiv.org/pdf/2012.10557)
- [Bhaskar et al., pseudo labels from VLMs](https://arxiv.org/pdf/2511.09955)
- [Zheng et al., Cerberus](https://arxiv.org/pdf/2510.16290)
- [Zhao et al., SmartHome-Bench](https://arxiv.org/abs/2506.12992)
- [Yan et al., Avas](https://arxiv.org/html/2505.00254v3)
- [Roboflow trackers comparison](https://trackers.roboflow.com/latest/trackers/comparison/) · [ByteTrack](https://arxiv.org/abs/2110.06864) · [OC-SORT](https://arxiv.org/abs/2203.14360)

**Hardware, pricing and scale**

- [Ultralytics Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi)
- [Sixfab Pi 5 AI HAT benchmark](https://sixfab.com/blog/raspberry-pi-5-ai-hat-benchmark-deepx-vs-hailo/)
- [TokenCost: Muse Spark pricing](https://tokencost.app/blog/meta-muse-spark-1-1-pricing)
- [CNBC: one billion surveillance cameras](https://www.cnbc.com/2019/12/06/one-billion-surveillance-cameras-will-be-watching-globally-in-2021.html)

**Broader impact**

- [WHO falls fact sheet](https://www.who.int/en/news-room/fact-sheets/detail/falls)
- [Home-cage monitoring review](https://www.frontiersin.org/journals/behavioral-neuroscience/articles/10.3389/fnbeh.2022.877323/pdf)
- [WSU: AI cuts wildlife review time](https://news.wsu.edu/?p=260863)
- [Computer vision in dairy farming](https://cris.unibo.it/retrieve/handle/11585/1046536/0198d1c4-e7c7-4dbe-b1d7-548678026540/animals-15-02508-v2.pdf)
- [Industrial defect detection with scarce labels](https://arxiv.org/pdf/2509.05796v1)
- [Sinha et al., real-time anomaly detection and reactive planning with LLMs](https://arxiv.org/abs/2407.08735)
- [EU AI Act Article 5 summary](https://presencis.com/regulations/eu-ai-act/article-5/)

Prices and market sizes come from third-party reports as of November 2026. Session figures
are the project's own and have not been independently verified.
