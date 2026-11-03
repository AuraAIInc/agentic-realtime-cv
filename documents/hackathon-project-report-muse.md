# Agentic Computer Vision for Real-Time Detection with Memory

## Hackathon Project Report — Meta Global AI Developer Hackathon 2026

**Track:** Indie Developer · **Problem statement:** Build for Your World
**Team:** Terry & Muyang (2 people)
**Demo window:** 28 Oct 2026, living-room camera (view 3), 16:33–16:40, cats Qb & Qbo
**Models:** Muse Spark 1.3 via the Meta Model API (supervisor agent and memory) · custom YOLO cat-identity model on a Raspberry Pi (edge detector)
**Date of this report:** 2 Nov 2026

> **One-sentence pitch.** A $35 Raspberry Pi watches two cats 24/7 with a tiny YOLO model. When the tracker is unsure — a label flipped by nearness, a box teleported, a visit voted on thin evidence — a Muse Spark agent looks at the real frames, decides, corrects the record, and writes what it learned into a plain-language spatial-temporal memory. The expensive model thinks only when it matters, so the system gets cheaper and smarter the longer it runs.

This report is the complete, standalone hackathon write-up. All evidence needed to evaluate the claims — measurements, costs, prompts, verdicts and failure cases — is included in the body and appendices. Where a claim is measured, the exact session and numbers are given. Where a claim is still a design goal, it is labelled as planned rather than shown.

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [How to read this report in 3 minutes (for judges)](#2-how-to-read-this-report-in-3-minutes-for-judges)
3. [The personal problem](#3-the-personal-problem)
4. [Why this matters now — market and research context](#4-why-this-matters-now--market-and-research-context)
5. [The fundamental gap: fast vs. smart vision](#5-the-fundamental-gap-fast-vs-smart-vision)
6. [Cost analysis: why “send everything to an LLM” cannot work](#6-cost-analysis-why-send-everything-to-an-llm-cannot-work)
7. [What the literature already proves — and what it leaves open](#7-what-the-literature-already-proves--and-what-it-leaves-open)
8. [System overview](#8-system-overview)
9. [Architecture in detail](#9-architecture-in-detail)
10. [The five barriers and how each module answers them](#10-the-five-barriers-and-how-each-module-answers-them)
11. [Meta model usage](#11-meta-model-usage)
12. [Implementation inventory](#12-implementation-inventory)
13. [What was built before vs. during the hackathon](#13-what-was-built-before-vs-during-the-hackathon)
14. [Experimental results](#14-experimental-results)
15. [Demo walkthrough](#15-demo-walkthrough)
16. [Cost and performance, measured](#16-cost-and-performance-measured)
17. [Comparison with alternatives](#17-comparison-with-alternatives)
18. [Broader impact](#18-broader-impact)
19. [Ethics, privacy and safety](#19-ethics-privacy-and-safety)
20. [Limitations — stated plainly](#20-limitations--stated-plainly)
21. [Roadmap](#21-roadmap)
22. [Deployment and operation](#22-deployment-and-operation)
23. [Team and acknowledgements](#23-team-and-acknowledgements)
24. [References](#24-references)
25. [Appendices](#25-appendices)

---

## 1. Executive summary

A home camera at 2 frames per second produces **172,800 frames a day**. At a working LLM rate of **$0.001 per frame** ($2 per 2,000 frames, ≈800 input tokens at Muse Spark list pricing), reviewing every frame with a vision LLM costs **$172.80 per day, $5,184 per month, $63,072 per year — for one camera**. No household, lab or small farm can pay that. Yet the cheap alternative — a small YOLO model on a Raspberry Pi — fails in exactly the ways that matter at home: it misses a sleeping cat curled in a bed, it swaps the identities of two similar cats for hundreds of frames at a time, and it has no memory of what “normal” looks like for *this* cat in *this* room.

This project closes that gap with a loop, not a cascade:

```
Raspberry Pi (YOLO, every frame)
        │
        ▼
TrackCorrector + PeriodCorrector  ──►  14 named uncertainty scenarios (T1–T7, P1–P7)
        │                                        │
        │  only when unsure                      │  plain-language prompt + output schema
        ▼                                        ▼
              ValidationWorker (background, batched) ──► Muse Spark supervisor agent
                                                              │  MCP tools: live camera,
                                                              │  stored images, tracker state
                                                              ▼
                                                   verdict (structured JSON) ──► validated rows
                                                              │
                                                              ▼
                              Spatial-temporal memory (short-term day log + trusted long-term YAML)
                                                              │
                                                              └──► briefs future escalations; becomes tomorrow's training data
```

**What is demonstrated today (one replayed 7-minute window, 1,319 frames, both cats on camera):**

| Measure | Result |
| --- | --- |
| Frames ingested | 1,319 at ~2.7 fps (view 3, 800×600) |
| Detections kept (>0.5 conf) | 1,016 (818 Qb, 198 Qbo) |
| Tracked rows after correctors | 1,265 live rows + 10 tombstones → 4 visits |
| Escalations raised | **12 (0.9% of frames)** — 8 frame-level, 4 period-level |
| Agent verdicts returned | 11 of 12 (1 lost to a Meta API 400, no retry yet) |
| Images the agent actually read | 55 |
| Validation wall time | ~9.4 min for 12 cases (~47 s / case, sequential) |
| Tokens / cost (validation only) | **517,607 tokens, $0.60** at Muse Spark 1.3 Standard pricing |
| Cost per completed review | **≈ $0.054** ( ≈ 54 frames at the $0.001 working rate) |
| Rows corrected / confirmed | 2 rows relabelled, 167 rows stamped confirmed |
| Memory calls (same run) | 29 additional LLM calls (period labelling, promotion, day merge) |
| Headline finding | Agent overturned **3/3** sampled nearness-only relabels; upheld **3/3** interpolated runs and the teleport split |

**What is not yet claimed:** no hand-labelled ground truth exists yet, so there is no mAP / precision / recall headline. The honest claim today is narrower and, for a hackathon, stronger: *the system names its own doubt in 14 specific ways, asks a vision LLM a precise question with the right frames attached, applies the answer as auditable data, and remembers the outcome in language a person (and a future agent) can read.* The tracker’s largest error in the demo window — **772 of 818 Qb detections relabelled as Qbo by the corrector, only 3 of which escalated** — is reported in the same breath, because fixing that trigger recall is the next engineering priority.

**Why it matters beyond cats:** the same “watch everything cheaply, think only when it matters, remember what you learn” loop applies to any fixed camera that watches for rare, meaningful moments — animal labs, wildlife transects, livestock barns, elder-care homes, factory lines. Section 18 maps each domain to the evidence that it shares the gap. The cat camera is the proof case because the author lives with the problem and can verify every verdict by walking into the next room.

---

## 2. How to read this report in 3 minutes (for judges)

**Stage 1 (pass/fail) — Meta model use.** All LLM reasoning, vision and tool-calling runs on **Muse Spark `muse-spark-1.3`** through the Meta Model API (`https://api.meta.ai/v1`, Responses endpoint). The `openai` Python package is only the HTTP client; no OpenAI model is called. The supervisor agent, the validation worker and the spatial-temporal memory all use Muse Spark 1.3 as configured for the agent and memory roles. The edge detector is YOLO (Ultralytics) — it is not an LLM and is not presented as one. Section 11 gives the full role table, including the SAM 3.1 / Muse Voice / Muse Image roles that are wired in config but not yet called in code (labelled as such).

**Stage 2 — the four weighted criteria:**

| Criterion (weight) | Where to find it | One-line answer |
| --- | --- | --- |
| **Impact (30%)** | §3, §4, §18 | 53 M US cat households (APPA 2026); 1.1 B cameras worldwide; a method that makes continuous, identity-aware pet (and, by generalisation, animal / care / industrial) monitoring affordable on a Pi instead of $63 k/yr/camera. |
| **Use of Meta AI tooling (25%)** | §9.4, §10, §11 | Muse Spark is not a chatbot bolt-on: it is the *supervisor* in a closed loop — it receives 14 scenario-specific prompts with schemas, calls 24 MCP tools + 10 local helpers + 5 skills (39 tools total), inspects real frames, returns structured verdicts that rewrite tracked rows, and writes the long-term memory that briefs its own future decisions. |
| **Demo (25%)** | §15 | An 8-stage walkthrough on a real 7-minute window (1,319 frames, both cats) from raw JSON stream to plain-language day log, with the original, revised and validated timeline charts, one full escalation prompt, and one agent trace. Replayable as described in Section 22. |
| **Technical depth & execution (20%)** | §9, §12, §14 | ~19 k lines of Python in the hackathon surface, 297 tests, 49 hackathon-period commits, two hand-built correctors with a 15 s hindsight buffer, a 14-code escalation vocabulary with per-code prompts/schemas, a batched validation worker, and a two-tier (short/long) spatial-temporal memory with a trust gate and anti-overfitting promotion. |

**What to watch for in the demo video (3 min):** (0:00–0:20) two near-identical cats and a camera that mixes them up; (0:20–1:00) ask the agent a question by chat and watch it fetch the right frames itself; (1:00–2:00) the loop — CV fails → escalation fires → Muse Spark corrects it → the corrected row and the memory entry appear; (2:00–2:40) “label the cat when it sits in the tent” — a new label created by conversation; (2:40–3:00) cost slide ($63,072/yr all-LLM vs. $0.60 for the demo window’s escalations) and what was built this week.

**A note on honesty.** This report follows a simple discipline: every number is tied to a measured session described in this report, and every unbuilt piece (automatic retraining, memory-only decisions, SAM/Voice/Image roles) is labelled *planned*, not *shown*. The failures are listed alongside the successes in Sections 14 and 20 — that is intentional.

---

## 3. The personal problem

> I live with two cats, Qb and Qbo, but I only see them for 2 to 3 hours on a workday. I work 8 hours a day, come home tired and go to bed early. The rest of their day is invisible to me.
> — Project author, on living with Qb and Qbo

The questions are simple and daily:

- When did he eat, and when did he use the litter box?
- When and how long did he sleep?
- When and how long did he play?
- How does today compare with his normal pattern over a month or a year?

A conventional smart camera cannot answer them. It can show a live feed and, at best, send a “pet detected” notification. It cannot reliably tell Qb from Qbo, it cannot summarise a day, and it does not get better at *this* home over time. The first prototype — a fixed YOLO pipeline on a Raspberry Pi — exposed the three problems that became this project:

1. **General models miss my cats in my own rooms.** Low light, odd poses (curled asleep in a bed at the frame edge), occlusion and breed similarity defeat an off-the-shelf detector. In a recorded audit on 31 October 2026, a sleeping cat plainly visible in a green bed at the bottom of an 800×600 frame was missed by *both* the custom cat-identity model and the generic COCO model — neither reported any detection on the live loop or on a fresh re-check.
2. **A 24/7 deployment has to run on cheap hardware.** A Pi 5 with an AI HAT averages 6.1–7.6 W in published benchmarks; a home cannot host a GPU server per camera, and should not need to.
3. **Sending every frame to an LLM costs too much.** Section 6 quantifies this: $63,072 per camera per year at the working rate.

### Why continuous tracking matters for cat care (and why the claim stays modest)

Cats hide illness. The American Association of Feline Practitioners’ senior-care brochure tells owners that cats are “masters of hiding signs of disease and pain” and that *any* change in normal behaviour or routine can be a sign of pain. Direct measurement beats owner recall: a 2024 study of litter-box use (6,555 daily records, 37 cats) found guardians reported *lower* urination frequency than video recording showed, and set healthy ranges at 1–7 box visits per day. Disease does shift these patterns, but modestly — diseased cats averaged 3.77 visits/day vs. 3.38 for healthy cats, with AUC below 0.6 from visit count and stay time alone. A Nestlé Purina / Ohio State study of 136 cats (published in *Animals*, reported by dvm360) found clearer differences with chronic kidney disease (CKD), yet CKD recall was only 57.1% in validation.

The honest product claim, carried through this entire report, is therefore: **long-term, per-cat tracking gives owners and vets better information — a prompt to call the vet, not a diagnosis.** The hackathon organisers list “AI medical advice” as a banned anti-project; this system is deliberately framed as behaviour and activity observation with a human deciding.

### Target users, in order

1. **Multi-cat households** (the author’s home: Qb, cat 25, short-haired white/silver; Qbo, cat 26, long-haired cream/beige with a dark plume tail — profiles in ).
2. **Pet households generally** — any home where identity (“which pet?”) and routine (“what changed?”) matter more than a motion alert.
3. **Any fixed-camera scene** where a specialist would otherwise be needed to label a new location — the method generalises; the detector, triggers and memory adapt per camera (see Section 18).

There are no external users yet. Impact is argued from the size of the problem and the cost curve, not from traction — exactly as the project plan prescribes.

---

## 4. Why this matters now — market and research context

This section adds current market and ecosystem context (November 2026) to the literature review in Section 7. It is the “why now” behind the Impact score.

### 4.1 The pet market is large, growing, and feline-led

- **US pet industry spend reached $158 billion in 2025**, up 3.7% year-on-year, with $165 billion projected for 2026 (APPA 2026 State of the Industry Report, released at Global Pet Expo, March 2026).
- **53 million US households owned a cat in 2025 — 39% of all households**, up 5% in one year after a 23% jump in 2024 (APPA 2026 Cat Report, from the 2025 National Pet Owners Survey). Cat ownership now outpaces most other pet categories, driven by Gen Z and Millennial owners. Ownership of *two* cats rose 7% year-on-year; households with three or more cats rose 45% since 2018. Multi-cat identity — the exact problem this system solves — is therefore the growth segment, not an edge case.
- **≈95 million US households own at least one pet** (≈70% of homes). The addressable base for a home-camera product is not a niche.
- **Smart pet cameras are already a billion-dollar category.** Estimates for 2025 range from $0.97 B (pet smart cameras) to $1.45–1.57 B (pet camera / home pet camera), with forecasts of $4–6 B by 2034–2035 at 10–15% CAGR, driven explicitly by “AI-powered behaviour recognition and health alerting” and subscription monetisation. Furbo (360° camera, treat tossing, “Furbo Nanny” pet ID for up to 3 pets), Petcube (AI recognition, owned by Google), Wyze, Eufy and Google Nest Cam all compete on *detection*, none on a per-scene learning loop with a verifiable memory. That gap is the product wedge (§17).

Implication: the user does not need to be convinced to put a camera in the living room — they already have. What they lack is a system that knows *which* cat, *what* it did, and *what changed*, without a person reviewing footage.

### 4.2 The camera installed base makes the cost problem universal

- **Over 1.1 billion surveillance cameras were in active operation worldwide by end-2025** (multiple industry reports; the 2019 figure was 770 M, forecast 1 B by 2021). Asia-Pacific accounts for the largest share; ≈72% of new installs are IP-based and ≈41% already claim some AI analytics.
- The **video surveillance market** was $54–69 B in 2024–2025 and is forecast at $89–99 B by 2032–2034. **AI in video surveillance** alone was $6.4 B in 2025, forecast $49 B by 2035 (22.6% CAGR), with AI analytics expected on 420 M+ cameras by 2025.
- At the working rate from Section 6, sending *all* of those cameras to an LLM is an arithmetic absurdity (≈$49 trillion/year for 770 M cameras) whose only purpose is to show scale: even a 100× price drop leaves ≈$631/year per camera. The bottleneck is not model quality; it is *sustained attention* — a frontier vision model can answer about one frame in under a second, but it cannot affordably watch every frame, on every camera, all day.

### 4.3 Edge AI has crossed the “good enough on a Pi” threshold

- Ultralytics documents **YOLO26n on a Raspberry Pi 5 CPU (NCNN, 640 px) at 67 ms per image** for inference alone — leaving headroom for 2 fps with pre/post-processing. A Pi 5 + Hailo-8L AI HAT (13 TOPS, ~$150 total) runs YOLO-class detectors at 30+ fps on 1080p in Raspberry Pi Foundation testing; a Jetson Orin Nano (40 TOPS, $249–399) is the next step up. The refined claim is: *“The Pi runs a small model at 2 FPS, not a large one.” An AI HAT extends this.*
- Industry cost models now state the crossover plainly: on-camera detection at 2–5 W beats cloud video analysis (≈$0.10/min for stored video) the moment cameras run more than a couple of hours a day; above ~50–80 cameras, edge hardware amortises to ≈$11/camera/month vs. $50–150/camera/month for cloud-only VMS. For a 24/7 home camera, edge-first is not a preference — it is the only viable default. Cloud/LLM review must therefore be *selective*.

### 4.4 Agentic vision is now a platform capability, not a research demo

- **Meta Muse Spark** (1.1 in July 2026, 1.3 as used in this system) is a multimodal reasoning model built for agentic tasks: 1 M-token context, native tool-calling, and top scores on Scale AI’s **MCP-Atlas** benchmark for Model Context Protocol tool use (88–89, ahead of contemporary Claude/GPT models in independent reporting). The Meta Model API opened in public preview alongside it. Muse Spark Open (816 B total / 42 B active MoE, released 15 Oct 2026) brings the same capability to self-hosted/partner inference (Baseten model `meta-models/muse-spark-open-1026`).
- The **Model Context Protocol (MCP)** has become the standard way for an agent to call external tools. This project was already built on MCP before the hackathon (three MCP servers: camera, images, tracking) — the hackathon work ported the agent from OpenAI to Muse Spark and closed the loop into tracking and memory. The fit is therefore native, not retrofitted.
- The research frontier confirms the direction: **Cerberus** (2025) cascades motion/CLIP filtering into a VLM and reports a 151.79× speedup for a 2.8% AUC drop at 1% anomaly share (on a GPU server); **SmartHome-Bench** (CVPRW 2025) benchmarks multimodal LLMs on 1,203 home-camera videos; **Avas** gives agents a memory of long video (at minutes-per-answer cost). None of the reviewed work joins *attention + knowledge flow-down + spatial-temporal memory* in a loop on low-cost hardware for long-term home pet tracking. That join is this project’s contribution, stated within the limits of a targeted (not exhaustive) review.

**Net:** the market (tens of millions of cat homes, a billion cameras), the hardware (Pi-class edge inference at 2 fps for watts), and the platform (Muse Spark + MCP) have all arrived at once. The missing piece is the system that decides *when* to spend the expensive model — and remembers what it decided. That is what was built.

---

## 5. The fundamental gap: fast vs. smart vision

Picture one night at home. Qb walks to the litter box for the third time in an hour. The detector on the Pi reports `cat, 91%`. It is correct, and it tells the owner almost nothing. The real question — *“Is he OK?”* — needs identity (which cat?), activity (what is he doing?), norm (what is usual for him?) and change (what is different today?). A vision LLM could reason about that frame, but nobody told it to look. **The fast model saw the moment and did not understand it. The smart model could have understood it and never saw it. That is the gap.**

The two kinds of model fail in opposite ways, so neither replaces the other:

|  | Real-time computer vision (YOLO on Pi) | Vision LLM (Muse Spark) |
| --- | --- | --- |
| Speed | 67 ms/image (YOLO26n, Pi 5 CPU, inference only) | 0.46–3.63 s/image (independent Gemini bounding-box test; Muse validation turns took 30–80 s *including* tool calls) |
| Cost of 24/7 | Fixed hardware, ~6–8 W average with AI HAT | Paid per frame — $63,072/yr/camera at the working rate |
| Scope | Only the classes it was trained on | Any question in plain language |
| Best at | Where things are, frame after frame | What things mean and why they matter |
| Precise boxes | Co-DETR ≈0.60 mAP on COCO (cited) | Gemini 3 Pro Preview 0.407 mAP on COCO, no class training |
| Adapting to a new scene | New labels + retraining (COCO alone took >70,000 worker hours) | A new prompt |
| How it fails | Misses what it never trained on | Can hallucinate; struggles with counting; context window too small for hundreds of hours of video |
| Sense of time | Tracks across seconds; no long-term understanding | No persistent memory across days by itself |

**Why better models alone will not close it.** The gap comes from scale. Each new scene restarts labelling for the CV side, and deployed models drift — Ekya (NSDI 2022) measured a **22% accuracy drop from data drift** on edge models, recovered only by retraining. Most cameras will never get a dedicated labelling team, including the one in the author’s living room. On the LLM side, even a 100× cheaper model still costs hundreds of dollars per camera-year and still sends private home footage to a cloud.

**The same gap, everywhere.** Any camera that watches for rare, meaningful moments faces it:

| Setting | What a fast model can do | What needs reasoning | Evidence |
| --- | --- | --- | --- |
| Senior care at home | Find a person in a room | Tell resting from a fall or a routine change | SmartHome-Bench includes this category |
| Baby monitoring | Track a child in a crib | Judge whether a situation is unsafe | Same benchmark |
| Wildlife monitoring | Spot an animal | Handle a new site and rare species | iWildCam macro F1 47.0 → 31.0 on unseen cameras |
| City traffic | Count vehicles | Judge whether a scene breaks a local rule | Cerberus tests rules drawn from each scene |
| Medical imaging | Screen tissue patches | Hold up at a new hospital | Camelyon17 accuracy 93.2% → 70.3% across sites |
| Robots | React in real time | Cope with new light, weather, places | WILDS surveys these shifts in robotics |

**The fundamental problem, stated as a design requirement:**

> How can a system see everything cheaply, think only when it matters, and grow more capable the longer it runs?

A simple cascade (NoScope-style filtering) answers only the first half: it filters frames for *one fixed question*. An agent adds what a filter lacks — it decides when to look, chooses tools, weighs context, asks a person when unsure, and keeps what it learns. Three things must cross the gap:

- **Attention flows up.** The fast model (and its tracker) finds the moments that deserve expensive thought.
- **Knowledge flows down.** Each verified answer becomes training data, so one camera learns its own world without a labelling team. *(Designed; the retraining step is not built yet — §20.)*
- **Memory spans both.** It turns isolated frames into a history, so “unusual” means unusual *for this cat, this room, this hour*.

Each piece exists in prior work — Mullapudi et al. taught small models from a costly teacher (7–17× lower runtime cost); Cerberus cascades into a VLM; Avas gives agents long-video memory. Within this targeted review, none joins all three in a loop on low-cost hardware. That loop is the project’s larger bet: **if it works, every additional hour of video should make the system cheaper and better at once.** A cat camera is where it can be proved, because the author cares about the answer and can check it when he gets home.

---

## 6. Cost analysis: why “send everything to an LLM” cannot work

One camera at 2 fps produces **172,800 frames/day** and ≈63 million frames/year. The working rate of **$0.001/frame** comes from the project’s figure of $2 per 2,000 frames. It equals ≈800 input tokens at the reported Muse Spark rate of $1.25 per million input tokens (Meta Model API Standard pricing as recorded in the model configuration: input $1.25 / cached $0.15 / output $4.25 per 1 M), *before* prompt or output text. Real calls cost more once the system prompt, tool schemas and the model’s answer are counted — the session measurements in §16 confirm this.

| Frames sent to the LLM | Frames/day | Cost/day | Cost/month (30 d) | Cost/year |
| --- | --- | --- | --- | --- |
| 100% (every frame) | 172,800 | $172.80 | $5,184 | **$63,072** |
| 20% | 34,560 | $34.56 | $1,037 | $12,614 |
| 5% | 8,640 | $8.64 | $259 | $3,154 |
| 1% | 1,728 | $1.73 | $52 | $631 |
| 0.1% | 173 | $0.17 | $5 | $63 |

The escalation rates in the lower rows are *scenarios*, not measurements — the demo’s job is to replace them with the real rate. Two facts make low rates plausible: a 28-cat accelerometer study found cats active ≈3% of the day and lying/sitting ≈74%, with elimination taking under 6 minutes/day; and the tracker only escalates on *decision conflicts*, not on every low-confidence frame (detections ≤0.5 are dropped before tracking by default, so they never reach the escalation rules — §10, Barrier 4).

**Caveats a judge should hear before asking:**

1. OpenRouter’s listing for Muse Spark does not state how many tokens an image consumes; the $0.001 rate must be confirmed from real API usage logs. The system does exactly that: every agent run records `input_tokens` / `output_tokens` / `cached_tokens` and an estimated cost from the prices in `model configuration`, both in the per-run execution trace and, when invocation logging is enabled, in the invocation log tables.
2. Meta’s reported *Contributor* tier cuts input to $0.10 per million (≈$5,046/year at 800 tokens/frame for all frames). That is still a large bill for one home camera, and it lets Meta train on the prompts — which matters for footage from inside a home. Selectivity is also a *privacy* feature: only escalated frames ever leave the house.
3. The demo window’s measured validation cost was **$0.60 for 12 escalations** (≈$0.054/completed review). At $0.001/frame, the same 1,319-frame window would have cost ≈$1.32 if every frame had gone to the LLM — so in this *busy* window (both cats active, many tracker conflicts) the agent saved only about half, and that $0.60 excludes memory’s own 29 LLM calls. A full-day run — mostly empty rooms and sleeping cats — is the real test of the saving, and it has not been run yet. The report does not hide this (§14, §20).

---

## 7. What the literature already proves — and what it leaves open

The assumption check below because it disciplines every claim in the demo. Each verdict summarises the supporting evidence.

| Assumption | Verdict | Evidence | Wording to use |
| --- | --- | --- | --- |
| General models lose accuracy in a new scene | **Supported** | iWildCam macro F1 47.0 → 31.0 on unseen cameras (WILDS); Beery et al. found poor generalisation to new locations, especially classification | Keep it, but note the evidence comes from fixed wildlife cameras, not homes |
| General models can *never* detect every frame | **Too strong** | No source supports “never”; evidence shows a gap, not an absolute limit | “General models miss too many frames in my rooms.” Show the measured miss rate |
| Low-quality images hurt models | **Partly supported** | Dodge & Karam: blur and noise hurt most (image classification); JPEG/contrast at typical levels hurt less | Name blur, noise and low light; report detector results on own night frames |
| Fine-tuning for one camera helps | **Supported** | Mullapudi et al. matched a costly teacher at 7–17× lower runtime cost; NoScope specialised models ran up to 340× faster | Keep it; add that a fixed camera sees a narrow slice of the visual world |
| 24/7 monitoring must run on cheap hardware | **Supported** | LLM-on-every-frame costs far more than a Pi (§6); Pi 5 + AI HAT 6.1–7.6 W | Keep it |
| A Pi cannot run a capable model | **Needs refinement** | YOLO26n on Pi 5 CPU/NCNN 67 ms/image at 640 px (inference only) leaves room for 2 fps | “The Pi runs a small model at 2 FPS, not a large one.” AI HAT extends this |
| Video LLMs can understand frame content | **Supported** | Cerberus: Qwen2.5-VL 82.51% AUC vs. 64.31% CLIP alone; VLMs can miss small/peripheral detail | Treat the LLM as a strong reviewer, not ground truth |
| Sending every frame to an LLM is too expensive | **Supported** | At the working rate, one 2 fps camera costs $172.80/day | Keep it; state $0.001/frame is the working rate |
| Combining LLM + CV balances cost and accuracy | **Supported in adjacent work** | Cerberus: 151.79× speedup for 2.8% AUC drop at 1% anomaly share (GPU server) | Say it is untested for home pet tracking; the demo measures it |

### Literature map (targeted review, not exhaustive)

| Theme | Work | Key result | What it gives this project |
| --- | --- | --- | --- |
| Scene shift | Beery et al., ECCV 2018 | Excellent at training locations, poor at new ones | Why a fixed-camera model degrades when moved |
| Scene shift | WILDS (iWildCam, Camelyon17), ICML 2021 | iWildCam F1 47.0 → 31.0; Camelyon17 accuracy 93.2% → 70.3% | A measured size for the gap |
| Image quality | Dodge & Karam, 2016 | Blur and noise degrade deep networks most | Why cheap-camera and night frames are hard |
| Specialisation | NoScope, VLDB 2017 | Difference detectors + specialised models; 1–5% accuracy loss on binary tasks; up to 1,000× speedup | The original cheap-filter cascade for fixed video |
| Teacher–student | Mullapudi et al., ICCV 2019 | Student trained online from an intermittent teacher; 7–17× lower cost | Closest match to an LLM teaching a Pi model |
| Drift | Ekya, NSDI 2022 | Data drift cut edge-model accuracy by 22%; retraining helps | Why a one-time fine-tune is not enough |
| Noisy LLM labels | Bhaskar et al., 2025 (preprint) | VLM pseudo-labels: mAP@0.5 31.12 → 46.61 on KITTI with noise handling; 57.97 with 10% human labels | Why human validation is worth its small cost |
| Cascaded VLM | Cerberus, 2025 (preprint) | Motion + CLIP filter before a VLM; rules improved by automated and user feedback | Closest system design — built for anomalies on a GPU server |
| Smart-home LLMs | SmartHome-Bench, CVPRW 2025 | 1,203 home-camera videos to test multimodal LLMs | LLMs on home video are an active research topic |
| Long-video agents | Avas, 2025 | Agent memory of long video; answers take minutes | Memory is necessary; latency must be managed off the real-time path |

*Preprints (Bhaskar, Cerberus) are not peer-reviewed in the versions read. Most evidence comes from wildlife, traffic and driving video. These sources are treated as analogies until the project’s own cat numbers exist.*

**The gap in the literature is the contribution:** no reviewed work combines (a) an always-on edge detector, (b) an agentic LLM supervisor that can steer the camera and call tracker/image tools, (c) a spatial-temporal memory that briefs future decisions, and (d) a path from verified verdicts back into training data — on low-cost hardware, for long-term home pet tracking. Each piece exists; the loop does not.

---

## 8. System overview

### 8.1 Design principle

> Treat LLM calls as a scarce budget and spend them only where the Pi is unsure.

The Pi detects cats cheaply, all day. A two-stage tracker cleans up the detections and — crucially — *reports its own doubt* in 14 named scenarios. Only those doubts become LLM calls. Each call is a precise question (“is the photo really cat 26, or did nearness steal a neighbour’s box?”) with the disputed frames attached and a strict JSON schema for the answer. Verified answers correct the tracked rows and, through labelled period activity, become the spatial-temporal memory. Memory then briefs the next escalation, so the agent reasons with history, not from scratch. Verified labels are designed to become training data so the Pi needs the LLM less over time — that retraining step is the main unbuilt piece (§20, §21).

### 8.2 End-to-end flow (one `agent.main` run)

All input modes feed the same ingestion routine. Each message carries the view identifier, the capture time and the detection list (cat identifier, bounding box and confidence for each detection).

```
detections ──► ingest ──► TrackCorrector ──► PeriodCorrector ──► charts and run summary
(live / sim /     (window,      (frame scope)      (period scope)
 file / DB)       min-conf)            │                    │
                                       │ escalations        │ escalations
                                       ▼                    ▼
                              ValidationWorker queue ──► supervisor agent ──► verdicts
                              (background thread)      (images and tracking     (validation journal,
                                                        tools and traces)       validated row set)
                                       │                    ▲
                                       ▼                    │ past context (brief ≤600 chars)
                              LongTermMemory ◄──────────────┘
                       (combined + labelled period activity, trust-gated)
                                       │
                              ShortTermMemory (full-fidelity day log, all confidences)
```

**Four detection sources, one message shape:** live streaming (the camera pushes detections over a WebSocket connection), file replay (a recorded detection stream), database replay (a read-only query over the detection archive; the shared database is never written), and camera simulation (archived frames replayed through the same camera interface as the real Pi, so live operation can be tested without hardware).

**Ingest gate:** only messages inside the requested time window are processed (the final minute is fully included). A detection is kept only when its confidence is strictly above the minimum threshold (default 0.5) and its cat identifier is known for this view. Empty frames are ingested too — they advance buffer timeouts and close visits.

### 8.3 The three MCP servers + local tools (the agent’s hands)

The supervisor agent does not “see the database”. It calls tools, exactly as a human operator would:

| Prefix | Server | Location | Tools |
| --- | --- | --- | --- |
| `camera__` | Live camera on the Pi | Raspberry Pi camera service (MCP port 8765) | `get_status`, `capture_frame`, `get_camera_modes`, `capture_high_res`, `list_models`, `detect`, `get_track_history`, `get_track_frame`, `list_buffered_frames`, `set_schedule`, `reset_schedule` (11) |
| `images__` | Stored images (local archive + cloud storage) | Image archive service (MCP port 10001) | `image_exists`, `images_exist`, `get_image`, `list_images` (4) |
| `tracking__` | Tracker and visit results | Tracking service inside the ingest server (MCP port 10002) | `list_views`, `get_live_status`, `get_last_seen`, `get_current_periods`, `get_periods`, `get_period_detail`, `get_arrival_path`, `get_last_movement`, `get_raw_detections` (9) |
| `local__*` | In-process helpers over the cat knowledge base | Supervisor agent process | `get_cat_profiles`, `find_references`, `get_reference_image`, `make_contact_sheet`, `analyze_trajectory`, `get_label_set`, `zones_for_box`, `get_zones`, `add_label`, `save_zone` (10) |
| `skill__*` | Task and chat workflows | Supervisor agent process | `validate-cat-identity`, `check-trajectory`, `assign-activity-label`, `check-live-detections`, `manage-labels` (5) |

**Total: 24 MCP tools + 10 local + 5 skills = 39 tools.** Default chat hides write/rare tools (`set_schedule`, `reset_schedule`, `list_models`, `get_camera_modes`, `get_raw_detections`, `add_label`, `save_zone`) and pipeline-only skills; `--full-tools` restores them. Each workflow declares its surface (task, chat, or both), which controls where it is offered — a fix that came directly from the trace reviews (§14.4).

### 8.4 System components

| Component | Responsibility | Maturity |
| --- | --- | --- |
| Ingest and tracking services | WebSocket ingest of camera detections, the two correctors, the tracking query service and the tracker visualisation | Correctors and ingest predate the hackathon (2025); the tracking query service and visualisation were added 23–25 October 2026 |
| Camera service | Always-on YOLO detection on the Pi plus the MCP control server (capture, zoom, high-resolution stills, re-detection, schedules) | Added 24–25 October 2026 |
| Image archive service | Query and retrieval over stored camera images (local archive for earlier captures, cloud storage for recent captures) | Added 25 October 2026 |
| Supervisor agent, tracking pipeline, escalation, validation and memory | The Muse Spark supervisor agent, the unified tracking pipeline, the 14-scenario escalation system, the background validation loop, the spatial-temporal memory, the camera simulator and the cat knowledge base | The agent core was built 23–25 October and ported to Meta models on 26 October; the unified pipeline followed on 30 October; escalation, validation and memory were built during the hackathon |

All components require Python 3.10 or later. A single environment configuration serves all components on the same machine; the Pi and the archive host each use only their own section of that configuration.

---

## 9. Architecture in detail

This section is the technical-depth core. All behaviour described here is summarised in full; no external document is needed to follow it.

### 9.1 Edge camera — the steerable camera service

The camera keeps the always-on YOLO loop from v2 and adds an MCP server beside it. The agent does not run every frame — a single agent review took 30–80 s in the session measurements — so it *steers* the loop only when needed. Every override expires within 24 hours; with no MCP traffic the camera behaves exactly as it always did.

**How the pieces fit:**

```
frame_source ──newest full-res frame──► detection_loop ──JSON──► ws_sender ──► ingest server (:8000)
   (camera)                                  │    ▲
                                             │    │ one-shot jobs + overrides
                              frame_buffer ◄─┘    │
                              track_history ◄─┘  controller ◄── camera_service ◄── mcp_server ◄── agent
```

- The **detection loop is the only code that runs YOLO**. Tools that need a model queue a job on the controller; the loop runs it between frames (delaying the default stream by one inference). Tools that only read pixels never touch the loop.
- The **capture thread** reads the camera continuously and keeps only the newest frame, so the loop never falls behind even on a slow schedule (1 frame / 30 s).
- **Zoom is digital:** `capture_frame(box=...)`, `detect(roi=...)` and ROI schedules crop from the full-resolution frame. `capture_high_res` closes the camera, reopens it at the requested size (from `v4l2-ctl --list-formats-ext`; unlisted sizes snap to the nearest), grabs one still, and reopens at the normal size — detection pauses ~2–3 s. All sizes show the full view (YUYV capture scales the full sensor at every size; MJPG centre-crops below 1920×1080). A box is aligned to the still by ORB + RANSAC feature matching, cached per size, falling back to plain scaling with a `warning`.
- **Two detector models are registered:** the default custom cat-identity model (its classes *are* the household cats, so detections carry a cat identifier) and a generic COCO model (80 everyday classes, no cat identity, useful only for “is it a cat at all?” — never for which cat). New models can be registered with a short description so the agent can discover them at runtime.

| Camera tool | What it gives the agent |
| --- | --- |
| `get_status` | Current settings, frame rate, last detections, models, buffer span, WebSocket state |
| `capture_frame` | Newest or buffered frame (`at=`), optional crop, resized to `max_side` (default 1280) |
| `get_camera_modes` | Resolutions `capture_high_res` supports |
| `capture_high_res` | One sharper still at another resolution, optional crop; pauses detection ~3 s |
| `list_models` | Every model available, with `cat_classes` and class names once loaded |
| `detect` | One pass with any model, region, input size or threshold, without touching the tracker |
| `get_track_history` | Per-track first/last seen, hits, max confidence, `identity_consistent` flag |
| `get_track_frame` | Buffered frame where a track first/last appeared (cropped) |
| `list_buffered_frames` | Which recent moments can still be revisited |
| `set_schedule` / `reset_schedule` | Temporary change to fps / roi / imgsz / model (always expires) / immediate return to defaults |

*Honest limits:* night/IR handling is prepared but not finished — the loop flags grey infrared frames (`is_grey`) and `extra_models` accepts an infrared model, but none ships today and nothing switches models automatically. High-res capture is unavailable on RTSP streams (native frame returned).

A **camera simulator** replays archived frames through the same camera interface as the real Pi, so live operation can be tested and demonstrated without hardware. The false-negative audit in Section 14.3 ran in this simulation mode.

### 9.2 Tracking — `TrackCorrector` + `PeriodCorrector` (the connected record)

YOLO gives one independent guess per image: “cat 1 is here, conf 0.62”. Those guesses flicker (missed frames), jump (one bad box) and swap labels. Two stages clean this up. Full mechanics are in this section. The frame-level corrector is about 700 lines and the visit-level corrector about 520 lines.

**TrackCorrector — “which boxes are the same animal?” (frame scope, ~15 s hindsight)**

Per cat it keeps a `buffer` (deque of the last ~15 s of rows, still rewritable), a `base` (last committed row — anchor in the past) and a `target` (most trusted buffer row — anchor in the future). Per frame:

1. **Clean labels** (`add_new_detection_to_buffer`): duplicate boxes with IoU > 0.7 are one animal labelled twice — keep the more confident. A detection overlapping its own cat’s buffer (IoU > 0.5) keeps its label. The rest are checked against other cats’ histories (`match_buffer_history`: IoU > 0.2 with last detection/base, *or* centre within 160 px when that cat was not detected this frame, *or* IoU > 0.8 with any buffer box) and relabelled if they fit — this is what fixes “cat 1 called cat 0 for a few frames”. If a cat had no live detection, still has a `base`, and the new box barely overlaps it (IoU < 0.2), the stale track is wiped and the period closed (**jump reset**).
2. **Pick a target** (`select_target`): first buffer box with IoU > 0.7 against `base`, else first box with conf > 0.5 or overlapping `base`, else highest-confidence box.
3. **Drop noise:** if neither buffer nor period ever reached conf 0.4, erase the buffer’s boxes.
4. **Smooth and fill gaps** (`interplate_buffer`): linear interpolation in time from `base` to `target`. Missed frames get interpolated boxes with `original_cat_id = None` (still distinguishable as synthesised); jitter is smoothed, blending real boxes with the interpolation weighted by confidence when base/target differ a lot (IoU < 0.7).
5. **Commit old rows** (`collect_buffer`): rows older than `min(now − 15 s, target time)` become final, each with the current `period_id`. A jump >300 px from the last committed confident box starts a **new period** and retracts trailing interpolated rows. Nothing detected for 60 s flushes the buffer and closes the period.

Key parameters: `buffer_range` 15 s, `max_undetected_gap` 60 s, `min_conf` 0.4, teleport 300 px, relabel distance 160 px.

**PeriodCorrector — “which cat was this visit, really?” (period scope)**

The tracker assigns each period to the label it was tracked under, which can be wrong for the whole visit (e.g. YOLO mostly called Qb “Qbo”). PeriodCorrector waits until a group of overlapping periods has ended, then votes over the whole period using the *original YOLO labels* (not the tracker’s relabels):

- Stats per candidate cat: `detect_length` (frames labelled as that cat), `detect_rate`, `detect_conf` (mean confidence), `period_area_count` (50×50 px grid cells where that label’s mean conf > 0.5 — rewards a label that holds up across positions).
- Score: highest `detect_conf` +3, highest `detect_length` +2, `detect_length` > 20 → +1, `detect_rate` > 0.5 → +1, > 0.3 → +1, `detect_conf` > 0.5 → +1, highest `period_area_count` +4.2; `detect_length` < 10 → whole score ×0.2.
- Assignment is greedy by best score; **one cat cannot hold two overlapping periods** — if the top cat is taken, that cat’s score is zeroed for this period and it retries with its next-best cat. Best remaining score < 1 → no cat (`None`, too weak to call).

**How this differs from open-source trackers** (and why it is not claimed to be “better” — §17.3): SORT-family trackers (ByteTrack, OC-SORT, BoT-SORT) answer “is this the same object as before?” with anonymous track IDs, frame-by-frame, with Kalman motion models and optional ReID, benchmarked on MOT17 (BoT-SORT 63.7 HOTA at defaults). The RCAT correctors answer “which *known* cat is this?” over a closed set of named cats, with 15 s hindsight, whole-visit voting, a one-cat-per-overlap rule, and — uniquely — a standardised report of their own doubt (next section). The Pi already calls `yolo.track()` with Ultralytics’ tracker; the correctors work on its detections and do not use its track IDs. No head-to-head test exists yet; §21 specifies the fair test.

### 9.3 Escalation — the 14-scenario doubt vocabulary (the bridge to language)

A vision pipeline emits boxes, timestamps and scores. Those numbers carry no account of *what went wrong* or *what decision is needed*, so an LLM cannot judge them directly. The escalation module converts tracker output into language an LLM can judge — and converts the answer back into data.

**Trigger vocabulary:** every escalation is numbered run-wide in emission order, continuous across both scopes (`E<seq>-<code>`, e.g. `E3-P5`), logged as a JSON record + reviewer prompt + output schema, and drawn on the charts as red lines (frames) / light-red spans (periods).

| Code | Scenario | Trigger |
| --- | --- | --- |
| T1 | Multi-match relabel | Detection fits 2+ cats’ histories; first match won |
| T2 | Weak-rule relabel | Box relabelled by centre distance alone, no IoU support |
| T3 | Duplicate tie | Duplicate boxes removed on near-tied confidence |
| T4 | Jump reset | New box missed the anchor (base); buffer wiped, visit restarts |
| T5 | Target fallback | No overlap with base and nothing confident; tracker guessed |
| T6 | Teleport split | Committed box jumped >300 px; visit split |
| T7 | Long synthesis | Long run of interpolated-only frames committed |
| P1 | Close vote | Top-two cat scores nearly tied |
| P2 | Weak winner | Winner barely passed the minimum, or the visit was short (×0.2 penalty) |
| P3 | Conflict reassign | Top cat taken by an overlapping visit; got 2nd choice |
| P4 | Long unknown | Long visit with no assigned cat |
| P5 | Low quality | Low mean confidence, or mostly missing (interpolated) frames |
| P6 | Merge candidate | Visit may continue the same cat’s previous visit |
| P7 | Static link | Sleeping cat may have been split by a long gap (both ends static, overlapping) |

A low-confidence box alone never triggers review: detections ≤0.5 are dropped before tracking, so triggers fire on *conflicts in the tracker’s decisions*, not on detector uncertainty in isolation. All thresholds are centralised in one configuration and fire at the exact decision point inside the correctors.

**Prompts** (about 700 lines, one builder per scenario code): Each states the conflict, the question, the context, the alternatives and which frames to inspect (~1/min key frames with boxes and pick reasons), plus the involved cats’ written profiles. The small on-camera YOLO is explicitly framed as *“a hint, not evidence”*. Real example (shortened from Section 15):

```text
Structured record
  code: T2   rule: distance   original_cat_id: 25   cat_id: 26
  source_distance_px: 128.6   source_threshold_px: 160

Generated prompt (excerpt)
  Conflict: the camera said cat 25, but the tracker re-labelled the box
  to cat 26 by nearness only — the boxes do not overlap at all.
  Validate: is the photo really cat 26?
  Alternatives: keep 25, pick another nearby cat, or drop the box.
```

**Output schemas:** one slim validated model per scenario code, one `verdict` enum plus at most one qualifier (`chosen_cat_id` / `cut_time`), keeping LLM generation short and parseable. Verdict sets include e.g. T2: `confirm / keep_original / reassign_other / drop`; T6: `confirm_split / merge_visits / drop_box`; P5: `keep / drop / trim_core / reassign`; P6/P7: `merge / keep_separate`. A failed parse becomes `{"verdict": "error"}` — never an exception — and every review carries a short `rationale`.

### 9.4 Supervisor agent — Muse Spark with tools (the reviewer)

The supervisor agent comprises the tool-calling loop (about 330 lines), the pipeline task runner (about 440 lines), the system prompt, the MCP hub, the tool adapter, the execution tracer and the invocation logger.

The agent runs on **Muse Spark `muse-spark-1.3`** via the Meta Model API Responses endpoint. The hub executes the model’s function calls (several at once when the model asks for several — `asyncio.gather`, so a turn’s tool cost is the slowest call, not the sum), returns text + real images, and repeats until the model answers. Every run produces a full execution trace (system prompt, every LLM input and output, every tool call with its complete result, per-turn timing and token counts, embedded images, and a statistics block with pricing and cost) plus every image the tools returned. When invocation logging is enabled, runs are also recorded in the invocation log from a background thread, adding no response latency; the record is dropped with a warning if the log store is unreachable.

**System prompt** (folded into the first user turn — see Section 14.5 for why it is never sent as a separate parameter): the agent is told to start from the source the question names, prefer cheap calls before pixels, fetch all needed images in one turn, prioritise key frames and *stop when the evidence settles the answer*, say only what is actually visible, treat a period’s `cat_id` as the identity to trust (a frame’s label is only a label), never cross-reference replay (`sim:true`) frames with live tracker data, treat a long-static box with high agreement as a possible decoy (ball/toy/cushion), treat two simultaneous boxes with the same `cat_id` as proof that one label is wrong, and never use a generic (COCO) detector for cat identity.

**Two ways the agent is invoked:**

1. **Free chat** (`python -m supervisor_agent "Is any cat on camera right now?"`, interactive REPL, `--list-tools`). The user asks questions; the agent picks tools. `check-live-detections` is the chat skill for the recurring “what is on camera and is the model right?” question (≈3 turns by design).
2. **Pipeline tasks:** code routes a task envelope to a handler that (a) does the cheap deterministic part itself and returns *without calling the model* when that already decides it, (b) otherwise loads *one* workflow as the instructions, exposes only that task’s tools, attaches one small crop, and asks for strict structured output, (c) validates the answer, retries once, then falls back to human review, and (d) records the outcome in the task log (and in the human-review queue when needed).

| Task | Question | Code-first shortcuts | Skill | Image / high-res budget |
| --- | --- | --- | --- | --- |
| `validate_detection` | Is this a cat, and which one? | Visit consensus (≥20 frames, ≥80% agreement, *never* for a long-static visit — the decoy guard) decides without the model | `validate-cat-identity` (6-rung cheapest-evidence ladder) | 3 / 1 |
| `check_trajectory` | Is this visit one plausible animal? | No physics flags + `accept_without_model_if_no_flags` decides without the model | `check-trajectory` (one contact sheet) | 2 / 0 |
| `assign_label` | What activity is this? | Label rules (motion state, zones, other-cat-nearby) filter candidates; exactly one surviving non-`unknown` candidate with no image need decides without the model | `assign-activity-label` | 2 / 0 |
| `label_visit` | Label a whole visit | One label per motion segment (≤8); a 40-min nap costs one image | (via `assign_label`) | per segment |
| `manage_labels` (chat) | “Label the cat when it sits in the tent” | — | `manage-labels` (draft → confirm → save; never saves without an explicit yes) | 2 / 0 |

The **identity ladder** (`validate-cat-identity`) is the clearest expression of the cost discipline: (1) metadata (exclusion — a cat confidently seen elsewhere at the same moment cannot be this one; visit consensus — unless static-suspect), (2) the attached crop vs. written profiles (name each cue actually seen; strongest quick cues: Qbo’s dark plume tail tip + fuzzy outline vs. Qb’s smooth bright-white shape), (3) a sharper view only if the deciding cue is high-res or the box is <~60 px, (4) owner photos via a single contact sheet, (5) confirmed camera crops (same angle usually beats a studio photo), (6) `send_to_human` — never force a call. Budgets enforce this: an identity check gets 3 images and 1 high-res capture; the agent must decide within that or ask a person.

*Scope note for judges:* today the **live escalation path** (the validation worker, §9.5) uses a narrower tool set — `images__get_image`, `images__list_images`, `tracking__get_period_detail` — and reviews stored frames. The full ladder exists as the `validate_detection` task, but the tracking pipeline does not call it yet. Both paths share the same agent, model and tool hub.

### 9.5 Validation loop — queue, batching, apply (the closed loop)

The validation subsystem comprises the background worker (about 430 lines), the batching queue, the verdict-application step and the period labeller.

Escalations leave the real-time path. Each escalation is enqueued as it is emitted; a single background validation worker consumes them, so **tracking never waits for the LLM**.

- **Batching queue:** a new case joins the *waiting* tail batch if it falls within **60 s** of that batch’s span, up to **4 cases** and **12,000 prompt characters**; otherwise it starts a new batch. Merging is opportunistic with zero added latency — the worker takes whatever is waiting the moment it is free (in replay the producer dumps all escalations in ~1 s while one validation takes 30–80 s, so merging captures nearly everything; in live mode sparse arrivals usually validate solo, correctly). A batch already taken is in-flight and can never gain members, which makes the design race-free with one `threading.Condition`. The design example turns the demo window’s 12 calls into 3 batches. *Status:* the batcher and merged-prompt path are built and tested; the headline session in §14 ran *without* merging (12 sequential calls) so the saving is designed and unit-tested, not yet measured end-to-end.
- **Per batch:** optional memory brief (§9.6) appended as *“Past context from memory (may help, may be stale)”*; `build_validation_query()` / `build_merged_query()` = full escalation prompt(s) + brief(s) + “return ONLY a JSON object”; the review JSON schema is passed as the LLM’s structured-output format; `MCPAgent.run()` with tools restricted to the three validation tools, tagged `validation`. The merged prompt shares frames across related cases (fetch each image once) and still returns one verdict per case, keyed by name. Partial failure is isolated: a missing/unknown/schema-failing item becomes an `error` op for that escalation only; siblings still journal their verdicts.
- **Verdict application** (after the queue drains at the end of a run): frame ops match rows by exact timestamp (+ cat), period ops by `period_id`; untouched rows pass through. Executable verdicts rewrite rows (relabel, drop, cut/trim spans, reassign winners); each touched row records `validated_by` (escalation name) and `verdict`, so every correction is auditable. Verdicts that cannot be executed precisely (splits, merges into an unnamed neighbour, free-text-span trims) keep the rows and record a note instead of guessing. Outputs are a validated row set plus a per-run validated database. The shared detection database is never written; the validated table definition is provided for deployments that want a database copy.
- **Failure behaviour:** if the supervisor backends are unreachable, the worker logs once and every queued escalation validates to `error` (tracked, not lost). A Meta API 400 currently drops the case with no retry and no trace — a known issue (§14.6, §20).

Verdict → row effect (examples): `keep_original` (T2) restores the detector’s label; `confirm_split` (T6) / `keep` (P5) stamp rows as confirmed; `reject_box` / `drop` removes the box; `merge_visits` (T6) rejoins two visits. Some verdicts are only logged today (e.g. `merge_neighbor` when the neighbour is unknown).

### 9.6 Memory — spatial-temporal, two-tier, trust-gated (the past that briefs the future)

Agent frameworks treat memory as a core tool, but it usually holds conversations. Video needs memory organised by *where* and *when*. The memory subsystem builds that store in plain language. Memory never blocks tracking: one `MemoryWorker` thread handles every write and LLM call (FIFO queue, promote coalescing, `bootstrap` jumps ahead of stores); reads return immediately under an `RLock`. A failing job is logged and dropped/degraded; tracking carries on. An idle timer triggers promotion after 5 minutes without new records. Memory is enabled as a pipeline option.

**From detections to knowledge.** Memory does not store every frame. Closed visits by the same cat, less than 5 minutes apart (`activity_combine_gap_minutes`), combine into one *activity record*. A vision LLM (the period labeller, on its own background thread with an image-service connection) labels each member period’s key frames (about one per minute, with the tracked box disambiguating multi-cat views) against the activity set listed in Section 12.4 (sleeping, resting, sitting, standing, walking, eating, drinking, grooming, playing, playing with another cat, unknown), returning activity + confidence + appearance + background. Any labelling failure degrades that member to `unknown` — never raises, never blocks tracking.

**Two memories:**

- **Short-term** (the daily memory): everything from the day, including weak records, escalations and verdicts (with rationales), as combined activity records with detections and key frames, in a plain-language daily log that is re-rendered on every change. New sessions open a new session section; midnight (event time, so replays behave like live days) merges the day into long-term memory; session end merges the session the same way, marked so nothing merges twice.
- **Long-term** (the durable memory): only trusted knowledge. A record enters only if the detector *and every* activity label reach **0.7** (the minimum store confidence) — anything weaker stays in short-term day context. Escalations and raw verdicts never enter long-term directly; they reach durable memory through the labelled period activity they confirm or correct. In the headline session a visit at 0.54 was refused and one at 0.71 was kept (§14.2). Every record carries its original image names, the detection snapshot (box, cat, activity, periods) and plain-language text, so any claim can be traced to evidence.

**The spatial pyramid.** The 800×600 view splits into **48 cells of 100 px** (L0, 8×6), rolling up into **12 zones** (L1, 4×3, each 2×2 cells), then **one full-view summary** (L2). Each unit holds plain text on environment, background, cat contrast, activity, appearance and behaviour, plus visit counts per cat (`stats: {visits, by_cat}` — programmatic counters, not prose, are the source of truth for dominance).

**The temporal pyramid.** Each level answers a different question: **5-minute slots** hold events, **hours** hold the activity mix, **days** hold routine, **weeks** hold habits and change. Retention: slots 7 days, hours 30, days 180, weeks indefinitely.

**Summaries that resist noise.** One odd event must not rewrite what the system knows:

- A pattern needs **≥3 observations** (`min_pattern_support`) to enter a unit’s stable text.
- One-offs stay in a bounded **recent buffer of 5** and age out.
- Stable text changes only after **3 consecutive contradicting batches** (`flip_threshold`); the LLM judges whether new evidence contradicts, but code does the counting — so the rules are unit-testable with the stub client.
- Word caps (L0 ~60, L1 ~100, L2 ~150; slot ~40, hour ~80, day ~120, week ~150) force each summary to keep the strongest patterns.
- Bootstrap: one vision call on the first full-view frame fills L2 + per-L1-zone environment/background/contrast; L0 cells inherit their L1 text marked `unobserved` until real observations refine them. Without a bootstrap image the store runs in text-only mode with vision fields marked unknown (the headline session ran this way — zone `environment`/`background` were still `Unknown`).

**How memory reaches a decision.** Each escalation retrieves the memory that fits its doubt, not a generic search (`invoke_for_escalation`, brief capped at 600 chars):

| Scenarios | Question for memory | What it retrieves |
| --- | --- | --- |
| T1, T2, T5 | Is this cat hard to see here? | Hit cell + neighbours: environment, background, contrast |
| T4, T6 | Was this cat recently nearby? | Wider ring of cells + current slot/hour |
| T3, T7 | Is the track continuous? | Current slot + hour |
| P1, P2, P5 | What is this cat’s routine? | Day/hour routine + same-slot history |
| P3 | Which cat is usually here now? | Same-slot history over 7 days |
| P4 | Is this an occluded spot? | Environment and contrast notes |
| P6, P7 | Same place, or the usual nap? | Gap-span + that place’s routine |

The brief is appended as *“may help, may be stale”* — memory informs the agent and does not overrule the image. Semantic search in v1 is plain-text token overlap over each unit’s own text (plus a phrase bonus); embeddings wait for a confirmed Meta embeddings endpoint (none is available in the current model configuration).

**Toward decisions without images (designed, not built).** The goal is a memory strong enough to replace *some* image reviews — e.g. memory says Qb has held one position for an hour, the next frame shows the same box moving the same way but the detector says Qbo: a label flicker that may not need an image. At ≈$0.054/review, each avoided call is a direct saving. A smaller version already exists in code: for `validate_detection`, a final visit with ≥20 frames and ≥80% detector agreement can settle a case with no model call, *unless* the visit is long and static (the decoy guard). Three risks guard any rollout: **decoys** (a box still for an hour with steady agreement is exactly what a toy/cushion looks like), **circular evidence** (the tracker wrote Qbo over 772 Qb detections in one session — a memory built on such labels would repeat the mistake with more confidence), and **missing fields** (memory stores cells and visit counts, not speed; live same-day records do not reach briefs yet). The safe path is shadow mode first — memory-only decisions compared against image-based verdicts — then skipping images only for scenario types where memory proves reliable, with periodic image checks to catch drift.

---

## 10. The five barriers and how each module answers them

The system is framed here as five barriers. This section is the claim-to-module map a judge can audit.

### Barrier 1: a camera that cannot adapt

*A Pi cannot run a large model at full resolution all day, so the stream runs at 800×600 with a small custom model. A fixed pipeline has no second option when that fails.*

**Answer — the steerable camera service (Section 9.1).** The always-on YOLO loop stays untouched; an MCP server beside it lets the agent capture a buffered frame, crop or high-res still, re-run any registered model on any region, inspect track history, and temporarily change fps / roi / imgsz / model — every change expiring automatically. The sleeping-cat audit is the motivating failure: both the custom model and the generic COCO model returned no detections on a frame where a cat is plainly visible in the green bed. The camera tools exist so the agent has a *second look* without a human touching the Pi. Automatic policy (the pipeline changing the camera by itself) is not built yet; the tools are.

### Barrier 2: detections nobody double-checks

**Answer — the Muse Spark supervisor agent (§9.4).** It reviews uncertain results against evidence, aiming for the *cheapest evidence that settles the case* (the identity ladder). Image budgets enforce the discipline. It runs on Muse Spark, sees the original frames, and can call on the tracker’s record of each visit. Its system prompt encodes the hard-won rules from the trace reviews: decoy suspicion, same-ID-twice-implies-error, COCO models never decide identity, replay-vs-live separation.

### Barrier 3: numbers that an agent cannot reason about

**Answer — the escalation bridge (Section 9.3).** There is one prompt builder per scenario; each states the conflict, the question, the context, the alternatives and which frames to inspect. The bridge runs both directions: each scenario also has its own output schema with a fixed verdict set, checked by a strict per-scenario output schema. Language goes out, structured data comes back to update the tracked rows, and every changed row records which escalation changed it.

### Barrier 4: deciding when to ask, at what cost

**Answer — the 14 triggers plus the background batched queue (Sections 9.3 and 9.5).** The tracker scores each visit on evidence such as average certainty, frame count and spatial spread; rules watch for decisions it could not make cleanly. Escalations leave the real-time path into a batching queue (60 s window, ≤4 cases, ≤12 k chars) so tracking never waits for the LLM, and one agent call can share frames across related cases while still returning one verdict per case. A low-confidence box alone never triggers review — the triggers fire on *conflicts in the tracker’s decisions*.

### Barrier 5: vision without memory

**Answer — spatial-temporal memory (Section 9.6).** YOLO sees one frame at a time; a tracker joins frames into a path, but that path is only numbers. Neither knows that Qb usually naps here, or what is unusual today. The memory turns frames into a per-cat timeline, gives the supervisor context fitted to its doubt, and supports month- and year-long comparisons (“how does today compare with his normal pattern?”). It also limits how much home footage reaches the LLM provider, because only escalated cases are ever sent.

---

## 11. Meta model usage

All LLM use goes through Meta models, configured per role (provider: Meta Model API; the agent calls its OpenAI-compatible Responses endpoint). The agent calls the API through its OpenAI-compatible Responses endpoint; the `openai` Python package is only the HTTP client.

| Role | Model | Use | Wired in code today? |
| --- | --- | --- | --- |
| `agent` | Muse Spark `muse-spark-1.3` | Agent reasoning, vision, tool calls; escalation teacher | **Yes** — supervisor agent + validation worker |
| `memory` | Muse Spark 1.3 | Memory store merges, promotion summaries, bootstrap vision, period activity labelling | **Yes** (falls back to the agent model when no memory model is configured) |
| `segmentation` | SAM 3.1 `sam-3.1` | Masks/boxes for training data (open-vocabulary finding is out of scope) | Not yet — config only |
| `voice` | Muse Voice Transcribe `muse-voice-transcribe-1.0` | Voice questions (if voice is built) | Not yet — config only |
| `creative` | Muse Image `muse-image-1.0` + Muse Wonder `muse-wonder-1.0` | Creative output, e.g. a daily “cat diary” comic (Best Creative Build) | Not yet — config only |
| `open_weight` | Muse Spark Open `meta-models/muse-spark-open-1026` (Baseten) / Muse Glimmer | Local or partner-hosted model (privacy; Best Open Weight) | Not yet — config only (`base_url https://inference.baseten.co/v1`) |
| `coding` | Muse Code | Coding (a product, not an API model) | n/a |

Standard pricing used for every cost calculation in this report (retrieved 26 October 2026): Muse Spark input $1.25, cached input $0.15 and output $4.25 per million tokens; the Contributor tier is input $0.10, cached $0.002 and output $0.20 per million; SAM 3.1 is $2.50 per 1,000 images; Muse Image is $0.01 per image; Muse Voice transcription is $0.18 per hour of audio. Every agent and memory LLM call logs its token counts and estimated cost from these figures, so cost claims in §16 are computed, not asserted.

**Why Muse Spark fits this system specifically:** (a) *vision + tool-calling in one model* — the supervisor must look at a crop and, in the same turn, decide to fetch a sharper view or a reference photo; (b) *1 M-token context* — a batched review of 4 escalations with shared frames fits without truncation; (c) *MCP-Atlas-leading tool use* — the workload is dozens of small, schema-typed tool calls per session, exactly what that benchmark measures; (d) *structured outputs* — the Responses `json_schema` format carries each scenario’s review schema, so verdicts arrive parseable. The port from OpenAI (`gpt-6.1-sol` in the pre-hackathon `mcp_agent`) to Muse Spark was the Stage-1-critical hackathon task; §14.5 documents the one API behaviour that required a code change (chained image turns).

---

## 12. Implementation inventory

Measured in October–November 2026 unless noted.

### 12.1 System scale

| Component | Scale |
| --- | --- |
| Complete system | 159 Python modules, about 19,300 lines |
| Core hackathon system (agent, tracking, escalation, validation, memory and tests) | About 19,100 lines |
| Main pipeline orchestrator (ingest, correctors, escalation, validation, memory, charts) | About 1,200 lines |
| Memory (durable store, daily log and LLM client) | About 1,070 + 530 + 580 lines |
| Tracking (frame corrector and visit corrector) | About 700 + 520 lines |
| Escalation (prompt builders and output schemas) | About 700 + 200 lines |
| Supervisor agent (tool-calling loop and task runner) | About 330 + 440 lines |
| Validation (worker, labeller, verdict application and batching queue) | About 430 + 300 + 290 + 130 lines |
| Automated tests | 28 test suites, about 7,000 lines, 297 test cases |

The evaluation evidence in this report comprises two complete session accountings, two independent trace reviews, a full false-negative audit with complete tool-call records, and prompt-plus-schema examples for all 14 escalation scenarios. The key numbers from each are reproduced in Sections 14–16.

### 12.2 Test coverage (fully offline)

All test suites run offline with fake or stub language-model clients; no API key and no database are required. Coverage includes: corrector mechanics; escalation triggering and configuration precedence; prompt construction and output-schema validation for every scenario code; supervisor task routing; agent conversation chaining; memory grid and time-bucket mathematics; durable-store round-trips and retention; worker ordering, promotion coalescing and clean shutdown; the trust gate; compression and weighting rules (dominant-pattern and contradiction-threshold behaviour); bootstrap marking; daily-log rendering and merging; validation batching (gap joining and splitting, size and character caps, in-flight isolation); merged-review parsing including per-case failure isolation; and an end-to-end replay test that exercises storing, promotion and retrieval over a real tracking session.

### 12.3 Outputs of one pipeline run

Each run produces: the canonical detection stream (plus the raw camera messages in live mode); the revised row set with retraction markers; the visit records; the image index; the escalation journal; and, when validation is enabled, the validation journal, the validated row set and a per-run validated database. It also produces the original, revised and validated timeline charts, optional per-frame process visualisations with a stitched video, and a run summary (arguments, window and counts, including escalations by scenario code and validations by verdict). Each agent validation produces an execution trace with embedded evidence images, and each memory language-model call produces its own audit trace. Run outputs are temporary; the durable memory store (the long-term structured store plus the daily logs) is retained across runs.

### 12.4 Configuration and reference data

- **Cats:** two household cats are profiled in full. Qb (tracker identifier 25 on the living-room camera) is a short-haired white cat with faint silver shading, a round face and blue-green eyes; distinguishing cues are the cool white-to-silver coat with no warm tones, the smooth sharp outline, the thin tail with grey rings or a grey tip, and white legs and paws. Qbo (tracker identifier 26) is a long-haired cream and golden-beige cat with darker brown paws, face and plume tail; distinguishing cues are the warm cream colour, the fuzzy outline with a fluffy halo, the large bushy grey-brown plume tail, and the darker brown lower legs and paws. Known confusions are documented for the agent: white blankets, towels and bags for Qb; cream plush toys, rugs and blankets for Qbo. Each cat has owner reference photographs (full frames and crops, in both posed and in-scene views) that the agent can retrieve for comparison.
- **Activity labels (version 1):** exactly one activity per observation — sleeping, resting, sitting, standing, walking, eating, drinking, grooming, playing, playing with another cat, or unknown — plus any number of context labels (for example, on the sofa, decided by a zone-overlap rule; inside the tent, decided by zone overlap plus a visual check). Each label declares what evidence it needs (an image, the motion state, a named zone, or another cat nearby) so deterministic rules can exclude impossible labels before any model call.
- **View zones and budgets:** the living-room view is 800×600 pixels with named zones for places such as the sofa and the tent. Task budgets, visit-consensus thresholds, trajectory-physics thresholds and label-rule switches are centralised in one task configuration (summarised in Appendix B). Supported camera resolutions are centralised in one camera configuration.
- **Secrets and credentials:** a single environment configuration supplies the Meta Model API key, one shared bearer token for all three MCP services, and the read-only archive credentials. No secrets, internal addresses or proprietary weights belong in the public submission.


---

## 13. What was built before vs. during the hackathon

The rules require that what is presented was built during the event (27 October – 4 November 2026), with pre-existing work clearly identified. The Indie track applies this strictly. The split below is what the demo video must state out loud; the public submission contains no secrets, internal addresses or proprietary weights.

### Before 27 October 2026 (existing infrastructure — not presented as new)

| Piece | Dating |
| --- | --- |
| YOLO detection pipeline and the two correctors in their original form | 2025; the corrector behaviour was unchanged by the later restructuring |
| WebSocket ingest server | Pre-hackathon |
| Tracking query service and tracker visualisation | 23–25 October 2026 |
| Camera service with MCP control and the image archive service | 24–25 October 2026 |
| Supervisor agent running on a non-Meta model, with pipeline tasks, workflows and the cat knowledge base | 23–25 October 2026 |
| Legacy labelling and training scripts | Pre-hackathon |

### During 27 October – 4 November 2026 (the hackathon build — what the demo highlights)

| Piece | Dating |
| --- | --- |
| **Port of the agent to Meta models** (Muse Spark 1.3 via the Meta Model API, including the chained-image-turn handling described in Section 14.5) | Hackathon period |
| **Unified tracking pipeline** (single entry point, database-free corrector modules, run store, charts and process visualisation) | 30 October 2026 |
| **Escalation points in the correctors** — the 14-scenario trigger vocabulary, thresholds, per-scenario prompts and slim validated output schemas (examples in Section 15) | 30 October – 1 November 2026 |
| **Live escalation validation loop** — background queue, supervisor review, verdict operations, validated row set and validated timeline chart | 1 November 2026 |
| **Escalation queue merging** — temporal batching (60-second window, at most 4 cases, at most 12,000 prompt characters) with a joint prompt and per-case failure isolation (design in Section 9.5) | 2 November 2026 |
| **Spatial-temporal memory** — 48 cells rolling up to 12 zones and one full view, plus 5-minute, hourly, daily and weekly time levels; durable structured store, background worker, trust gate, daily log with day-end merge, and period activity labelling | 31 October – 2 November 2026 |
| **Camera simulator** — replay of archived frames through the camera pipeline for hardware-free live testing and demonstration | 30 October 2026 |
| **Trace-review hardening** — slimmer tool contracts, workflow surface gating, chat budgets, the static-decoy guard, replay/live separation and prompt rework (findings in Section 14.4) | 1 November 2026 |
| **End-to-end session accountings** (the measured sessions and costs in Sections 14 and 16) | 1–2 November 2026 |

**Scale of the hackathon build:** 49 work steps between 27 October and 3 November (of 80 in total), touching 195 components with about 27,500 added lines. The demo therefore highlights the escalation loop, the memory, the validation and merging machinery and the Meta-model port — not the pre-existing correctors or MCP services, which are presented as infrastructure.

*Open compliance questions for the organisers (to raise through the event Q&A channel): whether building on clearly marked pre-existing work is acceptable in the Indie track, whether a registered company with no product users belongs in Indie, and whether a recorded-data replay is acceptable when the live system depends on a home camera.*


---

## 14. Experimental results

All results below are from real measured runs. No hand-labelled ground truth exists yet, so accuracy is reported as *agent-vs-tracker agreement on sampled decisions* and *qualitative frame inspection*, never as mAP/precision/recall. That limitation is stated wherever the numbers appear.

### 14.1 Headline session — full-loop replay, 28 Oct 16:33–16:40 (view 3)

The session replayed the 28 October 16:33–16:40 window from the detection archive with validation and memory enabled and timeline charts generated (a rerun of the same window with trusted-only long-term memory, the daily short-term memory, in-pipeline validated charts and full memory-call logging).

**Pipeline totals:**

| Stage | In | Out |
| --- | --- | --- |
| Ingest (MySQL, read-only) | 1,319 frames | 1,016 detections kept (818 Qb, 198 Qbo; 0 filtered) |
| TrackCorrector | 1,016 detections | 1,265 live rows + 10 tombstones; 8 frame escalations |
| PeriodCorrector | 1,265 rows | 4 periods; 4 period escalations |
| Escalation | 12 raised | 12 logged (record + prompt + schema), 12 queued |
| Supervisor validation | 12 prompts | 11 verdicts + 1 error (E9, Meta API 400) |
| Verdict application | 1,275 revised rows + 12 ops | 1,275 validated rows (167 stamped, 1,108 pass-through) |
| Long-term memory | 4 periods + 11 verdicts + day merge | Trusted-only long-term store rebuilt (484 lines); briefs on all 12 prompts |
| Short-term memory | 4 periods + 12 episodes | Day file with verdicts, merged to long-term at session end |

Tracking is bit-deterministic: the original timeline chart and the revised timeline chart have identical MD5s to the prior run. The validated timeline chart differs because the verdict set changed (see nondeterminism note below). Wall time ≈11.1 min total: tracking replay ≈1 s, validation of 12 escalations ≈9.4 min, short-term merge + memory promotion ≈1.7 min.

**The four periods:**

| Period | Span | Winner | Detections | Mean conf | Score |
| --- | --- | --- | --- | --- | --- |
| 1 | 16:33:06–16:34:01 | Qbo (26) | 44 | 0.54 | 11.2 |
| 2 | 16:34:05–16:40:44 | Qb (25) | 772 | 0.71 | 9.0 |
| 3 | 16:34:41–16:34:42 | none | 2 | — | 0.0 |
| 4 | 16:35:21 (1 frame) | Qbo (26) | 1 | 0.82 | 2.4 |

**The headline finding (reproduced exactly across both runs):** the tracker relabelled **772 of 818 Qb detections as Qbo**, only 3 escalated as T2 (**0.4% recall** on that error class), and the period voter awarded period 2 to Qb while the rows underneath say Qbo. The revised/validated charts are one unbroken Qbo band against Qb-heavy raw detections. This single fact drives the roadmap’s first priority (§21): *better triggers matter more than a better agent.*

**What the 12 escalations were:** E1/E4/E10 T7 (interpolated gaps), E2 T6 (362 px teleport split), E3 P5 (weak visit, 44/153 frames real), E5–E7 T2 (nearness relabels), E8 T4 (anchor wipe, 107 rows), E9 P6 (visit merge candidate), E11–E12 P2 (thin visits, 2 and 1 original frames).

### 14.2 What the supervisor agent did (12 sequential validation runs, `muse-spark-1.3`)

| Esc | Turns | Tools (ok/err) | Images | Verdict |
| --- | --- | --- | --- | --- |
| E1-T7 | 6 | 11/3 | 9 | `keep_synthesis` |
| E2-T6 | 3 | 8/0 | 6 | `confirm_split` (two cats visible at once — split upheld) |
| E3-P5 | 4 | 7/3 | 6 | `keep` (163 rows stamped) |
| E4-T7 | 5 | 12/1 | 10 | `keep_synthesis` |
| E5-T2 | 4 | 3/2 | 2 | `keep_original` (**row flipped 26→25**) |
| E6-T2 | 3 | 2/2 | 2 | `keep_original` (flip applied, then undone by E7 — same-frame cross-talk) |
| E7-T2 | 3 | 2/2 | 2 | `keep_original` (**row flipped 26→25**) |
| E8-T4 | 4 | 3/2 | 2 | `drop_box_as_outlier` (recorded only; wipe had already happened) |
| E9-P6 | 3 | 4/0 | 2 | **error** (API 400 — no trace recorded, merge lost) |
| E10-T7 | 3 | 10/0 | 8 | `keep_synthesis` |
| E11-P2 | 3 | 6/0 | 5 | `merge_neighbor` (no-op: neighbour id unknown) |
| E12-P2 | 4 | 2/1 | 1 | `keep` (1 row stamped) |
| **Total** | **45** | **70/16** | **55** | **2 rows changed, 167 stamped confirmed** |

**Upheld vs. overturned, by code:**

| Code | Upheld | Overturned | Reading |
| --- | --- | --- | --- |
| T2 (nearness relabel) | 0/3 | **3/3** | Nearness-only relabelling was wrong every time sampled |
| T4 (jump reset) | 0/1 | 1/1 | New box judged an outlier this run (continuation last run) — nondeterminism |
| T6 (teleport split) | 1/1 | 0/1 | Split right, again |
| T7 (interpolation) | 3/3 | 0/3 | Guessed runs fine, again |
| P5/P2 (period quality) | 2/3 | 1/3 | E11 now says visit 3 belongs to a neighbour (unapplied) |
| P6 (merge) | — | — | No verdict: API error dropped the escalation |

Short-term memory preserves the agent’s own one-line rationales, which is how these verdicts can be audited without opening traces, e.g. E5-T2: *“Disputed box is standing center cat: cool white, smooth sharp outline, thin upright tail = Qb 25, not warm cream/fuzzy/plume-tail Qbo 26. Source box is bed cat visible simultaneously — two cats co-exist, so 129 px nearness stole neighbour.”* The complete day log for 28 October contains the same one-line rationale for every episode in this session.

**What memory learned in this run** (first live run of the memory rework, store cleared beforehand): the trust gate held — period 1 (conf 0.54) skipped (`confidence 0.54 < 0.70`), period 3 skipped (no winner), E9’s error verdict skipped (0.00); period 2 (0.71), period 4 (0.82) and all 11 decisive verdicts stored with image names, detection snapshots and appearance lines. Long-term narratives included *“This zone has five visits in the afternoon around 16:34. qb appears here between 16:34 and 16:40 and qbo visits here around 16:35.”* and the day/week summary *“Qb and qbo visit on October 28 with 17 visits in total. Most visits happen in the afternoon.”* Final promotion: `calls=7 promoted=7 deferred=0 errors=0`. Memory LLM traffic: 29 calls (4 period + 11 verdict + 7 day-merge stores, 7 promotions), every one logging input and output. Caveats: zone `environment`/`background` still `Unknown` (no bootstrap image), and escalation names in recents collide across runs.

### 14.3 False-negative audit — the sleeping cat both models missed

Recorded audit, 31 October 2026 (with complete tool-call records retained for audit). Recorded audit, 31 October 2026, living-room camera in simulation/replay of 27 October 18:00–18:06. Query: *“fetch the latest camera view and identify any potential cat in the view. compare to the detection and report any inaccurect detection.”*

**Outcome: false negative confirmed.** A white-grey cat curled asleep in a small green ruffled bed at the bottom edge of the 800×600 frame (approx. box `[183,506,289,595]`) is plainly visible in the full frame and in the zoomed crop, yet:

- the always-on loop reported no detections, and a fresh detection pass with the custom model on the nearest buffered frame also reported none;
- a detection pass with the generic COCO model returned a bicycle, sports balls and a bed (the sofa) — no cat;
- track history showed the cat *had* been tracked (212 sightings, mostly as cat 26 with 5 flips to cat 25, maximum confidence 0.67) but was dropped after 18:01:18 despite no visible change — so the 26 identity was uncertain, and the agent’s visual read leaned Qb (25) without certainty.

The agent completed this in 5 turns, 11 tool calls and 2 images, in 128.8 seconds (99.4% language-model time), using 68,583 tokens at a cost of $0.0685. It also correctly discounted the live tracking status, whose clock (1 November, “ongoing visit, cat 26 top-centre”) described a different moment than the 27 October replay — the prompt rule added after the trace reviews working as intended. The two evidence frames are described in this section: a full 800×600 living-room frame and a zoomed crop of the green bed at the bottom edge, both showing the curled sleeping cat. **Implication:** silent misses never escalate (with no detection, nothing fires), so a periodic sampling check is needed in addition to conflict-driven escalation (§20).

### 14.4 Trace reviews — the 9-turn run that taught the system its rules

Two independent reviews of an interactive agent run on 1 November 2026 (“check the live camera view and find any cat… validate the detection accuracy”): the trace reviews in Section 14.4 and the trace reviews in Section 14.4. Headline metrics: **9 turns / 22 tool calls, 152.4 s wall (LLM 142.8 s, 94%), 158 k input tokens (72 k cached), $0.177.** The reviews’ value is the fixes they produced, now in code:

- **Correctness:** the run validated a false positive — an exercise ball tracked as “Qbo-2” for 26 minutes (box `[351,47,407,104]` on the pink/blue ball; net displacement 0.7 px; `yolov8s` called the same spot `sports ball`). Root cause: visit-consensus auto-accept (≥20 frames, ≥80% agreement) with no static guard. *Fix:* the task runner never auto-accepts a long-static visit (a visit lasting at least 600 seconds with at most 5 seconds of movement is flagged as a static suspect and the model must look). The same rule is in the `validate-cat-identity` skill and the system prompt.
- **Identity:** both visible cats labelled 26 — with two cats on screen, two simultaneous same-ID boxes prove at least one label wrong (the exclusion rule). Now a system-prompt rule and a `check-live-detections` step.
- **Tooling:** four image calls failed because the image service was missing its image-processing library (the failure was discovered only after the download). The image budget was charged for failed calls with no refund. The arrival-path query spent 6.4 seconds selecting key frames over a stationary 4,000-row path. *Fixes:* fail-fast/actionable errors, budget refund on error, path capping — tracked as P0 items in the reviews and reflected in the current tool descriptions.
- **Efficiency:** 38 tool schemas (~8 k tokens) resent every turn (≈45% of all input); verbose replies (60-point paths for a cat that moved 0.7 px, 50 buffer timestamps, S3 metadata) re-billed on every later turn; pipeline-only skills exposed in chat. *Fixes:* `surface` gating for skills, hidden write/rare tools in default chat, slimmed defaults (`final_limit`, `limit=0` span-only, `max_points` discipline) — see §8.3, §9.4.
- **Prompt hygiene:** a non-existent crop tool was advertised; a duration parameter was named inconsistently; and the generic COCO model was described as a “bigger model”. All three wordings were corrected in the system prompt and tool descriptions; the current prompt also caps answers at ~200 words and forbids pasting raw JSON.

The reviews estimate the same query done efficiently at **≈3 turns / ~4–6 calls / ~35–40 k input tokens (≈35–45 s)** vs. the observed 9/22/158 k — the gap is the cost of missing skills and slim contracts, not of the model.

### 14.5 Model-API chaining probe — why `instructions` is never sent

Probe summary: the API probe in Section 14.5. Interactive queries needing two tool rounds intermittently failed with a “referenced response not found or expired” error (HTTP 400) on the Meta Model API. Systematic probing (first with a single dummy tool, then with 34 dummy tools, varying one factor at a time) showed the failure needs an *image-bearing* chained turn plus an opening turn that carried the separate instructions parameter — necessary but not sufficient (6 of 10 runs with it failed, 0 of 7 without it failed, yet 4 runs with it also passed; a full replica of the failing shape also passed). The conclusion is intermittent gateway-side flakiness around chained image turns (most likely response-store replication lag or load dependence), with two different 400 messages as alternate surfacings of the same failure. **Fix applied:** the agent never sends the separate instructions parameter; the instruction text is folded into the first user turn, where it persists through the conversation chain. Live retests still failed 2/6 runs with the original error, so the fold is kept as harmless hardening, not claimed as the fix; the remaining client-side answer is resilience (retry once after a delay, fall back to an unchained flattened-history request, never exit the CLI on a failed query) — part of the §21 roadmap. Side findings (only automatic tool choice is accepted; images must travel inside the tool-output payload; a very small output-token limit returns reasoning-only responses) are reflected in the agent’s request construction.

### 14.6 Known issues from the headline session (ranked)

1. **Tracker relabelled 772/818 Qb→Qbo; only 3 escalated.** Still the core recall gap.
2. **Visit-detail lookups returned production visits for replay-local identifiers** (the four replay visits resolved to 2024 production visits; 11 wasted calls; every agent run noticed and judged from the raw images instead).
3. **Image server has no Pillow** in the running venv (16/16 zoom crops failed in the headline session; all 54 full-frame reads succeeded).
4. **Meta API failure drops an escalation with no retry and no trace** (E9; the visit merge was lost silently at the row level).
5. **Same-frame T2 cross-talk undoes verdicts** (two escalations share one frame; application order decides the final label; the shared verdict field keeps only the last writer).
6. **Recorded-but-unapplied verdicts:** the outlier-drop and neighbour-merge verdicts are journaled without changing rows (in the second case because the neighbour is unknown), so the journal says more than the rows do.
7. **Agent verdicts flip run-to-run on identical inputs** (E8, E11; both no-ops here, but this bounds what a single-run evaluation can claim).
8. **Stale memory recents across runs** (E-name collision; masked in the headline run by clearing the store).
9. Minor: period-3 winner wording; “1 original frames” plural wart.

---

## 15. Demo walkthrough

Judges score whether the product “holds up live”, but the camera is a Pi in the author’s home. The demo therefore runs on a recorded stretch of real time and is replayable by anyone as described in Section 22. It follows the 8-stage walkthrough below on the 28 October window (Qb and Qbo both on camera), the same window as the headline session. The walkthrough is pre-determined but comes from real data and real model behaviour, and the video says so.

| Stage | In | Out | On screen |
| --- | --- | --- | --- |
| 1. Ingest + YOLO | 1,319 frames | 1,016 detections (818 Qb, 198 Qbo) | Live detection view beside the JSON stream |
| 2. TrackCorrector | 1,016 detections | 1,265 tracked rows, 8 frame escalations | the process visualisation video (buffer animation), then the original timeline chart |
| 3. PeriodCorrector | 1,265 rows | 4 visits, 4 visit escalations | the original timeline chart beside the revised timeline chart (red escalation marks) |
| 4. Escalation → prompt | 12 escalations | 12 plain-language prompts + schemas | One record beside its generated prompt |
| 5. Supervisor agent | 12 prompts (+ memory briefs) | 11 verdicts | `execution trace` at the tool calls and the two decisive frames |
| 6. Apply verdicts | 1,275 rows + 11 verdicts | 2 rows changed, 167 confirmed | the revised timeline chart beside the validated timeline chart, one flipped row highlighted |
| 7. Was it better? | Charts + verdict table | Honest assessment (below) | Upheld/overturned table |
| 8. Memory as product | Visits + verdicts | Plain-text day log + long-term YAML | Day log, then one question answered from memory* |

*\*Step 8’s Q&A is close but not wired up: the agent has no tool that reads memory today, so the files are opened directly. `eating` is a defined label but needs a `food_bowl` zone and a camera view of the bowl. A small memory read tool would make this step fully live (§21).*

### Step 1 — frames become a data stream

The camera thread keeps only the newest frame, so the loop never falls behind. YOLO reads each frame on the Pi; each result goes over WebSocket as one JSON message. This session ran at ~2.7 fps (simulated camera replaying the same message shape):

```json
{"view_id": 3, "create_time": "2026-10-28-16-34-36-154433", "is_grey": 0,
 "detections": [{"cat_id": 25, "box": [296, 369, 371, 531], "confidence": 0.62}]}
```

(Message shape real; confidence illustrative.) Live runs, recordings and the simulated camera all use this format, so the demo can replay a real stretch of time on any machine.

### Step 2 — what YOLO sees, and where it fails

The Pi runs `custom cat-identity model`, whose classes *are* the cats. The agent can also call `yolov8s` (generic COCO) to ask whether something is a cat at all — never which cat. Recorded evidence of why the rest of the pipeline exists:

- **Gaps:** in visit 1, YOLO spotted a cat in 44 of 153 frames (the rest were interpolated).
- **Weak confidence:** that visit averaged 0.54, against 0.71 for visit 2 — and 0.54 is below the long-term memory trust gate, so it correctly stayed out of durable knowledge.
- **Misses:** the sleeping cat missed by both models (§14.3).
- **False tracks:** the exercise ball tracked as Qbo for 26 minutes (§14.4).

*No formal detector accuracy results exist yet (no mAP, precision, recall or Pi inference time) — those must come from the training run and are on the pre-recording checklist in Section 20.*

### Step 3 — the correctors build a connected record

TrackCorrector (15 s hindsight buffer: dedupe → relabel by history → pick a trusted future target → interpolate → commit; >300 px jump splits the visit) and PeriodCorrector (whole-visit vote over original labels; one cat cannot hold two overlapping visits) turn 1,016 flickery detections into 1,265 connected rows and 4 visits. When Qbo appeared to jump 362 px, the tracker split the visit — and the agent later confirmed two cats were visible at once, so the split was right.

### Step 4 — uncertainty becomes a question

Each of the 14 scenarios has fixed thresholds in the central escalation configuration and fires at the exact decision point. This session raised T2, T4, T6, T7, P2, P5, P6. Two real prompts (shortened; full set in Section 15):

```text
[E2-T6] Cat seemingly teleported; visit split.
Task: validate whether splitting visit 1 into visit 2 at 16:34:05 was
right: cat 26 jumped 362 px in one step, and the tracker deleted 10
guessed frames.
Validate: Is this one fast-moving cat (merge the visits back), or two
different cats or a false box (the split was right)?
Alternatives: merge visits 1 and 2 into one, or drop the jumped box.
```

```text
[E3-P5] Verdict rests on weak, patchy sightings.
Conflict: visit 1 (cat 26) has avg certainty 0.54; of 153 frames in
total, the camera actually spotted a cat in 44; the other 109 were
filled in by guessing.
Validate: real visit or noise padded with guesses?
Alternatives: drop, trim to the high-certainty core, or reassign.
```

Each prompt also lists the cats’ written descriptions and the key frames to inspect, and (with memory enabled) a ≤600-char memory brief marked “may help, may be stale”.

### Step 5 — the agent investigates

The agent receives the system prompt (§9.4), the escalation prompt, the memory brief and the output schema. For the teleport-split case it took 3 turns, 8 tool calls and 6 images, then confirmed the split because two cats were visible at once in the frames before and after the jump. The full execution trace holds the exact call order, every image and the token/cost accounting. *On screen: the execution trace opened at the tool calls and the two decisive frames.*

### Step 6 — the verdict returns to the pipeline

The answer is validated against the scenario’s schema, journaled to `validation journal`, and applied by `materialize()`. Language becomes data again: `keep_original` restores the detector’s label, `confirm_split`/`keep` stamp rows confirmed, `drop` removes a box, `merge_visits` rejoins visits. Every changed row records `validated_by` and `verdict` for audit. *On screen: the revised timeline chart beside the validated timeline chart.*

### Step 7 — was the final result better?

The pipeline clearly adds structure, but no measurement yet shows higher accuracy: there are no hand labels to score against, and the agent’s verdicts are not ground truth.

| Stage | What can be shown today | What cannot be claimed yet |
| --- | --- | --- |
| YOLO only | 1,016 detections with gaps and noise | mAP, recall on home frames |
| Correctors | 1,265 connected rows and 4 visits | Fewer identity errors |
| With the agent | 3/3 sampled relabels overturned; 2 rows fixed | Overall error reduction |
| With memory | Past context in all 12 prompts | That memory changed verdicts |

The best honest headline is narrower: **the agent found errors the tracker made, and explained each decision with evidence.** Because only 3 of ~770 relabels escalated, the final rows still carry most of those errors — fix the trigger gap before showing an overall accuracy number.

### Step 8 — memory as a product

Memory is a second output, and possibly the one a cat owner values most. It is readable without any tools. Short-term (day log):

```text
16:33–16:34: Qbo (cat 26) — 44 sightings, mean confidence 0.54.
16:34–16:40: Qb (cat 25) — 772 sightings, mean confidence 0.71. Activity: resting …
```

Long-term (room description learned from this one short session — format demonstration, not yet a real routine):

```text
This zone has five visits in the afternoon around 16:34. qb appears
here between 16:34 and 16:40 and qbo visits here around 16:35.

Qb and qbo visit on October 28 with 17 visits in total. Most visits
happen in the afternoon.
```

The promised richness — where each cat likes to play, when each eats, how today differs from the month — needs weeks of data and the memory read tool for the agent (§21).

### Suggested 3-minute video outline (from the project plan)

1. **0:00–0:20 Hook:** two near-identical cats, and a camera that mixes them up. Name the competitors (Furbo, Petcube, Nest) and the difference in one line.
2. **0:20–1:00 Talk to it:** ask, find, summarise — by chat (voice if Muse Voice is wired by recording day).
3. **1:00–2:00 The loop:** CV fails → escalation fixes it → (retrain teaser) → before/after numbers that exist (§14), with the trigger-recall caveat stated.
4. **2:00–2:40 Customise:** “label the cat when it sits in the tent” (`manage-labels`); note the framework generalises beyond cats.
5. **2:40–3:00 Close:** competitors vs. this system; Meta models used; what was built this week vs. before (§13).

*Pre-recording checklist: detector mAP/precision/recall + Pi inference time from the training run; hand-labelled clip compared with ByteTrack/BoT-SORT; memory read tool so the agent can answer daily questions; rerun with merging on and report real cost; tightened T2 trigger; charts, videos and traces saved from the run outputs.*

---

## 16. Cost and performance, measured

### 16.1 Validation cost (headline session, sequential, no merging)

Measured session detail: Pricing: Muse Spark Standard, input $1.25, cached input $0.15 and output $4.25 per million tokens. The failed case’s partial spend is unrecorded (no trace was produced) — roughly a $0.03–0.05 undercount by the yardstick of the cheapest completed cases.

| Esc | Wall (validating → verdict) | Turns | Tokens | Cost |
| --- | --- | --- | --- | --- |
| E1-T7 | 77 s | 6 | 98,565 | $0.105 |
| E2-T6 | 54 s | 3 | 54,533 | $0.055 |
| E3-P5 | 61 s | 4 | 55,659 | $0.071 |
| E4-T7 | 57 s | 5 | 78,767 | $0.075 |
| E5-T2 | 46 s | 4 | 30,485 | $0.044 |
| E6-T2 | 34 s | 3 | 16,712 | $0.024 |
| E7-T2 | 31 s | 3 | 16,309 | $0.023 |
| E8-T4 | 57 s | 4 | 47,304 | $0.061 |
| E9-P6 | 33 s (failed) | 3 | n/a | n/a |
| E10-T7 | 36 s | 3 | 44,049 | $0.048 |
| E11-P2 | 32 s | 3 | 33,256 | $0.044 |
| E12-P2 | 45 s | 4 | 41,968 | $0.047 |
| **Total** | **≈9.4 min** | **45** | **517,607** | **$0.60** |

Averages: **≈47 s, ≈47 k tokens, ≈$0.054 per completed escalation.** Tracking itself cost ≈1 s and $0; memory promotion ≈1.7 min (29 LLM calls, cost additional to the $0.60 — not totalled in the session accounting and therefore not totalled here).

**Reading the numbers honestly:**

- One review costs as much as **54 frames** at the $0.001 working rate. Escalation only pays when the escalated fraction is well under ~2% *and* the window is not unusually busy. The demo window escalated 0.9% of frames but was 7 minutes of peak two-cat activity; a full day (mostly empty room, sleeping cats, far fewer tracker conflicts) should escalate a far smaller fraction. That full-day measurement does not exist yet.
- At $0.001/frame, reviewing all 1,319 frames would have cost ≈$1.32, so this window saved only ≈half — *before* counting memory’s 29 calls. The report states this wherever the $0.60 figure appears.
- **Merging is the designed lever:** the batching queue (Section 9.5) should collapse this window’s 12 sequential calls into 3 joint calls sharing frames and one system prompt per batch. The expected direction is fewer turns and meaningfully fewer tokens per escalation than the current ~47 k, plus better same-frame coherence (the agent sees both sides of the E6/E7 swap at once). The saving is unit-tested, not yet measured end-to-end — the next run must rerun this window with merging on and report the real delta.
- **Latency:** validation took longer (9.4 min) than the footage it covered (7 min). This is acceptable *by design* — validation is a background thread off the tracking path, and ops apply at `finish()` — but it rules out any “real-time correction” claim. The product claim is *eventually-corrected record + memory*, not *live relabelling*.
- **Prompt-cache behaviour is erratic** (trace reviews, §14.4): cached tokens per turn swung from 113 to 24 k on near-identical prefixes. At $1.25 vs. $0.15 per 1 M, cache misses cost ~8× the cached rate; stabilising the cache (byte-stable tool ordering, image-turn interaction) is worth up to ~20–25% of run cost and is an open investigation.

### 16.2 Other measured runs

| Run | Shape | Wall | Tokens | Cost | Lesson |
| --- | --- | --- | --- | --- | --- |
| False-negative audit (§14.3) | 5 turns, 11 calls, 2 images | 128.8 s (99.4% LLM) | 68,583 | $0.0685 | Agent correctly overrules both detectors by looking; LLM time dominates everything |
| Trace-review run (§14.4) | 9 turns, 22 calls, 10 image attempts (4 failed) | 152.4 s (LLM 142.8 s) | 158 k in / 13.9 k out | $0.177 | Turns, not tools, cost the time; ~8 k tokens/turn is tool schemas alone |

### 16.3 Projected full-day economics (scenarios, to be replaced by measurement)

Assume a full day at 2 fps with the tracker escalating primarily on conflicts. Cats are active ~3% of the day; most frames are empty or a motionless sleeping cat that the tracker handles without conflict. If the *effective* escalation rate lands at 0.1–1% of frames with merging on, the LLM bill is **$0.17–$1.73/day ($63–$631/year)** plus memory labelling/promotion — versus $172.80/day for all-LLM review. The system’s bet is that the rate *falls over time* as verified labels retrain the edge model and as memory learns to settle flicker-class cases without images. Neither fall has been measured yet; the experiment that would measure it is specified in §21 (eval set + 2–3 retrain loops with before/after identity error, gap/jump rate and escalation rate).

---

## 17. Comparison with alternatives

### 17.1 “Just send frames to a VLM”

Covered in §5–§6: correct on hard frames, unaffordable and privacy-expensive on all frames, and still without long-term memory (context windows cannot hold hundreds of hours of video). This system keeps the VLM’s judgement and removes its need to watch.

### 17.2 Commercial pet/home cameras

| Product family | What it does well | What it does not do (the wedge) |
| --- | --- | --- |
| Furbo 360° / Furbo Nanny | Treat tossing, barking alerts, pet ID for up to 3 pets, multi-camera plans | No per-scene learning loop; no tracker-doubt vocabulary; no plain-language spatial-temporal memory the owner (or another agent) can query; identity is a cloud feature, not a verified, auditable correction |
| Petcube (Bites / Cam 360) | AI pet recognition, vet chat, treat dispensing | Same: detection-centric, subscription AI features, no verified-labels → training-data path, no memory that briefs future decisions |
| Google Nest Cam / Ring / Wyze / Eufy | Person/pet detection, activity zones, cloud recording at $30–100 hardware | Generic classes (“pet detected”); no named-individual identity over months; no escalation to a reasoning agent that can steer the camera; footage leaves the home by default |

*Positioning (from the project plan, to be stated in the video):* large vendors are adding LLM features — answering questions and summarising are no longer unique. What is new here is that the large model **acts proactively** (asks the user to confirm hard cases, asks which scenes/objects/pets matter and prioritises them, surfaces insights unasked) and **teaches the small model**, so the system improves per home. A companion differentiator: what the system knows about the home can feed a personal assistant (e.g. Muse) as a data source. Health/diagnosis claims are deliberately avoided (§3, §19).

### 17.3 Open-source trackers

|  | SORT family (ByteTrack, OC-SORT, BoT-SORT) | RCAT correctors |
| --- | --- | --- |
| Main question | Is this the same object as before? | Which *known* cat is this? |
| Identity | Anonymous track IDs | Closed set of named cats |
| Timing | Frame-by-frame, no waiting | ~15 s delay, with hindsight |
| Gap filling | Predicts forward during occlusion | Interpolates between real sightings |
| Whole-visit decision | Generally none | Votes across the visit (PeriodCorrector) |
| Two-cat logic | Generic association | One cat cannot be in two overlapping visits |
| Motion model | Kalman filter or similar (most) | None; fixed distance/IoU rules |
| Appearance | Optional ReID (some trackers) | YOLO labels only |
| Reports its own doubt | Not as a standard output | **14 named scenarios** |
| Tuning | Benchmarked defaults | Hand-set for one 800×600 view |
| Public benchmark | MOT17 and others (BoT-SORT 63.7 HOTA, defaults) | None |

The correctors are not proven better, and this report does not claim they are. They solve a different problem (known identities, in hindsight, with uncertainty reported), and they compose with — rather than replace — the Ultralytics tracker the Pi already runs. A fair test is cheap and is on the roadmap: hand-label one clip with both cats, run ByteTrack and BoT-SORT on the same detections, and compare identity accuracy, IDF1 and visit errors.

### 17.4 Prior systems (NoScope, Ekya, Cerberus, Avas)

NoScope proved specialised models + difference detectors accelerate fixed-video queries by orders of magnitude for *one fixed question*. Ekya proved edge models drift (22%) and need retraining infrastructure. Cerberus proved a motion/CLIP → VLM cascade is viable at scale on a GPU server. Avas proved agents need long-video memory. This project’s claim is the *join* on low-cost hardware with an agent that can act (steer the camera, ask a human, write memory) — and the honest caveat is that the join is demonstrated on one room, two cats and 7 minutes of analysed footage.

---

## 18. Broader impact

The lasting idea is bigger than cats: **a cheap camera that watches continuously, asks for help when unsure, and remembers what it learns.** Wherever people need long observation of something rare or slow, the same gap appears. Two cats are the first proof case; every domain below still needs its own testing.

### 18.1 What this method makes possible

| Before | With this method | Status |
| --- | --- | --- |
| A model either runs cheaply or understands, not both | Fast model watches; LLM reasons only on hard moments | Shown in one session |
| A detector fails silently | System names its doubt and asks a specific question | Shown, but triggers miss many errors |
| Long video needs someone to review it | Plain-language memory answers from a summary of the past | Memory built; agent query tool not yet |
| Boxes and tracks have no history | System knows what is normal for this place and individual | Built; needs long-term data |
| One fixed pipeline per camera | Agent can change model, region, resolution, frame rate | Tools built; automatic policy not yet |
| Each new scene needs a labelling team | Verified answers become that camera’s training data | Planned |

Together these open one new kind of task: *asking a question nobody planned for, about something no model was trained on, across months of video, at a cost one person can afford.*

### 18.2 Where it could apply

Strongest fits share three traits: fixed cameras, rare or slow events, and too much video for people to review.

| Domain | Problem today | What this method could add |
| --- | --- | --- |
| **Animal behaviour science** | Home-cage video struggles with occlusion and identity swaps, often needing manual correction; limited validation in group-housed cages | Named animals, corrected identity, flagged doubts, written activity logs for long studies |
| **Wildlife conservation** | Manual review took 6–7 months in one reported study vs. ~1 week with AI; rare/easily confused species remain hard | Keep automated sorting; send only hard/rare sightings to deeper review |
| **Livestock & farms** | Visual scoring is slow and subjective; models struggle with varied barns/lighting | Long-term baselines per animal, changes flagged for the farmer |
| **Pets & vet care** | Owners miss gradual changes; recall differs from recorded behaviour (§3) | A daily log to bring to the vet (prompt to call, not diagnosis) |
| **Care at home** | Falls caused ~842,000 deaths in 2023 (WHO), 71% among people 60+; families cannot watch all day | Consent-based alert on routine changes, minimal footage leaving the home |
| **Robotics** | Robots meet untrained situations; published work pairs fast anomaly detection with slower LLM reasoning | Escalation + tools + memory as a model for robots that know when to stop and think |
| **Factories** | Defect rates can be ~0.1%, so 1,000 balanced samples could take up to 2.5 years to collect | Fast checks on every part; LLM review of odd ones becomes training data |
| **Road safety** | Bellevue’s study used ~5,000 hours of video; near-misses predict where crashes may occur | Reasoned review of near-misses, local rules kept in memory |

The architecture is built around a detector and a set of scenarios, so cars, robots or machines could replace cats; the detector, triggers, prompts and memory would each need adapting. Open-vocabulary “find any object” is out of scope for now (SAM 3.1 is in the model config for masks/boxes for training data, not for search).

### 18.3 Why scale matters, and the likely path

Cheap attention is what makes this reach far. With over a billion cameras installed and most scenes never justifying a custom model or labelling team, a method that *adapts to each camera* brings understanding to the many small settings specialist projects never reach — one lab, one farm, one family.

1. **Now:** pets and homes, where one person can check the answers.
2. **Next:** research labs, conservation groups and farms, where long-term data has clear value.
3. **Later:** robots and industry, where safety and reliability demand far more proof.

For the pet market specifically (§4.1): 53 M US cat households, multi-cat homes growing fastest, and a billion-dollar pet-camera category whose incumbents compete on detection — the per-scene learning loop with verifiable memory is an open position, and the daily questions (“when did he eat / sleep / play, and what changed?”) are the retention hook a motion alert cannot provide.

---

## 19. Ethics, privacy and safety

The same capability that helps a family can harm people, so limits belong in the design — stated here as product constraints, not afterthoughts.

- **People need consent.** Watching people is a different act from watching cats. Any care-at-home use needs clear agreement from the person being watched, with the camera’s presence and data flows visible to them.
- **Some uses are prohibited.** The EU AI Act (Article 5) bans inferring emotions at work and in education (with medical/safety exceptions) and restricts live biometric identification by police in public spaces. This system identifies *household cats by owner-provided profiles* — it must not be repurposed for human biometric identification, emotion inference, or workplace/education monitoring.
- **Privacy comes first.** Keep frames local where possible (the Pi + SSD/S3 archive are household-controlled), send the fewest possible frames to the cloud (only escalated cases, only the needed crops), and keep memory *readable and deletable* — the store is plain structured text and daily Markdown logs that the owner can inspect, correct and erase (a full reset keeps an optional timestamped backup). The Contributor-tier pricing that lets the provider train on prompts (§6) should be declined for home footage.
- **Decisions stay human.** The system raises a flag (“Qbo hasn’t eaten by 10 am” is an observation; “may be sick” is a diagnosis and is not produced). The project plan explicitly avoids health claims because AI medical advice is on the organisers’ banned list; summaries say “ate less than usual”, never why.
- **Claims need testing.** Shown on two cats in one room. Each new domain needs its own accuracy and safety evidence before any operational claim. Sleep from video means *rest* (still, curled, eyes closed as labelled), not a measured physiological sleep state; the camera only sees what is in view, so food bowls and litter boxes must be covered by the view for those questions to be answerable.
- **Security basics.** All three MCP servers share one bearer token (`MCP_AUTH_TOKEN`); camera startup logs a SHA-256 fingerprint of the token so a 401 can be diagnosed without printing the secret. The public submission must contain no secrets, keys, internal addresses or customer data (checked on the submission checklist in Section 22).

---

## 20. Limitations — stated plainly

This section collects every known limit, ranked by how much it bounds the demo’s claims. Nothing here is hidden elsewhere in the report.

1. **No ground truth, therefore no accuracy headline.** No hand-labelled frames exist yet: no mAP, precision, recall, identity accuracy, IDF1 or Pi inference time. The agent’s verdicts are not ground truth. The strongest honest claim is “the agent found tracker errors and explained each with evidence” on sampled decisions. *Fix:* the §21 eval set.
2. **Trigger recall is the bottleneck.** 772/818 Qb→Qbo relabels, 3 escalated (0.4%). The final rows still carry most of that error class. Better triggers (e.g. escalating a *sample* of nearness relabels, or a visit-level identity-consistency check) matter more than a better agent.
3. **Silent misses never escalate.** With no detection, nothing fires (§14.3). A periodic sampling check (e.g. agent review of a frame every N minutes regardless of detections) is needed and not built.
4. **Savings are not proven.** The demo window saved ~half vs. all-LLM review before memory costs; the full-day rate is unmeasured. One review ≈54 frames at the working rate, so escalation only pays at low rates in typical (quiet) footage.
5. **Validation is slower than the footage** (9.4 min for 7 min, sequential). Background-by-design, but no live-correction claim is possible. Merging (built, unmeasured end-to-end) and parallelism are the levers.
6. **Agent nondeterminism.** Two verdicts flipped between runs on byte-identical inputs (E8, E11; both row no-ops here). Single-run evaluations are therefore weak evidence; safety-critical verdicts would need agreement across runs or a human.
7. **Reviews need hardening.** API 400 drops a case with no retry and no trace (E9 — the session’s only merge was lost). Same-frame T2 cross-talk (E6/E7) and the last-writer `verdict` field are reproduced apply bugs. Recorded-but-unapplied verdicts (`_noop`, unknown neighbour) journal more than rows show.
8. **External taxes in the demo rig:** replay-local period ids resolving to production visits (11 wasted tool calls), image-server Pillow missing (16/16 zooms failed; full frames worked), prompt-cache instability (~8× cost swing), `get_arrival_path` path-picking cost on static visits.
9. **Memory is young.** One 7-minute session cannot show a routine; zone environment/background were `Unknown` without a bootstrap image; recents collide across runs (E-name scoping); memory’s effect on verdicts is unmeasured (briefs were attached to all 12 prompts, but no A/B exists); live same-day records do not yet reach briefs.
10. **Unbuilt by design (labelled, not implied):** verified-labels → dataset → retraining (detection and activity models); automatic camera policy (the pipeline steering the camera by itself); memory-only decisions (shadow design only); the agent memory-read tool for daily Q&A; SAM 3.1 / Muse Voice / Muse Image / open-weight roles (config only); night/IR model switching; embeddings-based semantic search (token overlap in v1); multi-camera and multi-view generalisation beyond view 3.
11. **Correctors are hand-tuned** for one 800×600 view with no motion model and no benchmark. The session’s 772-error is the cost of hand-set rules meeting a hard clip.
12. **Literature is analogical.** Most supporting evidence comes from wildlife/traffic/driving video and two preprints; home-pet numbers must replace analogies before any general claim.

**Claims-to-measure table (demo narrative → metric → baseline):**

| Claim | Metric | Baseline to beat |
| --- | --- | --- |
| Hybrid improves on the general model | Recall + precision on labelled home frames, incl. night | Off-the-shelf YOLO (`yolov8s`) |
| Accuracy stays close to all-LLM | Event agreement on a labelled sample | Muse Spark on every sampled frame |
| Hybrid cuts LLM cost | Actual escalation rate + actual API spend/day | $172.80/day |
| The Pi needs less help over time | Escalation rate before/after each fine-tune | First-week rate |
| The two cats stay separate | Identity accuracy for Qb and Qbo | Random assignment |

---

## 21. Roadmap

Ordered by what most increases the strength of the central claim (“cheaper and better the longer it runs”) per unit of work. Dates are PT, aligned to the project-plan timeline (freeze 3 Nov, submit 4 Nov noon PT).

**Immediate (before recording):**

1. **Tighten the T2 trigger + sample-audit the rest.** Escalate on visit-level identity inconsistency (a visit whose rows disagree with its PeriodCorrector winner) and audit a random sample of non-escalated relabels, so recall on the 772-class is measured and then raised. This is the single highest-value change.
2. **Rerun the headline window with merging on** (validation batching as built) and report the real call/token/cost delta vs. §16.1; add one **retry with delay + partial-trace record** on API errors so no escalation is silently lost (E9 class).
3. **Detector numbers from the training run:** mAP, precision, recall (overall + night/low-light slice) and Pi inference time for `custom cat-identity model`; add to the project summary and video.
4. **Hand-label one clip** (both cats, hard cases on purpose: gaps, jumps, both cats close, low light — a few hundred frames) and run the fair tracker comparison (ByteTrack / BoT-SORT on the same detections: identity accuracy, IDF1, visit errors).
5. **Memory read tool for the agent** (query short-term day log + long-term YAML by time/cat/place) so demo Step 8 answers “when did Qbo last eat / sleep / play?” from memory; define the `food_bowl` zone and confirm bowl coverage in view.
6. **Fix the demo-rig taxes:** install Pillow in the image-server venv (fail-fast at startup), namespace replay visit identifiers so detail lookups cannot resolve to production visits, scope memory recents by run.

**Core loop completion (the “knowledge flows down” half):**

7. **Eval set + before/after experiment** : baseline YOLO + tracker on the eval set (identity error rate, gap/jump rate, activity accuracy once the activity model exists); run escalation on a separate window, build the dataset from its verified labels (with the human-confirmation policy decided (always, never, or only on low agent agreement)), retrain detection (using the existing cloud training path) and the separate pose/activity model, re-measure on the same eval set; 2–3 rounds so the onboarding walkthrough can show improvement step by step with real numbers. Report before/after in the project summary and video.
8. **Labels → dataset plumbing:** agent verdicts today go to the validation journal and task logs; wire them into a versioned training set that the labelling and training steps actually read, with the static-decoy and circular-evidence guards (Section 9.6) as dataset filters.
9. **Proactive agent behaviours:** ask the user to confirm/label hard cases (the `send_to_human` queue surfaced in chat), ask which scenes/objects/pets matter and prioritise optimisation there, and proactive alerts (“tell me if Qbo hasn’t eaten by 10 am”) over the memory. Voice questions via Muse Voice Transcribe if scope allows.

**Memory maturation:**

10. **Full-day and multi-week runs** to replace scenario economics (§16.3) with the measured escalation rate and spend/day, and to grow the temporal pyramid (hour/day/week summaries) into real routines; bootstrap vision from a real first frame so zone environment/background/contrast are populated.
11. **Memory-only decisions in shadow mode:** log memory-only verdicts alongside image-based ones per scenario code; graduate codes with proven agreement to image-skipping (with periodic image spot-checks for drift). Blend live short-term entries into validation briefs (read-through `invoke`).
12. **Robustness:** verdict-agreement across reruns for high-stakes codes; prompt-cache stabilisation investigation; arrival-path key-frame capping and run-scoped naming in memory recents.

**Generalisation (post-hackathon):**

13. Second camera view + second household; open-vocabulary extension via SAM 3.1 for masks/boxes in the training-data path (finding *any* object stays out of scope until the loop is proven); Muse Spark Open / Glimmer on Baseten or local hardware for the privacy tier (Best Open Weight); a daily “cat diary” via Muse Wonder + Muse Image (Best Creative Build) as the shareable artifact for the Most Viral / Community Vote prizes.

---

## 22. Deployment and operation

This section describes how the system is deployed and operated, in self-contained terms. No companion document is needed to follow it.

### 22.1 Configuration

The system is configured through environment variables (grouped by component) plus four central configurations: the model configuration (provider endpoint, per-role models and token pricing), the cat profiles (identities, appearance descriptions and reference photographs), the activity label set (activities, context labels and their evidence requirements) and the task configuration (per-task image budgets, consensus thresholds and physics thresholds). Python 3.10 or later is required throughout. A single environment configuration serves all components on the same machine; the camera and the archive host each use only their own section.

Secrets handling: the Meta Model API key, the single shared bearer token for the three MCP services and the read-only archive credentials are supplied only through the environment. They never appear in charts, traces or this report.

### 22.2 Running components

| Component | Role in operation | Connection |
| --- | --- | --- |
| Ingest server with tracking service | Receives camera detections, runs the ingest window and confidence gate, and serves tracker and visit queries | Detection stream on port 8000; tracking queries on MCP port 10002 |
| Camera service (on the Pi) | Always-on detection plus agent-steerable capture, zoom, re-detection and schedules | MCP port 8765 |
| Image archive service | Discovers, checks and retrieves stored frames by capture timestamp (local archive for earlier captures, cloud storage for recent captures) | MCP port 10001 |
| Supervisor agent | Interactive question answering, pipeline tasks and escalation review | Connects to the three MCP services; runs wherever the operator runs it |
| Tracking pipeline | One run over a chosen view and time window, in live, file-replay, database-replay or simulation mode, with optional validation, memory and charts | Consumes the detection stream; produces the outputs listed in Section 12.3 |

Operation notes: if an MCP service is unavailable, the agent warns and continues with the remaining services. The camera simulator allows the complete live path (camera, ingest, tracking, validation and memory) to be exercised with archived frames and no Pi hardware, which is how the demonstration in Section 15 is replayed.

### 22.3 Reproducing the headline measurement

To reproduce the headline session (Section 14.1), run one tracking pipeline session over the living-room view for 28 October 16:33–16:40, sourcing detections from the read-only detection archive, with validation and memory enabled and timeline charts generated. Expect about 10 minutes of wall time and about $0.60 in validation tokens for this 7-minute window in sequential mode without batch merging. For a tracking-only comparison, run the same window with validation and memory disabled. For a hardware-free live check, first start the camera simulator over an archived interval, then run the pipeline in live mode against it with validation and memory enabled.

Key operating parameters: the minimum detection confidence is 0.5 (strictly above); the pipeline window includes the full final minute; process visualisation renders per-frame buffer views plus a stitched video for a chosen cat; an optional starting frame can seed the memory vision descriptions; the live ingest port defaults to 8000.

### 22.4 Verification

All verification runs fully offline with fake or stub language-model clients; no API key and no database are required. The suites cover the areas listed in Section 12.2. A system is considered verified for demonstration when the offline suites pass and the headline window reproduces the counts in Section 14.1 (1,319 frames, 1,016 detections, 1,265 revised rows, 4 visits and 12 escalations) with validation producing 11 verdicts in about 9–10 minutes.

### 22.5 Submission checklist (deadline 4 November, 9:00 PM ET / 6:00 PM PT; aim for noon PT)

- [ ] Public code submission, accessible without login, whose summary covers the problem, the architecture, which Meta models are used and where, how the system is operated, and what was built during the hackathon versus before (Section 13)
- [ ] 3-minute demo video, publicly accessible, showing only what was built during the event
- [ ] Live demonstration or replay bundle that holds up when the home Pi cannot be reached (recorded window, charts, traces and day log as in Section 15)
- [ ] All team members listed; track (Indie Developer) and problem statement (Build for Your World) chosen
- [ ] Social post link submitted for the audience prize (post early on 4–5 November)
- [ ] No secrets, keys, internal addresses or customer data in the submission or the video
- [ ] Each team member has claimed the Meta Model API credits and set a billing spend limit (hard stop 4 November, 9:00 PM ET)

**Prizes in scope:** the Indie track prize pool ($150,000 / $100,000 / $75,000 / $50,000 / $25,000) plus the audience prize for social engagement, the agentic use-case prize (the strongest bonus fit for this system), the open-weight-model prize (if the open-weight Muse model is served for part of the pipeline), the community vote prize, and the creative-build prize (for a generated daily cat diary).

---

## 23. Team and acknowledgements

**Team (Indie, 2 people):** Terry and Muyang. Muyang owns the tracker failure-mode knowledge, the escalation-trigger definitions and the cat data (Qb and Qbo, view 3); the project plan assigns the open trigger/retraining/human-confirmation decisions accordingly. The Representative who submits is to be named on the submission page.

**Built with:** Meta Muse Spark (agent, memory, validation) and Muse Code (coding); Ultralytics YOLO (custom cat-identity model plus the generic COCO model) on Raspberry Pi; the Model Context Protocol (three MCP servers); Python, MySQL (read-only replay), SQLite (per-run validated output), SSD + AWS S3 image storage. Partner-credit options noted for the hackathon: Baseten (`meta-models/muse-spark-open-1026`), Modal, Fireworks.

**Prior work acknowledged:** NoScope, Mullapudi et al. (online distillation), Ekya, Cerberus, SmartHome-Bench, Avas, WILDS/iWildCam, Beery et al., Dodge & Karam, Bhaskar et al. — full citations in Section 24. The correctors, MCP servers and pre-hackathon agent lineage are the team’s own earlier work, marked as such in §13.

---

## 24. References

Prices and market figures are as reported in November 2026. Session figures are the project’s own measurements and have not been independently checked.

**Cat health and behaviour**

- AAFP senior cat care brochure — cats as “masters of hiding signs of disease and pain”
- Shin et al., *J Vet Clin* 2024 — litter-box study (6,555 records, 37 cats; healthy 1–7 visits/day; diseased 3.77 vs. healthy 3.38; AUC <0.6)
- Langenfeld-McCoy et al., *Animals* 2026 (Nestlé Purina / Ohio State, 136 cats; CKD recall 57.1% in validation) via dvm360
- Smit et al. 2024 (28-cat accelerometer study: active ~3%, lying/sitting ~74%, elimination <6 min/day) via AVSAB summary
- Nestlé Purina / Ohio State smart-litter-box CKD work, *Animals* (2023–2025 data, 136 cats) via PR Newswire / dvm360; Petivity defecation-frequency study, *PMC* 2025

**Computer vision and video systems**

- Beery, Van Horn & Perona, *Recognition in Terra Incognita*, ECCV 2018
- Koh et al., *WILDS*, ICML 2021 (iWildCam F1 47.0→31.0; Camelyon17 93.2%→70.3%)
- Dodge & Karam, *Understanding How Image Quality Affects Deep Neural Networks*, 2016
- Kang et al., *NoScope*, VLDB 2017
- Mullapudi et al., *Online Model Distillation*, ICCV 2019 (7–17× lower cost)
- Bhardwaj et al., *Ekya*, NSDI 2022 (22% drift drop)
- Bhaskar et al., *Robust Object Detection with Pseudo Labels from VLMs*, 2025 (preprint)
- Zheng et al., *Cerberus*, 2025 (preprint; 151.79× speedup, 2.8% AUC drop at 1% anomaly share)
- Zhao et al., *SmartHome-Bench*, CVPRW 2025 (1,203 home videos)
- Yan et al., *Avas: Empowering Agentic Video Analytics Systems with Video Language Models*, 2025
- Lin et al., *Microsoft COCO*, 2014 (>70,000 worker hours)
- Zhang et al., *ByteTrack*, 2022; Cao et al., *OC-SORT*, 2022; Roboflow Trackers MOT comparison (BoT-SORT 63.7 HOTA); Ultralytics track-mode docs

**Hardware, pricing and platform**

- Ultralytics Raspberry Pi guide (YOLO26n, Pi 5 CPU/NCNN, 67 ms/image at 640 px)
- Sixfab Raspberry Pi 5 AI HAT benchmark (6.1–7.6 W average); Raspberry Pi Foundation AI Kit testing (YOLOv5 30+ fps, Hailo-8L)
- Meta Model API Standard pricing used in this report (input $1.25, cached input $0.15, output $4.25 per million tokens; Contributor tier input $0.10, cached $0.002, output $0.20 per million), cross-checked against independent Muse Spark pricing listings
- Muse Spark 1.1 / 1.3 and Muse Spark Open (816 billion total / 42 billion active mixture-of-experts parameters, October 2026) reporting: Meta Superintelligence Labs via DataCamp and Pulse2, with independent MCP-Atlas (Scale AI) tool-use coverage; open-weight serving via Baseten
- Gemini bounding-box evaluation (independent blog test: 0.46–3.63 s/image; Co-DETR ~0.60 mAP, Gemini 3 Pro Preview 0.407 mAP on COCO)

**Market context (searched Nov 2026)**

- APPA 2026 State of the Industry Report ($158 B in 2025, $165 B projected 2026) and APPA 2026 Cat Report (53 M cat households, 39%, +5% YoY; multi-cat growth)
- Video surveillance / AI-in-video-surveillance market reports (1.1 B+ cameras end-2025; surveillance $54–69 B; AI video analytics $6.4 B 2025 → $49 B 2035); IHS Markit camera estimate via CNBC (770 M in 2019)
- Smart pet camera market reports ($0.97–1.57 B in 2025, $4–6 B by 2034–35; Furbo/Petcube/Wyze positioning)
- Edge-vs-cloud video cost models (on-camera 2–5 W vs. ~$0.10/min stored-video cloud; ~$11 vs. $50–150/camera/month above 50–80 cameras)

**Broader impact**

- *Frontiers in Behavioral Neuroscience*, home-cage monitoring review (occlusion/identity-swap limits)
- Washington State University, AI cuts wildlife tracking time (6–7 months manual → ~1 week)
- *Animals* 2025, computer vision in dairy farming (visual scoring limits)
- WHO falls fact sheet (~842,000 deaths in 2023, 71% aged 60+)
- Sinha et al., *Real-Time Anomaly Detection and Reactive Planning with LLMs*, 2024
- Industrial defect detection with scarce labels (0.1% defect rate; up to 2.5 years for 1,000 samples)
- City of Bellevue conflict and speed analysis (~5,000 hours of video)
- EU AI Act Article 5 (prohibited practices) summary — check the official text before any people-facing deployment

---

## 25. Appendices

### Appendix A — Escalation verdict reference

| Code | Allowed verdicts |
| --- | --- |
| T1 | `confirm_assigned` / `reassign_candidate` / `reject_box` |
| T2 | `confirm` / `keep_original` / `reassign_other` (+ `chosen_cat_id`) / `drop` |
| T3 | `confirm_survivor` / `keep_killed` / `keep_both` |
| T4 | `confirm_reset` / `link_continuation` / `drop_box_as_outlier` |
| T5 | `commit_target` / `wait_for_stronger` / `close_visit` |
| T6 | `confirm_split` / `merge_visits` / `drop_box` |
| T7 | `keep_synthesis` / `cut_at_gap_start` / `shorten_gap` (+ `cut_time`) |
| P1 | `confirm_winner` / `award_runner_up` / `split_visit` |
| P2 | `keep` / `drop` / `merge_neighbor` |
| P3 | `confirm_reassign` / `restore_wanted` / `mark_unknown` |
| P4 | `assign_cat` (+ `chosen_cat_id`) / `split_fragments` / `confirm_unknown` |
| P5 | `keep` / `drop` / `trim_core` / `reassign` (+ `chosen_cat_id`) |
| P6, P7 | `merge` / `keep_separate` |

Each verdict set is enforced by a strict structured-output schema (one per scenario code). In the live loop the reviewer may use only three tools: fetching a stored image, listing stored images, and reading visit detail.

### Appendix B — Skills reference

| Skill | Surface | Used by | Core discipline |
| --- | --- | --- | --- |
| `validate-cat-identity` | task | `validate_detection` | 6-rung cheapest-evidence ladder; cite a visible cue or do not verdict; static-decoy guard |
| `check-trajectory` | task | `check_trajectory` | Read physics flags first; one contact sheet; split at the flag where the animal changes |
| `assign-activity-label` | task | `assign_label` / `label_visit` | Choose only from code-filtered candidates; echo rule-decided context verbatim; `unknown` + suggestion when nothing fits |
| `check-live-detections` | chat | Free chat (“what is on camera, is the model right?”) | Wide frame first (own eyes before boxes), then one turn of crops + profiles, verdict table (TP/FP/missed/wrong-ID) |
| `manage-labels` | chat | `--labels` chat | Draft → show → explicit yes → save; never save unconfirmed; place labels need a confirmed zone |

Task budgets: validate 3 images / 1 high-res / 8 steps; trajectory 2/0/6; label 2/0/6; manage-labels 2/0/10. Visit consensus: ≥20 frames, ≥80% agreement, never when long-static (≥600 s, moving ≤5 s). Trajectory physics: teleport 3.0 body-lengths/s (min 1.5), long gap 8 s, size-jump ratio 2.5, edge margin 0.12, static ≥600 s.

### Appendix C — Memory configuration reference

Memory defaults (every value can be overridden by an environment variable):

```yaml
frame_size: [800, 600]
spatial_levels:  [{cells, 8x6}, {zones, 4x3}, {full_view, 1x1}]   # 48 → 12 → 1
temporal_levels: [{slot_5min, 5}, {hour, 60}, {day, 1440}, {week, 10080}]
retention_days:  {slot_5min: 7, hour: 30, day: 180, week: 0}      # 0 = keep
retrieval:       {spatial_chebyshev_radius: 1, temporal_history_days: 7, semantic_top_k: 5}
promotion:       {idle_seconds: 300, min_pattern_support: 3, stable_weight_cap: 50,
                  flip_threshold: 3, recent_kept: 5}
quality:         {min_store_confidence: 0.7, short_term_keyframes: 3}
activity_combine_gap_minutes: 5.0
llm:             {model_role: memory, max_units_per_call: 8}
```

The durable store is a human-readable structured long-term store plus a daily short-term log (both retained across runs). Every real memory language-model call produces an audit trace; the trace location can be overridden by configuration.

### Appendix D — Evidence index (all contained in this report)

| Question | Answered in |
| --- | --- |
| How the loop runs end to end | Sections 8 and 9, with the measured walkthrough in Section 15 |
| What a prompt and its answer schema look like | Sections 9.3 and 15, with the full verdict list in Appendix A |
| How the agent reasons | Sections 9.4 and 14.2–14.4, including tool counts, images read and rationales |
| What the memory contains | Sections 9.6 and 14.2, including the trust-gate decisions and learned summaries |
| What validation costs | Sections 14.1 and 16.1, with per-case time, tokens and cost |
| Where the system fails | Sections 14.3, 14.6 and 20, including the sleeping-cat miss and the trigger-recall gap |
| What is new in the hackathon period | Section 13, with dating for every piece |

---

*End of report.*
