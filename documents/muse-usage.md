# How we use Muse in this project

**Project:** Agentic Realtime Computer Vision · Meta Global AI Developer Hackathon 2026 ·
Summary date: 2026-11-04 · Details and measurements: [agentic-realtime-cv-report.md](submission/agentic-realtime-cv-report.md) (§9 and §10)

Muse is both the **model family the product runs on** and the **toolchain we built it with**.
This page summarizes both, with the numbers we measured.

---

## 1. In one paragraph

A YOLO nano detector on a Raspberry Pi watches every frame. When the tracker is unsure, it
raises a typed doubt (about 1% of frames). **Muse Spark 1.3** reviews those doubts as a
tool-calling vision agent and returns schema-checked verdicts that rewrite the record. Spark
also labels each visit's activity and writes the spatial-temporal memory. **Muse SAM 3.1**
produced the ground truth we measure against, **Muse Wonder** shaped the design, **Muse
Glimmer** is the on-device path for private deployment, and the code, docs and reports were
built in **Muse Code**.

## 2. Why the product needs Muse

| The product needs… | What Muse provides | Measured |
| --- | --- | --- |
| to tell two look-alike cats apart from pixels, and say why | Spark vision reasoning over crops plus each cat's written profile | identity accuracy 0.934 → **0.993** |
| to fetch only the evidence that settles a doubt | multi-turn tool calling, with images returned inside tool outputs | median run: 4 turns, 8 tool calls |
| answers that code can apply to rows | strict `json_schema` output, including schemas built at runtime | 12 of 12 verdicts parsed and applied |
| to stay cheap enough to run all day | everyday Spark pricing and cached input | 52% of input tokens cached; **$0.39** per 8 minutes |
| memory a person can read | Spark as the memory engine | activity labels at 0.92 confidence; ~$0.14 a session |

Without it, the doubts the Pi raises have nowhere to go. The cheap model stays cheap and
wrong (tracker rules alone *lower* identity accuracy, 0.934 → 0.920), or every frame goes to AI
and the system stops being cheap ($24.65 to label 8 minutes, about $1.2M a year per camera).

---

## 3. In the product (runtime)

| Where | Muse model | What it does | Status |
| --- | --- | --- | --- |
| Supervisor agent | **Muse Spark 1.3** | Reviews each doubt: picks frames, crops to boxes, compares the cats, returns a verdict and a rationale | built, measured |
| Joint review | **Muse Spark 1.3** | Reviews up to 4 related doubts in one call with a merged schema | built, measured |
| Activity labeler | **Muse Spark 1.3** (vision) | Labels each visit's key frames with activity, appearance and background | built, measured |
| Memory engine | **Muse Spark 1.3** | Checks new observations against stable memory; writes word-capped summaries for zones, hours, days and weeks | built, measured |
| Chat agent (CLI) | **Muse Spark 1.3** | "Is any cat on camera right now?"; adds labels and zones by request | built |
| Edge / private deployment | **Muse Glimmer** | On-device path so footage can stay home; a personal agent gets these "eyes" locally | built locally; roadmap |
| Voice questions | Muse Voice Transcribe | "When did Qbo last eat?" by voice | planned |
| Daily recap | Muse Image / Muse Wonder | A creative summary of each cat's day | planned |

**One doubt, end to end (demo run, E5-T2):** the tracker relabeled the bed cat by distance
alone and raised `T2` → the question went to Spark with both cats' profiles and a memory brief →
Spark pulled 4 frames, then 4 crops → answered `keep_original` ("cool white, smooth sharp
outline … = Qb") → the rows were rewritten with `validated_by: E5-T2` → the visit entered the
day log. Full walkthrough: [report §8.0](submission/agentic-realtime-cv-report.md#80-follow-one-doubt-through-the-loop).

**What it delivered on the demo window** (8 minutes, both cats, audited ground truth):

| | Raw detector | After Muse Spark review |
| --- | --- | --- |
| False positives | 107 | **8** (−93%) |
| Precision | 0.895 | **0.992** |
| Identity accuracy | 0.934 | **0.993** |
| Agent cost | — | **$0.39** for 12 doubts in 5 calls |

**Usage across the build** (Oct 27 – Nov 3, logged in the project database):
124 agent runs (93 validation, 31 interactive), all on Muse Spark 1.3 and all answered;
1,058 tool calls across 26 tools; 9.75M input / 0.63M output tokens, 52% of input cached;
**$9.22** in total. The three final replay runs reviewed 105 doubts in 43 calls for **$3.77**.

**Pricing and privacy.** All costs are at **Standard** rates. The Contributor tier was cheaper
but grants permission to train on the footage, and customers' home cameras are not ours to
give away.

---

## 4. In development

| Stage | Muse tool | How we used it |
| --- | --- | --- |
| Design | **Muse Wonder** | The doubt taxonomy, its prompts and verdict contracts, and the spatial-temporal memory architecture, argued out before any code |
| Implementation | **Muse Code** + Muse Spark | The pipeline, correctors, supervisor agent, memory system and tests, built in small verified steps (a failing test first, then the fix) |
| Multi-file work | **Muse Code workflow skills** | Large changes run as orchestrated tasks with focused subagents, e.g. the escalation system (trigger → prompt → schema → verdict applier, six files) in one task |
| Capturing ideas | **Muse Code voice input** | Speaking design ideas out loud and landing them as structured notes |
| Ground truth | **Muse SAM 3.1** (+ Spark to name the cat) | Labeled every frame of the demo window for evaluation: 66 min, $24.65, F1 0.794 vs Spark-only 0.612 ([labeling-sam-vs-spark-20261103.md](labeling-sam-vs-spark-20261103.md)) |
| Debugging the agent | Muse Spark traces | Every run writes a `trace.md`; two trace reviews became fixes (slimmer tools, skill gating, raw-pixel grounding, an explicit decoy rule) |
| Documentation | **Muse Code** | Design docs, runbooks, session reports, the evaluation report and the project report, written next to the code they describe |
| API learning | Muse Spark (probe) | A controlled probe found intermittent `referenced response not found` on chained image turns; we folded instructions into turn 1 and retry without chaining ([model-api-chaining-probe.md](model-api-chaining-probe.md)) |

**One family, matched to the job:**

| Model | Used for | Why this tier |
| --- | --- | --- |
| Muse Wonder | design | deepest reasoning, used rarely and early, where a wrong design costs the most |
| Muse Spark 1.3 | runtime agent, labeler, memory; everyday coding | strong vision and tool calling at everyday cost; strict structured output; cached pricing |
| Muse SAM 3.1 | ground-truth labels | mask-true boxes that VLM box regression does not match |
| Muse Glimmer | on-device, private deployment | latency and privacy at the edge |

---

## 5. What's next with Muse

- **Muse Glimmer locally**, so a personal agent can use these eyes without footage leaving home.
- **Teach the Pi:** turn Spark's verdicts into labeled data and retrain the small detector, so
  fewer doubts reach the agent.
- **Voice questions** (Muse Voice Transcribe) and a **daily recap** (Muse Image / Wonder).
- **Muse Spark Open** (`muse-spark-open-1026`) for private reasoning on own hardware.
