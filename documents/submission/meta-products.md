# Meta Products Used to Build Agentic Realtime Computer Vision

Final report: https://github.com/AuraAIInc/agentic-realtime-cv/blob/main/documents/submission/agentic-realtime-cv-report.md

Team: Muyang Lu, Terry Li · Meta Global AI Developer Hackathon 2026

## Overview

Agentic Realtime Computer Vision turns a cheap, imperfect camera into a record you can trust. A YOLO nano detector on a Raspberry Pi watches every frame. When the tracker is unsure, it raises a typed doubt (about 1% of frames). Meta's models review only those doubts, write verified answers back into the record, and build a plain-language memory of what is normal for each cat, in each place, at each hour.

Meta products appear at every stage of the project:

- **Muse Spark 1.3** runs the product: the supervisor agent, the activity labeler and the memory engine.
- **Muse SAM 3.1** produced the ground truth we measure against.
- **Muse Wonder** shaped the system design.
- **Muse Glimmer** is our path to local, private deployment.
- **Muse Code** is the environment we built and documented the project in.
- **The Meta Model API** (its Responses endpoint) connects all of it.

This document describes each product, what it does in the project, and how we used it.

## Muse Spark 1.3: the model the product runs on

Muse Spark 1.3 is a multimodal reasoning model that can see images, call tools across several turns and return strictly structured answers. In our system it plays four runtime roles.

**1. The supervisor agent.** When the tracker raises a doubt, Muse Spark receives a plain-English question. The question states the conflict, what to validate, the alternatives, the frames and boxes to inspect, and both cats' written appearance profiles. The agent then investigates with tools instead of receiving a pile of images:

- camera tools to capture a fresh or sharper frame, or re-run detection on a region;
- image tools to check whether a stored frame exists and fetch it cropped to a box;
- tracking tools to read a visit's structured history before any image is needed;
- local tools for each cat's written profile, owner reference photos, contact sheets and trajectory checks;
- skills, loaded on demand, that hold playbooks such as the identity-check evidence ladder.

The agent returns one verdict from a fixed list, with a written rationale, in a strict JSON schema. In the demo run, Muse Spark reviewed 12 doubts in 5 calls and returned 12 of 12 valid verdicts.

**2. Joint review of related doubts.** Doubts cluster in time. A queue merges up to four related doubts into one Muse Spark call, with an answer schema built at runtime from the union of the cases' allowed verdicts. The agent reasons across the cases, fetches each frame once and gives one consistent reading of who is who.

**3. The activity labeler.** Each trusted visit is labeled by a Muse Spark vision call from its key frames: one activity (such as sleeping, walking, eating or playing), a confidence, the cat's appearance and the background. A real answer: "playing, confidence 0.8; qbo is a fluffy white cat with a long dark bushy tail; a dim living room with a light sofa and a patterned rug with scattered wand toys."

**4. The memory engine.** Muse Spark writes the long-term spatial-temporal memory in plain language:

- a store-update call judges whether a new observation genuinely contradicts what a memory unit already knows;
- a summary call compresses child units into a parent (cells into zones, hours into days, days into weeks) under a word cap;
- a bootstrap call describes a camera's first frame as structured environment notes.

Code, not the model, enforces the counting rules that keep memory honest. Muse Spark supplies the language and the judgment.

**A fifth, interactive role.** The same agent also runs as a chat assistant over the live camera, history and tracker. It answers questions such as "Is any cat on camera right now?" and adds custom labels and zones when asked.

Across the build (October 27 to November 3), Muse Spark ran 124 agent sessions (93 validation, 31 interactive) and 1,058 tool calls across 26 tools, and every session returned an answer. It used 9.75 million input tokens, 52% of them served from cache, and cost $9.22 in total.

## Muse SAM 3.1: the ground truth behind every number

A system that claims to improve accuracy needs an honest yardstick. We used Muse SAM 3.1 to label every frame of the demo window:

- SAM 3.1 proposes cat regions from short concept prompts, with geometry taken from segmentation masks;
- Muse Spark then names the cat in each region, using both cats' written profiles, or rejects the region;
- the labels pass through the same tracker and validation pipeline in a labels-only replay mode, and are exported as validated ground truth.

In a head-to-head labeling comparison on the same 1,318 frames, the SAM 3.1 pipeline scored F1 0.794, against 0.612 for Muse Spark alone. It also had full coverage, tighter boxes and lower cost. That ground truth is what lets us state, for example, that false alarms fell by 93% and identity accuracy rose from 0.934 to 0.993. The same labels are the starting point for retraining the small edge detector.

## Muse Wonder: the design before the code

We used Muse Wonder, the deepest-reasoning tier, early and sparingly, where a wrong design would cost the most. Three designs started as argued notes with Wonder before any code existed:

- **The doubt taxonomy:** which tracker decisions deserve review, how to name them, and what the agent may answer for each (fifteen typed codes plus a routine spot-check).
- **The verdict contracts:** small, strict answer schemas that code can apply directly to tracked rows, so an answer is an operation, not an essay.
- **The spatial-temporal memory:** short-term and long-term memory, the spatial pyramid of cells, zones and view, the temporal pyramid of slots, hours, days and weeks, and the anti-overfitting rules.

## Muse Glimmer: local and private deployment

The project is built for compute-limited and privacy-bound settings. The camera server and the track correctors already run on the device. Muse Glimmer is the on-device model path, so a small local Muse model can be called without footage leaving the home. Our roadmap's first step builds on it: run Muse Glimmer locally, so a person's own AI agent can use these "eyes" to ask the camera questions and hear when a routine changes.

## Muse Code: the development environment

We built the project inside Muse Code:

- **Implementation.** The pipeline, the correctors, the supervisor agent, the memory system and the tests were written in small, verified steps: a focused failing test first, then the fix.
- **Workflow skills and subagents.** Larger multi-file changes ran as orchestrated tasks. For example, the escalation system (trigger, question builder, strict schema and verdict applier, across six files) was implemented as one coordinated task.
- **Voice input.** We spoke design ideas out loud and turned them into structured notes before they were forgotten.
- **Documentation.** Design documents, runbooks, session reports, the evaluation report and the project report were written next to the code they describe, so the documentation stays close to the implementation.

## The Meta Model API

All Muse calls go through the Meta Model API's Responses endpoint. The project relies on several of its features:

- **Images inside tool results**, so the agent sees only the frames and crops it asked for.
- **Multi-turn chaining** with a previous response ID, so the agent can investigate over several turns.
- **Strict JSON schema output**, including schemas assembled at runtime for merged calls.
- **Cached input pricing**, which kept repeated context such as system prompts and cat profiles cheap.

We also contributed a careful finding back to our own design. A controlled probe showed that chained turns carrying images sometimes return a "referenced response not found" error. It ruled out instructions, image size and payload size as causes. We folded instructions into the first turn and retry without chaining when that error appears.

**Pricing choice.** We priced and ran everything at Standard rates ($1.25 per million input tokens, $0.15 cached, $4.25 output). The Contributor tier would have been cheaper, but it grants permission to train on the footage, and customers' home cameras are not ours to give away.

## Planned Meta products

The model configuration already reserves roles for:

- **Muse Voice Transcribe**, so owners can ask by voice: "When did Qbo last eat?"
- **Muse Image with Muse Wonder**, for a daily, creative recap of each cat's day.
- **Muse Spark Open** (`muse-spark-open-1026`), for private reasoning on the owner's own hardware.

## Summary

One Meta model family covers the whole project. Muse Wonder for the design, Muse Code for the build, Muse SAM 3.1 for the ground truth, Muse Spark 1.3 for the running product, and Muse Glimmer for local, private deployment. Each is matched to the job it does best.
