# Project Plan: Meta Global AI Developer Hackathon 2026

Draft for the team (Terry, Muyang) to review. Event rules, dates and judging are in
[HACKATHON_GUIDE.md](HACKATHON_GUIDE.md). Chinese version: [PROJECT_PLAN.zh.md](PROJECT_PLAN.zh.md);
keep both in sync.

Every item is tagged:

- ✅ **Decided**: stated explicitly by Terry
- 💡 **Proposal**: not discussed yet, needs agreement
- ❓ **To define**: open, needs a decision, an experiment, or the organizers

---

## 0. Pitch

✅ **Direction:** our theme is everyday life, so the pitch **lands on how it changes
people's lives**: the emotional side, concrete everyday scenes, and the bigger impact. The
technical innovation is still stated clearly, as the "how".

💡 **Draft:**

> When you're away, you want to know your cats are OK: who ate, who slept all afternoon,
> whether the shy one came out at all. Today's smart cameras can't tell two cats apart,
> and they never get better at *your* home. Ours pairs a small CV model that watches 24/7
> with Meta's models. The large model doesn't just answer questions. It acts on its own:
> it asks you about the moments it isn't sure of, learns what matters in your home, and
> **teaches the small model**. Every week it knows your home a little better, so you get
> peace of mind without watching any footage.

Differentiators, in order of weight:

1. ✅ **Continuous per-scene learning** (the core): the large model is the teacher, the CV
   model is the student. This answers the question judges will ask: "why not send every
   frame to a VLM?" Our answer is cost, latency and privacy. The cheap model runs all the
   time, and the expensive model only runs on hard cases, which it turns into training
   data.
2. ✅ **A large model that acts on its own.** Answering questions, summaries and custom
   labels by conversation are no longer unique, because large camera vendors are adding
   LLM features. What is new is that the large model also acts proactively, in several
   ways:
   - asks the user to confirm or label hard cases, to strengthen the CV model;
   - asks the user which scenes, objects, people or pets matter, and focuses the
     optimization there;
   - improves the model, cuts latency, and surfaces insights without being asked.

   The goal is a better experience for the user, by giving richer and more accurate
   information.
3. ✅ **A data source for personal assistants.** What this system knows about the home can
   be an important part of a personal assistant's data. An assistant like Muse can
   combine with it to understand the user better.

✅ Name competitors explicitly in the demo video, to show the difference.

---

## 1. Decisions so far

| Topic | Decision |
| --- | --- |
| Track | ✅ **Indie Developer** |
| Team | ✅ 2 people: Terry, Muyang (max 4) |
| Problem statement | ✅ **Build for Your World** (below) |
| Target user | ✅ Multi-cat households first, then pet households, then general scenes (the method generalizes) |
| Real users | ✅ None yet |
| How to argue Impact | ✅ Land on the impact on people's everyday lives (§0); technology is the "how". 💡 With no users yet, don't argue from traction. |
| Proactive large model | ✅ Part of the product (§0, §2 Part 1) |
| Meta models | ✅ Use every Meta model that fits (table in §5) |
| What gets retrained | ✅ Both **detection** (YOLO) and **activity**. Activity needs a separate pose/activity CV model, because YOLO can't learn it. |
| Summaries and summarized text memory | ✅ Building it. Written by the large model (interval ❓) |
| Find *any* object | ✅ Not building it. The framework generalizes to any object, and we say so. |
| Demo approach | ✅ Past time window + onboarding walkthrough (§4) |
| Competitors | ✅ Mentioned in the video |

### Problem statement (✅)

**Build for Your World.** We live with the problem (our own cats Qb and Qbo), which is
exactly what this statement asks for: "Solve a problem you know firsthand", and "free up
time and mental space". The generalization story (pets, then any scene) carries the Impact
argument.

---

## 2. System: three parts plus the learning loop

```text
                    ┌──────────────── user: chat / voice ────────────────┐
                    ▼                                                    │
 camera (2 fps) ─► CV model (YOLO + activity) ─► tracker ─► MEMORY ◄─► AGENT (Meta models)
                         ▲                         │                     │
                         │                  hard case?                   │
                         │                         ▼                     │
                         │                  ESCALATION: large model + tools ─► label
                         │                                                 │
                         └──── retrain ◄── dataset ◄── (human confirm ❓) ◄┘
```

### Part 1: Agentic system

The user talks to the system: asks questions, requests summaries. The system also acts on
its own: it asks the user for labels and priorities, and sends alerts.

| | Status |
| --- | --- |
| Tool-calling agent over 3 MCP servers (camera on the Pi, stored images, tracker state) | Exists, pre-hackathon (`mcp_agent`, `tracking_mcp`) |
| Custom labels and zones by conversation (`--labels`) | Exists, pre-hackathon (PR #4) |
| Runs on Meta models instead of OpenAI | **New**, required (Stage 1 pass/fail) |
| Summaries ("what did the cats do today") over the memory | ✅ New |
| **Proactive agent**: asks the user to confirm or label hard cases; asks which scenes, objects, people or pets matter and prioritizes them | ✅ New. Exact behaviors 💡 |
| Proactive alerts ("tell me if Qbo hasn't eaten by 10 am") | 💡 New |
| Ask by voice (Muse Voice Transcribe) | 💡 New, worth considering |
| Find *any* object, not only cats (SAM 3.1 open-vocabulary segmentation) | ✅ Not building. The framework generalizes to it. |

### Part 2: Memory

What the agent can pull when answering. For example, "which cat is at this spot right
now?" pulls the last 5 minutes of detections plus each cat's state (moving or resting).

| Kind | Source | Status |
| --- | --- | --- |
| Short-term state | tracker buffer (~15 s), live status, current periods, last seen, last movement (`tracking_mcp`) | Exists |
| Long-term structured | MySQL `period`, `revised_detection`, daily/hourly stats (`daily_job.py`) | Exists |
| Images | image server (SSD) | Exists |
| Static text knowledge | cat profiles, reference photos, labels, zones (`agent_data/`) | Exists (PR #4) |
| **Summarized text memory** | written by the large model, retrievable by time / cat / text | ✅ **New**. Interval ❓ (per period? hourly? daily?), storage and retrieval ❓ |

💡 Avoid health claims in summaries ("ate less than usual" is fine; "may be sick" is not).
AI medical advice is on the organizers' banned list.

### Part 3: Escalation and continuous learning

Normally the CV model runs at about 2 fps with detections on every frame. In practice
there are **gaps** (missed detections) and **jumps** (boxes that drift off). When the
scene is hard, we escalate to a large model that uses tools to find the right label. Those
labels become extra training data for the CV models.

| | Status |
| --- | --- |
| Escalation tasks: identity check, trajectory physics check, activity labeling; rules first, then the model; image budgets; `send_to_human` | Exists, pre-hackathon (PR #4, not merged) |
| **Trigger**: what counts as a hard scene, decided automatically from the stream | **New**. Definition ❓, Muyang decides (gap length, jump size, identity flips, low confidence, unseen objects?) |
| **Labels → dataset**: agent output into a training set (today it only goes to jsonl logs; `flaskr/scr/auto_label.py` does not read it) | **New** |
| **Retrain detection** (YOLO) | **New**. Existing `auto_train.py` runs on AWS EC2 |
| **Retrain activity** (separate pose/activity model) | **New** |
| Human confirmation before training (this is where the proactive agent asks the user) | ❓ Muyang decides: always, never, or only on low agreement? |
| Train locally on the user's device, or upload to the cloud | ❓ Muyang decides; trade-off below |

**Local vs cloud (❓ Muyang decides; the analysis below is 💡):**

| | On-device / home box | Cloud (current EC2 path) |
| --- | --- | --- |
| Privacy story | Strong: footage never leaves home | Weaker |
| Bonus fit | Best Open Weight ($10K) if Muse Spark Open / Glimmer also runs locally | none |
| Engineering in 9 days | Harder (compute on the Pi / box) | Existing pipeline to adapt |

---

## 3. Experiments (💡)

We have no labeled data yet, so the claim "it gets better" needs an experiment. Smallest
version that proves it:

1. **Eval set.** Hand-label a few hundred frames from our own cameras, covering hard
   cases on purpose (gaps, jumps, both cats close, low light).
2. **Baseline.** Current YOLO plus tracker on the eval set: identity error rate,
   gap/jump rate, and activity accuracy (once the activity model exists).
3. **Loops.** Run escalation on a separate window, build the dataset from its labels,
   retrain, and re-measure on the same eval set. Do 2-3 rounds if time allows, so the
   onboarding demo (§4) can show the improvement step by step with real numbers.
4. **Report** before and after in the README and the video.

❓ Metrics and eval-set size: who decides? (💡 Muyang, who knows the tracker failure modes
best)

---

## 4. Demo (✅ approach)

Judges score whether the product "holds up live", but the camera is a Pi in our home. So
the demo runs on a recorded stretch of time:

1. **A past time window, with real events.** What the user can try:
   - **Insights**: ask the large model custom questions about that window and get
     answers.
   - **Tracking and labels** over the whole window.
2. **Onboarding walkthrough**: from a fresh install to fully labeled. It shows how the
   model interacts with the user to get labels (the proactive agent, §2 Part 1), and how
   its performance improves step by step.

The walkthrough is pre-determined, but it comes from real data and real model improvement
(the §3 runs), and we say so.

- 💡 Host it so anyone can try it from a page or from the repo.
- 💡 **3-minute video outline** (first draft; the script will keep changing):
  1. 0:00-0:20 Hook: two near-identical cats, and a camera that mixes them up.
  2. 0:20-1:00 Talk to it: ask, find, summarize, by voice.
  3. 1:00-2:00 The loop: CV fails → escalation fixes it → retrain → before/after numbers.
  4. 2:00-2:40 Customize: "label the cat when it sits in the tent". Generalizes beyond cats.
  5. 2:40-3:00 Competitors vs us; Meta models used; what we built this week.
- ❓ Are we OK with home footage being public (repo, video, X post)?

---

## 5. Meta models (✅ use where they fit; mapping is 💡)

| Role | Model |
| --- | --- |
| Agent reasoning, vision, tool calls; escalation teacher | Muse Spark (Meta Model API) |
| Masks/boxes for training data (open-vocabulary finding is out of scope, §2) | SAM 3.1 |
| Voice questions (if we build voice) | Muse Voice Transcribe |
| Creative output, e.g. a daily "cat diary" comic (Best Creative Build) | Muse Image / Muse Wonder |
| Local or partner-hosted model (Best Open Weight; privacy) | Muse Spark Open / Muse Glimmer (Baseten id `meta-models/muse-spark-open-1026`) |
| Coding | Muse Code |

❓ Nobody has tested yet whether Muse Spark's tool calling and vision can replace
`gpt-6.1-sol`. 💡 Test this first (day 1).

---

## 6. Built before vs during the hackathon (⚠️ disqualification risk)

The rules: "What you present at this Hackathon must have been built during the event."
The explicit allowance to build on existing code is written **for startups**. We are in
the Indie track, so this applies to us more strictly.

| Before Oct 27 (existing) | During Oct 27 - Nov 4 (💡 candidates) |
| --- | --- |
| YOLO pipeline, TrackCorrector / PeriodCorrector | Meta-model port of the agent |
| `tracking_mcp`, `mcp_agent` (OpenAI) | Escalation trigger on the live stream |
| Tracker visualization | Labels → dataset → retrain (detection + activity) |
| PR #4 escalation tasks (Oct 26) | Activity/pose model |
| `auto_label.py`, `auto_train.py` | Summaries and summarized memory |
| | Proactive agent (label requests, priorities, alerts) |
| | Voice (if in scope) |
| | Eval set + before/after experiment |
| | Demo: past window + onboarding walkthrough |

💡 Plan: tag `hackathon-baseline` before building, keep the public repo history starting
from it, and make the README and video say plainly what is new.

---

## 7. Questions

### For Muyang

1. Does the three-part description in §2 match how you see the system?
2. Escalation trigger (your call): which gap/jump patterns are worth escalating, given the
   tracker's real failure modes?
3. Retraining: which detection classes, and which pose/activity model? Is there any usable
   labeled data today?
4. Human confirmation before training (your call): always, never, or only on low
   agreement?
5. Local vs cloud training (your call).
6. Summarized memory: interval, storage, retrieval.
7. Demo: do we have a past time window with enough real events for §4? OK to make footage
   public?
8. Should PR #4 merge before the baseline tag? (It is pre-hackathon either way.)

### For the organizers (Discord `#questions`)

1. **Indie track and pre-existing code**: can we build on our own earlier code (tracker,
   MCP servers) if we clearly mark it as pre-existing and only present the new work?
2. **Indie track and a company**: we have a registered company but no product users. Is
   Indie the right track?
3. **Hardware demos**: is a recorded-data replay acceptable when the live system depends
   on a home camera?

---

## 8. Timeline (💡 rough)

| Days (PT) | Work |
| --- | --- |
| Mon Oct 26 | Kickoff; claim credits; ask the organizers the §7 questions; tag baseline |
| Tue Oct 27 - Wed Oct 28 | Port the agent to Muse Spark; test tool calling and vision; start the eval set; define the trigger |
| Thu Oct 29 - Sat Oct 31 | Trigger + labels → dataset → retrain (detection); activity model; summarized memory; proactive label requests |
| Sun Nov 1 - Mon Nov 2 | Before/after experiment (2-3 rounds); proactive alerts and voice if in; build the demo |
| Tue Nov 3 | Freeze features; README; record the video |
| Wed Nov 4 | Submit by noon PT (deadline 6 PM PT); post on X |
