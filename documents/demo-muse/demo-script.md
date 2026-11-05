# Demo Script — Production Design (3 minutes, 12 slides)

Voice-over: ~325 words at ~1.8 words/s, leaving room to pause on visuals. The first
7 seconds stack three surprises: two near-identical cats, a camera that confuses them,
and a ball it mistook for a cat. Every visual below is a real artifact from this
project — no stock footage, no mock-ups.

**Formats used:** PHOTO (full-bleed real frames) · CHART (measured numbers) ·
DIAGRAM (system flows) · TABLE (results card) · BLOCKS (quote / before-after) ·
VIDEO (boxed timelapse) · TEXT (closing card). Section titles appear only on
S4 and S10 — everywhere else the visual carries the slide.

---

## S1 · 0:00 · PHOTO (full-bleed, no title)

**Asset:** `assets/photo_both_cats.jpg` — real frame 2026-10-28 16:35, both cats
boxed as the tracker saw them; the pink ball sits top-center for the next line.

> These are Qb and Qbo. My camera can't tell them apart. Last week, it decided
> this *ball* was Qbo, for twenty-six minutes.

Production: open on the boxed frame at 100%, then a slow crop-zoom onto the ball.
Narration does the joke; the frame proves both cats really are near-identical.

## S2 · 0:08 · PHOTO (full-bleed)

**Asset:** `assets/photo_agent_view.jpg` — a daytime frame the supervisor agent
actually fetched during validation (one cat on the rug, one in the bed).

> I work all day. I just want to know simple things: did my cats eat, sleep,
> play? Cats hide illness, so small changes matter.

Production: unboxed version of the same view. Keep it quiet — this is the
"ordinary home" beat before any technology appears.

## S3 · 0:20 · CHART

**Asset:** `assets/chart_cost.png` — two-panel bar chart: per-window ($25.61 vs
$0.26) and per-year ($63,000 vs ~$650).

> Why not send every frame to an AI? One camera makes over a hundred and
> seventy thousand frames a day. That's about sixty-three thousand dollars a
> year. A tiny model on a Raspberry Pi is almost free, but it makes mistakes.

Production: bars animate in left-to-right; the near-zero bars are the point —
let them sit for a beat. Numbers are measured (§9 of the project report).

## S4 · 0:40 · DIAGRAM + SECTION TITLE

**Title card text:** "Cheap eyes on every frame — smart brain only on doubt"
**Asset:** `assets/diagram_architecture.png` — camera → doubt → Spark supervisor
→ verdict/memory, with the offline SAM labeling loop underneath.

> So we built both. The Pi watches every frame. When it's unsure, it says
> exactly why. It knows fourteen kinds of doubt, like "I relabeled this cat by
> distance alone."

## S5 · 1:00 · DIAGRAM + QUOTE BLOCK

**Asset:** `assets/diagram_escalation.png`, with the verdict appearing as a
quote block overlaid on the last panel:

> *Thin upright tail, cool white fur. This is Qb, not Qbo.* — **Overturned.**

Narration:

> That doubt becomes a plain-English question for Meta's Muse Spark. The agent
> pulls the frames it needs, compares the cats, and answers with a reason.
> Watch: "Thin upright tail, cool white fur. This is Qb, not Qbo." Overturned.

Production: if time allows, replace the diagram's final panel with a screen
capture of a real agent trace (frames + reasoning) — the trace text is the
evidence; keep the diagram if the capture is noisy.

## S6 · 1:25 · VIDEO

**Asset:** `assets/video_timelapse.mp4` — 25 s boxed timelapse of the full
16:33 window (299 real frames, ground-truth boxes), played at 2× under this beat.

> Related doubts get reviewed together, fetching each image once. Twelve
> questions, three calls. Here it caught the tracker inventing a cat on an
> empty floor, and deleted it.

Production: this is the only true video beat — tracking must *move* to be
believed. Freeze on an empty-floor stretch when "inventing a cat" lands.

## S7 · 1:45 · BLOCKS (before → after)

**On-screen blocks** (typeset, no asset file needed):

```text
BEFORE   {... "cat_id": 26, "verdict": "pass"}
AFTER    {... "cat_id": 25, "validated_by": "E5-T2", "verdict": "keep_original"}
```

> Every verdict rewrites the record, and every change is traceable.

Production: two lines only. The point is auditability — restraint reads as
confidence.

## S8 · 1:55 · PHOTO (full-bleed)

**Asset:** `assets/photo_sleeping_qbo.jpg` — Qbo in the green bed. Honest
detail: a second cat hides inside the tent house at left, unboxed — the kind
of miss the reviewer exists to catch.

> Then memory. Not boxes and timestamps, but a diary: "Qb slept in the green
> bed, four thirty-four to four forty." Over weeks, it learns what normal
> looks like for each cat, in each spot.

## S9 · 2:15 · TABLE (results card)

Render as a clean table, not a graphic file:

| | |
|---|---|
| Doubts in 8 minutes | 12 (under 1% of frames) |
| Cost to review them | $0.26, in 3 joint calls |
| Fleet so far | 124 agent runs · 1,058 tool calls · $9.22 total |
| Frames leaving home | Only the flagged ones |

> The numbers: twelve doubts in eight minutes, less than one percent of
> frames. Twenty-six cents to review them, about a fifth of the cost of
> sending every frame to the AI. And only those frames ever leave my home.

## S10 · 2:35 · CHART + SECTION TITLE

**Title card text:** "Teach the Pi from what the agent learns"
**Asset:** `assets/chart_labeling.png` — SAM 3.1 vs Spark 1.3 labeling
shootout (F1 0.794 vs 0.612).

> Next: catch more errors, run full days, and teach the Pi from what the
> agent learns.

Production: the chart is the roadmap's credibility — SAM labels ground truth,
the loop replays it through validation, and the edge model fine-tunes on what
survives. The annotation (both models lose the sleeping long-haired cat) is
the honest next problem.

## S11 · 2:45 · TEXT CARD (no title)

Three lines, appearing one at a time — an elderly parent alone at home;
wildlife on a trail camera; a factory line watching for rare defects — then:

> Cats are just the start. Any camera watching for rare moments has the same
> gap. Our answer: cheap eyes on every frame, a smart brain only when it
> matters, and a memory that keeps growing. Built this week, on Meta's Muse.

Production: no montage footage exists (and stock would break the all-real
rule) — typographic cards keep the promise that everything shown was real.

---

## Materials manifest

| File | Source | Used in |
|---|---|---|
| `assets/photo_both_cats.jpg` | retrieved frame 5836_2026-10-28-16-35-01, boxes from `muse_labels` | S1 |
| `assets/photo_agent_view.jpg` | frame fetched by the agent in a 2026-11-02 validation trace | S2 |
| `assets/photo_sleeping_qbo.jpg` | retrieved frame 5944_2026-10-28-16-35-41 | S8 |
| `assets/chart_cost.png` | measured window/year costs (project report §9–10) | S3 |
| `assets/chart_funnel.png` | 8-minute window funnel: 1,319 → 1,016 → 12 → 3 | S4 alt / backup |
| `assets/chart_labeling.png` | SAM vs Spark eval @ IoU 0.5 (1,318 frames) | S10 |
| `assets/diagram_architecture.png` | system flow, generated | S4 |
| `assets/diagram_escalation.png` | one-case anatomy, generated | S5 |
| `assets/video_timelapse.mp4` | 299 retrieved frames, GT boxes, 12 fps, 25 s | S6 |

## Production rules

- 16:9 throughout; photos full-bleed; charts/diagrams on white, palette
  navy `#2d3748` / blue `#2b6cb0` (Qb) / orange `#dd6b20` (Qbo) / green `#276749`.
- Every number spoken is measured; every frame shown is real. If a nicer
  visual would require inventing data, use type instead (S7, S9, S11).
- Regenerate charts/diagrams with the generator scripts used for this
  package; re-inspect every output visually before locking.
