# Corrector escalation points

When the tracker or the period voter faces a choice it cannot make confidently,
it flags that exact frame (or period) for human/AI review instead of silently
guessing. This file describes the strategy behind every flag and the threshold
that triggers it. All knobs live in
[`escalation_config.py`](escalation_config.py); the record format and collectors
live in [`escalation.py`](escalation.py).

## Principle

A point is escalated only when the corrector itself branches on weak evidence —
a near-tie, a fallback rule, a drastic reset. Confident decisions are never
flagged, no matter how low the raw YOLO confidence is:

- a relabel backed by strong box overlap → not flagged (`T1`/`T2` stay quiet);
- a duplicate kill with a clear winner → not flagged (`T3` stays quiet);
- an empty buffer, or a period closing after the 60 s no-detection timeout →
  not flagged (timeouts are not judgment calls);
- a short unknown blip → not flagged (`P4` requires a long track).

Every flag is one JSON line on stdout (greppable, agent-consumable) and one
record in the corrector's `escalations` list (drawn by `run_tracking_viz.py`).

```json
{"escalation": true, "time": "2025-12-18T15:03:20.123456", "end_time": null,
 "scope": "frame", "code": "T3",
 "reason": "duplicate boxes (IoU 1.00) kept cat 1 over cat 0 on a 0.02 confidence margin",
 "cat_id": 1, "period_id": null,
 "metrics": {"killed_cat_id": 0, "iou": 1.0, "conf_margin": 0.02}}
```

`scope` is `frame` (one image, drawn as a red vertical line) or `period` (a
`time`–`end_time` span, drawn as a light red band).

## Frame points (TrackCorrector, `track_corrector.py`)

| Code | Signal | Why it is a difficult choice | Threshold |
|---|---|---|---|
| `T1` | A relabeled detection fits **2 or more** cats' histories; first match wins | The winner depends on cat iteration order, not evidence. The label swap may belong to another cat. | none (count ≥ 2) |
| `T2` | Relabeled by **center distance alone** (`< 160 px`), with no IoU support | The weakest of the three match rules: proximity without any box overlap. A nearby passer-by steals the label. | none (rule name) |
| `T3` | Duplicate boxes (IoU > 0.7) killed on a **near-tied** confidence | Could be one animal labeled twice (kill is right) or two overlapping animals (kill deletes a real cat). | `duplicate_tie_margin` = 0.15 |
| `T4` | New box misses the base (IoU < 0.2) → **whole buffer wiped**, period closed | Either the old track went stale (wipe is right) or the new box is a misdetection that just destroyed a good track. | none (the wipe itself) |
| `T5` | Target is the **fallback guess** (nothing overlaps the base, nothing > 0.5 conf), or no target with an open period | Interpolation anchors on a guess, so every box between base and target is built on it. | none (fallback path) |
| `T6` | Committed box jumps **> 300 px** → period split, interpolated tail retracted | A real teleport (different animal) or one bad box that split a good period and deleted its tail. | none (the split itself; distance + box conf in metrics) |
| `T7` | Committed run of **interpolated-only frames** (no real detection) grows long | The track is hallucinated through a gap; the longer the run, the less it is worth trusting. Fires once per run. | `synth_gap_seconds` = 5.0 |

## Period points (PeriodCorrector, `period_corrector.py`)

| Code | Signal | Why it is a difficult choice | Threshold |
|---|---|---|---|
| `P1` | **Top-two vote margin** below threshold | The whole-period identity vote is a coin flip between two cats. | `vote_margin` = 2.0 |
| `P2` | Winner scores below threshold, or won with **< 10 original frames** (the ×0.2 penalty band) | A barely-above-minimum winner, or a verdict built on a handful of frames. | `weak_winner_score` = 3.0, `weak_winner_min_length` = 10 |
| `P3` | Top cat **taken by an overlapping period**; assigned second choice | The assignment is forced by the no-overlap constraint, not earned by evidence. | none (final ≠ original top) |
| `P4` | Assigned **no cat** but the period is long | A long track nobody could label is worth a look; a 3-frame blip is not. | `unknown_min_seconds` = 10.0 **or** `unknown_min_frames` = 30 |
| `P5` | Winner's **mean YOLO conf** low, or its **original-frame rate** low (rest interpolated/missing) | The verdict rests on weak or mostly synthesized evidence. | `low_mean_conf` = 0.55, `low_detect_rate` = 0.3 |
| `P6` | New period starts soon after the **same cat's previous period** ended, nearby | Likely one visit split by a gap rather than two visits. Shades the gap between them. | `merge_gap_seconds` = 60.0, `merge_max_distance` = 300.0 |
| `P7` | New period **starts static** where the previous period **ended static**, boxes largely overlapping, gap up to 10 min | A sleeping cat: weak/missed detections split one motionless stay into two periods. Checked at close (the head window is only complete then); shades the gap like `P6`. May co-fire with `P6` on short gaps — the two reasons are complementary. | `static_window_seconds` = 5.0, `static_max_px` = 15.0, `static_overlap_iou` = 0.5, `static_link_max_gap` = 600.0 |

Mirrors of tracker constants are deliberate: `merge_gap_seconds` matches the
tracker's 60 s `max_undetected_gap`, `merge_max_distance` matches its 300 px
teleport distance, and `weak_winner_min_length` matches the scorer's own
`detect_length < 10` ×0.2 rule. If those constants ever change, update the
escalation defaults to match.

## Tuning

1. **Environment (recommended for the live server):** set any `ESCALATION_*`
   variable in `.env` (see the commented list in the repo-root
   [`.env.example`](../../../.env.example)) and restart the ingest server.
   Names are the upper-snake form of the config field, e.g.
   `ESCALATION_VOTE_MARGIN=3.0`.
2. **Constructor (for scripts/tests):**
   `TrackCorrector(view_id, escalation_config=EscalationConfig(vote_margin=3.0))`.

Start by lowering noise, not raising sensitivity: if a code fires on stretches
you always agree with, tighten its threshold; the `metrics` object on every
line shows how far each case was from the boundary.

## Files

| File | Role |
|---|---|
| [`escalation_config.py`](escalation_config.py) | Every threshold + `ESCALATION_*` env overrides (this file is the single source of truth) |
| [`escalation.py`](escalation.py) | `Escalation` record, JSON serialization, emit-to-stdout collector |
| [`track_corrector.py`](track_corrector.py) | `T1`–`T7` hooks at the exact branch points; red-line/span drawing in `plot_2mins` |
| [`period_corrector.py`](period_corrector.py) | `P1`–`P6` hooks at vote/assign/close/open |
| [`test_escalation.py`](test_escalation.py) | Suite (no MySQL): `../.venv/bin/python -m pytest model/test_escalation.py -q` |
| [`../run_tracking_viz.py`](../run_tracking_viz.py) | Passes both correctors' `escalations` lists into the plot |
