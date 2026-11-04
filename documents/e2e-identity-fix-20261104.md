# Why the end-to-end flow mislabeled the cats, and what fixed it (2026-11-04)

Branch `fix-e2e-identity-vs-gt`. Three windows, replayed from read-only MySQL through the
full flow (correctors → escalations → Muse Spark review → apply → memory):

| Hour | Window | Ground truth |
| --- | --- | --- |
| 10-28 16 | 16:33–16:41, 1,319 frames | `labels/` (dense, both cats) |
| 10-25 04 | 04:13–04:39, 4,331 frames | `labels/` (validated only 04:19–04:37) |
| 10-26 02 | 02:00–02:31, 5,320 frames | `labels/` |

All scores are class-aware F1 at IoU 0.5 (`evaluation/evaluate.py`, right cat + box overlap).

## Results

**As given** = scored against `labels/` unchanged. **Audited** = with the documented
corrections in `evaluation/gt_audit.yaml` (see [Ground-truth problems](#ground-truth-problems)).

| Hour | Stage | Baseline (as given) | Final (as given) | Baseline (audited) | Final (audited) |
| --- | --- | --- | --- | --- | --- |
| 10-28 | tracker (revised) | 0.110 | **0.535** | 0.110 | **0.535** |
| 10-28 | end to end (validated) | 0.582 | **0.619** | 0.582 | **0.619** |
| 10-26 | tracker | 0.447 | 0.210 | 0.122 | **0.591** |
| 10-26 | end to end | 0.124 | 0.174 | 0.459 | **0.643** |
| 10-25 | tracker | 0.342 | 0.099 | 0.082 | 0.108 |
| 10-25 | end to end | 0.079 | 0.079 | 0.147 | 0.144 |

Class accuracy on matched boxes, end to end (audited):

| Hour | Baseline | Final | Note |
| --- | --- | --- | --- |
| 10-28 | 0.865 | **0.922** | |
| 10-26 | 0.935 | 0.935 | its F1 gain comes from removing ~1,000 ball false positives |
| 10-25 | 0.945 | 0.889 | |

**10-25 did not improve end to end.** Its three final-code runs scored 0.144, 0.116 and
0.144, against a 0.147 baseline. Its validated ground truth is small (536 boxes) and misses
many visible cats. The remaining identity errors there come from two cats sharing one label
(see "Checked and rejected" below).

Final = iteration 3 (`*_20261104-031539-*`). The two earlier runs differ only by the
apply-id guard and the P6 prompt, and land at the same place: 10-28 0.619 / 0.619,
10-26 0.638 (re-applied) / 0.639 (audited). In iteration 3 **no verdict broke a
single row on any hour** (`evaluation/attribute.py`). At baseline, two verdicts alone broke
682 and 607 rows. Validation cost per run is unchanged (about $1.4 / $3.9 / $3.9).

**Why "as given" falls on 10-25 and 10-26:** on those hours `labels/` calls the ball a cat
and calls Qbo "Qb". The final pipeline stops doing both, so it disagrees with those labels.
The table keeps both columns so nothing is hidden.

**What limits 10-28 now is recall, not identity.** 1,493 of its 2,466 GT boxes were never
boxed by the detector: 871 are Qbo inside the cat house and 464 are Qb in the bed during
detector dropouts. The review step can relabel, drop or interpolate, but it can't invent
sightings, so recall stays near 0.47.

## Root causes found and fixed

### 1. The exercise ball becomes "Qbo" and steals Qbo's identity (decoy)

The back-wall ball is detected as Qbo for minutes at a time: 1,016 boxes on 10-26, 660 on
10-25, and the 26-minute incident on 11-01. Its box barely moves (p90 1.0 px), but stillness
can't separate it from a sleeping cat (Qb in the bed: p90 2.8 px). Its **place** does
separate it: the median box is the same `[345–351, 46–47, 407–408, 104–106]` in all three
incidents.

What it broke:
- 10-26: the ball visit claimed cat 26, so PeriodCorrector's one-cat-per-time rule gave the
  real Qbo visit cat 25 (P3).
- 10-25: the reviewer looked at dark frames, said "stationary back cat is Qb", and
  reassigned the whole ball visit to Qb. That one `E45-P5 reassign` touched 1,277 rows.
- 1,168 false-positive ball rows passed straight through on 10-26.

**Fix:** `zones.yaml` gains `kind: decoy`; `agent/tracking/decoys.py` drops detections
overlapping a decoy zone at IoU ≥ 0.6 at ingest. A cat lying beside the ball gives a wider
box (IoU 0.3–0.5 against the ball alone) and is kept. The agent's zone tools accept the new
kind. Counted in `run.json` as `detections_decoy`.

### 2. Continuity hijack: the tracker files Qb under Qbo for 772 frames (10-28)

TrackCorrector keeps a detection under its own label only if it overlaps that cat's own
buffer; otherwise an overlap with another cat's last box (IoU > 0.2) relabels it to that
cat. When Qb settles in the bed where the "Qbo" track last was, every Qb box overlaps the
Qbo track and gets relabeled. Qb's buffer never refills, so the hijack feeds itself: 762
rows broken in revised output.

The pre-hackathon code had a guard for exactly this case, `check_buffer_label()`. It is
commented out, and it counts `b[6]` (confidence) where it should count `b[5]` (the camera's
original id).

**Fix:** `TrackCorrector.detector_insists()` vetoes a continuity relabel X→Y once Y's
15-second buffer holds ≥ 5 real boxes the camera called X, and at least twice as many as it
called Y. On 10-28 a single veto fired and tracker class accuracy went from 0.164 to 0.799;
the P8 "continuity hijacked the visit" escalations no longer fire.

The same veto swings 10-26 by about 555 rows. There the tracker had been overriding
consistent 0.82–0.87 "Qbo" detections of a cat with a dark plume tail; `labels/` says Qb.
See [Ground-truth problems](#ground-truth-problems).

### 3. Same-label collision at ingest

`build_img_detected` keeps one box per cat id. When two boxes share an id (the ball and the
real Qbo, both "26"), the last one silently won. It now keeps the more confident box
(`detections_dup_cat`). Replayed DB data had no collisions after the decoy filter, so this
didn't move scores; it matters live.

### 4. Verdicts naming a cat id the view doesn't have

The reviewer sees canonical profile ids (Qb 0, Qbo 2) alongside view-3 tracking ids
(25/26). View 1 uses 0/1. On 10-26 a P6 `different_cats` came back as cat 0, and apply wrote
cat 0 into 13 rows. `apply_ops` now normalizes `chosen_cat_id` against the run's cat ids:
- an unambiguous alias is mapped;
- anything else skips the op with a warning.

### 5. Identity calls without looking at the cat (P6 `different_cats`)

The P5 prompt already requires an appearance-first identity ruling, and on 10-28 that turned
`E2-P5` from "keep Qbo" into "reassign to Qb" (+153 rows). P6 didn't have that rule. In a
confirmation run, `E14-P6` relabeled a smooth white, thin-tailed Qb visit to Qbo because
"another visit already has this cat". It cited no appearance cue (−30 rows on 10-25). The P6
prompt and schema now require naming the new visit's cat from its own frames, quoting the
cues, and keeping the camera's label unless appearance rules it out.

### Checked and ruled out / rejected

- **Memory poisoning.** Long-term memory says "qbo … sleeps in the green bed", which is
  wrong, but the brief attached to the bad 10-28 verdict was neutral ("qb appears in
  period 2…").
- **Keeping far real boxes in `interplate_buffer` (rejected).** On 10-25 at 04:25:41 a
  walking Qbo (camera box `[515,426,643,598]`, conf 0.80) is committed with the back-wall box
  `[351,48,456,109]` of another cat that also carries label 26. When base and target overlap
  (IoU ≥ 0.7), the interpolation overwrites every buffered real box. Leaving real boxes alone
  when they are far from the path moved tracker F1 on 10-25 from 0.108 to 0.129. It cost
  10-28 0.535 → 0.494/0.522 (the overwrite also hides a second same-label cat there) and
  raised escalations from 11 to 17. Not shipped. The real fix is a tracker that can hold two
  boxes under one detector label.

## Ground-truth problems

Found by drawing `labels/` boxes on the frames wherever the pipeline and the labels
disagreed. Evidence images are in `documents/assets/e2e-identity-fix-20261104/`.

| Hour | Problem | Evidence | In the audit |
| --- | --- | --- | --- |
| 10-25 | Before 04:19, `labels/` is byte-identical to `labels_detector_backup/` (raw detector, never validated): 682 frames, all the ball as Qbo | `1025-gt-labels-ball-as-qbo.jpg` | window 04:19–04:38 |
| 10-25 | Many frames with a clearly visible cat have empty labels (04:20–04:26, 04:31–04:35) | `1025-real-cats-with-empty-gt.jpg` | **not corrected**; presence precision is understated |
| 10-25 | 04:35:48–04:36:08: one cat, long coat with dark plume tail (Qbo); labels say Qb | `1026-1025-agent-says-qbo-crops.jpg` | relabel |
| 10-26 | 02:05:30–02:09:15: one cat, dark plume tail, detector Qbo at 0.82–0.87, SAM labels 319 Qbo / 203 Qb; `labels/` says Qb for all 619 | `1026-qbo-dark-plume-tail-gt-says-qb.jpg` | relabel |
| 10-26 | Dark-IR frames with a cat on the rug and empty labels (02:11) | `1026-real-cats-with-empty-gt.jpg` | **not corrected** |

**Bias warning:** I only inspected frames where the pipeline and the labels disagree. Label
errors the pipeline shares are invisible to this method, so the audited scores lean
optimistic. The as-given column is the conservative number.

## Reproduce

```bash
# full flow (as run here)
.venv/bin/python -m agent.main --view-id 3 --start 2026-10-28-16-33 --end 2026-10-28-16-40 \
  --replay-db --validate --memory --plot --report
# score, as given and audited
.venv/bin/python evaluation/evaluate.py --run output/tracking_runs/<run> \
  --hour image_server/retrieved_images/2026-10-28-16 --gt-source labels --iou 0.5
.venv/bin/python evaluation/evaluate.py ... --audit evaluation/gt_audit.yaml
# which verdict fixed or broke which rows
.venv/bin/python evaluation/attribute.py --run <run> --hour <hour> --audit evaluation/gt_audit.yaml
# re-apply stored verdicts with current apply code (no LLM calls)
.venv/bin/python evaluation/rematerialize.py output/tracking_runs/<run> /tmp/out
```

## Next steps (not done)

1. **Two cats, one detector label.** The tracker holds one box per cat id, so when the
   camera calls both cats "26" one of them is pinned or lost (10-25 04:25). That's the
   largest remaining tracker error.
2. **Detector recall** is now the ceiling on 10-28: Qbo in the cat house and Qb in the bed
   during dropouts. That's the edge-model retraining target.
3. **Relabel the missing cats** in 10-25 and 10-26 `labels/` before using them as training
   ground truth. The `labels/` export inherits the pipeline errors fixed above (the ball,
   Qbo→Qb), so re-exporting from these fixed runs, then a human pass, would help.
4. **Correct the long-term memory text** that says Qbo sleeps in the green bed (it's Qb).
5. **Delete or fix `check_buffer_label()`** (wrong column, commented out) now that
   `detector_insists()` covers its case.
