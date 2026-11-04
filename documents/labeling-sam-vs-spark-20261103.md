# Labeling shootout: Muse SAM 3.1 vs Muse Spark 1.3 vs tracked ground truth

Window 2026-10-28 16:33–16:40 (view 3, cats Qb-2 / Qbo-2), 1318 frames.
Combines the SAM labeling session, the Spark labeling session, and the
head-to-head evaluation against validated tracking ground truth.

Sources:
- [SAM session](../image_server/grounding_truth/reports/session-20261103-002358.md)
  (+ [JSON](../image_server/grounding_truth/reports/session-20261103-002358.json))
- [Spark session](../image_server/grounding_truth/reports/session-spark-20261103-004154.md)
  (+ [JSON](../image_server/grounding_truth/reports/session-spark-20261103-004154.json))
- [Evaluation](../image_server/grounding_truth/reports/eval-20261103-223846.md)
  (+ [JSON](../image_server/grounding_truth/reports/eval-20261103-223846.json))

## TL;DR

| | SAM 3.1 pipeline (`muse_labels`) | Spark 1.3 (`spark_labels`) |
|---|---|---|
| Pipeline | 2 stages: SAM proposes, Spark classifies | 1 stage: single VLM call detects + identifies |
| Wall time (10 workers) | 66 min (**3.0 s/img**) | 99 min (**4.5 s/img**, timeout-inflated) |
| Cost (Standard pricing) | **$24.65 ($0.0187/img)** | **~$25.61 ($0.0195/img)** |
| Coverage | 1318/1318 (100%) | 1210/1318 (108 timeouts never retried) |
| F1 @ IoU 0.5 | **0.794** (P 0.936 / R 0.689) | **0.612** (P 0.656 / R 0.574) |
| Qb recall / Qbo recall | 0.977 / 0.380 | 0.821 / 0.310 |
| Mean IoU on matches | 1.000 (see circularity note) | 0.737 |

> **Pricing basis.** The Spark session was billed $1.25 on the Contributor
> tier, but this report prices it at Standard rates ($1.25/$4.25 per 1M
> tokens, ~$25.61). In real applications the images are customers' real
> camera views, so the Contributor tier — which grants Meta permission to
> train future models on your footage — is not an option for this
> experiment. All Spark cost figures below are Standard unless marked
> "as billed".

Verdict: the SAM pipeline wins on accuracy, geometry, coverage, speed, and
cost (~4% cheaper at equal pricing). Neither model sees Qbo reliably; both
miss the sleeping long-haired cat far more than Qb.

## Ground truth and eval method

GT is the validated tracking result
(`output/tracking_runs/view3_20261028-1633_1640_labels_20261103-222656-565341`),
exported to `labels/` as YOLO: **2465 boxes over 1318 frames** (Qb 1276,
Qbo 1189; 1189 frames with both cats, 87 with Qb only, 42 empty). Matching
is greedy same-class IoU ≥ 0.5; cross-class IoU ≥ 0.5 counts as misclassified
(1 FP + 1 FN); missing prediction files count as empty predictions.

Two caveats (see eval report for detail):

1. **Circularity favors SAM.** The tracking run ingested `muse_labels`, so
   SAM boxes the tracker kept verbatim match GT at IoU 1.0 by construction.
   SAM's precision is inflated; Spark is fully independent. SAM's *recall*
   gap remains meaningful: the tracker interpolated ~700 Qbo boxes across
   frames SAM left empty.
2. **42 tail frames** (16:40:44–59) are GT-empty only because the window cut
   tracking off — the ingested stream has boxes there and both predictors
   fire. Excluding them: SAM 0.962/0.689/0.803, Spark 0.678/0.574/0.622.

## 1. Pipeline complexity

**SAM: two stages, two models** (`label_with_sam.py`).
PROPOSE: SAM 3.1 segments generic concepts ("animal", "a cat" — short
concepts ground reliably; prose does not), unioned and IoU-deduplicated per
image (2636 requests → 2564 regions). CLASSIFY: one Spark call per image
with proposed boxes (1318 calls) assigns each region a cat identity from
the full cat-profile descriptions, or rejects it — 748 regions (29%)
rejected, 1816 boxes written. Extra machinery: IoU dedup, aspect-space
validation (mismatched boxes dropped, never rescaled), YOLO reference boxes
as prompt context only. Geometry comes from segmentation masks; identity
comes from the VLM.

**Spark: one stage, one model** (`label_with_spark.py`). A single
`muse-spark-1.3-contributor` call per image detects AND identifies every
cat, constrained by a strict pydantic JSON schema (output always parses),
boxes in 0-1000 scale converted to YOLO. 1208 calls → 2157 boxes. Far less
code and no cross-model plumbing — but geometry is VLM-regressed
coordinates, visibly looser (mean IoU 0.737 on matches vs SAM's
mask-derived boxes).

## 2. Time cost (10 workers each)

| | Wall | Per image | API calls |
|---|---|---|---|
| SAM | 3963 s (66 min) | **3.0 s** | 2636 SAM + 1318 Spark = 3954 |
| Spark | 5934 s (99 min) | **4.5 s** | 1208 |

Spark is slower per image despite fewer calls: each call waits on a long
structured generation (~5.7M output tokens total), and 108 requests burned
full timeouts during a sustained 504 outage (frames 16:38:12–16:39:16)
before failing.
SAM's per-call latency is lower and its lone failure was a transient DNS
blip (recovered; final coverage 100%).

## 3. Money cost (Standard pricing for both)

| | Total | Per image | Tokens (in/out) |
|---|---|---|---|
| SAM pipeline | **$24.65** ($6.59 SAM + $18.06 Spark) | **$0.0187** | 5.06M (1.15M / 3.91M, all Spark; SAM bills per image) |
| Spark 1.3 | **~$25.61** | **~$0.0195** | 6.72M (0.98M / 5.74M) |

Spark burns *more* tokens than the SAM pipeline's classify stage, so at
Standard rates it costs ~4% more per image — roughly $85 per hour of
footage at this frame rate for either approach. (As billed, the Spark
session paid $1.25 on the Contributor tier, but that tier is excluded
from this comparison: customer camera footage cannot be granted for
model training.)

Cost per true positive (@ IoU 0.5): SAM **$0.0145**, Spark **$0.0181**.
Cost per exact-match frame: SAM **$0.046**, Spark **$0.061**.

## 4. Accuracy @ IoU 0.5

| | Boxes (Qb/Qbo) | TP | FP | FN | Miscls. | P | R | F1 | Exact frames |
|---|---|---|---|---|---|---|---|---|---|
| GT | 2465 (1276/1189) | — | — | — | — | — | — | — | — |
| SAM | 1816 (1332/484) | 1699 | 117 | 766 | 68 | 0.936 | 0.689 | **0.794** | 539/1318 |
| Spark | 2158 (1262/896) | 1415 | 743 | 1050 | 132 | 0.656 | 0.574 | **0.612** | 420/1318 |

(Eval counts 2158 Spark boxes on eval frames — one more than the session's
2157 written; the extra box sits in a pre-existing skipped file.)

Per class: both models find Qb (recall 0.977 / 0.821) and lose Qbo (recall
0.380 / 0.310). Qbo — long-haired cream, asleep in the bed most of the
window — is systematically under-detected: SAM drew only 484 Qbo boxes
against 1189 in GT (the tracker interpolated the rest); Spark drew 896 but
only 368 matched (precision 0.411 — it fires Qbo boxes that don't overlap
anything, plus 215 Qb FPs).

Failure-mode split:

- **SAM's errors are misses, not hallucinations**: 117 FP vs 766 FN.
  The 29% Spark-reject rate in stage 2 buys precision at the cost of
  recall — rejected regions include real Qbo instances the classifier
  wouldn't commit to.
- **Spark's errors are both**: 743 FP (6.3x SAM's) and 1050 FN. 216 of its
  FN boxes (21%) sit on the 108 timeout-missing frames — infrastructure,
  not model quality. Covered-only recall rises to **0.629** (F1 0.642),
  still well under SAM.
- Both models fire on all 42 GT-empty tail frames (see caveat 2) — 49 SAM
  / 72 Spark boxes that are GT artifacts more than model errors.

## 5. Reliability

- SAM: 1/1319 failed (transient DNS), recovered — 100% final coverage,
  0 bad lines.
- Spark: 108/1319 failed (sustained HTTP 504s over ~1 min of frames),
  never retried — 91.8% coverage, capping overall recall by ~5.5 points. A
  retry pass would likely recover most of them; nothing in the failure
  pattern suggests per-image difficulty.

## Recommendation

1. **Ship GT with the SAM pipeline.** Best accuracy, tightest boxes, full
   coverage, fastest wall time, Standard-tier privacy. ~$0.019/img.
2. **Attack Qbo recall next**, not the pipeline choice: both models miss
   the sleeping long-haired cat. Cheapest levers are SAM-side — lower the
   stage-2 reject threshold for Qbo-pattern regions, or add a "sleeping
   cat in bed" concept to stage 1 — then re-run this eval to confirm.
3. **No Contributor shortcut for customer footage.** The $0.0009/img
   Contributor price is off the table wherever images show customers'
   real views, since it grants training rights over the footage. At the
   Standard pricing such applications require, Spark costs more than SAM
   per image, per true positive, and per exact-match frame — while
   trailing on every accuracy metric. It has no case here.
4. **Keep this eval wired**: `export_validated_gt.py` + `evaluate_labels.py`
   reproduce every number above from any future run; re-run after prompt
   or threshold changes.
