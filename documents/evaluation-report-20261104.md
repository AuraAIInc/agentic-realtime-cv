# Evaluation report: three replay_db experiment runs (2026-11-04)

Raw camera detections replayed from read-only MySQL through the full pipeline
(correctors → supervisor validation → validated outputs → charts → report),
compared against the labels-mode runs of the same windows (ground-truth YOLO
labels as input). All three runs completed with full artifacts.

| Run | Window | Frames | Raw det. | Periods | Esc. | Val. | Wall |
|---|---|---|---|---|---|---|---|
| A | 10-28 16:33–16:41 | 1,319 | 1,016 (median conf 0.72) | 4 | 12 | 12 | 10.2 min |
| B | 10-25 04:13–04:39 | 4,331 | 1,310 (median conf 0.58) | 19 | 63 | 63 | 32.1 min |
| C | 10-26 02:00–02:31 | 5,320 | 1,805 (median conf 0.75) | 23 | 48 | 48 | 27.7 min |

## 1. Escalation burden follows input noise, not footage length

Escalations per 1k frames: A 9.1, B 14.5, C 9.0 on raw detections — versus
0.8 / 13.3 / 7.1 on ground-truth labels for the same windows. The dense
16:33 window is the extreme case: clean labels produce **zero** corrector
escalations (one routine spot-check), while the raw detector's output on the
identical frames produces 12 (T2 ×3, P8 ×2, P2 ×2, …). The tracker is quiet
when the input is honest; nearly all escalation volume is detector noise
being adjudicated, which is the design working as intended.

Run B is interpolation-heavy: 4,874 revised rows from 1,310 detections —
roughly three of four rows are tracker guesses (vs ~20% in A, ~26% in C).
That is where T7 (×8), T4 (×8) and T6 (×6) concentrate.

## 2. The supervisor overturns about half of what it reviews

Verdict balance (upheld = tracker's decision stands; overturned = rows
rewritten; abstain = flag/unknown/split-unexecutable):

| Run | Upheld | Overturned | Abstain | Cost (val. only) | $/verdict |
|---|---|---|---|---|---|
| A | 5 | 6 | 1 | $0.93 | $0.078 |
| B | 32 | 29 | 2 | $3.51 | $0.056 |
| C | 24 | 23 | 1 | $2.94 | $0.061 |

Total batch cost $9.17 including memory ($7.38 validation + $1.79 memory);
778 images analyzed. A ~50% overturn rate means validation is not a rubber
stamp: e.g. all three T2 nearness relabels in run A were rejected
(`keep_original` ×3), consistent with every earlier run. New verdict
behaviors fired in production: P6 `different_cats` (A ×1), transitive
merges with gap re-interpolation (B: 41 gap-filled rows), split confirms.

Materialization shrank or reshaped every run (revised → validated rows):
A 1,275 → 1,250 (4 periods), B 4,874 → 3,117 (19 → 14 periods),
C 2,445 → 2,333 (23 → 19). Row-level stamps show the work: B carries
1,326 `reassign` + 348 `relabel_to_vote` + 41 `merge` rows.

## 3. Evidence quality recovered; one stale tool remains

Unlike the earlier degraded batch, these runs fetched real frames
(68/293/229 validation images) and rationales cite specific timestamps,
boxes and appearance cues. Zero rationales report servers unavailable.
The recurring wart is `tracking__get_period_detail` serving 2024-stale
data (mentioned in 1/6/9 rationales across the runs); agents noticed and
judged from images instead — the prompt caveat is doing its job, but the
tool itself should be fixed or scoped out of replay validation.

## 4. Issues found

1. **`split_mixed` stamps 1,096 rows in run A.** A P8 "cats mixed at
   different times" verdict is unexecutable (boundaries are free text), so
   rows pass through — but they inherit verdict `split_mixed`, making
   row-level verdict distributions misleading (87% of A's validated rows
   carry a verdict that changed nothing). Unexecutable verdicts should
   stamp a distinct marker (e.g. `verdict_note`) instead of the verdict.
2. **Stale period detail in replay** (above) — fix server-side or drop the
   tool from validation in replay mode.
3. **Wall time is validation-bound**: 10–32 min per run, dominated by
   sequential LLM review turns (median run ≈ 60 s). Fine for offline
   evaluation; live mode needs the batching headroom watched as escalation
   rates rise with detector noise.
4. **Memory adds ~24% on top of validation cost** ($1.79 of $9.17) — worth
   it for the diary, but it should be a visible line item in run budgets.

## 5. Recommendations

- Fix or scope-out `tracking__get_period_detail` for replays; re-run B
  afterward to measure verdict changes with trustworthy detail data.
- Stamp unexecutable verdicts (`split_mixed`, visit splits) as notes, not
  row verdicts, so validated-row statistics stay honest.
- Attack run B's class of input (sparse, low-conf night detections):
  63 escalations and ~73% interpolated rows say the edge detector, not the
  reviewer, is the bottleneck — this is the fine-tuning target the
  validated ground truth now feeds.
- Keep the labels-mode replay as the regression harness: it isolates
  tracker/validator behavior from detector noise at ~1/10 the wall time.

*Method note: verdict classes follow each code's proposal semantics (see
per-code prompts); counts come from `run.json`, `validations.jsonl` and
`validated.jsonl` of the three runs. Run B's labels-mode comparison uses a
04:19 start (label span) vs 04:13 here, so its cross-mode numbers are
indicative, not exact.*
