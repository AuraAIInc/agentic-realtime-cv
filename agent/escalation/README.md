# escalation

Hard scenes go to a large model with tools; its verified labels become training data
(Project Plan, Part 3: Escalation and continuous learning).

Already exists: the escalation tasks in `../supervisor_agent/tasks.py` (identity check,
trajectory physics check, activity labels; rules first, then the model; image budgets;
`send_to_human`).

New here: the **trigger** that decides automatically from the live stream what counts as a
hard scene (gap length, jump size, identity flips, low confidence, unseen objects), and the
path from escalation labels to the retraining dataset.

## Trigger vocabulary

`Escalation` / `emit`, the pure helpers (`top_two_margin`, `box_distance`,
`box_iou`, `max_displacement`, `window_is_static`), and `EscalationConfig`
live in this package (`escalation.py`, `escalation_config.py`, re-exported
from `__init__.py`). The tracking correctors import from here. Behavior is
identical to the legacy `existing_work_before_hackathon/model/` sources.

Every escalation is numbered run-wide in emission order, continuous across both
scopes: `seq` 1, 2, 3… and display name `E<seq>-<code>` (e.g. `E3-T5`,
`E7-P1`). The name is printed on the stdout JSON line, stored in
`escalations.jsonl`, and drawn below each red line/span when visualization is
enabled. Unnumbered escalations (legacy records) fall back to the bare code.

Frame codes (TrackCorrector):

| Code | Name | Meaning |
|------|------|---------|
| T1 | multi_match_relabel | detection fits 2+ cats' histories; first match won |
| T2 | weak_rule_relabel | relabeled by center distance alone, no IoU support |
| T3 | duplicate_tie | duplicate boxes killed with near-tied confidence |
| T4 | jump_reset | new box missed the base, whole buffer wiped |
| T5 | target_fallback | no overlap with base and nothing confident; guessed |
| T6 | teleport_split | committed box jumped > 300 px; period split |
| T7 | long_synthesis | committed a long run of interpolated-only frames |

Period codes (PeriodCorrector):

| Code | Name | Meaning |
|------|------|---------|
| P1 | close_vote | top-two cat scores near-tied |
| P2 | weak_winner | winner barely above min score, or short (x0.2) period |
| P3 | conflict_reassign | top cat taken by an overlapping period; got 2nd choice |
| P4 | long_unknown | assigned no cat but the period is long |
| P5 | low_quality | winner's mean conf low or mostly missing frames |
| P6 | merge_candidate | period may connect to the same cat's previous period |
| P7 | static_link | both ends static and overlapping; sleeping cat split by a long gap |
| P8 | tracker_vote_split | tracker's frame labels disagree with the period vote winner |

Heartbeat code (TrackingRun, period scope over the quiet window):

| Code | Name | Meaning |
|------|------|---------|
| H1 | quiet_spot_check | no escalation for `heartbeat_seconds` (default 300 s); routine audit |

The heartbeat fires at most once per quiet stretch: when stream time passes
`heartbeat_seconds` with no escalation since the window start, and once at
`finish()` when the whole run stayed quiet (`heartbeat_at_end`, default on).
Set `ESCALATION_HEARTBEAT_SECONDS=0` to disable the periodic check and
`ESCALATION_HEARTBEAT_AT_END=0` to disable the end-of-run fallback. `H1`
verdicts (`confirm` / `flag_issue`) never rewrite rows.

Tests: `tests/test_escalation_points.py` (run `pytest agent/escalation/tests`
from the repo root). `test_explicit_config_wins_over_env` stays with the
tracking phase since it needs `TrackCorrector`.
