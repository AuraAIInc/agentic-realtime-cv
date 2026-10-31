"""Escalation points: frames and periods the correctors could not decide confidently.

TrackCorrector (frame scope) flags the exact image where it faced a difficult
choice while building a track; PeriodCorrector (period scope) flags whole
periods whose verdict is shaky. Each point is one JSON line on stdout
(machine-readable for the human/AI review trigger) and one record in the
corrector's ``escalations`` list (drawn by run_tracking_viz.py).

Frame codes (TrackCorrector):
    T1 multi_match_relabel  detection fits 2+ cats' histories; first match won
    T2 weak_rule_relabel    relabeled by center distance alone, no IoU support
    T3 duplicate_tie        duplicate boxes killed with near-tied confidence
    T4 jump_reset           new box missed the base, whole buffer wiped
    T5 target_fallback      no overlap with base and nothing confident; guessed
    T6 teleport_split       committed box jumped > 300 px; period split
    T7 long_synthesis       committed a long run of interpolated-only frames

Period codes (PeriodCorrector):
    P1 close_vote           top-two cat scores near-tied
    P2 weak_winner          winner barely above min score, or short (x0.2) period
    P3 conflict_reassign    top cat taken by an overlapping period; got 2nd choice
    P4 long_unknown         assigned no cat but the period is long
    P5 low_quality          winner's mean conf low or mostly missing frames
    P6 merge_candidate      period may connect to the same cat's previous period
    P7 static_link          both ends static and overlapping; sleeping cat split by a long gap
"""

from __future__ import annotations

import datetime
import json
import math
from dataclasses import dataclass, field

from agent.escalation.escalation_config import EscalationConfig  # noqa: F401 (re-exported; thresholds live in escalation_config.py)


@dataclass
class Escalation:
    time: datetime.datetime
    scope: str  # "frame" or "period"
    code: str  # T1..T7, P1..P6
    reason: str
    end_time: datetime.datetime | None = None  # period scope: the flagged span
    cat_id: int | None = None
    period_id: int | None = None
    metrics: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "escalation": True,
                "time": self.time.isoformat(timespec="microseconds"),
                "end_time": self.end_time.isoformat(timespec="microseconds") if self.end_time else None,
                "scope": self.scope,
                "code": self.code,
                "reason": self.reason,
                "cat_id": _jsonable(self.cat_id),
                "period_id": _jsonable(self.period_id),
                "metrics": {k: _jsonable(v) for k, v in self.metrics.items()},
            }
        )


def emit(escalations: list, esc: Escalation) -> Escalation:
    """Record the point for the visualizer and print one JSON line for review."""
    escalations.append(esc)
    print(esc.to_json(), flush=True)
    return esc


def top_two_margin(score) -> tuple[int, float]:
    """(winner index, top1 - top2) of a per-cat score array. Single cat: margin inf."""
    import numpy as np

    score = np.asarray(score, dtype=float)
    top = int(score.argmax())
    if len(score) < 2:
        return top, float("inf")
    second = np.partition(score, -2)[-2]
    return top, float(score[top] - second)


def box_distance(boxA, boxB) -> float:
    """Center distance in px (same formula as TrackCorrector.get_distance)."""
    ax = (boxA[0] + boxA[2]) / 2
    ay = (boxA[1] + boxA[3]) / 2
    bx = (boxB[0] + boxB[2]) / 2
    by = (boxB[1] + boxB[3]) / 2
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def box_iou(boxA, boxB) -> float:
    """Box overlap 0..1 (same formula as TrackCorrector.get_iou)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    areaA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    areaB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return inter / float(areaA + areaB - inter)


def max_displacement(boxes) -> float | None:
    """Largest center movement between any two boxes; None with fewer than 2 boxes."""
    boxes = list(boxes)
    if len(boxes) < 2:
        return None
    centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    return max(math.dist(a, b) for i, a in enumerate(centers) for b in centers[i + 1 :])


def window_is_static(boxes, max_px: float) -> bool:
    """True when 2+ boxes barely move (a single box cannot prove stillness)."""
    disp = max_displacement(boxes)
    return disp is not None and disp < max_px


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, datetime.datetime):
        return v.isoformat(timespec="microseconds")
    mod = v.__class__.__module__
    if mod == "numpy":
        return v.item() if hasattr(v, "item") else list(v)
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v
