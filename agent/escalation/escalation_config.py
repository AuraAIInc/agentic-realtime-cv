"""All escalation knobs in one place.

Every threshold below has a code default and an ``ESCALATION_*`` environment
override (read by ``from_env``). The correctors use ``from_env()`` unless the
caller passes an explicit ``EscalationConfig``, so tuning is a ``.env`` edit
plus a restart — no code change. See README.md next to this file for what each
knob means and why the default was chosen.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw is not None else default


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw is not None else default


@dataclass(frozen=True)
class EscalationConfig:
    duplicate_tie_margin: float = 0.15  # T3: ESCALATION_DUPLICATE_TIE_MARGIN
    synth_gap_seconds: float = 5.0  # T7: ESCALATION_SYNTH_GAP_SECONDS
    vote_margin: float = 2.0  # P1: ESCALATION_VOTE_MARGIN
    weak_winner_score: float = 3.0  # P2: ESCALATION_WEAK_WINNER_SCORE
    weak_winner_min_length: int = 10  # P2: ESCALATION_WEAK_WINNER_MIN_LENGTH
    unknown_min_seconds: float = 10.0  # P4: ESCALATION_UNKNOWN_MIN_SECONDS
    unknown_min_frames: int = 30  # P4: ESCALATION_UNKNOWN_MIN_FRAMES
    low_mean_conf: float = 0.55  # P5: ESCALATION_LOW_MEAN_CONF
    low_detect_rate: float = 0.3  # P5: ESCALATION_LOW_DETECT_RATE
    merge_gap_seconds: float = 60.0  # P6: ESCALATION_MERGE_GAP_SECONDS
    merge_max_distance: float = 300.0  # P6: ESCALATION_MERGE_MAX_DISTANCE
    static_window_seconds: float = 5.0  # P7: ESCALATION_STATIC_WINDOW_SECONDS
    static_max_px: float = 15.0  # P7: ESCALATION_STATIC_MAX_PX
    static_overlap_iou: float = 0.5  # P7: ESCALATION_STATIC_OVERLAP_IOU
    static_link_max_gap: float = 600.0  # P7: ESCALATION_STATIC_LINK_MAX_GAP (10 min)

    @classmethod
    def from_env(cls) -> "EscalationConfig":
        return cls(
            duplicate_tie_margin=_float("ESCALATION_DUPLICATE_TIE_MARGIN", cls.duplicate_tie_margin),
            synth_gap_seconds=_float("ESCALATION_SYNTH_GAP_SECONDS", cls.synth_gap_seconds),
            vote_margin=_float("ESCALATION_VOTE_MARGIN", cls.vote_margin),
            weak_winner_score=_float("ESCALATION_WEAK_WINNER_SCORE", cls.weak_winner_score),
            weak_winner_min_length=_int("ESCALATION_WEAK_WINNER_MIN_LENGTH", cls.weak_winner_min_length),
            unknown_min_seconds=_float("ESCALATION_UNKNOWN_MIN_SECONDS", cls.unknown_min_seconds),
            unknown_min_frames=_int("ESCALATION_UNKNOWN_MIN_FRAMES", cls.unknown_min_frames),
            low_mean_conf=_float("ESCALATION_LOW_MEAN_CONF", cls.low_mean_conf),
            low_detect_rate=_float("ESCALATION_LOW_DETECT_RATE", cls.low_detect_rate),
            merge_gap_seconds=_float("ESCALATION_MERGE_GAP_SECONDS", cls.merge_gap_seconds),
            merge_max_distance=_float("ESCALATION_MERGE_MAX_DISTANCE", cls.merge_max_distance),
            static_window_seconds=_float("ESCALATION_STATIC_WINDOW_SECONDS", cls.static_window_seconds),
            static_max_px=_float("ESCALATION_STATIC_MAX_PX", cls.static_max_px),
            static_overlap_iou=_float("ESCALATION_STATIC_OVERLAP_IOU", cls.static_overlap_iou),
            static_link_max_gap=_float("ESCALATION_STATIC_LINK_MAX_GAP", cls.static_link_max_gap),
        )
