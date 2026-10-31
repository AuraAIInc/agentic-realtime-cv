"""Escalation trigger vocabulary (migrated from legacy model/ code, Phase 1 of 5)."""

from agent.escalation.escalation import (
    Escalation,
    box_distance,
    box_iou,
    emit,
    max_displacement,
    top_two_margin,
    window_is_static,
)
from agent.escalation.escalation_config import EscalationConfig

__all__ = [
    "Escalation",
    "emit",
    "EscalationConfig",
    "top_two_margin",
    "box_distance",
    "box_iou",
    "max_displacement",
    "window_is_static",
]
