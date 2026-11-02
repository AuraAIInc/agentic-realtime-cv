"""Escalation trigger vocabulary (migrated from legacy model/ code, Phase 1 of 5)."""

from agent.escalation.escalation import (
    Escalation,
    box_distance,
    box_iou,
    emit,
    esc_label,
    max_displacement,
    top_two_margin,
    window_is_static,
)
from agent.escalation.escalation_config import EscalationConfig
from agent.escalation.prompting import build_prompt
from agent.escalation.review_schema import REVIEW_SCHEMAS, review_model_for, review_schema_json

__all__ = [
    "Escalation",
    "REVIEW_SCHEMAS",
    "build_prompt",
    "emit",
    "review_model_for",
    "review_schema_json",
    "esc_label",
    "EscalationConfig",
    "top_two_margin",
    "box_distance",
    "box_iou",
    "max_displacement",
    "window_is_static",
]
