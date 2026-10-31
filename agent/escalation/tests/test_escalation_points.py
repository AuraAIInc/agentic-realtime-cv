"""Escalation-only trigger vocabulary: JSON format, helpers, config.

Ported from existing_work_before_hackathon/model/test_escalation.py — only the
tests that touch the escalation modules alone (no TrackCorrector/PeriodCorrector).

Run from repo root:
    .venv/bin/python -m pytest agent/escalation/tests -q
"""

from __future__ import annotations

import datetime
import io
import json
import os
from contextlib import redirect_stdout

import numpy as np

from agent.escalation import (
    Escalation,
    EscalationConfig,
    box_distance,
    box_iou,
    emit,
    max_displacement,
    top_two_margin,
    window_is_static,
)

T0 = datetime.datetime(2025, 12, 18, 15, 3, 0)


def _capture(fn, *args, **kwargs):
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = fn(*args, **kwargs)
    return out, buf.getvalue()


# ---------------------------------------------------------------- pure helpers


def test_escalation_json_line_shape():
    esc = Escalation(
        time=T0, scope="frame", code="T3", reason="tie", cat_id=np.int64(1),
        period_id=7, metrics={"margin": np.float64(0.02), "when": T0},
    )
    line = esc.to_json()
    obj = json.loads(line)  # one line, valid JSON
    assert obj["escalation"] is True
    assert obj["code"] == "T3" and obj["scope"] == "frame"
    assert obj["cat_id"] == 1 and obj["period_id"] == 7
    assert obj["metrics"] == {"margin": 0.02, "when": T0.isoformat(timespec="microseconds")}
    assert "\n" not in line


def test_top_two_margin():
    top, margin = top_two_margin([9.2, 8.0])
    assert top == 0 and abs(margin - 1.2) < 1e-9
    top, margin = top_two_margin([5.0, 5.0])
    assert margin == 0.0
    top, margin = top_two_margin([4.0])
    assert (top, margin) == (0, float("inf"))


def test_box_distance():
    assert box_distance([0, 0, 10, 10], [3, 4, 13, 14]) == 5.0


def test_box_iou():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert box_iou([0, 0, 10, 10], [50, 50, 60, 60]) == 0.0
    assert abs(box_iou([0, 0, 10, 10], [5, 5, 15, 15]) - 36 / 206) < 1e-9


def test_window_static():
    assert max_displacement([[0, 0, 10, 10]]) is None
    assert window_is_static([[0, 0, 10, 10]], 15.0) is False  # one box proves nothing
    assert window_is_static([[0, 0, 10, 10], [2, 1, 12, 11]], 15.0) is True
    assert window_is_static([[0, 0, 10, 10], [100, 100, 110, 110]], 15.0) is False
    assert max_displacement([[0, 0, 10, 10], [3, 4, 13, 14]]) == 5.0


def test_emit_appends_and_prints_json():
    col = []
    esc, printed = _capture(emit, col, Escalation(time=T0, scope="period", code="P1", reason="close"))
    assert col == [esc]
    assert json.loads(printed.strip())["code"] == "P1"


def _with_env(vars):
    old = {k: os.environ.get(k) for k in vars}
    os.environ.update(vars)
    return old


def _restore_env(old):
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_config_defaults():
    cfg = EscalationConfig()
    assert cfg.vote_margin == 2.0
    assert cfg.weak_winner_score == 3.0 and cfg.weak_winner_min_length == 10
    assert cfg.merge_gap_seconds == 60.0 and cfg.merge_max_distance == 300.0
    assert cfg.static_window_seconds == 5.0 and cfg.static_max_px == 15.0
    assert cfg.static_overlap_iou == 0.5 and cfg.static_link_max_gap == 600.0


def test_config_from_env_overrides():
    old = _with_env({"ESCALATION_VOTE_MARGIN": "3.5", "ESCALATION_UNKNOWN_MIN_FRAMES": "12"})
    try:
        cfg = EscalationConfig.from_env()
    finally:
        _restore_env(old)
    assert cfg.vote_margin == 3.5
    assert cfg.unknown_min_frames == 12 and isinstance(cfg.unknown_min_frames, int)
    assert cfg.synth_gap_seconds == 5.0  # untouched knobs keep defaults
