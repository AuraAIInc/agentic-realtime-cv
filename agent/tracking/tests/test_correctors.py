"""Corrector behavior tests (ported from legacy model/test_escalation.py).

No MySQL, no stubs: correctors are constructed directly with cat_ids and the
default in-memory sink. Run from the repo root::

    ../.venv/bin/python -m pytest agent/tracking/tests -q
"""

from __future__ import annotations

import datetime
import io
import json
import os
from collections import deque
from contextlib import redirect_stdout

import numpy as np

from agent.escalation import EscalationConfig
from agent.tracking import PeriodCorrector, TrackCorrector

T0 = datetime.datetime(2025, 12, 18, 15, 3, 0)


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


def test_explicit_config_wins_over_env():
    old = _with_env({"ESCALATION_VOTE_MARGIN": "99.0"})
    try:
        default_tc = TrackCorrector(cat_ids=[0, 1])
        explicit_tc = TrackCorrector(cat_ids=[0, 1], escalation_config=EscalationConfig())
    finally:
        _restore_env(old)
    assert default_tc.escalation_config.vote_margin == 99.0
    assert explicit_tc.escalation_config.vote_margin == 2.0


# ---------------------------------------------------------------- TrackCorrector


def test_remove_multi_labeling_reports_tie_kills():
    from agent.tracking.track_corrector import remove_multi_labeling

    box = [10, 10, 100, 100]
    det = {0: box + [0, 0.60], 1: box + [1, 0.62]}
    kills = []
    out = remove_multi_labeling(det, kills)
    assert set(out) == {1}  # lower conf killed, behavior unchanged
    assert len(kills) == 1
    killed, kept, iou, margin = kills[0]
    assert (killed, kept) == (0, 1) and iou == 1.0 and abs(margin - 0.02) < 1e-9

    far = {0: [10, 10, 50, 50, 0, 0.9], 1: [400, 400, 450, 450, 1, 0.9]}
    kills = []
    assert remove_multi_labeling(far, kills) == far and kills == []


def _skeleton_tracker():
    tc = TrackCorrector.__new__(TrackCorrector)
    tc.img_width = 640
    tc.last_detection = [None, None]
    tc.base = [None, None]
    tc.buffer = [deque([]), deque([])]
    return tc


def _row(t, box, orig, conf):
    x1, y1, x2, y2 = box
    return np.array([t, x1, y1, x2, y2, orig, conf, None], dtype=object)


def test_match_rule_classification():
    tc = _skeleton_tracker()
    new_box = [100, 100, 200, 200]

    tc.last_detection[0] = _row(T0, [110, 110, 210, 210], 0, 0.9)  # IoU > 0.2
    assert tc.match_buffer_history_rule(9, {9: new_box + [9, 0.7]}, 0) == "iou_last"
    assert tc.match_buffer_history(9, {9: new_box + [9, 0.7]}, 0) is True

    tc.last_detection[1] = _row(T0, [210, 210, 270, 270], 1, 0.9)  # IoU 0, center 127 px away
    assert tc.match_buffer_history_rule(0, {0: new_box + [0, 0.7]}, 1) == "distance"

    tc.last_detection[1] = None
    tc.base[1] = _row(T0, [500, 400, 560, 460], 1, 0.9)  # far in every way
    tc.buffer[1].append(_row(T0, [102, 102, 198, 198], 1, 0.9))  # IoU > 0.8 with one box
    assert tc.match_buffer_history_rule(0, {0: new_box + [0, 0.7], 1: [0, 0, 1, 1, 1, 0.5]}, 1) == "iou_buffer"

    tc2 = _skeleton_tracker()  # no history at all
    assert tc2.match_buffer_history_rule(0, {0: new_box + [0, 0.7]}, 0) is None


def test_track_duplicate_tie_escalates_with_json_log():
    tc = TrackCorrector(cat_ids=[0, 1])
    box = [10, 10, 100, 100]
    img = {0: box + [0, 0.60], 1: box + [1, 0.62]}
    buf = io.StringIO()
    with redirect_stdout(buf):
        list(tc.process_images(T0, img, 0, 0, 0, 0))
    codes = [e.code for e in tc.escalations]
    assert "T3" in codes
    lines = [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]
    assert any(o.get("code") == "T3" and o["escalation"] is True for o in lines)


def test_track_target_fallback_escalates_on_weak_boxes():
    tc = TrackCorrector(cat_ids=[0, 1])
    buf = io.StringIO()
    with redirect_stdout(buf):
        # all conf < 0.5 and pairwise IoU 0: rule-3 fallback guess from frame 2
        for s in range(3):
            box = [10 + 100 * s, 10, 100 + 100 * s, 100]
            list(tc.process_images(T0 + datetime.timedelta(seconds=s), {0: box + [0, 0.45]}, 0, 0, 0, 0))
    assert "T5" in [e.code for e in tc.escalations]


# ---------------------------------------------------------------- PeriodCorrector


def _det9(t, box, orig, conf, cat, pid):
    x1, y1, x2, y2 = box
    return np.array([t, x1, y1, x2, y2, orig, conf, cat, pid], dtype=object)


def _close8(t, cat):
    return np.array([t, None, None, None, None, None, 0, cat], dtype=object)


def _feed_period(pc, pid, cat, start, n, box, label_fn, conf=0.9, step_s=1.0):
    for i in range(n):
        t = start + datetime.timedelta(seconds=i * step_s)
        list(pc.process_detection(_det9(t, box, label_fn(i), conf, cat, pid)))
    return start + datetime.timedelta(seconds=(n - 1) * step_s)


def test_period_close_vote_escalates():
    pc = PeriodCorrector(cat_ids=[0, 1])
    box = [10, 10, 100, 100]
    last = _feed_period(pc, 101, 0, T0, 24, box, lambda i: i % 2)  # 12 frames each label
    buf = io.StringIO()
    with redirect_stdout(buf):
        list(pc.process_detection(_close8(last + datetime.timedelta(seconds=1), 0)))
    codes = [e.code for e in pc.escalations]
    assert "P1" in codes
    p1 = next(e for e in pc.escalations if e.code == "P1")
    assert p1.scope == "period" and p1.end_time is not None and p1.end_time > p1.time
    assert "P2" not in codes and "P5" not in codes  # strong tie: only the vote is unsure


def test_period_merge_candidate_escalates():
    pc = PeriodCorrector(cat_ids=[0, 1])
    box = [10, 10, 100, 100]
    last = _feed_period(pc, 101, 0, T0, 12, box, lambda i: 0)
    list(pc.process_detection(_close8(last + datetime.timedelta(seconds=1), 0)))
    buf = io.StringIO()
    with redirect_stdout(buf):
        _feed_period(pc, 102, 0, last + datetime.timedelta(seconds=21), 3, box, lambda i: 0)
    p6 = [e for e in pc.escalations if e.code == "P6"]
    assert len(p6) == 1
    assert p6[0].metrics["prev_period_id"] == 101
    assert p6[0].metrics["gap_seconds"] == 21.0  # vs the last detection row, not the close row


def test_period_distant_revisit_does_not_merge():
    pc = PeriodCorrector(cat_ids=[0, 1])
    box = [10, 10, 100, 100]
    last = _feed_period(pc, 101, 0, T0, 12, box, lambda i: 0)
    list(pc.process_detection(_close8(last + datetime.timedelta(seconds=1), 0)))
    buf = io.StringIO()
    with redirect_stdout(buf):
        _feed_period(pc, 102, 0, last + datetime.timedelta(seconds=21), 3, [400, 400, 500, 500], lambda i: 0)
    assert "P6" not in [e.code for e in pc.escalations]


def _feed_static_pair(pc, gap_seconds, box=(10, 10, 100, 100), move_new=0):
    """Two same-box periods `gap_seconds` apart; move_new shifts the new head per frame."""
    last = _feed_period(pc, 401, 0, T0, 8, list(box), lambda i: 0)
    list(pc.process_detection(_close8(last + datetime.timedelta(seconds=1), 0)))
    start2 = last + datetime.timedelta(seconds=gap_seconds)
    for i in range(8):
        b = [box[0] + move_new * i, box[1], box[2] + move_new * i, box[3]]
        list(pc.process_detection(_det9(start2 + datetime.timedelta(seconds=i), b, 0, 0.9, 0, 402)))
    list(pc.process_detection(_close8(start2 + datetime.timedelta(seconds=9), 0)))
    return start2


def test_period_static_link_escalates_across_long_gap():
    pc = PeriodCorrector(cat_ids=[0, 1])
    buf = io.StringIO()
    with redirect_stdout(buf):
        _feed_static_pair(pc, 300.0)
    codes = [e.code for e in pc.escalations]
    assert "P6" not in codes  # 300 s is beyond P6's 60 s window
    p7 = [e for e in pc.escalations if e.code == "P7"]
    assert len(p7) == 1
    assert p7[0].metrics["gap_seconds"] == 300.0
    assert p7[0].metrics["overlap_iou"] == 1.0
    assert p7[0].metrics["prev_period_id"] == 401
    assert (p7[0].end_time - p7[0].time).total_seconds() == 300.0  # the gap is the span


def test_period_moving_head_does_not_static_link():
    pc = PeriodCorrector(cat_ids=[0, 1])
    buf = io.StringIO()
    with redirect_stdout(buf):
        _feed_static_pair(pc, 300.0, move_new=60)
    assert "P7" not in [e.code for e in pc.escalations]


def test_period_gap_beyond_ten_minutes_does_not_static_link():
    pc = PeriodCorrector(cat_ids=[0, 1])
    buf = io.StringIO()
    with redirect_stdout(buf):
        _feed_static_pair(pc, 601.0)
    assert "P7" not in [e.code for e in pc.escalations]
