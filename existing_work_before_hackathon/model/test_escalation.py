"""Escalation points: JSON format, thresholds, and branch-point hooks.

No MySQL: corrector DB access is stubbed. Heavy deps (numpy/pandas/matplotlib)
come from the repo venv.

Run from existing_work_before_hackathon/:
    ../.venv/bin/python -m pytest model/test_escalation.py -q
    ../.venv/bin/python model/test_escalation.py
"""

from __future__ import annotations

import datetime
import io
import json
import os
import sys
from collections import deque
from contextlib import redirect_stdout

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("LOG_LEVEL", "ERROR")

import numpy as np

from model.escalation import Escalation, EscalationConfig, box_distance, box_iou, emit, max_displacement, top_two_margin, window_is_static

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
    from model.escalation_config import EscalationConfig

    cfg = EscalationConfig()
    assert cfg.vote_margin == 2.0
    assert cfg.weak_winner_score == 3.0 and cfg.weak_winner_min_length == 10
    assert cfg.merge_gap_seconds == 60.0 and cfg.merge_max_distance == 300.0
    assert cfg.static_window_seconds == 5.0 and cfg.static_max_px == 15.0
    assert cfg.static_overlap_iou == 0.5 and cfg.static_link_max_gap == 600.0


def test_config_from_env_overrides():
    from model.escalation_config import EscalationConfig

    old = _with_env({"ESCALATION_VOTE_MARGIN": "3.5", "ESCALATION_UNKNOWN_MIN_FRAMES": "12"})
    try:
        cfg = EscalationConfig.from_env()
    finally:
        _restore_env(old)
    assert cfg.vote_margin == 3.5
    assert cfg.unknown_min_frames == 12 and isinstance(cfg.unknown_min_frames, int)
    assert cfg.synth_gap_seconds == 5.0  # untouched knobs keep defaults


def test_explicit_config_wins_over_env():
    from model.escalation_config import EscalationConfig

    old = _with_env({"ESCALATION_VOTE_MARGIN": "99.0"})
    try:
        default_tc = _make_track_corrector()
        explicit_tc = _make_track_corrector(escalation_config=EscalationConfig())
    finally:
        _restore_env(old)
    assert default_tc.escalation_config.vote_margin == 99.0
    assert explicit_tc.escalation_config.vote_margin == 2.0


# ---------------------------------------------------------------- TrackCorrector


class _FakeDB:
    def __init__(self):
        self._ids = iter(range(10**12, 10**12 + 10**6))

    def execute_query(self, *a, **k):
        return [("0,1",)]

    def execute_insert(self, *a, **k):
        return next(self._ids)

    def execute_update(self, *a, **k):
        return None

    def execute_many_insert(self, *a, **k):
        return None

    def execute_delete(self, *a, **k):
        return None


def _make_track_corrector(**kwargs):
    import model.track_corrector as mod

    real_db = mod.MysqlDB
    mod.MysqlDB = _FakeDB
    try:
        tc = mod.TrackCorrector(view_id=0, **kwargs)
    finally:
        mod.MysqlDB = real_db
    return tc


def test_remove_multi_labeling_reports_tie_kills():
    from model.track_corrector import remove_multi_labeling

    box = [10, 10, 100, 100]
    det = {0: box + [0, 0.60], 1: box + [1, 0.62]}
    kills = []
    out, _ = _capture(remove_multi_labeling, det, kills)
    assert set(out) == {1}  # lower conf killed, behavior unchanged
    assert len(kills) == 1
    killed, kept, iou, margin = kills[0]
    assert (killed, kept) == (0, 1) and iou == 1.0 and abs(margin - 0.02) < 1e-9

    far = {0: [10, 10, 50, 50, 0, 0.9], 1: [400, 400, 450, 450, 1, 0.9]}
    kills = []
    assert remove_multi_labeling(far, kills) == far and kills == []


def _skeleton_tracker():
    from model.track_corrector import TrackCorrector

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
    tc = _make_track_corrector()
    box = [10, 10, 100, 100]
    img = {0: box + [0, 0.60], 1: box + [1, 0.62]}
    buf = io.StringIO()
    with redirect_stdout(buf):
        list(tc.process_images(T0, img, 0, 0, 0, 0, save_db=False))
    codes = [e.code for e in tc.escalations]
    assert "T3" in codes
    lines = [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]
    assert any(o.get("code") == "T3" and o["escalation"] is True for o in lines)


def test_track_target_fallback_escalates_on_weak_boxes():
    tc = _make_track_corrector()
    buf = io.StringIO()
    with redirect_stdout(buf):
        # all conf < 0.5 and pairwise IoU 0: rule-3 fallback guess from frame 2
        for s in range(3):
            box = [10 + 100 * s, 10, 100 + 100 * s, 100]
            list(tc.process_images(T0 + datetime.timedelta(seconds=s), {0: box + [0, 0.45]}, 0, 0, 0, 0, save_db=False))
    assert "T5" in [e.code for e in tc.escalations]


# ---------------------------------------------------------------- PeriodCorrector


def _make_period_corrector(**kwargs):
    import model.period_corrector as mod

    real_db = mod.MysqlDB
    mod.MysqlDB = _FakeDB
    try:
        pc = mod.PeriodCorrector(view_id=0, **kwargs)
    finally:
        mod.MysqlDB = real_db
    return pc


def _det9(t, box, orig, conf, cat, pid):
    x1, y1, x2, y2 = box
    return np.array([t, x1, y1, x2, y2, orig, conf, cat, pid], dtype=object)


def _close8(t, cat):
    return np.array([t, None, None, None, None, None, 0, cat], dtype=object)


def _feed_period(pc, pid, cat, start, n, box, label_fn, conf=0.9, step_s=1.0):
    for i in range(n):
        t = start + datetime.timedelta(seconds=i * step_s)
        list(pc.process_detection(_det9(t, box, label_fn(i), conf, cat, pid), view_id=0))
    return start + datetime.timedelta(seconds=(n - 1) * step_s)


def test_period_close_vote_escalates():
    pc = _make_period_corrector()
    box = [10, 10, 100, 100]
    last = _feed_period(pc, 101, 0, T0, 24, box, lambda i: i % 2)  # 12 frames each label
    buf = io.StringIO()
    with redirect_stdout(buf):
        list(pc.process_detection(_close8(last + datetime.timedelta(seconds=1), 0), view_id=0))
    codes = [e.code for e in pc.escalations]
    assert "P1" in codes
    p1 = next(e for e in pc.escalations if e.code == "P1")
    assert p1.scope == "period" and p1.end_time is not None and p1.end_time > p1.time
    assert "P2" not in codes and "P5" not in codes  # strong tie: only the vote is unsure


def test_period_merge_candidate_escalates():
    pc = _make_period_corrector()
    box = [10, 10, 100, 100]
    last = _feed_period(pc, 101, 0, T0, 12, box, lambda i: 0)
    list(pc.process_detection(_close8(last + datetime.timedelta(seconds=1), 0), view_id=0))
    buf = io.StringIO()
    with redirect_stdout(buf):
        _feed_period(pc, 102, 0, last + datetime.timedelta(seconds=21), 3, box, lambda i: 0)
    p6 = [e for e in pc.escalations if e.code == "P6"]
    assert len(p6) == 1
    assert p6[0].metrics["prev_period_id"] == 101
    assert p6[0].metrics["gap_seconds"] == 21.0  # vs the last detection row, not the close row


def test_period_distant_revisit_does_not_merge():
    pc = _make_period_corrector()
    box = [10, 10, 100, 100]
    last = _feed_period(pc, 101, 0, T0, 12, box, lambda i: 0)
    list(pc.process_detection(_close8(last + datetime.timedelta(seconds=1), 0), view_id=0))
    buf = io.StringIO()
    with redirect_stdout(buf):
        _feed_period(pc, 102, 0, last + datetime.timedelta(seconds=21), 3, [400, 400, 500, 500], lambda i: 0)
    assert "P6" not in [e.code for e in pc.escalations]


def _feed_static_pair(pc, gap_seconds, box=(10, 10, 100, 100), move_new=0):
    """Two same-box periods `gap_seconds` apart; move_new shifts the new head per frame."""
    last = _feed_period(pc, 401, 0, T0, 8, list(box), lambda i: 0)
    list(pc.process_detection(_close8(last + datetime.timedelta(seconds=1), 0), view_id=0))
    start2 = last + datetime.timedelta(seconds=gap_seconds)
    for i in range(8):
        b = [box[0] + move_new * i, box[1], box[2] + move_new * i, box[3]]
        list(pc.process_detection(_det9(start2 + datetime.timedelta(seconds=i), b, 0, 0.9, 0, 402), view_id=0))
    list(pc.process_detection(_close8(start2 + datetime.timedelta(seconds=9), 0), view_id=0))
    return start2


def test_period_static_link_escalates_across_long_gap():
    pc = _make_period_corrector()
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
    pc = _make_period_corrector()
    buf = io.StringIO()
    with redirect_stdout(buf):
        _feed_static_pair(pc, 300.0, move_new=60)
    assert "P7" not in [e.code for e in pc.escalations]


def test_period_gap_beyond_ten_minutes_does_not_static_link():
    pc = _make_period_corrector()
    buf = io.StringIO()
    with redirect_stdout(buf):
        _feed_static_pair(pc, 601.0)
    assert "P7" not in [e.code for e in pc.escalations]


if __name__ == "__main__":
    for name, fn in sorted({k: v for k, v in globals().items() if k.startswith("test_")}.items()):
        if name == "test_escalation_json_line_shape":
            pass
        fn()
        print("ok", name)
