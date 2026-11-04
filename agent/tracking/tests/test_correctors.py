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


def test_track_duplicate_tie_escalates_with_json_log(caplog):
    import logging

    tc = TrackCorrector(cat_ids=[0, 1])
    box = [10, 10, 100, 100]
    img = {0: box + [0, 0.60], 1: box + [1, 0.62]}
    with caplog.at_level(logging.INFO, logger="agent.escalation.escalation"):
        list(tc.process_images(T0, img, 0, 0, 0, 0))
    codes = [e.code for e in tc.escalations]
    assert "T3" in codes
    lines = [
        json.loads(t.split(": ", 1)[1])
        for t in (r.getMessage() for r in caplog.records)
        if t.startswith("Escalation E") and ": {" in t
    ]
    assert any(o.get("code") == "T3" and o["escalation"] is True for o in lines)


def test_shared_counter_numbers_frame_and_period_continuously(caplog):
    import itertools
    import logging

    seq = itertools.count(1)
    tc = TrackCorrector(cat_ids=[0, 1], escalation_seq=seq)
    pc = PeriodCorrector(cat_ids=[0, 1], escalation_seq=seq)
    with caplog.at_level(logging.INFO, logger="agent.escalation.escalation"):
        tc._escalate("T5", "frame one", T0)
        pc._escalate("P1", "period one", T0, T0)
        tc._escalate("T3", "frame two", T0)
    assert [e.name for e in tc.escalations] == ["E1-T5", "E3-T3"]
    assert [e.name for e in pc.escalations] == ["E2-P1"]
    lines = [
        json.loads(t.split(": ", 1)[1])
        for t in (r.getMessage() for r in caplog.records)
        if t.startswith("Escalation E") and ": {" in t
    ]
    assert [o["name"] for o in lines] == ["E1-T5", "E2-P1", "E3-T3"]


def test_private_counters_start_at_one_each():
    tc = TrackCorrector(cat_ids=[0, 1])
    pc = PeriodCorrector(cat_ids=[0, 1])
    buf = io.StringIO()
    with redirect_stdout(buf):
        tc._escalate("T5", "t", T0)
        pc._escalate("P1", "p", T0, T0)
    assert tc.escalations[0].name == "E1-T5" and pc.escalations[0].name == "E1-P1"


def test_track_target_fallback_escalates_on_weak_boxes():
    tc = TrackCorrector(cat_ids=[0, 1])
    buf = io.StringIO()
    with redirect_stdout(buf):
        # all conf < 0.5 and pairwise IoU 0: rule-3 fallback guess from frame 2
        for s in range(3):
            box = [10 + 100 * s, 10, 100 + 100 * s, 100]
            list(tc.process_images(T0 + datetime.timedelta(seconds=s), {0: box + [0, 0.45]}, 0, 0, 0, 0))
    assert "T5" in [e.code for e in tc.escalations]
    t5 = next(e for e in tc.escalations if e.code == "T5")
    assert t5.metrics["target_box"] is not None and len(t5.metrics["target_box"]) == 4


def test_synth_run_sampler_collects_interp_boxes_only():
    tc = TrackCorrector(cat_ids=[0])
    for i in range(10):
        t = T0 + datetime.timedelta(seconds=i)
        tc.revised_detection[0].append(np.array([t, 10 + i, 10, 100 + i, 100, None, 0.4, 0, 1], dtype=object))
    tc.revised_detection[0].append(np.array([T0 + datetime.timedelta(seconds=10), 10, 10, 100, 100, 0, 0.9, 0, 1], dtype=object))
    frames = tc._sample_synth_run(0, T0, T0 + datetime.timedelta(seconds=10), cap=4)
    assert len(frames) == 4
    assert frames[0] == {"t": T0.isoformat(timespec="microseconds"), "box": [10.0, 10.0, 100.0, 100.0]}
    assert frames[-1]["t"] == (T0 + datetime.timedelta(seconds=9)).isoformat(timespec="microseconds")
    assert all(set(f) == {"t", "box"} and len(f["box"]) == 4 for f in frames)


def test_jump_frame_context_captures_pre_post_and_mids():
    from agent.tracking.track_corrector import jump_frame_context

    pre = np.array([T0, 10, 10, 100, 100, 0, 0.9, 0, 1], dtype=object)
    post = np.array([T0 + datetime.timedelta(seconds=5), 400, 10, 490, 100, 0, 0.52, 0, 2], dtype=object)
    mids = [np.array([T0 + datetime.timedelta(seconds=i), 100 + i, 10, 190 + i, 100, None, 0.4, 0, 1], dtype=object) for i in range(1, 7)]
    ctx = jump_frame_context(pre, post, mids)
    assert ctx["pre_time"] == T0.isoformat(timespec="microseconds")
    assert ctx["pre_box"] == [10.0, 10.0, 100.0, 100.0]
    assert ctx["post_box"] == [400.0, 10.0, 490.0, 100.0]
    assert len(ctx["retracted_frames"]) == 4  # sampled down from 6
    assert ctx["retracted_frames"][0]["t"] == (T0 + datetime.timedelta(seconds=1)).isoformat(timespec="microseconds")


def test_p5_carries_counts_box_and_key_frames():
    # Strong winner (11 real sightings, score 9.0) but low quality: P5 fires
    # alone — a weak winner would take P2 instead (see dedup test below).
    pc = PeriodCorrector(cat_ids=[25, 26])
    for i in range(12):
        t = T0 + datetime.timedelta(seconds=i * 10)
        box = [10 + i, 10, 100 + i, 100] if i != 11 else [None, None, None, None]
        pc.detections.append(np.array([t, *box, 26, 0.5, 26, 7], dtype=object))
    assert pc._median_winner_box(7, 26) == [15.0, 10.0, 105.0, 100.0]
    frames = pc._sample_period_frames(7, cap=4)
    assert len(frames) == 4
    assert frames[0]["real"] is True and frames[0]["box"] == [10.0, 10.0, 100.0, 100.0]
    assert frames[-1]["box"] is None and frames[-1]["real"] is False
    stats = {"detect_length": np.array([0, 11]), "detect_conf": np.array([0.0, 0.5]),
             "detect_rate": np.array([0.0, 0.1]), "length": 12}
    buf = io.StringIO()
    with redirect_stdout(buf):
        pc._check_vote_escalations(7, stats, np.array([0.0, 9.0]), 1, 9.0, (T0, T0 + datetime.timedelta(seconds=110)))
    assert [e.code for e in pc.escalations] == ["P5"]
    p5 = pc.escalations[0]
    assert p5.metrics["runner_up_cat_id"] == 25  # score-ranked, not count-ranked
    assert p5.metrics["total_frames"] == 12
    assert p5.metrics["detect_counts"] == {"26": 11}
    assert p5.metrics["winner_box"] == [15.0, 10.0, 105.0, 100.0]
    assert len(p5.metrics["key_frames"]) == 6


def test_nearness_source_reports_reference_position():
    tc = TrackCorrector(cat_ids=[25, 26])
    assert tc._nearness_source(0, [400, 20, 490, 200]) == {}
    tc.last_detection[0] = [T0, 350, 25, 440, 205, 25, 0.9, 25]
    src = tc._nearness_source(0, [400, 20, 490, 200])
    assert src["source_time"] == T0.isoformat(timespec="microseconds")
    assert src["source_box"] == [350.0, 25.0, 440.0, 205.0]
    assert src["source_distance_px"] == (50**2 + 5**2) ** 0.5  # centers (445, 110) vs (395, 115)
    assert src["source_threshold_px"] == 160.0
    tc.last_detection[0] = None
    tc.base[0] = [T0, 350, 25, 440, 205, 25, 0.9, 25]
    assert tc._nearness_source(0, [400, 20, 490, 200])["source_box"] == [350.0, 25.0, 440.0, 205.0]


def test_p6_carries_prev_span_and_boundary_boxes():
    pc = PeriodCorrector(cat_ids=[25])
    prev_start = T0 - datetime.timedelta(minutes=2)
    pc.prev_period_end[25] = (T0, [10, 20, 100, 200], 101, [[10, 20, 100, 200]])
    pc.connected_periods[101] = [prev_start, T0]
    det = np.array([T0 + datetime.timedelta(seconds=39), 120, 30, 210, 210, 25, 0.8, 25, 102], dtype=object)
    buf = io.StringIO()
    with redirect_stdout(buf):
        pc._check_merge_candidate(25, 102, det)
    p6 = next(e for e in pc.escalations if e.code == "P6")
    assert p6.metrics["prev_start"] == prev_start
    assert p6.metrics["prev_end_box"] == [10.0, 20.0, 100.0, 200.0]
    assert p6.metrics["new_start_box"] == [120.0, 30.0, 210.0, 210.0]


def test_p2_carries_key_frames():
    pc = PeriodCorrector(cat_ids=[25, 26])
    for i in range(12):
        t = T0 + datetime.timedelta(seconds=i)
        pc.detections.append(np.array([t, 10 + i, 10, 100 + i, 100, 25, 0.4, 25, 7], dtype=object))
    stats = {"detect_length": np.array([12, 0]), "detect_conf": np.array([0.4, 0.0]),
             "detect_rate": np.array([1.0, 0.0]), "length": 12}
    buf = io.StringIO()
    with redirect_stdout(buf):
        pc._check_vote_escalations(7, stats, np.array([2.0, 0.0]), 0, 9.0, (T0, T0 + datetime.timedelta(seconds=11)))
    p2 = next(e for e in pc.escalations if e.code == "P2")
    assert len(p2.metrics["key_frames"]) == 6
    assert p2.metrics["key_frames"][0]["box"] == [10.0, 10.0, 100.0, 100.0]


def test_weak_winner_suppresses_duplicate_p5():
    pc = PeriodCorrector(cat_ids=[25, 26])
    for i in range(12):
        t = T0 + datetime.timedelta(seconds=i)
        pc.detections.append(np.array([t, 10 + i, 10, 100 + i, 100, 25, 0.4, 25, 7], dtype=object))
    # weak winner (score 2.0) AND low quality (conf 0.4): same visit, same key
    # frames, same keep/drop question — only P2 fires (it keeps the merge option).
    stats = {"detect_length": np.array([12, 0]), "detect_conf": np.array([0.4, 0.0]),
             "detect_rate": np.array([1.0, 0.0]), "length": 12}
    buf = io.StringIO()
    with redirect_stdout(buf):
        pc._check_vote_escalations(7, stats, np.array([2.0, 0.0]), 0, 9.0, (T0, T0 + datetime.timedelta(seconds=11)))
    assert [e.code for e in pc.escalations] == ["P2"]


def _synth_poop(t, box):
    x1, y1, x2, y2 = box
    return np.array([t, x1, y1, x2, y2, None, 0.4, 0], dtype=object)


def test_t7_skips_static_run_but_fires_on_motion():
    tc = TrackCorrector(cat_ids=[0])
    still = [10, 10, 100, 100]
    tc._synth_run[0] = [T0, False, list(still), 0.0]
    buf = io.StringIO()
    with redirect_stdout(buf):
        # 6 s of drift within 2 px: silent, flag stays False.
        tc._maybe_escalate_synth_run(0, _synth_poop(T0 + datetime.timedelta(seconds=6), [11, 10, 101, 100]), T0, still, 0.0)
    assert tc.escalations == [] and tc._synth_run[0][1] is False
    with redirect_stdout(buf):
        # the same run starts moving: fires, with displacement in metrics.
        tc._maybe_escalate_synth_run(
            0, _synth_poop(T0 + datetime.timedelta(seconds=7), [200, 10, 290, 100]), T0, still, tc._synth_run[0][3]
        )
    assert [e.code for e in tc.escalations] == ["T7"]
    assert tc.escalations[0].metrics["run_displacement_px"] >= 15.0
    assert tc.escalations[0].metrics["gap_seconds"] == 7.0
    assert tc._synth_run[0][1] is True


def _seed_jump_reset(tc, detecting):
    tc.right_time = T0 + datetime.timedelta(seconds=60)
    tc.base[0] = [T0, 10, 10, 100, 100, 25, 0.9, 25]
    tc.detecting[0] = detecting
    for i in range(3):
        t = T0 + datetime.timedelta(seconds=50 + i)
        tc.buffer[0].append(np.array([t, None, None, None, None, None, 0, None], dtype=object))
        tc.buffer_origin[0].append(np.array([t, None, None, None, None, None, 0, None], dtype=object))


def test_jump_reset_carries_anchor_and_sighting_boxes():
    tc = TrackCorrector(cat_ids=[25])
    _seed_jump_reset(tc, True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        tc.add_new_detection_to_buffer({25: [400, 400, 500, 500, 25, 0.6]})
    t4 = next(e for e in tc.escalations if e.code == "T4")
    assert t4.metrics["base_box"] == [10.0, 10.0, 100.0, 100.0]
    assert t4.metrics["base_time"] == T0.isoformat(timespec="microseconds")
    assert t4.metrics["box"] == [400.0, 400.0, 500.0, 500.0]
    assert t4.metrics["wiped_rows"] == 3
    assert t4.metrics["had_open_visit"] is True
    assert "closed the period" in t4.reason
    assert tc.base[0] is None and tc.detecting[0] is False


def test_jump_reset_without_open_visit_says_so():
    tc = TrackCorrector(cat_ids=[25])
    _seed_jump_reset(tc, False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        tc.add_new_detection_to_buffer({25: [400, 400, 500, 500, 25, 0.6]})
    t4 = next(e for e in tc.escalations if e.code == "T4")
    assert t4.metrics["had_open_visit"] is False
    assert "no visit was open to close" in t4.reason


def test_multi_match_carries_disputed_box_and_candidate_refs():
    tc = TrackCorrector(cat_ids=[0, 1, 2])
    tc.right_time = T0
    tc.last_detection[0] = [T0, 10, 10, 100, 100, 0, 0.9, 0]
    tc.last_detection[1] = [T0, 400, 400, 500, 500, 1, 0.9, 1]
    tc.buffer[1].append(np.array([T0, 12, 12, 102, 102, 1, 0.9, 1], dtype=object))
    buf = io.StringIO()
    with redirect_stdout(buf):
        tc.add_new_detection_to_buffer({2: [12, 12, 102, 102, 2, 0.55]})
    t1 = next(e for e in tc.escalations if e.code == "T1")
    assert t1.metrics["box"] == [12.0, 12.0, 102.0, 102.0]
    refs = t1.metrics["candidate_refs"]
    assert refs["0"]["rule"] == "iou_last"
    assert refs["0"]["source_box"] == [10.0, 10.0, 100.0, 100.0]
    assert refs["1"]["rule"] == "iou_buffer"


def test_duplicate_tie_carries_both_boxes():
    tc = TrackCorrector(cat_ids=[0, 1])
    tc.right_time = T0
    box = [10, 10, 100, 100]
    buf = io.StringIO()
    with redirect_stdout(buf):
        tc.add_new_detection_to_buffer({0: box + [0, 0.60], 1: box + [1, 0.62]})
    t3 = next(e for e in tc.escalations if e.code == "T3")
    assert t3.metrics["kept_box"] == [10.0, 10.0, 100.0, 100.0]
    assert t3.metrics["killed_box"] == [10.0, 10.0, 100.0, 100.0]
    assert t3.metrics["kept_conf"] == 0.62 and t3.metrics["killed_conf"] == 0.60


# ---------------------------------------------------------------- PeriodCorrector


def test_conflict_reassign_names_taking_period():
    pc = PeriodCorrector(cat_ids=[0, 1])
    box_a = [10, 10, 100, 100]
    box_b = [300, 300, 400, 400]
    _feed_period(pc, 101, 0, T0, 24, box_a, lambda i: 0)
    _feed_period(pc, 102, 1, T0 + datetime.timedelta(seconds=10), 24, box_b, lambda i: i % 2)
    buf = io.StringIO()
    with redirect_stdout(buf):
        list(pc.process_detection(_close8(T0 + datetime.timedelta(seconds=34), 0)))
        list(pc.process_detection(_close8(T0 + datetime.timedelta(seconds=34), 1)))
    p3 = next(e for e in pc.escalations if e.code == "P3")
    assert p3.period_id == 102
    assert p3.metrics["wanted_cat_id"] == 0
    assert p3.metrics["taken_by_period_id"] == 101
    assert p3.metrics["taken_by_span"][0] == T0.isoformat(timespec="microseconds")
    assert len(p3.metrics["key_frames"]) == 6


def test_p4_carries_counts_and_key_frames():
    pc = PeriodCorrector(cat_ids=[25, 26])
    for i in range(40):
        t = T0 + datetime.timedelta(seconds=i * 2)
        orig = 26 if i % 2 == 0 else None
        pc.detections.append(np.array([t, 10 + i, 10, 100 + i, 100, orig, 0.5, 26, 7], dtype=object))
    buf = io.StringIO()
    with redirect_stdout(buf):
        pc._check_unknown_escalation(7, (T0, T0 + datetime.timedelta(seconds=80)), 40)
    p4 = next(e for e in pc.escalations if e.code == "P4")
    assert p4.metrics["detect_counts"] == {"26": 20}
    assert len(p4.metrics["key_frames"]) == 6


def test_p7_carries_prev_span_and_boundary_boxes():
    pc = PeriodCorrector(cat_ids=[25])
    prev_start = T0 - datetime.timedelta(minutes=2)
    pc.prev_period_end[25] = (T0, [10, 20, 100, 200], 101, [[10, 20, 100, 200], [11, 21, 101, 201]])
    pc.connected_periods[101] = [prev_start, T0]
    pc._open_prev[102] = pc.prev_period_end[25]
    for i in range(4):
        t = T0 + datetime.timedelta(seconds=120 + i)
        pc.detections.append(np.array([t, 10 + i, 20, 100 + i, 200, 25, 0.8, 25, 102], dtype=object))
    buf = io.StringIO()
    with redirect_stdout(buf):
        pc._check_static_link(102)
    p7 = next(e for e in pc.escalations if e.code == "P7")
    assert p7.metrics["prev_start"] == prev_start
    assert p7.metrics["prev_end_box"] == [10.0, 20.0, 100.0, 200.0]
    assert p7.metrics["new_start_box"] == [10.0, 20.0, 100.0, 200.0]


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
    assert set(p1.metrics["votes"]) == {"0", "1"}
    assert p1.metrics["runner_up_cat_id"] in (0, 1)
    assert len(p1.metrics["key_frames"]) == 6


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


def test_p8_fires_when_tracker_disagrees_with_vote():
    pc = PeriodCorrector(cat_ids=[0, 1])
    box = [10, 10, 100, 100]
    # Tracked as cat 1 but original labels mostly 0: vote 0 vs tracked 1.
    last = _feed_period(pc, 101, 1, T0, 24, box, lambda i: 0 if i % 4 else 1)
    buf = io.StringIO()
    with redirect_stdout(buf):
        list(pc.process_detection(_close8(last + datetime.timedelta(seconds=1), 1)))
    p8 = next(e for e in pc.escalations if e.code == "P8")
    assert p8.scope == "period" and p8.period_id == 101
    assert p8.metrics["tracked_cat_id"] == 1 and p8.metrics["voted_cat_id"] == 0
    assert p8.metrics["tracked_frames"] == 24
    assert p8.metrics["relabeled_frames"] == 18
    assert p8.metrics["votes"]["0"]["frames"] == 18
    # P8 samples denser than the shared cap-6: an 8-frame backbone plus jump clusters.
    assert len(p8.metrics["key_frames"]) == 8


def test_p8_key_frames_cluster_around_track_jump():
    pc = PeriodCorrector(cat_ids=[0, 1])
    rug, bed = [300, 100, 360, 170], [170, 450, 280, 585]
    # 40 s on the rug, then a jump to the bed for 200 s: a short disputed
    # span inside a long tail, like the 16:34 window.
    _feed_period(pc, 101, 1, T0, 40, rug, lambda i: 0, step_s=1.0)
    _feed_period(pc, 101, 1, T0 + datetime.timedelta(seconds=40), 200, bed, lambda i: 0, step_s=1.0)
    frames = pc._sample_period_frames_for_split(101)
    times = [datetime.datetime.fromisoformat(f["t"]) for f in frames]
    assert times == sorted(times)  # time order, endpoints kept
    assert times[0] == T0 and times[-1] == T0 + datetime.timedelta(seconds=239)
    jump = T0 + datetime.timedelta(seconds=40)
    near = [t for t in times if abs((t - jump).total_seconds()) <= 20]
    assert len(near) >= 8  # dense around the handover, not just the tail
    assert all(f["box"] is not None and f["real"] is True for f in frames)


def test_p8_silent_when_tracker_matches_vote():
    pc = PeriodCorrector(cat_ids=[0, 1])
    box = [10, 10, 100, 100]
    last = _feed_period(pc, 101, 0, T0, 24, box, lambda i: 0)
    buf = io.StringIO()
    with redirect_stdout(buf):
        list(pc.process_detection(_close8(last + datetime.timedelta(seconds=1), 0)))
    assert "P8" not in [e.code for e in pc.escalations]


def test_jump_reset_links_restart_period_on_next_mint():
    tc = TrackCorrector(cat_ids=[25])
    _seed_jump_reset(tc, True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        tc.add_new_detection_to_buffer({25: [400, 400, 500, 500, 25, 0.6]})
    t4 = next(e for e in tc.escalations if e.code == "T4")
    assert "continued_period_id" not in t4.metrics
    pid = tc._mint_period(0)
    assert t4.metrics["continued_period_id"] == pid
    assert tc._mint_period(0) != pid  # pending link consumed by the first mint
    assert t4.metrics["continued_period_id"] == pid


def test_jump_reset_without_open_visit_leaves_no_link():
    tc = TrackCorrector(cat_ids=[25])
    _seed_jump_reset(tc, False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        tc.add_new_detection_to_buffer({25: [400, 400, 500, 500, 25, 0.6]})
    t4 = next(e for e in tc.escalations if e.code == "T4")
    tc._mint_period(0)
    assert "continued_period_id" not in t4.metrics
