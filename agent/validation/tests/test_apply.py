"""Verdict ops rewrite revised rows; validated.jsonl + validated.db are written."""

import datetime
import json
import sqlite3

from agent.validation.apply import apply_ops, materialize

T0 = datetime.datetime(2026, 10, 28, 16, 35, 21, 976102)
T1 = datetime.datetime(2026, 10, 28, 16, 34, 42, 769881)


def _row(ts, cat_id, period_id, conf=0.8):
    return {
        "create_time": ts.isoformat(timespec="microseconds"),
        "box_x1": 1.0, "box_y1": 2.0, "box_x2": 3.0, "box_y2": 4.0,
        "orig_cat_id": cat_id, "conf": conf, "cat_id": cat_id,
        "period_id": period_id, "view_id": 3,
    }


def _op(code, verdict, esc_extra=None, review_extra=None):
    esc = {"name": f"E1-{code}", "code": code, "time": T0.isoformat(timespec="microseconds"),
           "cat_id": 25, "period_id": 4, "metrics": {}}
    esc.update(esc_extra or {})
    review = {"escalation": f"E1-{code}", "verdict": verdict, "confidence": 0.9,
              "explanation": "saw it", "frames_checked": [], "needs_human": False}
    review.update(review_extra or {})
    return {"escalation": esc, "review": review}


def test_frame_relabel_rewrites_cat():
    rows = [_row(T0, 25, 4), _row(T1, 25, 4)]
    esc_extra = {"metrics": {"original_cat_id": 26}}
    out = apply_ops(rows, [_op("T2", "keep_original", esc_extra)])
    assert [r["cat_id"] for r in out] == [26, 25]
    assert out[0]["validated_by"] == "E1-T2" and out[0]["verdict"] == "keep_original"
    assert out[1]["validated_by"] == "pass-through"


def test_frame_drop_removes_row():
    rows = [_row(T0, 25, 4), _row(T1, 25, 4)]
    out = apply_ops(rows, [_op("T2", "drop", {"metrics": {"original_cat_id": 26}})])
    assert len(out) == 1 and out[0]["create_time"].startswith("2026-10-28T16:34")


def test_t6_merge_rewrites_period():
    rows = [_row(T0, 25, 4)]
    out = apply_ops(rows, [_op("T6", "merge_visits", {"metrics": {"old_period_id": 3}})])
    assert out[0]["period_id"] == 3


def test_t7_cut_drops_gap_rows():
    mid = datetime.datetime(2026, 10, 28, 16, 35, 0)
    rows = [_row(T1, 25, 4), _row(mid, 25, 4), _row(T0, 25, 4)]
    esc_extra = {"metrics": {"gap_start": T1.isoformat(timespec="microseconds")}}
    out = apply_ops(rows, [_op("T7", "cut_at_gap_start", esc_extra)])
    assert [r["create_time"] for r in out] == [T1.isoformat(timespec="microseconds")]


def test_period_relabel_and_drop():
    rows = [_row(T0, 25, 4), _row(T1, 26, 9)]
    out = apply_ops(rows, [_op("P1", "award_runner_up", review_extra={"chosen_cat_id": 26})])
    assert [r["cat_id"] for r in out] == [26, 26]
    out = apply_ops(rows, [_op("P2", "drop")])
    assert len(out) == 1 and out[0]["period_id"] == 9


def test_p3_restore_and_unknown():
    rows = [_row(T0, 25, 4)]
    out = apply_ops(rows, [_op("P3", "restore_wanted", {"metrics": {"wanted_cat_id": 26}})])
    assert out[0]["cat_id"] == 26
    out = apply_ops(rows, [_op("P4", "confirm_unknown")])
    assert out[0]["cat_id"] is None


def test_unexecutable_verdicts_keep_rows():
    rows = [_row(T0, 25, 4)]
    for code, verdict in [("P1", "split_visit"), ("P2", "merge_neighbor"), ("P5", "trim_core"),
                          ("P4", "split_fragments"), ("T4", "link_continuation")]:
        out = apply_ops(rows, [_op(code, verdict)])
        assert len(out) == 1 and out[0]["cat_id"] == 25, (code, verdict)


def test_error_ops_pass_through():
    rows = [_row(T0, 25, 4)]
    out = apply_ops(rows, [_op("T2", "error")])
    assert out == [{**rows[0], "validated_by": "pass-through", "verdict": "pass"}]


def test_materialize_writes_jsonl_and_sqlite(tmp_path):
    rows = [_row(T0, 25, 4)]
    jsonl_path, db_path, out = materialize(tmp_path, rows, [_op("P5", "keep")])
    assert jsonl_path.name == "validated.jsonl" and db_path.name == "validated.db"
    disk = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert disk == out and disk[0]["verdict"] == "keep"
    conn = sqlite3.connect(db_path)
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(validated_detection)").fetchall()]
        assert cols == ["create_time", "box_x1", "box_y1", "box_x2", "box_y2", "orig_cat_id",
                        "conf", "cat_id", "period_id", "view_id", "validated_by", "verdict"]
        got = conn.execute("SELECT cat_id, validated_by, verdict FROM validated_detection").fetchall()
        assert got == [(25, "E1-P5", "keep")]
    finally:
        conn.close()
