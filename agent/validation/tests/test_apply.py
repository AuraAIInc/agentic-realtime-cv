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
    review = {"verdict": verdict}
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


def test_t7_cut_drops_gap_rows_and_splits_visit():
    mid = datetime.datetime(2026, 10, 28, 16, 35, 0)
    after = T0 + datetime.timedelta(seconds=30)
    rows = [_row(T1, 25, 4), _row(mid, 25, 4), _row(T0, 25, 4), _row(after, 25, 4)]
    esc_extra = {"metrics": {"gap_start": T1.isoformat(timespec="microseconds")}}
    out = apply_ops(rows, [_op("T7", "cut_at_gap_start", esc_extra)])
    assert [r["create_time"] for r in out] == [
        T1.isoformat(timespec="microseconds"), after.isoformat(timespec="microseconds")]
    assert [r["period_id"] for r in out] == [4, 5]  # post-gap rows resume as a new visit
    assert out[1]["validated_by"] == "E1-T7" and out[1]["verdict"] == "cut_at_gap_start"


def test_t7_shorten_drops_tail_and_splits_visit():
    mid1 = datetime.datetime(2026, 10, 28, 16, 34, 50)
    mid2 = datetime.datetime(2026, 10, 28, 16, 35, 10)
    after = T0 + datetime.timedelta(seconds=30)
    rows = [_row(T1, 25, 4), _row(mid1, 25, 4), _row(mid2, 25, 4), _row(after, 25, 4)]
    esc_extra = {"metrics": {"gap_start": T1.isoformat(timespec="microseconds")}}
    review_extra = {"cut_time": mid1.isoformat(timespec="microseconds")}
    out = apply_ops(rows, [_op("T7", "shorten_gap", esc_extra, review_extra)])
    assert [r["create_time"] for r in out] == [
        T1.isoformat(timespec="microseconds"), mid1.isoformat(timespec="microseconds"),
        after.isoformat(timespec="microseconds")]
    assert [r["period_id"] for r in out] == [4, 4, 5]


def test_t7_split_rows_skipped_by_later_merge():
    mid = datetime.datetime(2026, 10, 28, 16, 35, 0)
    after = T0 + datetime.timedelta(seconds=30)
    rows = [_row(T1, 25, 4), _row(mid, 25, 4), _row(after, 25, 4)]
    cut = _op("T7", "cut_at_gap_start", {"metrics": {"gap_start": T1.isoformat(timespec="microseconds")}})
    merge = _op("P6", "merge", {"metrics": {"prev_period_id": 3}})
    out = apply_ops(rows, [cut, merge])
    by_time = {r["create_time"]: r for r in out}
    assert by_time[T1.isoformat(timespec="microseconds")]["period_id"] == 3  # pre-gap merges
    assert by_time[after.isoformat(timespec="microseconds")]["period_id"] == 5  # post-gap stays split


def test_period_relabel_and_drop():
    rows = [_row(T0, 25, 4), _row(T1, 26, 9)]
    out = apply_ops(rows, [_op("P1", "award_runner_up", review_extra={"chosen_cat_id": 26})])
    assert [r["cat_id"] for r in out] == [26, 26]
    out = apply_ops(rows, [_op("P2", "drop")])
    assert len(out) == 1 and out[0]["period_id"] == 9
    out = apply_ops(rows, [_op("P2", "reassign", review_extra={"chosen_cat_id": 26})])
    assert [r["cat_id"] for r in out] == [26, 26]
    out = apply_ops(rows, [_op("P2", "confirm")])
    assert [r["cat_id"] for r in out] == [25, 26]
    assert out[0]["validated_by"] == "E1-P2" and out[0]["verdict"] == "confirm"


def test_p3_confirm_reassign_drop():
    rows = [_row(T0, 25, 4)]
    out = apply_ops(rows, [_op("P3", "reassign", review_extra={"chosen_cat_id": 26})])
    assert out[0]["cat_id"] == 26 and out[0]["verdict"] == "reassign"
    out = apply_ops(rows, [_op("P3", "confirm")])
    assert out[0]["cat_id"] == 25 and out[0]["validated_by"] == "E1-P3"
    out = apply_ops(rows, [_op("P3", "drop")])
    assert out == []
    out = apply_ops(rows, [_op("P4", "confirm_unknown")])
    assert out[0]["cat_id"] is None


def test_unexecutable_verdicts_keep_rows():
    rows = [_row(T0, 25, 4)]
    for code, verdict in [("P1", "split_visit"),
                          ("P4", "split_fragments"), ("T4", "link_continuation")]:
        out = apply_ops(rows, [_op(code, verdict)])
        assert len(out) == 1 and out[0]["cat_id"] == 25, (code, verdict)


def test_error_ops_pass_through():
    rows = [_row(T0, 25, 4)]
    out = apply_ops(rows, [_op("T2", "error")])
    assert out == [{**rows[0], "validated_by": "pass-through", "verdict": "pass"}]


def _box_row(ts, cat_id, period_id, box, orig=None):
    row = _row(ts, cat_id, period_id)
    row.update({"box_x1": box[0], "box_y1": box[1], "box_x2": box[2], "box_y2": box[3]})
    if orig is not None:
        row["orig_cat_id"] = orig
    return row


def test_frame_ops_match_disputed_box_not_just_timestamp_cat():
    # Regression: two boxes share one timestamp (E6/E7). Each op must land on
    # its own box even after an earlier op rewrote the other row's cat.
    box_a = [145.0, 357.0, 234.0, 462.0]
    box_b = [174.0, 457.0, 279.0, 585.0]
    rows = [_box_row(T0, 25, 4, box_a, orig=26), _box_row(T0, 26, 2, box_b, orig=25)]
    ops = [
        _op("T2", "keep_original", {"name": "E6-T2", "cat_id": 25,
                                    "metrics": {"original_cat_id": 26, "box": box_a}}),
        _op("T2", "keep_original", {"name": "E7-T2", "cat_id": 26,
                                    "metrics": {"original_cat_id": 25, "box": box_b}}),
    ]
    out = apply_ops(rows, ops)
    assert [r["cat_id"] for r in out] == [26, 25]
    assert out[0]["validated_by"] == "E6-T2" and out[1]["validated_by"] == "E7-T2"


def test_frame_ops_match_tracked_cat_across_rewrites():
    # E6 rewrites the row's cat before E8's drop runs; the drop must still
    # find its row by the tracked (pre-apply) cat, not the live one.
    box = [145.0, 357.0, 234.0, 462.0]
    rows = [_box_row(T0, 25, 4, box, orig=26)]
    ops = [
        _op("T2", "keep_original", {"name": "E6-T2", "cat_id": 25,
                                    "metrics": {"original_cat_id": 26, "box": box}}),
        _op("T4", "drop_box_as_outlier", {"name": "E8-T4", "cat_id": 25,
                                          "metrics": {"box": box}}),
    ]
    assert apply_ops(rows, ops) == []


def test_frame_op_ignores_non_overlapping_box():
    rows = [_box_row(T0, 25, 4, [1.0, 2.0, 3.0, 4.0])]
    op = _op("T2", "keep_original", {"metrics": {"original_cat_id": 26, "box": [400.0, 400.0, 500.0, 500.0]}})
    out = apply_ops(rows, [op])
    assert out[0]["cat_id"] == 25 and out[0]["validated_by"] == "pass-through"


def test_t4_confirm_and_drop():
    box = [145.0, 357.0, 234.0, 462.0]
    rows = [_box_row(T0, 25, 4, box), _box_row(T1, 25, 4, box)]
    out = apply_ops(rows, [_op("T4", "confirm_reset", {"metrics": {"box": box}})])
    assert len(out) == 2 and out[0]["verdict"] == "confirm_reset"
    assert out[1]["validated_by"] == "pass-through"
    out = apply_ops(rows, [_op("T4", "drop_box_as_outlier", {"metrics": {"box": box}})])
    assert [r["create_time"] for r in out] == [T1.isoformat(timespec="microseconds")]


def test_t4_link_continuation_renumbers_restart_period():
    box = [145.0, 357.0, 234.0, 462.0]
    rows = [_box_row(T1, 25, 3, box), _box_row(T0, 25, 4, box)]
    op = _op("T4", "link_continuation", {"period_id": 3, "metrics": {"box": box, "continued_period_id": 4}})
    out = apply_ops(rows, [op])
    assert [r["period_id"] for r in out] == [3, 3]
    assert out[1]["verdict"] == "link_continuation"


def test_t4_link_without_target_records_note_and_keeps_rows():
    box = [145.0, 357.0, 234.0, 462.0]
    rows = [_box_row(T0, 25, 4, box)]
    out = apply_ops(rows, [_op("T4", "link_continuation", {"period_id": 3, "metrics": {"box": box}})])
    assert len(out) == 1 and out[0]["period_id"] == 4
    assert out[0]["validated_by"] == "E1-T4" and "link target unknown" in out[0]["note"]


def test_p8_relabel_keep_and_split():
    rows = [_row(T0, 26, 2), _row(T1, 26, 2), _row(T0, 25, 3)]
    esc_extra = {"period_id": 2, "metrics": {"tracked_cat_id": 26, "voted_cat_id": 25}}
    out = apply_ops(rows, [_op("P8", "relabel_to_vote", esc_extra)])
    assert [r["cat_id"] for r in out] == [25, 25, 25]
    assert out[0]["validated_by"] == "E1-P8" and out[2]["validated_by"] == "pass-through"
    out = apply_ops(rows, [_op("P8", "keep_tracked", esc_extra)])
    assert [r["cat_id"] for r in out] == [26, 26, 25]
    assert out[0]["verdict"] == "keep_tracked"


def test_p8_split_mixed_labels_each_side():
    rows = [_row(T1, 26, 2), _row(T0, 26, 2), _row(T0, 25, 3)]
    esc_extra = {"period_id": 2, "metrics": {"tracked_cat_id": 26, "voted_cat_id": 25}}
    review_extra = {"chosen_cat_id": 25, "cut_time": T1.isoformat(timespec="microseconds")}
    out = apply_ops(rows, [_op("P8", "split_mixed", esc_extra, review_extra)])
    assert [(r["period_id"], r["cat_id"]) for r in out] == [(2, 26), (4, 25), (3, 25)]
    assert out[1]["validated_by"] == "E1-P8" and out[1]["verdict"] == "split_mixed"
    # missing cut keeps rows but records why
    out = apply_ops(rows, [_op("P8", "split_mixed", esc_extra, {"chosen_cat_id": 25})])
    assert [(r["period_id"], r["cat_id"]) for r in out] == [(2, 26), (2, 26), (3, 25)]
    assert "cut_time" in out[0]["note"] and out[2].get("note") is None


def test_chained_p7_merges_collapse_transitively():
    rows = [_row(T0, 25, 1), _row(T0, 25, 2), _row(T0, 25, 3), _row(T0, 25, 4)]
    ops = [
        _op("P7", "merge", {"name": "E4-P7", "period_id": 4, "metrics": {"prev_period_id": 3}}),
        _op("P7", "merge", {"name": "E3-P7", "period_id": 3, "metrics": {"prev_period_id": 2}}),
        _op("P7", "merge", {"name": "E2-P7", "period_id": 2, "metrics": {"prev_period_id": 1}}),
    ]
    out = apply_ops(rows, ops)  # reverse op order: result must not depend on arrival order
    assert [r["period_id"] for r in out] == [1, 1, 1, 1]
    assert all(r["verdict"] == "merge" for r in out[1:])


def test_p7_merge_fills_gap_with_interpolated_rows():
    t_a = datetime.datetime(2026, 10, 28, 16, 35, 20, 500000)
    t_b = datetime.datetime(2026, 10, 28, 16, 35, 24, 500000)
    row_a = _row(t_a, 25, 1)
    row_a.update({"box_x1": 0.0, "box_y1": 0.0, "box_x2": 10.0, "box_y2": 10.0})
    row_b = _row(t_b, 25, 2)
    row_b.update({"box_x1": 40.0, "box_y1": 0.0, "box_x2": 50.0, "box_y2": 10.0})
    out = apply_ops([row_a, row_b], [_op("P7", "merge", {"period_id": 2, "metrics": {"prev_period_id": 1}})])
    fills = [r for r in out if r.get("gap_filled")]
    assert [r["create_time"] for r in fills] == [
        "2026-10-28T16:35:21.000000",
        "2026-10-28T16:35:22.000000",
        "2026-10-28T16:35:23.000000",
        "2026-10-28T16:35:24.000000",
    ]
    assert [r["create_time"] for r in out] == sorted(r["create_time"] for r in out)  # slotted in order
    assert all(r["period_id"] == 1 for r in out)
    first, last = fills[0], fills[-1]
    assert (first["box_x1"], first["box_x2"]) == (5.0, 15.0)  # f=0.125 across the 4 s span
    assert (last["box_x1"], last["box_x2"]) == (35.0, 45.0)  # f=0.875
    assert all(r["conf"] == 0 and r["orig_cat_id"] is None for r in fills)
    assert all(r["validated_by"] == "E1-P7" and r["verdict"] == "merge" for r in fills)


def test_merge_fill_skipped_on_cat_disagreement():
    t_a = datetime.datetime(2026, 10, 28, 16, 35, 20, 500000)
    t_b = datetime.datetime(2026, 10, 28, 16, 35, 24, 500000)
    out = apply_ops(
        [_row(t_a, 25, 1), _row(t_b, 26, 2)],
        [_op("P7", "merge", {"period_id": 2, "metrics": {"prev_period_id": 1}})],
    )
    assert [r["period_id"] for r in out] == [1, 1]  # renumber stands
    assert all(not r.get("gap_filled") for r in out)  # but nothing is synthesized


def test_t6_confirm_split_drops_whole_interp_bridge():
    t0 = datetime.datetime(2026, 10, 28, 16, 35, 20)
    t1 = datetime.datetime(2026, 10, 28, 16, 35, 21)
    t2 = datetime.datetime(2026, 10, 28, 16, 35, 22)
    t3 = datetime.datetime(2026, 10, 28, 16, 35, 23)
    pre = _row(t0, 25, 3)
    tomb1 = {**_row(t1, 25, 3), "orig_cat_id": None, "deleted": True}
    live_interp = {**_row(t1 + datetime.timedelta(microseconds=500000), 25, 3), "orig_cat_id": None}
    tomb2 = {**_row(t2, 25, 3), "orig_cat_id": None, "deleted": True}
    other_cat = {**_row(t2, 26, 3), "orig_cat_id": None}  # same span, other cat: survives
    post = _row(t3, 25, 4)
    op = _op("T6", "confirm_split", {"time": t3.isoformat(), "metrics": {"pre_time": t0.isoformat()}})
    out = apply_ops([pre, tomb1, live_interp, tomb2, other_cat, post], [op])
    iso = lambda t: t.isoformat(timespec="microseconds")
    assert [r["create_time"] for r in out] == [iso(t0), iso(t2), iso(t3)]
    assert [r["cat_id"] for r in out] == [25, 26, 25]
    assert out[-1]["validated_by"] == "E1-T6" and out[-1]["verdict"] == "confirm_split"


def test_t6_confirm_split_falls_back_to_sampled_stamps():
    t1 = datetime.datetime(2026, 10, 28, 16, 35, 21)
    t2 = datetime.datetime(2026, 10, 28, 16, 35, 22)
    sampled = {**_row(t1, 25, 3), "orig_cat_id": None, "deleted": True}
    unsampled = {**_row(t2, 25, 3), "orig_cat_id": None, "deleted": True}
    op = _op(
        "T6",
        "confirm_split",
        {"time": t2.isoformat(), "metrics": {"retracted_frames": [{"t": t1.isoformat()}]}},
    )
    out = apply_ops([sampled, unsampled], [op])
    assert [r["create_time"] for r in out] == [t2.isoformat(timespec="microseconds")]  # partial but safe


def test_p6_different_cats_relabels_new_visit_only():
    rows = [_row(T0, 25, 3), _row(T1, 25, 4)]
    op = _op(
        "P6",
        "different_cats",
        {"period_id": 4, "metrics": {"prev_period_id": 3}},
        {"chosen_cat_id": 26},
    )
    out = apply_ops(rows, [op])
    assert [(r["period_id"], r["cat_id"]) for r in out] == [(3, 25), (4, 26)]
    assert out[0]["validated_by"] == "pass-through"  # previous visit keeps its label
    assert out[1]["validated_by"] == "E1-P6" and out[1]["verdict"] == "different_cats"


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
