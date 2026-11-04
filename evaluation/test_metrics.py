"""Unit tests for evaluation/metrics.py and evaluate.py helpers."""

import datetime
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import (  # noqa: E402
    collect_cost,
    load_stage,
    load_yolo_labels,
    resolve_hour_dir,
    stem_to_datetime,
    stem_to_time,
)
from metrics import Box, evaluate_frames, iou, match_frame, prf  # noqa: E402


def test_iou_identical():
    assert iou(Box(0, 0, 10, 10, 0), Box(0, 0, 10, 10, 0)) == 1.0


def test_iou_partial():
    # 5x10 overlap over 100+100-50 union
    assert abs(iou(Box(0, 0, 10, 10, 0), Box(5, 0, 15, 10, 0)) - 50 / 150) < 1e-9


def test_iou_disjoint():
    assert iou(Box(0, 0, 1, 1, 0), Box(5, 5, 6, 6, 0)) == 0.0


def test_prf_zero_division():
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)
    assert prf(0, 2, 3) == (0.0, 0.0, 0.0)


def test_match_prefers_confidence_order():
    # Low-conf box overlaps gt perfectly but high-conf box claims it first
    # with a still-qualifying IoU; low-conf becomes FP.
    gt = [Box(0, 0, 10, 10, 0)]
    hi = Box(1, 1, 11, 11, 0, conf=0.9)  # iou ~0.68
    lo = Box(0, 0, 10, 10, 0, conf=0.1)  # iou 1.0 but second
    m = match_frame([lo, hi], gt, 0.5, class_aware=True)
    assert (m.tp, m.fp, m.fn) == (1, 1, 0)


def test_match_class_aware_rejects_wrong_class():
    m = match_frame(
        [Box(0, 0, 10, 10, 1, conf=0.9)], [Box(0, 0, 10, 10, 0)], 0.5, class_aware=True
    )
    assert (m.tp, m.fp, m.fn) == (0, 1, 1)


def test_match_class_agnostic_accepts_wrong_class():
    m = match_frame(
        [Box(0, 0, 10, 10, 1, conf=0.9)], [Box(0, 0, 10, 10, 0)], 0.5, class_aware=False
    )
    assert (m.tp, m.fp, m.fn) == (1, 0, 0)
    assert m.agnostic_pairs == [(1, 0)]


def test_match_below_threshold_is_fp_fn():
    m = match_frame(
        [Box(0, 0, 10, 10, 0, conf=0.9)], [Box(50, 50, 60, 60, 0)], 0.5, class_aware=True
    )
    assert (m.tp, m.fp, m.fn) == (0, 1, 1)


def test_evaluate_frames_aggregates():
    frames = {
        "a": ([Box(0, 0, 10, 10, 0, 0.9)], [Box(0, 0, 10, 10, 0)]),  # TP
        "b": ([Box(0, 0, 10, 10, 1, 0.9)], [Box(0, 0, 10, 10, 0)]),  # class error
        "c": ([], [Box(0, 0, 10, 10, 0)]),  # FN
    }
    out = evaluate_frames(frames, 0.5, classes=[0, 1])
    assert (out["tp"], out["fp"], out["fn"]) == (1, 1, 2)
    assert out["agnostic"]["tp"] == 2  # frame b matches spatially
    assert out["class_accuracy_on_matched"] == 0.5
    assert out["per_class"][0]["tp"] == 1
    assert out["per_class"][1]["fp"] == 1


def test_stem_to_time():
    assert (
        stem_to_time("5520_2026-10-28-16-33-06-797209")
        == "2026-10-28T16:33:06.797209"
    )
    assert stem_to_time("classes") is None


def test_stem_to_datetime_pads_short_fractions():
    full = stem_to_datetime("1_2026-10-28-16-33-00-304643")
    short = stem_to_datetime("1_2026-10-28-16-33-00-3")
    assert full == datetime.datetime(2026, 10, 28, 16, 33, 0, 304643)
    assert short == datetime.datetime(2026, 10, 28, 16, 33, 0, 300000)
    assert stem_to_datetime("classes.txt") is None


def test_load_stage_filters_tombstones_and_interp(tmp_path):
    rows = [
        {"create_time": "2026-10-28T16:33:00.304643", "box_x1": 1, "box_y1": 1,
         "box_x2": 5, "box_y2": 5, "cat_id": 25, "conf": 0.9},  # stream-style: kept
        {"create_time": "2026-10-28T16:33:00.304643", "box_x1": 1, "box_y1": 1,
         "box_x2": 5, "box_y2": 5, "cat_id": 25, "orig_cat_id": 25},  # real: kept
        {"create_time": "2026-10-28T16:33:01.000000", "box_x1": 1, "box_y1": 1,
         "box_x2": 5, "box_y2": 5, "cat_id": 25, "orig_cat_id": None},  # interp: out
        {"create_time": "2026-10-28T16:33:01.000000", "deleted": True,
         "box_x1": 1, "box_y1": 1, "box_x2": 5, "box_y2": 5, "cat_id": 25},  # tomb: out
        {"create_time": "2026-10-28T16:33:02.000000", "cat_id": 25},  # boxless: out
    ]
    path = tmp_path / "revised.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    preds = load_stage(str(tmp_path), "revised.jsonl")
    assert list(preds) == [datetime.datetime(2026, 10, 28, 16, 33, 0, 304643)]
    assert len(preds[list(preds)[0]]) == 2
    assert load_stage(str(tmp_path), "missing.jsonl") == {}


def test_load_yolo_labels_tolerates_conf_and_bad_ids(tmp_path):
    (tmp_path / "1_2026-10-28-16-33-00-304643.txt").write_text(
        "0 0.5 0.5 0.1 0.1\n1 0.5 0.5 0.1 0.1 0.99\n2 0.5 0.5 0.1 0.1\nbad line\n"
    )
    (tmp_path / "classes.txt").write_text("Qb\nQbo\n")
    gt = load_yolo_labels(str(tmp_path), [25, 26], 800, 600)
    assert len(gt) == 1
    boxes = next(iter(gt.values()))
    assert [b.cls for b in boxes] == [25, 26]  # 6-col kept, id 2 dropped


def test_resolve_hour_dir_precedence(tmp_path):
    from evaluate import repo_root_of_run

    repo = tmp_path / "repo"
    run = repo / "output" / "tracking_runs" / "r1"
    (repo / "image_server" / "retrieved_images" / "2026-10-28-16").mkdir(parents=True)
    run.mkdir(parents=True)
    assert repo_root_of_run(str(run)) == str(repo)
    meta_labels = {"args": {"hour_dir": "image_server/retrieved_images/2026-10-28-16"}}
    assert resolve_hour_dir(str(run), meta_labels, None, "/data") == os.path.join(
        str(repo), "image_server/retrieved_images/2026-10-28-16"
    )
    meta_db = {"args": {}, "window": {"start": "2026-10-28T16:33"}}
    assert resolve_hour_dir(str(run), meta_db, None, "/data") == "/data/2026-10-28-16"
    assert resolve_hour_dir(str(run), meta_db, "/x", "/data") == "/x"


def test_collect_cost_without_validations(tmp_path):
    cost = collect_cost(str(tmp_path), {"elapsed_s": 12.5})
    assert cost["validations"] == 0 and cost["est_cost_usd"] == 0.0
    assert cost["elapsed_s"] == 12.5


def _fake_stage(present=True):
    return {
        "present": present,
        "metrics": {
            "iou_0.5": {
                "precision": 0.9,
                "recall": 0.4,
                "f1": 0.55,
                "agnostic": {"precision": 0.95, "recall": 0.42, "f1": 0.58},
                "mean_matched_iou": 0.77,
                "class_accuracy_on_matched": 0.93,
                "tp": 10,
                "fp": 1,
                "fn": 15,
                "per_class": {
                    25: {"precision": 1.0, "recall": 0.5, "f1": 0.667, "tp": 5, "fp": 0, "fn": 5},
                },
            }
        },
    }


def _col_counts(table_text):
    return [len(line.strip().strip("|").split("|")) for line in table_text.splitlines()]


def test_md_table_pads_and_aligns():
    from evaluate import md_table

    text = md_table(["name", "P"], [["muse", "0.936"], ["s", "1"]])
    lines = text.splitlines()
    assert _col_counts(text) == [2, 2, 2, 2]
    assert lines[1].startswith("| :") and lines[1].rstrip().endswith(": |")
    assert all(len(line) == len(lines[0]) for line in [lines[0]] + lines[2:])


def test_report_tables_have_matching_column_counts():
    from evaluate import delta_table, label_table, per_class_table, stage_table

    res = {
        "stages": {"raw": _fake_stage(), "revised": _fake_stage(), "validated": _fake_stage(False)},
        "label_sets": {"muse": _fake_stage(), "spark": _fake_stage(False)},
    }
    tables = [
        (label_table(res, "iou_0.5", "0.5"), 12),
        (stage_table(res, "iou_0.5", "0.5"), 12),
        (delta_table(res, "iou_0.5"), 9),
        (per_class_table(res, "iou_0.5", {"25": "Qb"}), 8),
    ]
    for text, ncols in tables:
        assert _col_counts(text) == [ncols] * len(text.splitlines())


def test_chart_lines_embeds_copied_charts():
    from evaluate import chart_lines

    res = {
        "charts": {
            "original": {"present": True, "file": "r1_original.png", "src": "/x/original.png"},
            "revised": {"present": False, "file": None, "src": None},
            "validated": {"present": True, "file": None, "src": "/x/validated.png"},
            "gt": {"present": True, "file": "r1_gt.png", "src": "/x/r1_gt.png"},
        }
    }
    text = "\n".join(chart_lines(res))
    assert "## Detection charts" in text
    assert "![Original detections](r1_original.png)" in text
    assert "![Validated detections](/x/validated.png)" in text
    assert "![Ground truth detections](r1_gt.png)" in text
    assert "Revised" not in text


def test_chart_lines_no_charts_note_and_absent_key():
    from evaluate import chart_lines

    res = {"charts": {k: {"present": False} for k in ("original", "revised", "validated")}}
    assert "(no charts" in "\n".join(chart_lines(res))
    assert chart_lines({}) == []


def test_discover_and_copy_charts(tmp_path):
    from evaluate import copy_charts, discover_charts

    run = tmp_path / "run"
    run.mkdir()
    (run / "original.png").write_bytes(b"fake-png")
    out = tmp_path / "out"
    out.mkdir()
    charts = discover_charts(str(run))
    assert charts["original"]["present"] and not charts["revised"]["present"]
    assert charts["validated"]["src"] is None
    assert charts["gt"]["present"] is False  # rendered, not discovered
    copy_charts(charts, str(out), "r1")
    assert (out / "r1_original.png").read_bytes() == b"fake-png"
    assert charts["original"]["file"] == "r1_original.png"
    assert charts["revised"]["file"] is None


def _write_eval_fixture(tmp_path):
    run = tmp_path / "r1"
    run.mkdir()
    (run / "run.json").write_text(
        json.dumps({"mode": "replay_db", "window": {"start": "2026-10-28T16:33"}})
    )
    (run / "images.jsonl").write_text(
        json.dumps({"create_time": "2026-10-28T16:33:00.304643"}) + "\n"
    )
    hour = tmp_path / "hour"
    (hour / "labels").mkdir(parents=True)
    (hour / "classes_map.json").write_text(
        json.dumps({"order": [25, 26], "names": {"25": "Qb", "26": "Qbo"}})
    )
    (hour / "labels" / "1_2026-10-28-16-33-00-304643.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    return run, hour


def test_render_gt_chart_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    from evaluate import render_gt_chart

    t = datetime.datetime(2026, 10, 28, 16, 33, 0, 304643)
    gt = {t: [Box(320, 240, 480, 360, 25, 1.0)]}
    dest = tmp_path / "r1_gt.png"
    assert render_gt_chart(gt, [t], 800, [25, 26], {"25": "Qb"}, str(dest)) is True
    assert dest.stat().st_size > 0


def test_evaluate_run_renders_gt_chart_and_embeds_it(tmp_path):
    pytest.importorskip("matplotlib")
    from evaluate import evaluate_run, render_report

    run, hour = _write_eval_fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    dest = str(out / "r1_gt.png")
    res = evaluate_run(str(run), str(hour), "labels", [0.5], gt_chart_path=dest)
    assert res["eval_frames"] == 1
    assert res["charts"]["gt"] == {"present": True, "src": dest, "file": "r1_gt.png"}
    assert (out / "r1_gt.png").stat().st_size > 0
    report = render_report(res, [0.5])
    assert "### Ground truth detections" in report
    assert "![Ground truth detections](r1_gt.png)" in report


def test_evaluate_run_without_gt_chart_path_skips_gt(tmp_path):
    from evaluate import evaluate_run

    run, hour = _write_eval_fixture(tmp_path)
    res = evaluate_run(str(run), str(hour), "labels", [0.5])
    assert res["charts"]["gt"]["present"] is False


def test_load_run_escalations_missing_and_present(tmp_path):
    from evaluate import load_run_escalations

    assert load_run_escalations(str(tmp_path)) is None
    (tmp_path / "escalations.jsonl").write_text(
        json.dumps({"name": "E1-T6", "time": "2026-10-28T16:34:05", "scope": "frame"}) + "\n"
    )
    rows = load_run_escalations(str(tmp_path))
    assert rows and rows[0]["name"] == "E1-T6"
