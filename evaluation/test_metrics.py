"""Unit tests for evaluation/metrics.py and evaluate.py helpers."""

import datetime
import json
import os
import sys

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
