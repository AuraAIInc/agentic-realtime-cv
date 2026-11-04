"""Unit tests for evaluation/metrics.py and evaluate.py helpers."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import stem_to_time  # noqa: E402
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
