"""Decoy zones at ingest, same-id collisions, and the detector-insists relabel veto."""

from __future__ import annotations

import datetime

from agent.main import build_img_detected
from agent.tracking import TrackCorrector
from agent.tracking.decoys import load_decoys, match_decoy

T0 = datetime.datetime(2026, 10, 28, 16, 33, 0)
BALL = [348.0, 46.0, 408.0, 106.0]


def _det(cat_id, box, conf):
    return {"cat_id": cat_id, "confidence": conf, "box": box}


def test_load_decoys_reads_only_decoy_kind(tmp_path):
    p = tmp_path / "zones.yaml"
    p.write_text(
        "views:\n"
        "  '3':\n"
        "    zones:\n"
        "      - {name: sofa, kind: place, box: [0, 80, 320, 420]}\n"
        "      - {name: exercise_ball, kind: decoy, box: [348, 46, 408, 106]}\n"
        "      - {name: strict, kind: decoy, box: [0, 0, 10, 10], min_iou: 0.9}\n"
    )
    decoys = load_decoys(3, p)
    assert [d["name"] for d in decoys] == ["exercise_ball", "strict"]
    assert decoys[0]["iou"] == 0.6 and decoys[1]["iou"] == 0.9
    assert load_decoys(4, p) == []
    assert load_decoys(3, tmp_path / "missing.yaml") == []


def test_match_decoy_keeps_a_cat_lying_beside_the_object():
    decoys = [{"name": "exercise_ball", "box": BALL, "iou": 0.6}]
    assert match_decoy([345, 46, 408, 106], decoys) == "exercise_ball"  # the ball itself (10-25 median box)
    assert match_decoy([351, 48, 458, 108], decoys) is None  # ball + cat next to it: IoU ~0.5, kept
    assert match_decoy([100, 300, 200, 400], decoys) is None


def test_build_img_detected_drops_decoys_and_keeps_most_confident_duplicate():
    decoys = [{"name": "exercise_ball", "box": BALL, "iou": 0.6}]
    dets = [
        _det(26, [345, 46, 408, 106], 0.75),  # the ball, labeled Qbo
        _det(26, [140, 350, 230, 460], 0.82),  # the real Qbo
        _det(25, [170, 450, 280, 590], 0.60),
        _det(25, [300, 360, 370, 530], 0.80),  # same id twice: the more confident box wins
    ]
    img, stats = build_img_detected(dets, min_conf=0.5, known_ids=[25, 26], decoys=decoys)
    assert img[26][:4] == [140.0, 350.0, 230.0, 460.0]
    assert img[25][:4] == [300.0, 360.0, 370.0, 530.0]
    assert stats["decoy"] == 1 and stats["dup_cat"] == 1 and stats["kept"] == 2
    # without decoys the ball competes with the real Qbo and loses on confidence; nothing is dropped
    img2, stats2 = build_img_detected(dets, min_conf=0.5, known_ids=[25, 26])
    assert stats2["decoy"] == 0 and img2[26][5] == 0.82


def _feed(tc, start, n, img_fn, step=0.36):
    rows = []
    for i in range(n):
        t = start + datetime.timedelta(seconds=i * step)
        rows += [r for _, r in tc.process_images(t, img_fn(i), 0, 0, 3, 0)]
    return rows


def test_detector_insists_stops_a_continuity_hijack():
    """Cat 1 leaves a spot; cat 0 settles there and the camera keeps saying cat 0.

    Without the veto every cat-0 box overlaps cat 1's last box and is filed under cat 1 for
    as long as cat 0 stays (2026-10-28: 772 of 818 Qb boxes). With it, the relabel stops after
    DETECTOR_INSIST_MIN frames and cat 0 gets its own track.
    """
    spot = [200, 500, 300, 598]
    tc = TrackCorrector(cat_ids=[0, 1])
    _feed(tc, T0, 10, lambda i: {1: spot + [1, 0.55]})  # cat 1 sleeps here
    later = T0 + datetime.timedelta(seconds=4)
    _feed(tc, later, 60, lambda i: {0: spot + [0, 0.75]})  # camera now says cat 0, every frame
    assert tc.relabel_vetoes >= 1
    last = [b for b in tc.buffer[0] if b[1] is not None]
    assert last, "cat 0 should own the recent boxes"
    assert all(b[5] == 0 for b in last)
    assert not any(b[1] is not None and b[5] == 0 for b in list(tc.buffer[1])[-10:])


def test_detector_insists_needs_a_consistent_majority():
    tc = TrackCorrector(cat_ids=[0, 1])
    spot = [200, 500, 300, 598]
    # cat 1's buffer: 4 boxes the camera called 0, 6 it called 1 -> not insistent
    _feed(tc, T0, 6, lambda i: {1: spot + [1, 0.7]})
    for i in range(4):
        tc.buffer[1].append(tc.buffer[1][-1].copy())
        tc.buffer[1][-1][5] = 0
    assert not tc.detector_insists(0, 1)
