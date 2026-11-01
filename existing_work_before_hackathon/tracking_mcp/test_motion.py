"""Checks for motion.py and serialize.py on synthetic tracks (no DB).

    python -m tracking_mcp.test_motion
"""

import datetime

import numpy as np

from tracking_mcp import motion
from tracking_mcp.serialize import parse_time, row_to_dict

T0 = datetime.datetime(2026, 10, 25, 10, 0, 0)


def track(spec, fps=3, jitter=0.0, seed=0):
    """spec: [(seconds, px_per_s), ...] moving along x; jitter adds +-px noise to every box."""
    rng = np.random.default_rng(seed)
    out, t, x = [], T0, 100.0
    for secs, v in spec:
        for _ in range(secs * fps):
            j = rng.uniform(-jitter, jitter, 4) if jitter else np.zeros(4)
            out.append((t, [x + j[0], 200 + j[1], x + 50 + j[2], 250 + j[3]]))
            t += datetime.timedelta(seconds=1 / fps)
            x += v / fps
    return out


def states(m):
    return [s["state"] for s in m["segments"]]


def test_walk_pause_walk_merges():
    m = motion.compute_motion(track([(10, 0), (5, 20), (2, 0), (5, 20), (20, 0)]))
    assert states(m) == ["resting", "moving", "resting"], m["segments"]
    assert m["segments"][1]["net_displacement_px"] > 150


def test_jitter_is_not_movement():
    m = motion.compute_motion(track([(60, 0)], jitter=3))
    assert states(m) == ["resting"], m["segments"]


def test_twitch_that_returns_is_not_movement():
    m = motion.compute_motion(track([(10, 0), (2, 15), (2, -15), (10, 0)]))
    assert "moving" not in states(m), m["segments"]


def test_edges_and_crop():
    assert motion.edge_of([0, 200, 50, 250]) == "left"
    assert motion.edge_of([300, 200, 350, 250]) == "center"
    assert motion.edge_of([300, 400, 350, 478]) == "bottom"
    assert motion.pad_box([0, 0, 100, 100]) == [0, 0, 125, 125]


def test_picks_by_distance_skip_resting_frames():
    pts = track([(30, 0), (5, 40)])
    picks = motion.picks_by_distance(pts, 4)
    assert picks[0] == 0 and len(picks) == 4
    assert all(k >= 30 * 3 - 1 for k in picks[1:]), picks  # nothing sampled from the long rest


def test_row_to_dict_handles_numpy_and_interpolated():
    row = np.array([T0, 1.0, 2, 3, 4, None, np.float64(0.73), "0", np.int64(5)], dtype=object)
    d = row_to_dict(row)
    assert d["interpolated"] and d["cat_id"] == 0 and d["period_id"] == 5
    assert d["image_name"] == "2026-10-25-10-00-00-000000.jpg"


def test_parse_time_formats():
    assert parse_time("2026-10-25 10:00") == datetime.datetime(2026, 10, 25, 10, 0)
    assert parse_time("2026-10-25T10:00:01.5") == datetime.datetime(2026, 10, 25, 10, 0, 1, 500000)
    assert parse_time("2026-10-25-10-00-01-000500.jpg") == datetime.datetime(2026, 10, 25, 10, 0, 1, 500)


def test_compress_path_collapses_static_runs():
    pts = track([(60, 0)])  # 180 identical centers
    out = motion.compress_path(pts, 60)
    assert len(out) == 2, len(out)  # first + last only
    assert out[0]["time"] < out[-1]["time"]


def test_compress_path_keeps_moves_and_endpoints():
    pts = track([(10, 0), (5, 40), (10, 0)])
    out = motion.compress_path(pts, 60)
    xs = [p["x"] for p in out]
    assert len(out) < len(pts) and max(xs) - min(xs) > 150
    assert out[0]["x"] == pts[0][1][0] + 25 and out[-1]["x"] == pts[-1][1][0] + 25


def test_travel_span_and_empty_evidence():
    from tracking_mcp import server as srv

    rows = [([T0 + datetime.timedelta(seconds=i), 100, 100, 150, 150, 26, 0.6, 26, 1], True, "period") for i in range(5)]
    assert srv._travel_span(rows) == 0.0
    rows[-1][0][1] += 100
    rows[-1][0][3] += 100  # whole box moves 100 px right
    assert srv._travel_span(rows) == 100.0
    periods = [{"evidence_by_cat": {"25": {"detections": 0, "avg_conf": 0.0, "score": 0.0},
                                    "26": {"detections": 5, "avg_conf": 0.6, "score": 3.0}}}]
    srv._drop_empty_evidence(periods)
    assert periods[0]["evidence_by_cat"] == {"26": {"detections": 5, "avg_conf": 0.6, "score": 3.0}}


def test_utcnow_str_labels_utc():
    from tracking_mcp import server as srv

    assert srv._utcnow_str(datetime.datetime(2026, 11, 1, 7, 35, 3)).endswith("+00:00")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
