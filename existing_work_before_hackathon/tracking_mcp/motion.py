"""Motion analysis over one period's boxes. Pure functions, no DB or corrector access.

Speed is box-center distance / time between consecutive rows, the same method as
flaskr/scr/live_detect.py:get_active_duration (threshold 3 px/s there).
"""

from __future__ import annotations

import datetime
import math
import statistics

from tracking_mcp.serialize import image_name

Point = tuple[datetime.datetime, list[float]]  # (time, [x1, y1, x2, y2])


def center(box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def center_distance(a, b) -> float:
    (ax, ay), (bx, by) = center(a), center(b)
    return math.hypot(bx - ax, by - ay)


def edge_of(box, width: float = 640, height: float = 480, margin: float = 0.15) -> str:
    """Image edge the box is closest to ('left', 'right', 'top', 'bottom'), or 'center'
    if it is more than margin x frame size away from every edge."""
    gaps = {
        "left": box[0] / width,
        "right": (width - box[2]) / width,
        "top": box[1] / height,
        "bottom": (height - box[3]) / height,
    }
    side = min(gaps, key=gaps.get)
    return side if gaps[side] <= margin else "center"


def pad_box(box, width: float = 640, height: float = 480, scale: float = 1.5) -> list[int]:
    """Box scaled around its center and clamped to the frame (a crop that keeps some context)."""
    cx, cy = center(box)
    hw, hh = (box[2] - box[0]) * scale / 2, (box[3] - box[1]) * scale / 2
    return [int(max(0, cx - hw)), int(max(0, cy - hh)), int(min(width, cx + hw)), int(min(height, cy + hh))]


def downsample(points: list[Point], n: int) -> list[dict]:
    """At most n evenly spaced box centers (always keeps the first and last)."""
    if not points:
        return []
    if len(points) > n > 1:
        step = (len(points) - 1) / (n - 1)
        idx = sorted({round(i * step) for i in range(n)})
    else:
        idx = range(len(points))
    out = []
    for i in idx:
        t, b = points[i]
        x, y = center(b)
        out.append({"time": t.isoformat(), "x": round(x, 1), "y": round(y, 1)})
    return out


def picks_by_distance(points: list[Point], n: int) -> list[int]:
    """n indices at equal steps of cumulative center distance, so a cat that sat still
    for a long time isn't over-sampled. Always includes the first and last index."""
    if not points:
        return []
    if n <= 1 or len(points) == 1:
        return [0]
    cum = [0.0]
    for (_, a), (_, b) in zip(points, points[1:]):
        cum.append(cum[-1] + center_distance(a, b))
    total = cum[-1]
    if total == 0:
        return sorted({0, len(points) - 1})
    out, j = [], 0
    for k in range(n):
        goal = total * k / (n - 1)
        while j < len(cum) - 1 and cum[j] < goal:
            j += 1
        out.append(j)
    return sorted(set(out))


def compute_motion(
    points: list[Point], speed_threshold: float = 3.0, min_moving_s: float = 2.0, merge_gap_s: float = 3.0, window_s: float = 2.0,
    min_distance_px: float = 20.0,
) -> dict:
    """Split a period into moving / resting segments.

    1. Each step between consecutive boxes is moving if the center speed over the trailing
       window_s seconds is > speed_threshold px/s. At ~3 fps, frame-to-frame box jitter alone
       is ~3 px/s, so per-step speed (window_s=0, the get_active_duration method) calls a
       sleeping cat "moving"; over 2 s the jitter drops to ~0.6 px/s median.
    2. Resting runs of <= merge_gap_s between two moving runs become moving (brief pauses).
    3. Moving runs shorter than min_moving_s, or whose net center displacement is under
       min_distance_px, become resting (one bad box; grooming or the box resizing in place
       travels a path but gets nowhere).
    """
    points = [p for k, p in enumerate(points) if k == 0 or p[0] > points[k - 1][0]]
    steps, j = [], 0
    for i in range(1, len(points)):
        (t0, b0), (t1, b1) = points[i - 1], points[i]
        while j < i - 1 and (t1 - points[j][0]).total_seconds() > window_s:
            j += 1
        tw, bw = points[j]
        dist = center_distance(b0, b1)
        speed = center_distance(bw, b1) / (t1 - tw).total_seconds()
        steps.append({"t0": t0, "t1": t1, "b0": b0, "b1": b1, "dist": dist, "dt": (t1 - t0).total_seconds(), "speed": speed})
    params = {"speed_threshold_px_s": speed_threshold, "min_moving_s": min_moving_s, "merge_gap_s": merge_gap_s, "window_s": window_s, "min_distance_px": min_distance_px}
    if not steps:
        return {"params": params, "totals": _totals(steps, []), "segments": []}

    moving = [s["speed"] > speed_threshold for s in steps]
    runs = _runs(steps, moving)
    for k in range(1, len(runs) - 1):
        state, i, j = runs[k]
        if not state and runs[k - 1][0] and runs[k + 1][0] and _span(steps, i, j) <= merge_gap_s:
            moving[i : j + 1] = [True] * (j - i + 1)
    for state, i, j in _runs(steps, moving):
        if state and (_span(steps, i, j) < min_moving_s or center_distance(steps[i]["b0"], steps[j]["b1"]) < min_distance_px):
            moving[i : j + 1] = [False] * (j - i + 1)

    segments = []
    for state, i, j in _runs(steps, moving):
        dist = sum(s["dist"] for s in steps[i : j + 1])
        dur = _span(steps, i, j)
        segments.append(
            {
                "state": "moving" if state else "resting",
                "start": steps[i]["t0"].isoformat(),
                "end": steps[j]["t1"].isoformat(),
                "duration_s": round(dur, 2),
                "distance_px": round(dist, 1),
                "net_displacement_px": round(center_distance(steps[i]["b0"], steps[j]["b1"]), 1),
                "mean_speed_px_s": round(dist / dur, 2) if dur else 0.0,
                "start_box": [round(v, 1) for v in steps[i]["b0"]],
                "end_box": [round(v, 1) for v in steps[j]["b1"]],
                "start_image": image_name(steps[i]["t0"]),
                "end_image": image_name(steps[j]["t1"]),
            }
        )
    return {"params": params, "totals": _totals(steps, moving), "segments": segments}


def _runs(steps, flags) -> list[tuple[bool, int, int]]:
    runs, start = [], 0
    for k in range(1, len(flags) + 1):
        if k == len(flags) or flags[k] != flags[start]:
            runs.append((flags[start], start, k - 1))
            start = k
    return runs


def _span(steps, i, j) -> float:
    return (steps[j]["t1"] - steps[i]["t0"]).total_seconds()


def _totals(steps, moving) -> dict:
    speeds = [s["speed"] for s in steps]
    return {
        "duration_s": round(sum(s["dt"] for s in steps), 2),
        "moving_s": round(sum(s["dt"] for s, m in zip(steps, moving) if m), 2),
        "resting_s": round(sum(s["dt"] for s, m in zip(steps, moving) if not m), 2),
        "distance_px": round(sum(s["dist"] for s in steps), 1),
        "moving_distance_px": round(sum(s["dist"] for s, m in zip(steps, moving) if m), 1),
        "speed_mean_px_s": round(statistics.fmean(speeds), 2) if speeds else 0.0,
        "speed_median_px_s": round(statistics.median(speeds), 2) if speeds else 0.0,
    }
