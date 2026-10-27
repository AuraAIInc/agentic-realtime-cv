"""Task 2: physics checks on one visit, no images.

A real cat cannot vanish from one spot and appear in another. Given a visit's detail
(tracking__get_period_detail) and the raw detector boxes over the same window
(tracking__get_raw_detections, all cats in the view), this module

1. rebuilds the visit's own detections (the raw box closest to the tracked path at each time),
2. flags anything physically implausible (teleport, mid-frame entry/exit, long gap with
   displacement, size jump, identity flip, another cat close by, a box that never moves),
3. picks a few keyframes that test each doubt, spread over space.

Distances are in body lengths (median box diagonal) so near and far cats are judged alike.
Pure functions: unit-testable without servers.
"""

from __future__ import annotations

import bisect
import datetime
import math
import statistics
from collections import Counter

DEFAULTS = {
    "teleport_body_lengths_per_s": 3.0,
    "teleport_min_body_lengths": 1.5,
    "long_gap_s": 8.0,
    "size_jump_ratio": 2.5,
    "edge_margin": 0.12,
    "overlap_body_lengths": 1.0,
    "static_min_s": 600,
    "identity_min_frames": 3,
    "max_keyframes": 6,
}
STRONG = {"teleport", "impossible_entry", "impossible_exit", "identity_flip", "size_jump"}


# ---------------------------------------------------------------------- small helpers
def parse_t(s) -> datetime.datetime:
    if isinstance(s, datetime.datetime):
        return s
    s = str(s).strip().removesuffix(".jpg").replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d-%H-%M-%S-%f"):
        try:
            return datetime.datetime.strptime(s, fmt)
        except ValueError:
            pass
    return datetime.datetime.fromisoformat(s)


def image_name(t: datetime.datetime) -> str:
    return t.strftime("%Y-%m-%d-%H-%M-%S-%f") + ".jpg"


def center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def diag(b):
    return math.hypot(b[2] - b[0], b[3] - b[1])


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def dist(a, b):
    (ax, ay), (bx, by) = center(a), center(b)
    return math.hypot(ax - bx, ay - by)


def pad_box(b, w, h, scale=1.6):
    cx, cy = center(b)
    hw, hh = (b[2] - b[0]) * scale / 2, (b[3] - b[1]) * scale / 2
    return [int(max(0, cx - hw)), int(max(0, cy - hh)), int(min(w, cx + hw)), int(min(h, cy + hh))]


def in_zone(point, zone) -> bool:
    x, y = point
    if "box" in zone:
        x1, y1, x2, y2 = zone["box"]
        return x1 <= x <= x2 and y1 <= y <= y2
    poly = zone.get("polygon") or []
    inside, j = False, len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi:
            inside = not inside
        j = i
    return inside


def box_zone_overlap(box, zone) -> float:
    """Fraction of the box's area inside the zone (polygons approximated on a 10x10 grid)."""
    if "box" in zone:
        x1, y1, x2, y2 = zone["box"]
        ix = max(0.0, min(box[2], x2) - max(box[0], x1))
        iy = max(0.0, min(box[3], y2) - max(box[1], y1))
        return round(ix * iy / max(area(box), 1e-9), 3)
    n, hits = 10, 0
    for i in range(n):
        for j in range(n):
            px = box[0] + (i + 0.5) * (box[2] - box[0]) / n
            py = box[1] + (j + 0.5) * (box[3] - box[1]) / n
            hits += in_zone((px, py), zone)
    return round(hits / (n * n), 3)


def edge_of(b, w, h, margin):
    gaps = {"left": b[0] / w, "right": (w - b[2]) / w, "top": b[1] / h, "bottom": (h - b[3]) / h}
    side = min(gaps, key=gaps.get)
    return side if gaps[side] <= margin else "center"


# ---------------------------------------------------------------------- core
def own_detections(path: list[dict], raw: list[dict], body_hint: float | None) -> tuple[list[dict], list[dict]]:
    """Split raw detections into the visit's own boxes and everything else.

    path: [{time, x, y}] tracked centres (may be downsampled); raw: [{time, cat_id, box, conf}].
    A raw box belongs to the visit if it is the closest box to the interpolated path centre at
    its time and within ~1.5 body lengths of it.
    """
    if not path:
        return [], raw
    pts = sorted(((parse_t(p["time"]), p["x"], p["y"]) for p in path), key=lambda p: p[0])
    times = [p[0] for p in pts]

    def path_at(t):
        i = bisect.bisect_left(times, t)
        if i <= 0:
            return pts[0][1:], (times[0] - t).total_seconds()
        if i >= len(pts):
            return pts[-1][1:], (t - times[-1]).total_seconds()
        (t0, x0, y0), (t1, x1, y1) = pts[i - 1], pts[i]
        span = (t1 - t0).total_seconds() or 1e-9
        a = (t - t0).total_seconds() / span
        return (x0 + a * (x1 - x0), y0 + a * (y1 - y0)), 0.0

    bl = body_hint or statistics.median([diag(r["box"]) for r in raw]) if raw else (body_hint or 100.0)
    by_time: dict[datetime.datetime, list[dict]] = {}
    for r in raw:
        by_time.setdefault(parse_t(r["time"]), []).append(r)
    own, other = [], []
    for t, rows in sorted(by_time.items()):
        (px, py), outside_s = path_at(t)
        best, best_d = None, None
        for r in rows:
            cx, cy = center(r["box"])
            d = math.hypot(cx - px, cy - py)
            if best is None or d < best_d:
                best, best_d = r, d
        if outside_s <= 1.0 and best_d is not None and best_d <= 1.5 * bl:
            own.append({**best, "t": t})
            other += [{**r, "t": t} for r in rows if r is not best]
        else:
            other += [{**r, "t": t} for r in rows]
    return own, other


def analyze(detail: dict, raw: list[dict], view_size: tuple[int, int], zones: list[dict] | None = None, cfg: dict | None = None) -> dict:
    c = {**DEFAULTS, **(cfg or {})}
    zones = zones or []
    w, h = view_size
    period = detail.get("period") or {}
    status = detail.get("status") or period.get("status")
    best = (detail.get("best_frame") or {}).get("box")
    own, other = own_detections(detail.get("path") or [], raw, diag(best) if best else None)
    flags: list[dict] = []
    if not own:
        return {
            "period_id": detail.get("period_id"), "view_id": detail.get("view_id"), "status": status,
            "continuous": None, "flags": [{"type": "no_detections", "detail": "no raw detections match this visit's path"}],
            "body_length_px": None, "own_detections": 0, "identity_runs": [], "keyframes": [], "strong_flags": 1,
        }
    bl = statistics.median(diag(o["box"]) for o in own) or 1.0
    passable = [z for z in zones if z.get("kind") in ("occluder", "entry")]

    def covered(*boxes):
        return any(in_zone(center(b), z) for b in boxes for z in passable)

    # teleports, long gaps, size jumps
    for a, b in zip(own, own[1:]):
        dt = max((b["t"] - a["t"]).total_seconds(), 1e-3)
        jump = dist(a["box"], b["box"]) / bl
        if jump >= c["teleport_min_body_lengths"] and jump / max(dt, 0.34) > c["teleport_body_lengths_per_s"] and not covered(a["box"], b["box"]):
            flags.append(_flag("teleport", a, b, jump_body_lengths=round(jump, 2), gap_s=round(dt, 2)))
        elif dt > c["long_gap_s"] and jump > 1.0:
            flags.append(_flag("long_gap", a, b, jump_body_lengths=round(jump, 2), gap_s=round(dt, 2), behind_occluder=covered(a["box"], b["box"])))
        ratio = max(area(a["box"]), 1) / max(area(b["box"]), 1)
        if dt <= 2.0 and max(ratio, 1 / ratio) > c["size_jump_ratio"]:
            flags.append(_flag("size_jump", a, b, area_ratio=round(max(ratio, 1 / ratio), 2)))

    # entry / exit
    m = c["edge_margin"]
    first, last = own[0], own[-1]
    if edge_of(first["box"], w, h, m) == "center" and not covered(first["box"]):
        flags.append({"type": "impossible_entry", "time": first["t"].isoformat(), "image_name": image_name(first["t"]), "box": first["box"],
                      "detail": "first box is mid-frame, not at an edge, doorway or occluder"})
    if status in ("final", "ended_pending") and edge_of(last["box"], w, h, m) == "center" and not covered(last["box"]):
        flags.append({"type": "impossible_exit", "time": last["t"].isoformat(), "image_name": image_name(last["t"]), "box": last["box"],
                      "detail": "visit ended with the last box mid-frame"})

    # identity runs
    runs = _identity_runs(own)
    majority = Counter(o.get("cat_id") for o in own).most_common(1)[0][0]
    for r in runs:
        if r["detector_cat_id"] != majority and r["frames"] >= c["identity_min_frames"]:
            flags.append({"type": "identity_flip", "time": r["from"], "until": r["to"], "detector_cat_id": r["detector_cat_id"],
                          "majority_cat_id": majority, "frames": r["frames"], "image_name": image_name(parse_t(r["from"]))})

    # another animal close by
    close = []
    for o in other:
        i = bisect.bisect_left([x["t"] for x in own], o["t"])
        for j in (i - 1, i):
            if 0 <= j < len(own) and abs((own[j]["t"] - o["t"]).total_seconds()) <= 0.5 and dist(own[j]["box"], o["box"]) / bl <= c["overlap_body_lengths"]:
                close.append((o["t"], o))
                break
    for start, end, sample in _windows(close, gap_s=3.0):
        flags.append({"type": "other_cat_close", "time": start.isoformat(), "until": end.isoformat(),
                      "other_detector_cat_id": sample.get("cat_id"), "image_name": image_name(start), "box": sample["box"]})

    # never moves
    totals = (detail.get("motion") or {}).get("totals") or {}
    dur = totals.get("duration_s") or period.get("duration_s") or 0
    if dur >= c["static_min_s"] and (totals.get("moving_s") or 0) == 0:
        flags.append({"type": "static", "detail": "box never moved; a sleeping cat or a decoy (needs a look, never auto-reject)",
                      "duration_s": dur, "image_name": image_name(own[len(own) // 2]["t"]), "box": own[len(own) // 2]["box"]})

    keyframes = pick_keyframes(own, flags, detail, c["max_keyframes"], (w, h))
    strong = sum(f["type"] in STRONG for f in flags)
    return {
        "period_id": detail.get("period_id"),
        "view_id": detail.get("view_id"),
        "status": status,
        "tracked_cat_id": period.get("cat_id"),
        "detector_agrees_fraction": (detail.get("frames") or {}).get("detector_agrees_fraction"),
        "continuous": not flags,
        "strong_flags": strong,
        "flags": flags,
        "body_length_px": round(bl, 1),
        "own_detections": len(own),
        "other_detections": len(other),
        "identity_runs": runs,
        "keyframes": keyframes,
    }


def _flag(kind, a, b, **extra) -> dict:
    return {"type": kind, "time": b["t"].isoformat(), "before": {"time": a["t"].isoformat(), "image_name": image_name(a["t"]), "box": a["box"]},
            "after": {"time": b["t"].isoformat(), "image_name": image_name(b["t"]), "box": b["box"]}, **extra}


def _identity_runs(own) -> list[dict]:
    runs = []
    for o in own:
        cid = o.get("cat_id")
        if runs and runs[-1]["detector_cat_id"] == cid:
            runs[-1]["to"], runs[-1]["frames"] = o["t"].isoformat(), runs[-1]["frames"] + 1
        else:
            runs.append({"detector_cat_id": cid, "from": o["t"].isoformat(), "to": o["t"].isoformat(), "frames": 1})
    return runs


def _windows(items, gap_s):
    out = []
    for t, o in sorted(items, key=lambda x: x[0]):
        if out and (t - out[-1][1]).total_seconds() <= gap_s:
            out[-1][1] = t
        else:
            out.append([t, t, o])
    return [tuple(x) for x in out]


def pick_keyframes(own: list[dict], flags: list[dict], detail: dict, k: int, size) -> list[dict]:
    """Frames that test the specific doubts first, then spread over space, then the best
    frame per motion segment. Returns at most k, in time order, numbered as tiles."""
    w, h = size
    by_time = {o["t"]: o for o in own}
    times = sorted(by_time)
    chosen: dict[datetime.datetime, str] = {}

    def add(t, role):
        if t in by_time and t not in chosen and len(chosen) < k:
            chosen[t] = role

    def nearest(ts):
        i = bisect.bisect_left(times, ts)
        cands = [times[j] for j in (i - 1, i) if 0 <= j < len(times)]
        return min(cands, key=lambda x: abs((x - ts).total_seconds())) if cands else None

    for f in flags:  # 1. around each doubt
        if "before" in f:
            add(parse_t(f["before"]["time"]), f"before_{f['type']}")
            add(parse_t(f["after"]["time"]), f"after_{f['type']}")
        elif f.get("time"):
            t = nearest(parse_t(f["time"]))
            if t:
                add(t, f["type"])
    add(times[0], "entry")
    add(times[-1], "last")
    # 2. spatial spread: farthest-point sampling over box centres
    picked = list(chosen) or [times[0]]
    while len(chosen) < k:
        far = max(times, key=lambda t: min(dist(by_time[t]["box"], by_time[p]["box"]) for p in picked))
        if far in chosen or min(dist(by_time[far]["box"], by_time[p]["box"]) for p in picked) < 1:
            break
        add(far, "spread")
        picked.append(far)
    # 3. best conf per motion segment
    for seg in (detail.get("motion") or {}).get("segments", []):
        s, e = parse_t(seg["start"]), parse_t(seg["end"])
        inside = [t for t in times if s <= t <= e]
        if inside:
            add(max(inside, key=lambda t: by_time[t].get("conf") or 0), f"best_{seg['state']}")
    out = []
    for i, t in enumerate(sorted(chosen), 1):
        o = by_time[t]
        out.append({"tile": i, "role": chosen[t], "time": t.isoformat(), "image_name": image_name(t), "box": o["box"],
                    "crop_box": pad_box(o["box"], w, h), "conf": o.get("conf"), "detector_cat_id": o.get("cat_id")})
    return out


def motion_state_at(detail: dict, t: datetime.datetime) -> dict | None:
    """The motion segment (moving / resting) covering time t, if any."""
    for seg in (detail.get("motion") or {}).get("segments", []):
        if parse_t(seg["start"]) <= t <= parse_t(seg["end"]):
            return seg
    return None
