"""Read-only views of the TrackCorrector / PeriodCorrector memory in detection_ingest_server.py.

Everything here runs on the websocket server's event loop. process_image() is synchronous
on that loop too, so as long as these functions don't await, they see a consistent state
between two frames. They copy what they return and never call a corrector method that
writes (process_*, collect_buffer, finish_up, save_*). The only corrector methods used,
coordinate_period_groups / get_period_stats, are pure.
"""

from __future__ import annotations

import datetime
from collections import defaultdict, deque

from tracking_mcp.serialize import (
    copy_row,
    has_box,
    image_name,
    is_missing,
    period_summary,
    py,
    row_to_dict,
    seconds_between,
    to_float,
    to_int,
)

# PeriodCorrector.save_period_to_db yields rows in this order
PERIOD_ROW_FIELDS = [
    "start_time", "end_time", "coordinated_cat_id", "detect_length", "detect_conf", "detect_rate",
    "detect_area_count", "score", "all_detect_lengths", "all_detect_confs", "all_detect_rates",
    "all_detect_area_counts", "all_scores", "view_id", "period_id",
]  # fmt: skip


class LiveState:
    def __init__(self, track_pool: dict, period_pool: dict):
        self.track_pool = track_pool  # view_id -> TrackCorrector (same dict detection_ingest_server uses)
        self.period_pool = period_pool  # view_id -> PeriodCorrector
        self.last_frame: dict = {}  # view key -> last message received for that view
        self.recent_periods: dict = defaultdict(lambda: deque(maxlen=100))  # view key -> finalized periods
        self.started_at = datetime.datetime.now()

    # ------------------------------------------------------------------ hooks
    def record_frame(self, detection: dict, received_time: datetime.datetime, min_confidence: float) -> None:
        """Called by detection_ingest_server for every message, before tracking."""
        dets = [
            {"cat_id": to_int(d.get("cat_id")), "box": py(d.get("box")), "conf": to_float(d.get("confidence"))}
            for d in detection.get("detections", [])
        ]
        self.last_frame[detection["view_id"]] = {
            "create_time": detection["create_time"],  # str "YYYY-mm-dd-HH-MM-SS-ffffff" as sent
            "received_time": received_time,
            "model_id": detection.get("model_id"),
            "device_id": detection.get("device_id"),
            "is_grey": detection.get("is_grey"),
            "detections": dets,
            "passed_threshold": [d for d in dets if d["conf"] is not None and d["conf"] > min_confidence],
            "min_confidence": min_confidence,
        }

    def record_period(self, view_id, row) -> None:
        """Called by detection_ingest_server for every period the PeriodCorrector coordinates."""
        if row is None:
            return
        p = dict(zip(PERIOD_ROW_FIELDS, row))
        out = period_summary(
            p["period_id"], p["view_id"], "final", p["start_time"], p["end_time"], p["coordinated_cat_id"],
            self.cat_order(view_id), p["all_detect_lengths"], p["all_detect_confs"], p["all_scores"],
        )  # fmt: skip
        self.recent_periods[view_id].appendleft({**out, "finalized_at": py(datetime.datetime.now())})

    # ------------------------------------------------------------------ lookup
    def _find(self, pool: dict, view_id):
        for k in pool:
            if str(k) == str(view_id):
                return k
        return None

    def view_ids(self) -> list:
        keys = {str(k) for k in self.track_pool} | {str(k) for k in self.period_pool} | {str(k) for k in self.last_frame}
        return sorted((to_int(k) if to_int(k) is not None else k for k in keys), key=str)

    def tracker(self, view_id):
        k = self._find(self.track_pool, view_id)
        return self.track_pool[k] if k is not None else None

    def periods(self, view_id):
        k = self._find(self.period_pool, view_id)
        return self.period_pool[k] if k is not None else None

    def frame(self, view_id) -> dict | None:
        k = self._find(self.last_frame, view_id)
        return self.last_frame.get(k) if k is not None else None

    def finalized(self, view_id) -> list[dict]:
        k = self._find(self.recent_periods, view_id)
        return list(self.recent_periods[k]) if k is not None else []

    def cat_order(self, view_id) -> list:
        """Cat ids in the order of the PeriodCorrector's per-cat arrays."""
        pc = self.periods(view_id)
        return [pc.idx_to_id[k] for k in sorted(pc.idx_to_id)] if pc is not None else []

    def views_with_cat(self, cat_id: int) -> list:
        return [v for v in self.view_ids() if (tc := self.tracker(v)) is not None and cat_id in tc.id_to_idx]

    # ------------------------------------------------------------------ snapshots
    def frame_snapshot(self, view_id, now: datetime.datetime) -> dict | None:
        f = self.frame(view_id)
        if f is None:
            return None
        t = _parse_create_time(f["create_time"])
        return {
            "time": py(t),
            "image_name": image_name(t),
            "seconds_ago": seconds_between(t, now),
            "detections": f["detections"],
            "min_confidence": f["min_confidence"],
        }

    def tracker_cat(self, view_id, cat_id: int) -> dict | None:
        tc = self.tracker(view_id)
        if tc is None or cat_id not in tc.id_to_idx:
            return None
        i = tc.id_to_idx[cat_id]
        real = [r for r in tc.buffer[i] if has_box(r) and not is_missing(r[5])]
        on_screen = bool(tc.detecting[i])
        return {
            "cat_id": cat_id,
            "on_screen": on_screen,
            "current_period_id": to_int(tc.period_id[i]) if on_screen else None,
            "last_period_id": to_int(tc.period_id[i]) or None,
            "latest_detection": row_to_dict(copy_row(real[-1]), final=False, cat_id=cat_id) if real else None,
        }

    def pending_rows(self, view_id, period_id: int) -> list[list]:
        """Boxes still in the tracker buffer that will be committed to this (open) period
        unless a teleport or jump reset reassigns them. Not in the DB yet."""
        tc = self.tracker(view_id)
        if tc is None:
            return []
        out = []
        for i, cat_id in tc.idx_to_id.items():
            if tc.detecting[i] and to_int(tc.period_id[i]) == period_id:
                for r in tc.buffer[i]:
                    if has_box(r):
                        r = copy_row(r)[:8]
                        r[7] = cat_id
                        out.append(r + [period_id])
        return out

    def committed_rows(self, view_id, period_id: int | None = None, cat_id: int | None = None) -> list[list]:
        """Committed rows (9 fields) the PeriodCorrector holds for today, optionally filtered."""
        pc = self.periods(view_id)
        if pc is None:
            return []
        return [
            copy_row(d)
            for d in pc.detections
            if (period_id is None or to_int(d[8]) == period_id) and (cat_id is None or to_int(d[7]) == cat_id)
        ]

    def find_period_view(self, period_id: int):
        """The view whose correctors hold this period (open, pending or committed today)."""
        for v in self.view_ids():
            pc = self.periods(v)
            if pc is not None and (period_id in pc.connected_periods or any(to_int(d[8]) == period_id for d in pc.detections)):
                return v
            tc = self.tracker(v)
            if tc is not None and any(tc.detecting[i] and to_int(tc.period_id[i]) == period_id for i in tc.idx_to_id):
                return v
        return None

    def period_states(self, view_id, now: datetime.datetime, with_scores: bool = True) -> list[dict]:
        """Periods the PeriodCorrector hasn't coordinated yet (open, or closed and waiting
        for the rest of their overlapping group), with provisional scores."""
        pc = self.periods(view_id)
        if pc is None:
            return []
        wanted = {to_int(k) for k in pc.connected_periods}
        by_period = defaultdict(list)
        for d in pc.detections:
            if to_int(d[8]) in wanted:
                by_period[to_int(d[8])].append(copy_row(d))
        cat_of = {to_int(pid): cat for cat, pid in pc.cat_period.items() if pid is not None}
        scores = self.provisional_scores(view_id) if with_scores else {}
        cat_ids = self.cat_order(view_id)
        out = []
        for pid, span in pc.connected_periods.items():
            pid = to_int(pid)
            rows = by_period.get(pid, [])
            start = span[0]
            end = span[1] if len(span) > 1 else pc.last_period_time.get(pid, rows[-1][0] if rows else start)
            sc = scores.get(pid) or {}
            cat = sc.get("cat_id")
            if cat is None:  # no decision possible yet: fall back to the track's label
                cat = to_int(rows[-1][7]) if rows else cat_of.get(pid)
            p = period_summary(
                pid, view_id, "ongoing" if pid in pc.started_period_starts else "ended_pending", start, end, cat,
                cat_ids, sc.get("lengths"), sc.get("confs"), sc.get("scores"),
            )  # fmt: skip
            p.update(
                {
                    "frames": len(rows),
                    "first_frame": row_to_dict(rows[0], final=True) if rows else None,
                    "last_frame": row_to_dict(rows[-1], final=True) if rows else None,
                }
            )
            out.append(p)
        return sorted(out, key=lambda p: p["start"] or "")

    def provisional_scores(self, view_id) -> dict:
        """period_id -> what PeriodCorrector would decide if the group were coordinated now
        (cat_id plus per-cat lengths / confs / scores in cat_order). Uses
        coordinate_period_groups(), which only reads corrector state."""
        pc = self.periods(view_id)
        if pc is None or not pc.connected_periods:
            return {}
        try:
            coordinated, stats, score = pc.coordinate_period_groups()
        except Exception:  # asserts / empty periods while a group is half built
            return {}
        return {
            to_int(pid): {"cat_id": to_int(coordinated.get(pid)), "lengths": st["detect_length"], "confs": st["detect_conf"], "scores": score[pid]}
            for pid, st in stats.items()
        }


def _parse_create_time(v):
    if isinstance(v, datetime.datetime):
        return v
    try:
        return datetime.datetime.strptime(v, "%Y-%m-%d-%H-%M-%S-%f")
    except (TypeError, ValueError):
        return None
