"""MySQL reads for finalized tracking data. Synchronous; the server calls these in a thread
so a slow query doesn't stall frame processing on the websocket event loop."""

from __future__ import annotations

import datetime
import time

from model.db import MysqlDB
from tracking_mcp.serialize import period_summary, py, to_float, to_int

# Same layout as the corrector rows: [create_time, x1, y1, x2, y2, original_cat_id, conf, cat_id, period_id]
REVISED_COLS = "rd.create_time, rd.box_x1, rd.box_y1, rd.box_x2, rd.box_y2, rd.original_cat_id, rd.conf, rd.cat_id, rd.period_id"
PERIOD_COLS = [
    "period_id", "start_time", "end_time", "coordinated_cat_id", "detect_length", "detect_conf", "detect_rate",
    "detect_area_account", "score", "all_detect_lengths", "all_detect_confs", "all_detect_rates",
    "all_detect_area_accounts", "all_scores", "view_id",
]  # fmt: skip
VIEW_CACHE_S = 60


class History:
    def __init__(self, db: MysqlDB):
        self.db = db
        self._views: dict | None = None
        self._cats: dict | None = None
        self._loaded_at = 0.0

    # ------------------------------------------------------------------ views / cats
    def _load(self) -> None:
        if self._views is not None and time.time() - self._loaded_at < VIEW_CACHE_S:
            return
        views = {}
        for view_id, name, cat_ids, width, height, device_id in self.db.execute_query(
            "SELECT view_id, view_name, cat_ids, width, height, device_id FROM view"
        ):
            views[int(view_id)] = {
                "view_id": int(view_id),
                "view_name": name,
                "cat_ids": [int(c) for c in str(cat_ids).split(",") if c.strip() != ""] if cat_ids else [],
                "width": to_int(width) or 640,
                "height": to_int(height) or 480,
                "device_id": device_id,
            }
        self._cats = {int(c): n for c, n in self.db.execute_query("SELECT cat_id, name FROM cats")}
        self._views, self._loaded_at = views, time.time()

    def views(self) -> list[dict]:
        self._load()
        return [{**v, "cats": {str(c): self._cats.get(c) for c in v["cat_ids"]}} for v in self._views.values()]

    def view(self, view_id) -> dict:
        self._load()
        return self._views.get(to_int(view_id)) or {"view_id": to_int(view_id), "cat_ids": [], "width": 640, "height": 480}

    def _view_filter(self, col: str, view_id) -> tuple[str, list]:
        """view_id = x, or view_id IN (every view): the (view_id, create_time, cat_id) indexes
        on detection / revised_detection / images only help when view_id is constrained."""
        if view_id is not None:
            return f"{col} = %s", [view_id]
        self._load()
        ids = list(self._views) or [-1]
        return f"{col} IN ({', '.join(['%s'] * len(ids))})", ids

    def cat_names(self) -> dict[int, str]:
        self._load()
        return dict(self._cats)

    # ------------------------------------------------------------------ periods
    def _period_dict(self, r) -> dict:
        p = dict(zip(PERIOD_COLS, r))
        cat_ids = self.view(p["view_id"])["cat_ids"] if p["view_id"] is not None else []
        return period_summary(
            p["period_id"], p["view_id"], "final" if p["start_time"] is not None else "ended_pending",
            p["start_time"], p["end_time"], p["coordinated_cat_id"],
            cat_ids, p["all_detect_lengths"], p["all_detect_confs"], p["all_scores"],
        )  # fmt: skip

    def period(self, period_id: int) -> dict | None:
        rows = self.db.execute_query(f"SELECT {', '.join(PERIOD_COLS)} FROM period WHERE period_id = %s", (period_id,))
        return self._period_dict(rows[0]) if rows else None

    def periods(
        self,
        view_id=None,
        cat_id=None,
        time_from: datetime.datetime | None = None,
        time_to: datetime.datetime | None = None,
        limit: int = 20,
        min_duration_s: float = 0,
    ) -> list[dict]:
        where, args = ["start_time IS NOT NULL"], []
        if view_id is not None:
            where.append("view_id = %s")
            args.append(view_id)
        if cat_id is not None:
            where.append("coordinated_cat_id = %s")
            args.append(cat_id)
        if time_from is not None:
            where.append("end_time >= %s")
            args.append(time_from)
        if time_to is not None:
            where.append("start_time <= %s")
            args.append(time_to)
        if min_duration_s:
            where.append("TIMESTAMPDIFF(MICROSECOND, start_time, end_time) >= %s")
            args.append(int(min_duration_s * 1_000_000))
        sql = f"SELECT {', '.join(PERIOD_COLS)} FROM period WHERE {' AND '.join(where)} ORDER BY start_time DESC LIMIT %s"
        return [self._period_dict(r) for r in self.db.execute_query(sql, (*args, int(limit)))]

    def period_rows(self, period_id: int) -> list[list]:
        sql = f"SELECT {REVISED_COLS} FROM revised_detection rd WHERE rd.period_id = %s ORDER BY rd.create_time"
        return [list(r) for r in self.db.execute_query(sql, (period_id,))]

    # ------------------------------------------------------------------ sightings
    def last_revised(self, cat_id: int, view_id=None, since: datetime.datetime | None = None) -> tuple[list, dict] | None:
        """Newest committed row tracked as cat_id since `since`, plus its period's coordination."""
        vf, vargs = self._view_filter("rd.view_id", view_id)
        where, args = [vf, "rd.cat_id = %s", "rd.box_x1 IS NOT NULL"], [*vargs, cat_id]
        if since is not None:
            where.append("rd.create_time >= %s")
            args.append(since)
        sql = f"""
        SELECT {REVISED_COLS}, rd.view_id, p.coordinated_cat_id, p.start_time
        FROM revised_detection rd LEFT JOIN period p ON rd.period_id = p.period_id
        WHERE {' AND '.join(where)} ORDER BY rd.create_time DESC LIMIT 1
        """
        rows = self.db.execute_query(sql, tuple(args))
        if not rows:
            return None
        r = list(rows[0])
        return r[:9], {"view_id": to_int(r[9]), "period_cat_id": to_int(r[10]), "period_final": r[11] is not None}

    def last_raw(self, cat_id: int, view_id=None, since: datetime.datetime | None = None) -> list | None:
        vf, vargs = self._view_filter("view_id", view_id)
        where, args = [vf, "cat_id = %s"], [*vargs, cat_id]
        if since is not None:
            where.append("create_time >= %s")
            args.append(since)
        sql = f"""
        SELECT create_time, box_x1, box_y1, box_x2, box_y2, cat_id, conf, cat_id
        FROM detection WHERE {' AND '.join(where)} ORDER BY create_time DESC LIMIT 1
        """
        rows = self.db.execute_query(sql, tuple(args))
        return list(rows[0]) + [None] if rows else None

    def raw_detections(self, time_from, time_to, cat_id=None, view_id=None, limit: int = 500) -> list[dict]:
        vf, vargs = self._view_filter("view_id", view_id)
        where, args = [vf, "create_time >= %s", "create_time <= %s"], [*vargs, time_from, time_to]
        if cat_id is not None:
            where.append("cat_id = %s")
            args.append(cat_id)
        sql = f"""
        SELECT create_time, cat_id, box_x1, box_y1, box_x2, box_y2, conf, model_id, view_id
        FROM detection WHERE {' AND '.join(where)} ORDER BY create_time LIMIT %s
        """
        return [
            {"time": py(t), "cat_id": to_int(c), "box": [round(float(v), 1) for v in (x1, y1, x2, y2)], "conf": to_float(conf),
             "model_id": model_id, "view_id": to_int(v)}
            for t, c, x1, y1, x2, y2, conf, model_id, v in self.db.execute_query(sql, (*args, int(limit)))
        ]  # fmt: skip

    def last_empty_frame(self, view_id, before: datetime.datetime, lookback_s: float) -> datetime.datetime | None:
        """Newest stored frame with no detection above threshold before `before`."""
        vf, vargs = self._view_filter("view_id", view_id)
        where = [vf, "detected = 0", "create_time < %s", "create_time > %s"]
        args = [*vargs, before, before - datetime.timedelta(seconds=lookback_s)]
        rows = self.db.execute_query(f"SELECT MAX(create_time) FROM images WHERE {' AND '.join(where)}", tuple(args))
        return rows[0][0] if rows else None
