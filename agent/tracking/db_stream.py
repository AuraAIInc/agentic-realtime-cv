"""Read-only MySQL stream source for `--replay-db` (Phase 3 of 5).

Yields camera-format messages straight from the `images`/`detection`
tables — no intermediate file, no writes, ever. Empty frames are kept so
buffer timeouts advance exactly like the live stream.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

CAMERA_TIME_FORMAT = "%Y-%m-%d-%H-%M-%S-%f"


def fetch_messages(db, view_id: int, start: datetime.datetime, end_exclusive: datetime.datetime):
    """Yield one camera message per frame in [start, end_exclusive), time-ordered.

    ``db`` is anything with ``execute_query(sql, args)`` (``MysqlDB`` in
    production, a stub in tests). Read-only: a single SELECT.
    """
    sql = """
    SELECT img.create_time, det.cat_id, det.box_x1, det.box_y1, det.box_x2, det.box_y2, det.conf
    FROM images img
    LEFT JOIN detection det
    ON img.create_time = det.create_time AND det.view_id <=> img.view_id
    WHERE (img.view_id = %s OR img.view_id IS NULL) AND img.create_time >= %s AND img.create_time < %s
    ORDER BY img.create_time
    """
    frames: dict = {}
    for create_time, cat_id, x1, y1, x2, y2, conf in db.execute_query(sql, (view_id, start, end_exclusive)):
        if isinstance(create_time, str):
            create_time = datetime.datetime.fromisoformat(create_time)
        msg = frames.get(create_time)
        if msg is None:
            msg = {"create_time": create_time.strftime(CAMERA_TIME_FORMAT), "view_id": view_id, "detections": []}
            frames[create_time] = msg
        if cat_id is not None:
            msg["detections"].append(
                {"cat_id": int(cat_id), "confidence": float(conf), "box": [float(x1), float(y1), float(x2), float(y2)]}
            )
    for create_time in sorted(frames):
        yield frames[create_time]


def connect():
    """Connect to MySQL (read-only use). Imports legacy model.db on demand."""
    legacy = Path(__file__).resolve().parents[2] / "existing_work_before_hackathon"
    if str(legacy) not in sys.path:
        sys.path.insert(0, str(legacy))
    from model.db import MysqlDB

    return MysqlDB()
