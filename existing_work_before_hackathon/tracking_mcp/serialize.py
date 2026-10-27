"""Corrector rows and DB values -> JSON-safe values.

Row layout shared by TrackCorrector, PeriodCorrector and the revised_detection table:
    [create_time, x1, y1, x2, y2, original_cat_id, conf, cat_id(, period_id)]
original_cat_id is None for boxes the tracker interpolated (no YOLO detection behind them).
"""

from __future__ import annotations

import datetime
import decimal
import json
import math
import re

import numpy as np

TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d-%H-%M-%S-%f",
    "%Y-%m-%d-%H-%M-%S",
    "%Y-%m-%d-%H-%M",
    "%Y-%m-%d",
)


def py(v):
    """numpy / Decimal / datetime / NaN -> plain JSON values."""
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return [py(x) for x in v.tolist()]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, decimal.Decimal):
        v = float(v)
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (datetime.datetime, datetime.date)):
        return v.isoformat()
    if isinstance(v, (list, tuple)):
        return [py(x) for x in v]
    if isinstance(v, dict):
        return {str(k): py(x) for k, x in v.items()}
    return v


def is_missing(v) -> bool:
    return v is None or (isinstance(v, (float, np.floating)) and math.isnan(v))


def to_int(v):
    if is_missing(v):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def to_float(v, ndigits: int = 4):
    return None if is_missing(v) else round(float(v), ndigits)


def image_name(t: datetime.datetime | None) -> str | None:
    """Stored image name for a capture time (the name images__get_image takes)."""
    return t.strftime("%Y-%m-%d-%H-%M-%S-%f") + ".jpg" if t is not None else None


def parse_time(s: str | None) -> datetime.datetime | None:
    """Accept 'YYYY-MM-DD HH:MM[:SS[.ffffff]]', ISO ('T' separator) or image-name style."""
    if s is None or s == "":
        return None
    s = s.strip().removesuffix(".jpg").replace("T", " ")
    s = re.sub(r"(Z|[+-]\d{2}:\d{2})$", "", s)
    for fmt in TIME_FORMATS:
        try:
            return datetime.datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"unrecognized time {s!r}; use 'YYYY-MM-DD HH:MM:SS'")


def copy_row(row) -> list:
    """Snapshot a corrector row (np object arrays are rewritten in place by interpolation)."""
    return list(row)


def has_box(row) -> bool:
    return not is_missing(row[1])


def box_of(row) -> list[float] | None:
    return [round(float(b), 1) for b in row[1:5]] if has_box(row) else None


def row_to_dict(row, final: bool | None = None, cat_id=None) -> dict | None:
    if row is None:
        return None
    row = list(row)
    t = row[0]
    boxed = has_box(row)
    d = {
        "time": py(t),
        "image_name": image_name(t),
        "box": box_of(row),
        "conf": to_float(row[6]) if boxed else None,
        "detector_cat_id": to_int(row[5]),
        "cat_id": to_int(cat_id if cat_id is not None else row[7]),
        "interpolated": boxed and is_missing(row[5]),
    }
    if len(row) > 8:
        d["period_id"] = to_int(row[8])
    if final is not None:
        d["final"] = final
    return d


def seconds_between(a: datetime.datetime | None, b: datetime.datetime | None) -> float | None:
    return round((b - a).total_seconds(), 3) if a is not None and b is not None else None


def parse_list(v):
    """'[1, 2, 3]' strings from the period table's all_* columns -> lists."""
    if v is None or isinstance(v, list):
        return v
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return v


def period_summary(period_id, view_id, status: str, start, end, cat_id, cat_ids, lengths, confs, scores) -> dict:
    """One period in the shape every tool returns. lengths / confs / scores are per-cat
    values in cat_ids order (lists, numpy arrays or the DB's '[...]' strings)."""
    vals = [parse_list(v) for v in (lengths, confs, scores)]
    evidence = None
    if cat_ids and all(v is not None and not isinstance(v, str) and len(v) == len(cat_ids) for v in vals):
        evidence = {
            str(c): {"detections": to_int(vals[0][i]), "avg_conf": to_float(vals[1][i]), "score": to_float(vals[2][i], 3)}
            for i, c in enumerate(cat_ids)
        }
    return {
        "period_id": to_int(period_id),
        "view_id": to_int(view_id),
        "status": status,
        "start": py(start),
        "end": py(end),
        "duration_s": seconds_between(start, end),
        "cat_id": to_int(cat_id),
        "evidence_by_cat": evidence,
    }
