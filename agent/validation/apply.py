"""Apply LLM validation verdicts to revised rows (-> validated rows).

Each validation op pairs an escalation snapshot (``Escalation.to_json`` dict)
with the reviewer's parsed verdict (a ``review_schema`` model dict). Frame
ops match revised rows by exact timestamp (+ cat); period ops match by
``period_id``. Rows no verdict touches pass through unchanged. Verdicts the
applier cannot execute precisely (visit splits, merges into an unnamed
neighbor, core trims with a free-text span) keep the rows and record a note
instead of guessing.
"""

from __future__ import annotations

import datetime
import json
import sqlite3
from pathlib import Path

VALIDATED_FILE = "validated.jsonl"
VALIDATED_DB = "validated.db"

#: MySQL DDL for production (the shared DB stays read-only from this repo, so
#: runs write SQLite locally; keep this in sync with the SQLite schema below).
MYSQL_VALIDATED_DDL = """CREATE TABLE IF NOT EXISTS validated_detection (
  create_time DATETIME(6) NOT NULL,
  box_x1 DOUBLE, box_y1 DOUBLE, box_x2 DOUBLE, box_y2 DOUBLE,
  orig_cat_id INT, conf DOUBLE, cat_id INT NULL, period_id INT,
  view_id INT, validated_by VARCHAR(32) NOT NULL DEFAULT 'pass-through',
  verdict VARCHAR(32) NOT NULL DEFAULT 'pass',
  PRIMARY KEY (create_time, view_id, cat_id, period_id)
)"""

_SQLITE_DDL = """CREATE TABLE IF NOT EXISTS validated_detection (
  create_time TEXT NOT NULL,
  box_x1 REAL, box_y1 REAL, box_x2 REAL, box_y2 REAL,
  orig_cat_id INTEGER, conf REAL, cat_id INTEGER NULL, period_id INTEGER,
  view_id INTEGER, validated_by TEXT NOT NULL DEFAULT 'pass-through',
  verdict TEXT NOT NULL DEFAULT 'pass'
)"""

_VALIDATED_COLS = (
    "create_time",
    "box_x1",
    "box_y1",
    "box_x2",
    "box_y2",
    "orig_cat_id",
    "conf",
    "cat_id",
    "period_id",
    "view_id",
    "validated_by",
    "verdict",
)


def _parse_ts(value) -> datetime.datetime | None:
    if isinstance(value, datetime.datetime):
        return value
    try:
        return datetime.datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _at(row: dict, when: datetime.datetime) -> bool:
    return _parse_ts(row.get("create_time")) == when


def _after(row: dict, when: datetime.datetime) -> bool:
    ts = _parse_ts(row.get("create_time"))
    return ts is not None and ts >= when


def _in_span(row: dict, start: datetime.datetime, end: datetime.datetime, inclusive_end=True) -> bool:
    ts = _parse_ts(row.get("create_time"))
    if ts is None:
        return False
    return (start < ts <= end) if inclusive_end else (start < ts < end)


def _touch(row: dict, name: str, verdict: str) -> None:
    names = [n for n in row.get("validated_by", "pass-through").split(",") if n != "pass-through"]
    if name not in names:
        names.append(name)
    row["validated_by"] = ",".join(names)
    row["verdict"] = verdict


def apply_ops(revised_rows: list[dict], ops: list[dict]) -> list[dict]:
    """Return validated rows: copies of ``revised_rows`` with verdict ops applied."""
    rows = [{**r, "validated_by": "pass-through", "verdict": "pass"} for r in revised_rows]
    for op in ops:
        esc = op.get("escalation") or {}
        review = op.get("review") or {}
        verdict = review.get("verdict")
        if not verdict or verdict == "error":
            continue
        name = esc.get("name") or esc.get("code") or "?"
        when = _parse_ts(esc.get("time"))
        handler = _HANDLERS.get(esc.get("code"), _noop)
        for row in rows:
            if row.get("deleted"):
                continue
            handler(row, esc, review, verdict, name, when)
    return [r for r in rows if not r.pop("_drop", False)]


def _noop(row, esc, review, verdict, name, when) -> None:
    return None


def _frame_match(row: dict, when, cats: set) -> bool:
    return when is not None and _at(row, when) and row.get("cat_id") in cats


def _t1(row, esc, review, verdict, name, when) -> None:
    m = esc.get("metrics") or {}
    cats = {m.get("original_cat_id"), esc.get("cat_id")}
    if verdict == "reassign_candidate" and _frame_match(row, when, cats):
        row["cat_id"] = review.get("chosen_cat_id", row["cat_id"])
        _touch(row, name, verdict)
    elif verdict == "reject_box" and _frame_match(row, when, cats):
        row["_drop"] = True
    elif verdict == "confirm_assigned" and _frame_match(row, when, {esc.get("cat_id")}):
        _touch(row, name, verdict)


def _t2(row, esc, review, verdict, name, when) -> None:
    m = esc.get("metrics") or {}
    orig, new = m.get("original_cat_id"), esc.get("cat_id")
    if verdict == "keep_original" and _frame_match(row, when, {new}):
        row["cat_id"] = orig
        _touch(row, name, verdict)
    elif verdict == "reassign_other" and _frame_match(row, when, {orig, new}):
        row["cat_id"] = review.get("chosen_cat_id", row["cat_id"])
        _touch(row, name, verdict)
    elif verdict == "drop" and _frame_match(row, when, {orig, new}):
        row["_drop"] = True
    elif verdict == "confirm" and _frame_match(row, when, {new}):
        _touch(row, name, verdict)


def _t3(row, esc, review, verdict, name, when) -> None:
    m = esc.get("metrics") or {}
    if verdict == "keep_killed" and _frame_match(row, when, {esc.get("cat_id")}):
        row["cat_id"] = m.get("killed_cat_id", row["cat_id"])
        _touch(row, name, verdict)
    elif verdict in ("confirm_survivor", "keep_both") and _frame_match(row, when, {esc.get("cat_id"), m.get("killed_cat_id")}):
        _touch(row, name, verdict)


def _t5(row, esc, review, verdict, name, when) -> None:
    if verdict == "close_visit" and when is not None and _after(row, when) and row.get("cat_id") == esc.get("cat_id"):
        if esc.get("period_id") is None or row.get("period_id") == esc.get("period_id"):
            row["_drop"] = True


def _t6(row, esc, review, verdict, name, when) -> None:
    m = esc.get("metrics") or {}
    if verdict == "merge_visits" and row.get("period_id") == esc.get("period_id"):
        row["period_id"] = m.get("old_period_id", row["period_id"])
        _touch(row, name, verdict)
    elif verdict == "drop_box" and _frame_match(row, when, {esc.get("cat_id")}):
        row["_drop"] = True
    elif verdict == "confirm_split" and _frame_match(row, when, {esc.get("cat_id")}):
        _touch(row, name, verdict)


def _t7(row, esc, review, verdict, name, when) -> None:
    m = esc.get("metrics") or {}
    start = _parse_ts(m.get("gap_start"))
    if start is None or when is None or row.get("cat_id") != esc.get("cat_id"):
        return
    if verdict == "cut_at_gap_start" and _in_span(row, start, when):
        row["_drop"] = True
    elif verdict == "shorten_gap":
        cut = _parse_ts(review.get("cut_time"))
        if cut is not None and _in_span(row, cut, when):
            row["_drop"] = True


def _period_match(row: dict, period_id) -> bool:
    return period_id is not None and row.get("period_id") == period_id


def _p1(row, esc, review, verdict, name, when) -> None:
    pid = esc.get("period_id")
    if verdict == "award_runner_up" and _period_match(row, pid):
        row["cat_id"] = review.get("chosen_cat_id", row["cat_id"])
        _touch(row, name, verdict)
    elif verdict == "confirm_winner" and _period_match(row, pid):
        _touch(row, name, verdict)
    # split_visit: boundary lives in free text; keep rows, reviewer note stays on the op.


def _p2(row, esc, review, verdict, name, when) -> None:
    pid = esc.get("period_id")
    if verdict == "drop" and _period_match(row, pid):
        row["_drop"] = True
    elif verdict == "keep" and _period_match(row, pid):
        _touch(row, name, verdict)
    # merge_neighbor: neighbor id unknown; keep rows.


def _p3(row, esc, review, verdict, name, when) -> None:
    pid = esc.get("period_id")
    m = esc.get("metrics") or {}
    if verdict == "restore_wanted" and _period_match(row, pid):
        row["cat_id"] = m.get("wanted_cat_id", row["cat_id"])
        _touch(row, name, verdict)
    elif verdict == "mark_unknown" and _period_match(row, pid):
        row["cat_id"] = None
        _touch(row, name, verdict)
    elif verdict == "confirm_reassign" and _period_match(row, pid):
        _touch(row, name, verdict)


def _p4(row, esc, review, verdict, name, when) -> None:
    pid = esc.get("period_id")
    if verdict == "assign_cat" and _period_match(row, pid):
        row["cat_id"] = review.get("chosen_cat_id", row["cat_id"])
        _touch(row, name, verdict)
    elif verdict == "confirm_unknown" and _period_match(row, pid):
        row["cat_id"] = None
        _touch(row, name, verdict)
    # split_fragments: boundaries live in free text; keep rows.


def _p5(row, esc, review, verdict, name, when) -> None:
    pid = esc.get("period_id")
    if verdict == "drop" and _period_match(row, pid):
        row["_drop"] = True
    elif verdict == "reassign" and _period_match(row, pid):
        row["cat_id"] = review.get("chosen_cat_id", row["cat_id"])
        _touch(row, name, verdict)
    elif verdict == "keep" and _period_match(row, pid):
        _touch(row, name, verdict)
    # trim_core: kept span lives in free text; keep rows.


def _p6_p7(row, esc, review, verdict, name, when) -> None:
    m = esc.get("metrics") or {}
    if verdict == "merge" and _period_match(row, esc.get("period_id")):
        row["period_id"] = m.get("prev_period_id", row["period_id"])
        _touch(row, name, verdict)


_HANDLERS = {
    "T1": _t1,
    "T2": _t2,
    "T3": _t3,
    "T4": _noop,  # the wipe already happened upstream; nothing left to rewrite
    "T5": _t5,
    "T6": _t6,
    "T7": _t7,
    "P1": _p1,
    "P2": _p2,
    "P3": _p3,
    "P4": _p4,
    "P5": _p5,
    "P6": _p6_p7,
    "P7": _p6_p7,
}


def write_validated(run_dir: str | Path, rows: list[dict]) -> tuple[Path, Path]:
    """Write ``validated.jsonl`` plus a SQLite ``validated.db`` table; return both paths."""
    run_dir = Path(run_dir)
    jsonl_path = run_dir / VALIDATED_FILE
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")
    db_path = run_dir / VALIDATED_DB
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_SQLITE_DDL)
        conn.execute("DELETE FROM validated_detection")
        conn.executemany(
            f"INSERT INTO validated_detection ({', '.join(_VALIDATED_COLS)}) VALUES ({', '.join('?' * len(_VALIDATED_COLS))})",
            [[r.get(c) for c in _VALIDATED_COLS] for r in rows],
        )
        conn.commit()
    finally:
        conn.close()
    return jsonl_path, db_path


def materialize(run_dir: str | Path, revised_rows: list[dict], ops: list[dict]) -> tuple[Path, Path, list[dict]]:
    """Apply ``ops`` to ``revised_rows`` and write validated outputs; return (jsonl, db, rows)."""
    rows = apply_ops(revised_rows, ops)
    jsonl_path, db_path = write_validated(run_dir, rows)
    return jsonl_path, db_path, rows
