"""Apply LLM validation verdicts to revised rows (-> validated rows).

Each validation op pairs an escalation snapshot (``Escalation.to_json`` dict)
with the reviewer's parsed verdict (a ``review_schema`` model dict). Frame
ops match revised rows by exact timestamp (+ tracked cat, + box overlap
when the escalation names its disputed box); period ops match by tracked
``period_id``. Matching uses pre-apply values so earlier ops cannot steal
or hide later ops' rows. Rows no verdict touches pass through unchanged.
Verdicts the applier cannot
execute precisely (visit splits, merges into an unnamed neighbor, core
trims with a free-text span) keep the rows and record a note instead of
guessing.

A decided merge also bridges the gap: one interpolated row per whole second
between the two boundary detections, marked ``gap_filled`` (``conf`` 0 like
tracker-interpolated rows) so the merged visit reads as one track.
"""

from __future__ import annotations

import bisect
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


def _touch(row: dict, name: str, verdict: str, note: str | None = None) -> None:
    names = [n for n in row.get("validated_by", "pass-through").split(",") if n != "pass-through"]
    if name not in names:
        names.append(name)
    row["validated_by"] = ",".join(names)
    row["verdict"] = verdict
    if note:
        prev = row.get("note")
        row["note"] = f"{prev}; {note}" if prev else note


#: Minimum box IoU for a frame op to claim a row when the escalation names
#: its disputed box. Timestamp+cat alone misfires when two boxes share one
#: frame (one op's rewrite can match a later op's cat filter).
_FRAME_BOX_IOU = 0.5


def _iou(box_a, box_b) -> float:
    try:
        ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
        bx1, by1, bx2, by2 = (float(v) for v in box_b)
    except (TypeError, ValueError):
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _row_box(row: dict):
    box = [row.get(k) for k in ("box_x1", "box_y1", "box_x2", "box_y2")]
    return box if all(isinstance(v, (int, float)) for v in box) else None


#: (code, verdict, metrics key holding the merge target) for verdicts that
#: renumber one period into another.
_MERGE_VERDICTS = {
    "P6": ("merge", "prev_period_id"),
    "P7": ("merge", "prev_period_id"),
    "T6": ("merge_visits", "old_period_id"),
}


def _canonical_merge_targets(ops: list[dict]) -> dict[int, int]:
    """Map each merged-away period id to its final representative (the min id).

    Chained merges (2->1, 3->2, ...) must collapse transitively: sequential
    application would cascade-shift ids instead of merging. Union-find over
    all merge pairs first, then every row lands on the set minimum whatever
    order the ops arrive in.
    """
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    merged: set[int] = set()
    for op in ops:
        esc = op.get("escalation") or {}
        review = op.get("review") or {}
        spec = _MERGE_VERDICTS.get(esc.get("code"))
        if spec is None or review.get("verdict") != spec[0]:
            continue
        pid = esc.get("period_id")
        target = (esc.get("metrics") or {}).get(spec[1])
        if isinstance(pid, int) and isinstance(target, int) and pid != target:
            union(pid, target)
            merged.add(pid)
    return {pid: find(pid) for pid in merged}


def _row_time(row: dict) -> datetime.datetime | None:
    """Parsed row timestamp as naive local time (None when unparseable)."""
    ts = _parse_ts(row.get("create_time"))
    if ts is not None and ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)
    return ts


def _boxed(row: dict) -> bool:
    return all(isinstance(row.get(k), (int, float)) for k in ("box_x1", "box_y1", "box_x2", "box_y2"))


def _merge_pair_names(ops: list[dict]) -> tuple[dict[tuple[int, int], str], dict[int, str]]:
    """Map merge ops to names: exact (prev, pid) pairs plus pid fallbacks."""
    exact: dict[tuple[int, int], str] = {}
    by_later: dict[int, str] = {}
    for op in ops:
        esc = op.get("escalation") or {}
        review = op.get("review") or {}
        spec = _MERGE_VERDICTS.get(esc.get("code"))
        if spec is None or review.get("verdict") != spec[0]:
            continue
        pid = esc.get("period_id")
        prev = (esc.get("metrics") or {}).get(spec[1])
        if not isinstance(pid, int) or not isinstance(prev, int) or pid == prev:
            continue
        name = esc.get("name") or esc.get("code") or "merge"
        exact.setdefault((prev, pid), name)
        by_later.setdefault(pid, name)
    return exact, by_later


def _fill_merged_gaps(rows: list[dict], orig_pids: list, ops: list[dict], merge_targets: dict[int, int]) -> list[dict]:
    """Bridge each merged gap with 1-second interpolated rows between its boundary detections.

    ``rows`` are post-verdict (drops removed, ids renumbered); ``orig_pids``
    aligns by index so gap endpoints are found on pre-merge membership while
    anchors use post-verdict boxes/cats. Filled rows slot into time order and
    carry ``gap_filled=True`` plus the merging op's name.
    """
    members: dict[int, set[int]] = {}
    for pid in set(merge_targets) | set(merge_targets.values()):
        members.setdefault(merge_targets.get(pid, pid), set()).add(pid)
    sets = [m for m in members.values() if len(m) > 1]
    if not sets:
        return rows
    exact_names, later_names = _merge_pair_names(ops)
    by_orig: dict[int, list[dict]] = {}
    for row, pid in zip(rows, orig_pids):
        if isinstance(pid, int):
            by_orig.setdefault(pid, []).append(row)
    fills: list[dict] = []
    for group in sets:
        any_member = next(iter(group))
        rep = merge_targets.get(any_member, any_member)
        ordered = sorted(group, key=lambda p: min((_row_time(r) or datetime.datetime.max for r in by_orig.get(p, [])), default=datetime.datetime.max))
        for earlier, later in zip(ordered, ordered[1:]):
            a_rows = [r for r in by_orig.get(earlier, []) if _boxed(r) and _row_time(r) is not None]
            b_rows = [r for r in by_orig.get(later, []) if _boxed(r) and _row_time(r) is not None]
            if not a_rows or not b_rows:
                continue
            anchor_a = max(a_rows, key=_row_time)
            anchor_b = min(b_rows, key=_row_time)
            cat = anchor_a.get("cat_id")
            if cat is None or anchor_b.get("cat_id") != cat:
                continue  # identity disagreement: renumber stands, no synthesis
            t_a, t_b = _row_time(anchor_a), _row_time(anchor_b)
            span = (t_b - t_a).total_seconds()
            if span <= 0:
                continue
            name = exact_names.get((earlier, later), later_names.get(later, "merge-chain"))
            box_a = [anchor_a[k] for k in ("box_x1", "box_y1", "box_x2", "box_y2")]
            box_b = [anchor_b[k] for k in ("box_x1", "box_y1", "box_x2", "box_y2")]
            t = t_a.replace(microsecond=0) + datetime.timedelta(seconds=1)
            while t < t_b:
                f = (t - t_a).total_seconds() / span
                box = [a + f * (b - a) for a, b in zip(box_a, box_b)]
                fills.append(
                    {
                        "create_time": t.isoformat(timespec="microseconds"),
                        "box_x1": box[0],
                        "box_y1": box[1],
                        "box_x2": box[2],
                        "box_y2": box[3],
                        "orig_cat_id": None,
                        "conf": 0,
                        "cat_id": cat,
                        "period_id": rep,
                        "view_id": anchor_a.get("view_id"),
                        "validated_by": name,
                        "verdict": "merge",
                        "gap_filled": True,
                    }
                )
                t += datetime.timedelta(seconds=1)
    if not fills:
        return rows
    keys = [_row_time(r) or datetime.datetime.max for r in rows]
    for fill in fills:
        key = _row_time(fill)
        i = bisect.bisect_right(keys, key)
        rows.insert(i, fill)
        keys.insert(i, key)
    return rows


def apply_ops(revised_rows: list[dict], ops: list[dict], fill_merged_gaps: bool = True) -> list[dict]:
    """Return validated rows: copies of ``revised_rows`` with verdict ops applied.

    Ops match rows by their pre-apply (tracked) cat/period: verdicts were
    raised against the tracker's labels, and earlier ops rewrite them, so
    matching live values misfires across ops sharing a frame or period.
    """
    rows = [{**r, "validated_by": "pass-through", "verdict": "pass"} for r in revised_rows]
    orig_cats = [r.get("cat_id") for r in revised_rows]
    orig_pids = [r.get("period_id") for r in revised_rows]
    merge_targets = _canonical_merge_targets(ops)
    for op in ops:
        esc = op.get("escalation") or {}
        review = op.get("review") or {}
        verdict = review.get("verdict")
        if not verdict or verdict == "error":
            continue
        name = esc.get("name") or esc.get("code") or "?"
        when = _parse_ts(esc.get("time"))
        handler = _HANDLERS.get(esc.get("code"), _noop)
        spec = _MERGE_VERDICTS.get(esc.get("code"))
        if spec is not None and verdict == spec[0] and isinstance(esc.get("period_id"), int):
            resolved = merge_targets.get(esc["period_id"])
            if resolved is not None and resolved != esc["period_id"]:
                esc = {**esc, "metrics": {**(esc.get("metrics") or {}), spec[1]: resolved}}
        for row, snap_cat, snap_pid in zip(rows, orig_cats, orig_pids):
            if row.get("deleted"):
                continue
            handler(row, esc, review, verdict, name, when, snap_cat, snap_pid)
    kept = [(row, pid) for row, pid in zip(rows, orig_pids) if not row.pop("_drop", False)]
    rows = [row for row, _ in kept]
    if fill_merged_gaps:
        rows = _fill_merged_gaps(rows, [pid for _, pid in kept], ops, merge_targets)
    return rows


def _noop(row, esc, review, verdict, name, when, snap_cat, snap_pid) -> None:
    return None


def _frame_match(row: dict, when, cats: set, box, snap_cat) -> bool:
    if when is None or not _at(row, when) or snap_cat not in cats:
        return False
    if box is None:
        return True
    mine = _row_box(row)
    return mine is not None and _iou(mine, box) >= _FRAME_BOX_IOU


def _t1(row, esc, review, verdict, name, when, snap_cat, snap_pid) -> None:
    m = esc.get("metrics") or {}
    cats = {m.get("original_cat_id"), esc.get("cat_id")}
    box = m.get("box")
    if verdict == "reassign_candidate" and _frame_match(row, when, cats, box, snap_cat):
        row["cat_id"] = review.get("chosen_cat_id", row["cat_id"])
        _touch(row, name, verdict)
    elif verdict == "reject_box" and _frame_match(row, when, cats, box, snap_cat):
        row["_drop"] = True
    elif verdict == "confirm_assigned" and _frame_match(row, when, {esc.get("cat_id")}, box, snap_cat):
        _touch(row, name, verdict)


def _t2(row, esc, review, verdict, name, when, snap_cat, snap_pid) -> None:
    m = esc.get("metrics") or {}
    orig, new = m.get("original_cat_id"), esc.get("cat_id")
    box = m.get("box")
    if verdict == "keep_original" and _frame_match(row, when, {new}, box, snap_cat):
        row["cat_id"] = orig
        _touch(row, name, verdict)
    elif verdict == "reassign_other" and _frame_match(row, when, {orig, new}, box, snap_cat):
        row["cat_id"] = review.get("chosen_cat_id", row["cat_id"])
        _touch(row, name, verdict)
    elif verdict == "drop" and _frame_match(row, when, {orig, new}, box, snap_cat):
        row["_drop"] = True
    elif verdict == "confirm" and _frame_match(row, when, {new}, box, snap_cat):
        _touch(row, name, verdict)


def _t4(row, esc, review, verdict, name, when, snap_cat, snap_pid) -> None:
    m = esc.get("metrics") or {}
    box = m.get("box")
    if verdict == "confirm_reset" and _frame_match(row, when, {esc.get("cat_id")}, box, snap_cat):
        _touch(row, name, verdict)
    elif verdict == "drop_box_as_outlier" and _frame_match(row, when, {esc.get("cat_id")}, box, snap_cat):
        row["_drop"] = True
    elif verdict == "link_continuation":
        old_pid, cont_pid = esc.get("period_id"), m.get("continued_period_id")
        if (
            isinstance(old_pid, int)
            and isinstance(cont_pid, int)
            and old_pid != cont_pid
            and snap_pid == cont_pid
        ):
            row["period_id"] = old_pid
            _touch(row, name, verdict)
        elif _frame_match(row, when, {esc.get("cat_id")}, box, snap_cat):
            _touch(row, name, verdict, note="link target unknown; rows kept as-is")


def _t3(row, esc, review, verdict, name, when, snap_cat, snap_pid) -> None:
    m = esc.get("metrics") or {}
    box = m.get("kept_box")
    if verdict == "keep_killed" and _frame_match(row, when, {esc.get("cat_id")}, box, snap_cat):
        row["cat_id"] = m.get("killed_cat_id", row["cat_id"])
        _touch(row, name, verdict)
    elif verdict in ("confirm_survivor", "keep_both") and _frame_match(row, when, {esc.get("cat_id"), m.get("killed_cat_id")}, box, snap_cat):
        _touch(row, name, verdict)


def _t5(row, esc, review, verdict, name, when, snap_cat, snap_pid) -> None:
    if verdict == "close_visit" and when is not None and _after(row, when) and snap_cat == esc.get("cat_id"):
        if esc.get("period_id") is None or snap_pid == esc.get("period_id"):
            row["_drop"] = True


def _t6(row, esc, review, verdict, name, when, snap_cat, snap_pid) -> None:
    m = esc.get("metrics") or {}
    post_box = m.get("post_box")
    if verdict == "merge_visits" and snap_pid == esc.get("period_id"):
        row["period_id"] = m.get("old_period_id", row["period_id"])
        _touch(row, name, verdict)
    elif verdict == "drop_box" and _frame_match(row, when, {esc.get("cat_id")}, post_box, snap_cat):
        row["_drop"] = True
    elif verdict == "confirm_split" and _frame_match(row, when, {esc.get("cat_id")}, post_box, snap_cat):
        _touch(row, name, verdict)


def _t7(row, esc, review, verdict, name, when, snap_cat, snap_pid) -> None:
    m = esc.get("metrics") or {}
    start = _parse_ts(m.get("gap_start"))
    if start is None or when is None or snap_cat != esc.get("cat_id"):
        return
    if verdict == "cut_at_gap_start" and _in_span(row, start, when):
        row["_drop"] = True
    elif verdict == "shorten_gap":
        cut = _parse_ts(review.get("cut_time"))
        if cut is not None and _in_span(row, cut, when):
            row["_drop"] = True


def _period_match(row: dict, period_id, snap_pid) -> bool:
    return period_id is not None and snap_pid == period_id


def _p1(row, esc, review, verdict, name, when, snap_cat, snap_pid) -> None:
    pid = esc.get("period_id")
    if verdict == "award_runner_up" and _period_match(row, pid, snap_pid):
        row["cat_id"] = review.get("chosen_cat_id", row["cat_id"])
        _touch(row, name, verdict)
    elif verdict == "confirm_winner" and _period_match(row, pid, snap_pid):
        _touch(row, name, verdict)
    # split_visit: boundary lives in free text; keep rows, reviewer note stays on the op.


def _p2(row, esc, review, verdict, name, when, snap_cat, snap_pid) -> None:
    pid = esc.get("period_id")
    if verdict == "drop" and _period_match(row, pid, snap_pid):
        row["_drop"] = True
    elif verdict == "keep" and _period_match(row, pid, snap_pid):
        _touch(row, name, verdict)
    # merge_neighbor: neighbor id unknown; keep rows.


def _p3(row, esc, review, verdict, name, when, snap_cat, snap_pid) -> None:
    pid = esc.get("period_id")
    m = esc.get("metrics") or {}
    if verdict == "restore_wanted" and _period_match(row, pid, snap_pid):
        row["cat_id"] = m.get("wanted_cat_id", row["cat_id"])
        _touch(row, name, verdict)
    elif verdict == "mark_unknown" and _period_match(row, pid, snap_pid):
        row["cat_id"] = None
        _touch(row, name, verdict)
    elif verdict == "confirm_reassign" and _period_match(row, pid, snap_pid):
        _touch(row, name, verdict)


def _p4(row, esc, review, verdict, name, when, snap_cat, snap_pid) -> None:
    pid = esc.get("period_id")
    if verdict == "assign_cat" and _period_match(row, pid, snap_pid):
        row["cat_id"] = review.get("chosen_cat_id", row["cat_id"])
        _touch(row, name, verdict)
    elif verdict == "confirm_unknown" and _period_match(row, pid, snap_pid):
        row["cat_id"] = None
        _touch(row, name, verdict)
    # split_fragments: boundaries live in free text; keep rows.


def _p5(row, esc, review, verdict, name, when, snap_cat, snap_pid) -> None:
    pid = esc.get("period_id")
    if verdict == "drop" and _period_match(row, pid, snap_pid):
        row["_drop"] = True
    elif verdict == "reassign" and _period_match(row, pid, snap_pid):
        row["cat_id"] = review.get("chosen_cat_id", row["cat_id"])
        _touch(row, name, verdict)
    elif verdict == "keep" and _period_match(row, pid, snap_pid):
        _touch(row, name, verdict)
    # trim_core: kept span lives in free text; keep rows.


def _p6_p7(row, esc, review, verdict, name, when, snap_cat, snap_pid) -> None:
    m = esc.get("metrics") or {}
    if verdict == "merge" and _period_match(row, esc.get("period_id"), snap_pid):
        row["period_id"] = m.get("prev_period_id", row["period_id"])
        _touch(row, name, verdict)


def _p8(row, esc, review, verdict, name, when, snap_cat, snap_pid) -> None:
    pid = esc.get("period_id")
    m = esc.get("metrics") or {}
    if verdict == "relabel_to_vote" and _period_match(row, pid, snap_pid):
        row["cat_id"] = m.get("voted_cat_id", row["cat_id"])
        _touch(row, name, verdict)
    elif verdict == "mark_unknown" and _period_match(row, pid, snap_pid):
        row["cat_id"] = None
        _touch(row, name, verdict)
    elif verdict == "keep_tracked" and _period_match(row, pid, snap_pid):
        _touch(row, name, verdict)


_HANDLERS = {
    "T1": _t1,
    "T2": _t2,
    "T3": _t3,
    "T4": _t4,
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
    "P8": _p8,
    "H1": _noop,  # spot-check audit: verdicts are advisory, rows stay as-is
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
