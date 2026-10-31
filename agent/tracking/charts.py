"""Timeline strip charts for tracking runs (Phase 3 of 5).

One lane per cat, x = time, each detection a vertical bar spanning its
``box_x1..box_x2`` normalized by the image width. Frame-scope escalations
are red vertical lines; period-scope escalations are red spans clipped to
the plotted window.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from agent.tracking.process_viz import CAT_COLORS


def _parse_time(value):
    if isinstance(value, datetime.datetime):
        return value
    return datetime.datetime.fromisoformat(value)


def _esc_span(esc):
    """Normalize an Escalation object or escalation dict to (start, end, scope)."""
    if isinstance(esc, dict):
        start = _parse_time(esc["time"]) if esc.get("time") else None
        end = _parse_time(esc["end_time"]) if esc.get("end_time") else None
        return start, end or start, esc.get("scope", "frame")
    start = getattr(esc, "time", None)
    end = getattr(esc, "end_time", None) or start
    return start, end, getattr(esc, "scope", "frame")


def _plot_strip(rows, path, title, cat_ids, names, escalations, width):
    idx_of = {cid: i for i, cid in enumerate(cat_ids)}
    times, bottoms, tops, colors = [], [], [], []
    for row in rows:
        if row.get("deleted"):
            continue
        cid = row.get("cat_id")
        if cid not in idx_of:
            continue
        x1, x2 = row.get("box_x1"), row.get("box_x2")
        if x1 is None or x2 is None:
            continue
        try:
            t = _parse_time(row["create_time"])
        except (KeyError, ValueError, TypeError):
            continue
        lane = idx_of[cid]
        times.append(t)
        bottoms.append(lane + min(x1, x2) / width)
        tops.append(lane + max(x1, x2) / width)
        colors.append(CAT_COLORS[lane % len(CAT_COLORS)])

    fig, ax = plt.subplots(figsize=(14, 2 + len(cat_ids)))
    if times:
        ax.vlines(times, bottoms, tops, colors=colors, linewidths=2)
        lo, hi = min(times), max(times)
        pad = max((hi - lo).total_seconds(), 1.0) * 0.02
        lo -= datetime.timedelta(seconds=pad)
        hi += datetime.timedelta(seconds=pad)
        for esc in escalations or []:
            start, end, scope = _esc_span(esc)
            if start is None or end < lo or start > hi:
                continue
            if scope == "period":
                ax.axvspan(max(start, lo), min(end, hi), color="red", alpha=0.12)
            else:
                ax.axvline(start, color="red", lw=1.5, alpha=0.85)
        ax.set_xlim(lo, hi)
        ax.set_title(title)
    else:
        ax.set_title(f"{title} (no data)")
    ax.set_ylim(-0.5, len(cat_ids) - 0.5)
    ax.set_yticks(range(len(cat_ids)))
    ax.set_yticklabels([names.get(cid, str(cid)) for cid in cat_ids])
    ax.set_xlabel("time")
    ax.set_ylabel("cat")
    fig.autofmt_xdate()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_detection_charts(
    original_rows,
    revised_rows,
    out_dir,
    cat_ids,
    cat_names=None,
    escalations=None,
    width=640,
):
    """Write ``original.png`` + ``revised.png`` strip charts into ``out_dir``.

    ``original_rows`` / ``revised_rows`` are canonical row dicts (see
    :mod:`agent.tracking.store`); each needs ``create_time``, ``box_x1``,
    ``box_x2``, ``cat_id``. ``cat_names`` maps cat id -> label (or a list
    parallel to ``cat_ids``). ``escalations`` is a list of ``Escalation``
    objects or escalation dicts. Empty input still writes a chart whose
    title says "no data". Returns ``(original_path, revised_path)``.
    """
    cat_ids = list(cat_ids)
    if cat_names is None:
        names = {}
    elif isinstance(cat_names, dict):
        names = dict(cat_names)
    else:
        names = dict(zip(cat_ids, cat_names))
    out_dir = Path(out_dir)
    original_path = _plot_strip(
        list(original_rows), out_dir / "original.png", "Original detections", cat_ids, names, escalations, width
    )
    revised_path = _plot_strip(
        list(revised_rows), out_dir / "revised.png", "Revised detections", cat_ids, names, escalations, width
    )
    return original_path, revised_path


__all__ = ["plot_detection_charts"]
