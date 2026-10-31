"""Run store: JSONL persistence for tracking runs (Phase 3 of 5).

A run is a directory ``<base_dir>/<run_name>/`` with append-only JSONL logs
plus a one-shot ``run.json`` metadata file. ``FileSink`` adapts the Phase-2
:class:`~agent.tracking.sinks.Sink` protocol (positional row lists coming out
of the correctors) onto the canonical row-dict formats below.

Canonical row-dict JSON formats (one dict per JSONL line):

``stream.jsonl`` — raw detector output (``save_original`` rows):
    ``{"create_time": "<ISO-8601>", "cat_id": int, "box_x1": float,
    "box_y1": float, "box_x2": float, "box_y2": float, "conf": float,
    "model_id": int | None, "view_id": int}``

``revised.jsonl`` — TrackCorrector committed rows (``save_revised`` rows):
    ``{"create_time": "<ISO-8601>", "box_x1": float | None, ...,
    "box_y2": float | None, "orig_cat_id": int | None, "conf": float,
    "cat_id": int | None, "period_id": int, "view_id": int}``
    Retractions (``delete_revised``) append the same dict plus
    ``"deleted": true`` (a tombstone; readers skip it).

``periods.jsonl`` — PeriodCorrector verdicts (``save_period`` rows):
    ``{"period_id": int, "start": "<ts>", "end": "<ts>",
    "coordinated_cat_id": int | None,
    "stats": {"detect_length": [...], "detect_conf": [...],
    "detect_rate": [...], "period_area_count": [...], "score": [...]},
    "winner": {...} | None, "view_id": int}``
    (``winner`` repeats the winning cat's scalars; ``None`` when unassigned.)

``images.jsonl`` — per-frame capture log (``save_image`` rows):
    ``{"create_time": "<ISO-8601>", "detected": int, "is_grey": bool,
    "view_id": int, "device_id": str | int | None}``

``escalations.jsonl`` — one ``Escalation.to_json()`` object per line.

``run.json`` — metadata dict written once via :meth:`RunStore.write_run_json`.
"""

from __future__ import annotations

import ast
import datetime
import json
import math
from pathlib import Path

DEFAULT_RUNS_DIR = Path(__file__).resolve().parent.parent / "output" / "tracking_runs"


def _jsonable(v):
    """Convert datetime/numpy values from corrector rows to JSON scalars."""
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, datetime.datetime):
        return v.isoformat(timespec="microseconds")
    if isinstance(v, datetime.date):
        return v.isoformat()
    mod = v.__class__.__module__
    if mod == "numpy":
        return v.item() if hasattr(v, "item") else list(v)
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


class RunStore:
    """Append-only JSONL logs plus metadata for one tracking run."""

    STREAM_FILE = "stream.jsonl"
    REVISED_FILE = "revised.jsonl"
    PERIODS_FILE = "periods.jsonl"
    IMAGES_FILE = "images.jsonl"
    ESCALATIONS_FILE = "escalations.jsonl"
    RUN_FILE = "run.json"

    def __init__(self, run_name: str, base_dir: str | Path | None = None):
        base = Path(base_dir) if base_dir is not None else DEFAULT_RUNS_DIR
        self.run_dir = base / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.counters = {
            "stream": 0,
            "revised": 0,
            "periods": 0,
            "images": 0,
            "escalations": 0,
            "deleted": 0,
        }
        self._handles = {}
        for key, fname in (
            ("stream", self.STREAM_FILE),
            ("revised", self.REVISED_FILE),
            ("periods", self.PERIODS_FILE),
            ("images", self.IMAGES_FILE),
            ("escalations", self.ESCALATIONS_FILE),
        ):
            self._handles[key] = open(self.run_dir / fname, "a", encoding="utf-8")

    def _append(self, key: str, row: dict) -> None:
        self._handles[key].write(json.dumps(_jsonable(row)) + "\n")
        self.counters[key] += 1

    def append_stream(self, row: dict) -> None:
        """Append one canonical stream row to ``stream.jsonl``."""
        self._append("stream", row)

    def append_revised(self, row: dict) -> None:
        """Append one canonical revised row to ``revised.jsonl``."""
        self._append("revised", row)

    def append_period(self, row: dict) -> None:
        """Append one canonical period row to ``periods.jsonl``."""
        self._append("periods", row)

    def append_image(self, row: dict) -> None:
        """Append one canonical image row to ``images.jsonl``."""
        self._append("images", row)

    def append_escalation(self, esc) -> None:
        """Append one escalation via ``esc.to_json()`` to ``escalations.jsonl``."""
        self._handles["escalations"].write(esc.to_json() + "\n")
        self.counters["escalations"] += 1

    def append_deleted_revised(self, row: dict) -> None:
        """Append a ``deleted: true`` tombstone for a retracted revised row."""
        self._append("revised", {**row, "deleted": True})
        self.counters["deleted"] += 1

    def write_run_json(self, meta: dict) -> Path:
        """Write the run metadata dict to ``run.json``; returns its path."""
        path = self.run_dir / self.RUN_FILE
        path.write_text(json.dumps(_jsonable(meta), indent=2) + "\n", encoding="utf-8")
        return path

    def flush(self) -> None:
        for fh in self._handles.values():
            fh.flush()

    def close(self) -> None:
        for fh in self._handles.values():
            if not fh.closed:
                fh.close()

    def __enter__(self) -> "RunStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):  # best-effort; prefer close() or the context manager
        try:
            self.close()
        except Exception:
            pass


def _parse_stats_list(value):
    """Period rows carry per-cat stat vectors as ``str(list)``; parse them back."""
    if isinstance(value, str):
        try:
            return list(ast.literal_eval(value))
        except (ValueError, SyntaxError):
            return value
    return _jsonable(value)


class FileSink:
    """A Phase-2 sink-protocol adapter that writes through a :class:`RunStore`.

    Accepts the same positional row shapes as ``InMemorySink`` and converts
    each to the canonical row-dict JSON format documented above.
    """

    def __init__(self, store: RunStore):
        self.store = store

    @property
    def counters(self) -> dict:
        return self.store.counters

    def save_revised(self, row: list) -> None:
        # [create_time, x1, y1, x2, y2, orig_cat_id, conf, cat_id, period_id, view_id]
        self.store.append_revised(
            {
                "create_time": _jsonable(row[0]),
                "box_x1": _jsonable(row[1]),
                "box_y1": _jsonable(row[2]),
                "box_x2": _jsonable(row[3]),
                "box_y2": _jsonable(row[4]),
                "orig_cat_id": _jsonable(row[5]),
                "conf": _jsonable(row[6]),
                "cat_id": _jsonable(row[7]),
                "period_id": _jsonable(row[8]),
                "view_id": _jsonable(row[9]) if len(row) > 9 else None,
            }
        )

    def save_original(self, rows: list) -> None:
        # each: (create_time, cat_id, x1, y1, x2, y2, conf, model_id, view_id)
        for r in rows:
            self.store.append_stream(
                {
                    "create_time": _jsonable(r[0]),
                    "cat_id": _jsonable(r[1]),
                    "box_x1": _jsonable(r[2]),
                    "box_y1": _jsonable(r[3]),
                    "box_x2": _jsonable(r[4]),
                    "box_y2": _jsonable(r[5]),
                    "conf": _jsonable(r[6]),
                    "model_id": _jsonable(r[7]),
                    "view_id": _jsonable(r[8]),
                }
            )

    def save_image(self, row: tuple) -> None:
        # (create_time, detected, is_grey, view_id, device_id)
        self.store.append_image(
            {
                "create_time": _jsonable(row[0]),
                "detected": _jsonable(row[1]),
                "is_grey": _jsonable(row[2]),
                "view_id": _jsonable(row[3]),
                "device_id": _jsonable(row[4]),
            }
        )

    def delete_revised(self, row: list) -> None:
        # Retraction of a 9-field revised row (no view_id); tombstone it.
        self.store.append_deleted_revised(
            {
                "create_time": _jsonable(row[0]),
                "box_x1": _jsonable(row[1]),
                "box_y1": _jsonable(row[2]),
                "box_x2": _jsonable(row[3]),
                "box_y2": _jsonable(row[4]),
                "orig_cat_id": _jsonable(row[5]),
                "conf": _jsonable(row[6]),
                "cat_id": _jsonable(row[7]),
                "period_id": _jsonable(row[8]),
                "view_id": _jsonable(row[9]) if len(row) > 9 else None,
            }
        )

    def save_period(self, row: list) -> None:
        # [start, end, coord_cat, det_len, det_conf, det_rate, area, score,
        #  str(det_len), str(det_conf), str(det_rate), str(area), str(score),
        #  view_id, period_id]
        coord = _jsonable(row[2])
        winner = None
        if coord is not None:
            winner = {
                "detect_length": _jsonable(row[3]),
                "detect_conf": _jsonable(row[4]),
                "detect_rate": _jsonable(row[5]),
                "period_area_count": _jsonable(row[6]),
                "score": _jsonable(row[7]),
            }
        self.store.append_period(
            {
                "period_id": _jsonable(row[14]),
                "start": _jsonable(row[0]),
                "end": _jsonable(row[1]),
                "coordinated_cat_id": coord,
                "stats": {
                    "detect_length": _parse_stats_list(row[8]),
                    "detect_conf": _parse_stats_list(row[9]),
                    "detect_rate": _parse_stats_list(row[10]),
                    "period_area_count": _parse_stats_list(row[11]),
                    "score": _parse_stats_list(row[12]),
                },
                "winner": winner,
                "view_id": _jsonable(row[13]),
            }
        )

    def flush(self) -> None:
        self.store.flush()


__all__ = ["RunStore", "FileSink", "DEFAULT_RUNS_DIR"]
