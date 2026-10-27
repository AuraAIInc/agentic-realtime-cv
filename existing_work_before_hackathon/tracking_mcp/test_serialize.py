"""period_summary on DB-shaped rows: string times, '[...]'-string arrays (no DB).

Run from the existing_work_before_hackathon/ folder:
    python -m tracking_mcp.test_serialize
"""

from __future__ import annotations

import datetime

from tracking_mcp.live_state import LiveState
from tracking_mcp.serialize import period_summary

# A finalized row as PeriodCorrector yields it (and as MySQL returns it):
# [start, end, cat, length, conf, rate, area_count, score,
#  all_lengths, all_confs, all_rates, all_area_counts, all_scores, view_id, period_id]
DB_ROW = [
    "2026-10-27 07:05:27.049494", "2026-10-27 07:06:26.520786", 25, 175, 0.6946, 0.9943, 1, 13.2,
    "[175, 0]", "[0.6946, 0.0]", "[0.9943, 0.0]", "[1, 0]", "[13.2, 0.0]", 3, 102481,
]


def test_period_summary_accepts_db_string_times():
    out = period_summary(
        DB_ROW[14], DB_ROW[13], "final", DB_ROW[0], DB_ROW[1], DB_ROW[2],
        [25, 0], DB_ROW[8], DB_ROW[9], DB_ROW[12],
    )
    assert out["period_id"] == 102481
    assert out["view_id"] == 3
    assert out["start"] == "2026-10-27T07:05:27.049494", out["start"]
    assert out["end"] == "2026-10-27T07:06:26.520786", out["end"]
    assert out["duration_s"] == 59.471, out["duration_s"]
    assert out["evidence_by_cat"]["25"] == {"detections": 175, "avg_conf": 0.6946, "score": 13.2}


def test_record_period_accepts_db_row():
    state = LiveState({}, {})
    state.record_period(3, DB_ROW)
    (got,) = state.finalized(3)
    assert got["period_id"] == 102481
    assert got["duration_s"] == 59.471, got["duration_s"]


def test_period_summary_keeps_datetime_inputs():
    start = datetime.datetime(2026, 10, 27, 7, 5, 27, 49494)
    end = datetime.datetime(2026, 10, 27, 7, 6, 26, 520786)
    out = period_summary(1, 3, "ongoing", start, end, 25, [25], [175], [0.6946], [13.2])
    assert out["duration_s"] == 59.471
    assert out["start"] == "2026-10-27T07:05:27.049494"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
