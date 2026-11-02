"""TrackingRun helpers without a full run. Run from the repo root:
    .venv/bin/python -m pytest agent/tests/test_tracking_run.py -q
"""

from __future__ import annotations

import datetime
import logging
from types import SimpleNamespace

from agent.main import TrackingRun

T0 = datetime.datetime(2026, 10, 28, 16, 33, 0)


def _row():
    # [start, end, coord_cat, det_len, det_conf, det_rate, area, score, ..., view_id, period_id]
    return [T0, T0 + datetime.timedelta(minutes=5), 25, 30, 0.8, 0.9, 10, 12.5,
            "s", "s", "s", "s", "s", 3, 7]


def test_log_period_row_summarizes(caplog):
    run = SimpleNamespace(id_to_name={25: "Qb"})
    with caplog.at_level(logging.INFO, logger="agent.main"):
        TrackingRun._log_period_row(run, _row())
    assert "period closed: Period 7 (16:33-16:38): Qb (cat 25). 30 detections, mean conf 0.80, score 12.5." in caplog.text


def test_log_period_row_skips_padding_nones(caplog):
    run = SimpleNamespace(id_to_name={})
    with caplog.at_level(logging.INFO, logger="agent.main"):
        TrackingRun._log_period_row(run, None)
    assert "period closed" not in caplog.text
