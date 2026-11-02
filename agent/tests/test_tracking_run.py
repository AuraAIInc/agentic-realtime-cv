"""TrackingRun helpers without a full run. Run from the repo root:
    .venv/bin/python -m pytest agent/tests/test_tracking_run.py -q
"""

from __future__ import annotations

import datetime
import logging
from types import SimpleNamespace

from agent.main import TrackingRun, escalation_to_memory_event, period_row_to_memory_event

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


def test_period_memory_event_carries_confidence_and_evidence():
    event = period_row_to_memory_event(_row(), view_id=3, id_to_name={25: "Qb"}, box=(1, 2, 3, 4))
    assert event.confidence == 0.8
    assert event.image_names == [
        "2026-10-28-16-33-00-000000.jpg",
        "2026-10-28-16-35-30-000000.jpg",
        "2026-10-28-16-38-00-000000.jpg",
    ]
    assert event.detection["cat_id"] == 25 and event.detection["period_id"] == 7
    assert event.appearance.startswith("Qb:")


def test_escalation_memory_event_is_untrusted_short_term_only():
    from agent.escalation import Escalation

    esc = Escalation(time=T0, scope="frame", code="T2", reason="r", cat_id=25, seq=6)
    event = escalation_to_memory_event(esc, view_id=3)
    assert event.kind == "escalation" and event.confidence is None
    assert event.image_names == ["2026-10-28-16-33-00-000000.jpg"]
