"""TrackingRun helpers without a full run. Run from the repo root:
    .venv/bin/python -m pytest agent/tests/test_tracking_run.py -q
"""

from __future__ import annotations

import datetime
import logging
from types import MethodType, SimpleNamespace

import pytest

from agent.main import TrackingRun, escalation_to_memory_event, period_row_to_memory_event
from agent.memory import ActivityLabel, PeriodActivityAggregator

T0 = datetime.datetime(2026, 10, 28, 16, 33, 0)


def _row():
    # [start, end, coord_cat, det_len, det_conf, det_rate, area, score, ..., view_id, period_id]
    return [T0, T0 + datetime.timedelta(minutes=5), 25, 30, 0.8, 0.9, 10, 12.5,
            "s", "s", "s", "s", "s", 3, 7]


def _period_row(pid, start, minutes=1, cat=25, n=30, conf=0.8):
    return [start, start + datetime.timedelta(minutes=minutes), cat, n, conf, 0.9, 10, 12.5,
            "s", "s", "s", "s", "s", 3, pid]


class _Mem:
    def __init__(self):
        self.stored = []

    def store(self, event):
        self.stored.append(event)


class _Short:
    def __init__(self):
        self.periods = []

    def record_period(self, **kw):
        self.periods.append(kw)
        return None


class _Labeler:
    def __init__(self):
        self.jobs = []
        self.stopped = False

    def label_period(self, pid, **kw):
        self.jobs.append((pid, kw))

    def stop(self, timeout=None):
        self.stopped = True


def _wired_run(**kw):
    args = {
        "memory": _Mem(),
        "short_term": _Short(),
        "_activity_agg": PeriodActivityAggregator(gap_minutes=5.0),
        "_labeler": _Labeler(),
        "_resolved": {},
        "view_id": 3,
        "id_to_name": {25: "Qb", 26: "Qbo"},
        "_period_box": lambda pid: (1.0, 2.0, 3.0, 4.0),
    }
    args.update(kw)
    run = SimpleNamespace(**args)
    run._legacy_memory_store = MethodType(TrackingRun._legacy_memory_store, run)
    run._store_activity_record = MethodType(TrackingRun._store_activity_record, run)
    return run


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


def _label(activity="resting", conf=0.9, **kw):
    args = {"activity": activity, "confidence": conf,
            "appearance": "white cat", "background": "rug"}
    args.update(kw)
    return ActivityLabel(**args)


def test_close_periods_combine_and_store_labeled_record():
    run = _wired_run()
    labeler = run._labeler
    TrackingRun._memory_store_period_row(run, _period_row(1, T0, conf=0.8))
    TrackingRun._memory_store_period_row(run, _period_row(2, T0 + datetime.timedelta(minutes=1, seconds=30), conf=0.9, n=20))
    assert [pid for pid, _ in labeler.jobs] == [1, 2]  # every member submitted
    assert run.memory.stored == []  # nothing complete yet
    TrackingRun._on_period_label(run, 1, _label(), ["r1.jpg"])
    TrackingRun._on_period_label(run, 2, _label(conf=0.85), ["r2.jpg"])
    TrackingRun._finish_activity_labels(run)
    assert labeler.stopped is True
    assert len(run.memory.stored) == 1
    event = run.memory.stored[0]
    assert event.kind == "period" and event.period_id == 1
    assert event.detection["period_ids"] == [1, 2]
    assert event.detection["activity"] == "resting"
    assert event.confidence == pytest.approx(0.84)  # min(detector 0.84, labels 0.9/0.85)
    assert "Qb (cat 25)" in event.text and "resting" in event.text
    assert "Looks like: white cat." in event.text and "Background: rug." in event.text
    assert event.image_names == ["r1.jpg", "r2.jpg"]  # real names win
    (entry,) = run.short_term.periods
    assert entry["period_ids"] == [1, 2] and entry["activity"] == "resting"
    assert entry["activity_confidence"] == 0.85 and entry["background"] == "rug"


def test_far_period_closes_record_mid_stream():
    run = _wired_run()
    TrackingRun._memory_store_period_row(run, _period_row(1, T0))
    TrackingRun._memory_store_period_row(run, _period_row(2, T0 + datetime.timedelta(minutes=6)))
    TrackingRun._on_period_label(run, 1, _label(), ["r1.jpg"])
    assert len(run.memory.stored) == 1  # closed record emits on labeling
    assert run.memory.stored[0].detection["period_ids"] == [1]
    TrackingRun._finish_activity_labels(run)
    assert len(run.memory.stored) == 2  # leftover flushes as unknown
    assert run.memory.stored[1].confidence == 0.0
    assert run.short_term.periods[1]["activity"] is None  # short-term keeps it anyway


def test_degraded_label_drops_long_term_but_keeps_short_term():
    run = _wired_run()
    TrackingRun._memory_store_period_row(run, _period_row(5, T0))
    TrackingRun._on_period_label(run, 5, ActivityLabel(activity=None, degraded=True), [])
    TrackingRun._finish_activity_labels(run)
    assert run.memory.stored[0].confidence == 0.0  # trust gate will drop it
    assert run.short_term.periods[0]["activity"] is None


def test_legacy_path_stores_directly_without_aggregator():
    run = _wired_run(_activity_agg=None, _labeler=None)
    TrackingRun._memory_store_period_row(run, _row())
    assert run.memory.stored[0].confidence == 0.8
    assert "activity" not in run.short_term.periods[0]
