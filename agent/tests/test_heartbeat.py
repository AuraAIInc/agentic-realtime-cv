"""Quiet-window heartbeat: a fully quiet run still gets one H1 spot-check.

Periodic H1 fires after ``heartbeat_seconds`` of stream time with no
escalation; the end-of-run H1 fires at ``finish()`` when the run stayed
quiet throughout. Run from the repo root:
    .venv/bin/python -m pytest agent/tests/test_heartbeat.py -q
"""

from __future__ import annotations

import datetime
import json

from agent.main import TrackingRun
from agent.tracking.store import RunStore

T0 = datetime.datetime(2026, 10, 28, 16, 33, 0)


def _msg(when: datetime.datetime, detections=None) -> dict:
    return {
        "view_id": 3,
        "create_time": when.strftime("%Y-%m-%d-%H-%M-%S-%f"),
        "detections": detections if detections is not None else [],
        "is_grey": 0,
    }


def _run(tmp_path, name="run1", **kw):
    args = {
        "view_id": 3,
        "start": T0,
        "end_exclusive": T0 + datetime.timedelta(minutes=10),
        "cat_ids": [25, 26],
        "id_to_name": {25: "Qb", 26: "Qbo"},
        "plot": False,
    }
    args.update(kw)
    return TrackingRun(store=RunStore(name, base_dir=tmp_path), **args)


def _finish_counts(run):
    run_dir = run.finish()
    meta = json.loads((run_dir / "run.json").read_text())
    return run_dir, meta["counts"]


def test_periodic_heartbeat_repeats_each_quiet_interval(tmp_path):
    run = _run(tmp_path, heartbeat_seconds=60.0, heartbeat_at_end=False)
    for offset in (0, 30, 61, 122):
        assert run.process_message(_msg(T0 + datetime.timedelta(seconds=offset))) == "processed"
    codes = [e.code for e in run._escs.all]
    assert codes == ["H1", "H1"]
    first, second = run._escs.all
    assert (first.name, second.name) == ("E1-H1", "E2-H1")
    assert first.scope == "period" and first.end_time > first.time
    assert first.metrics["kind"] == "periodic"
    assert first.metrics["quiet_seconds"] >= 60.0
    assert first.metrics["cat_ids"] == [25, 26]
    assert second.time == first.end_time  # window restarts where the last check ended
    run_dir, counts = _finish_counts(run)
    assert counts["escalations_by_code"] == {"H1": 2}
    assert len((run_dir / "escalations.jsonl").read_text().strip().splitlines()) == 2


def test_end_heartbeat_when_run_stays_quiet(tmp_path):
    run = _run(tmp_path, heartbeat_seconds=3600.0, heartbeat_at_end=True)
    run.process_message(_msg(T0))
    run.process_message(_msg(T0 + datetime.timedelta(seconds=10)))
    assert run._escs.all == []  # nothing mid-run: interval never reached
    run_dir, counts = _finish_counts(run)
    assert counts["escalations_by_code"] == {"H1": 1}
    (esc,) = run._escs.all
    assert esc.name == "E1-H1" and esc.metrics["kind"] == "end_of_run"
    assert esc.time == T0 and esc.end_time == T0 + datetime.timedelta(seconds=10)
    assert len((run_dir / "escalations.jsonl").read_text().strip().splitlines()) == 1


def test_real_escalation_suppresses_heartbeat(tmp_path):
    run = _run(tmp_path, heartbeat_seconds=60.0, heartbeat_at_end=True)
    run.process_message(_msg(T0))
    # a real frame escalation, drained on the next message like the live loop does
    run.track._escalate("T2", "test relabel", T0, cat_id=25, metrics={"original_cat_id": 26})
    run.process_message(_msg(T0 + datetime.timedelta(seconds=10)))
    run.process_message(_msg(T0 + datetime.timedelta(seconds=20)))
    _, counts = _finish_counts(run)
    assert counts["escalations_by_code"] == {"T2": 1}


def test_heartbeat_disabled_stays_quiet(tmp_path):
    run = _run(tmp_path, heartbeat_seconds=0, heartbeat_at_end=False)
    for offset in (0, 200, 400):
        run.process_message(_msg(T0 + datetime.timedelta(seconds=offset)))
    _, counts = _finish_counts(run)
    assert counts["escalations_total"] == 0
    assert counts["escalations_by_code"] == {}


def test_heartbeat_enqueued_for_validation(tmp_path):
    class FakeValidator:
        def __init__(self):
            self.enqueued = []

        def start(self):
            pass

        def enqueue(self, esc):
            self.enqueued.append(esc)

        def stop(self, timeout=None):
            return []

    validator = FakeValidator()
    run = _run(tmp_path, validator=validator, heartbeat_seconds=60.0, heartbeat_at_end=False)
    run.process_message(_msg(T0))
    run.process_message(_msg(T0 + datetime.timedelta(seconds=61)))
    assert [e.code for e in validator.enqueued] == ["H1"]
    run.finish()


def test_no_end_heartbeat_without_messages(tmp_path):
    run = _run(tmp_path, heartbeat_seconds=3600.0, heartbeat_at_end=True)
    _, counts = _finish_counts(run)
    assert counts["escalations_total"] == 0
