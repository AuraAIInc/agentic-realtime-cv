"""TrackingRun enqueues live escalations and materializes validated outputs at finish."""

import datetime

from agent.escalation import Escalation
from agent.main import TrackingRun
from agent.tracking.store import RunStore

T0 = datetime.datetime(2026, 10, 28, 16, 35, 21, 976102)


class FakeValidator:
    def __init__(self, ops):
        self.ops = ops
        self.enqueued = []
        self.started = False

    def start(self):
        self.started = True

    def enqueue(self, esc):
        self.enqueued.append(esc)

    def stop(self, timeout=None):
        return self.ops


def _run(tmp_path, validator, **kw):
    store = RunStore("run1", base_dir=tmp_path)
    return TrackingRun(
        view_id=3,
        start=datetime.datetime(2026, 10, 28, 16, 33),
        end_exclusive=datetime.datetime(2026, 10, 28, 16, 36),
        cat_ids=[25, 26],
        id_to_name={25: "Qb", 26: "Qbo"},
        store=store,
        plot=False,
        validator=validator,
        **kw,
    )


def test_escalations_enqueued_live_and_validated_at_finish(tmp_path):
    op = {"escalation": {"name": "E1-T2", "code": "T2", "time": T0.isoformat(timespec="microseconds"),
                         "cat_id": 25, "metrics": {"original_cat_id": 26}},
          "review": {"verdict": "confirm"}}
    validator = FakeValidator([op])
    run = _run(tmp_path, validator)
    assert validator.started

    esc = Escalation(time=T0, scope="frame", code="T2", reason="r", cat_id=25, seq=1)
    run.track.escalations.append(esc)
    run._persist_new_escalations()
    assert validator.enqueued == [esc]

    run_dir = run.finish()
    assert (run_dir / "validated.jsonl").is_file()
    assert (run_dir / "validated.db").is_file()
    assert (run_dir / "validations.jsonl").is_file()
    import json

    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["counts"]["validations_total"] == 1
    assert meta["counts"]["validations_by_verdict"] == {"confirm": 1}


def test_no_validator_no_validated_outputs(tmp_path):
    run = _run(tmp_path, None)
    assert run.validator is None
    run_dir = run.finish()
    assert not (run_dir / "validated.jsonl").exists()
    assert not (run_dir / "validated.db").exists()


def test_tracking_run_resolves_view_size_into_tracker_and_meta(tmp_path):
    import json

    run = _run(tmp_path, None)
    assert run.img_size == (800, 600)
    assert (run.track.img_width, run.track.img_height) == (800, 600)
    run_dir = run.finish()
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["img_size"] == [800, 600]


def test_tracking_run_explicit_img_size_overrides_resolve(tmp_path):
    run = _run(tmp_path, None, img_size=(640, 480))
    assert run.img_size == (640, 480)
    assert (run.track.img_width, run.track.img_height) == (640, 480)


def test_validated_chart_written_with_plot(tmp_path):
    import json

    from agent.tracking.store import RunStore

    store = RunStore("run1", base_dir=tmp_path)
    run = TrackingRun(
        view_id=3,
        start=datetime.datetime(2026, 10, 28, 16, 33),
        end_exclusive=datetime.datetime(2026, 10, 28, 16, 36),
        cat_ids=[25, 26],
        id_to_name={25: "Qb", 26: "Qbo"},
        store=store,
        plot=True,
        validator=FakeValidator([]),
    )
    run_dir = run.finish()
    assert (run_dir / "validated.png").is_file()
    meta = json.loads((run_dir / "run.json").read_text())
    assert "validated.png" in meta["artifacts"]


def test_no_validated_chart_without_plot(tmp_path):
    run = _run(tmp_path, FakeValidator([]))
    run_dir = run.finish()
    assert (run_dir / "validated.jsonl").is_file()
    assert not (run_dir / "validated.png").exists()
