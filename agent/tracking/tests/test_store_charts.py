"""Phase 3 tests: RunStore/FileSink, strip charts, view->cat resolution."""

import datetime
import json

import pytest

from agent.escalation import Escalation
from agent.tracking.cat_config import resolve_cat_ids
from agent.tracking.charts import plot_detection_charts
from agent.tracking.store import FileSink, RunStore

T0 = datetime.datetime(2026, 10, 30, 12, 0, 0)


def _rev_row(i, cat_id=25, period_id=1, view_id=3):
    t = T0 + datetime.timedelta(seconds=i * 5)
    return [t, 100 + i * 10, 50.0, 200 + i * 10, 150.0, cat_id, 0.9, cat_id, period_id, view_id]


def test_runstore_creates_run_dir_and_jsonl(tmp_path):
    store = RunStore("r1", base_dir=tmp_path)
    store.append_stream({"create_time": T0.isoformat(), "cat_id": 25})
    store.append_revised({"create_time": T0.isoformat(), "cat_id": 25})
    store.append_period({"period_id": 1})
    store.append_image({"create_time": T0.isoformat()})
    store.append_escalation(Escalation(time=T0, scope="frame", code="T5", reason="test"))
    meta_path = store.write_run_json({"run": "r1", "view_id": 3})
    store.close()

    run_dir = tmp_path / "r1"
    for fname in ("stream.jsonl", "revised.jsonl", "periods.jsonl", "escalations.jsonl", "run.json"):
        assert (run_dir / fname).is_file(), fname
    assert json.loads(meta_path.read_text())["view_id"] == 3
    assert store.counters == {"stream": 1, "revised": 1, "periods": 1, "images": 1, "escalations": 1, "deleted": 0}
    esc_line = json.loads((run_dir / "escalations.jsonl").read_text().splitlines()[0])
    assert esc_line["code"] == "T5" and esc_line["escalation"] is True


def test_filesink_accepts_sink_protocol_calls(tmp_path):
    store = RunStore("r2", base_dir=tmp_path)
    sink = FileSink(store)
    sink.save_revised(_rev_row(0))
    sink.save_original([(T0, 25, 100.0, 50.0, 200.0, 150.0, 0.9, 7, 3)])
    sink.save_image((T0, 1, False, 3, "dev0"))
    sink.delete_revised(_rev_row(0)[:9])
    sink.save_period(
        [
            "2026-10-30 12:00:00.000000",
            "2026-10-30 12:01:00.000000",
            25,
            10,
            0.9,
            0.8,
            3,
            7.2,
            "[10, 0]",
            "[0.9, 0.0]",
            "[0.8, 0.0]",
            "[3, 0]",
            "[7.2, 0.0]",
            3,
            1,
        ]
    )
    sink.flush()
    store.close()

    run_dir = tmp_path / "r2"
    revised = [json.loads(line) for line in (run_dir / "revised.jsonl").read_text().splitlines()]
    assert revised[0]["cat_id"] == 25 and revised[0]["period_id"] == 1
    assert revised[0]["create_time"].startswith("2026-10-30T12:00:00")
    assert revised[1].get("deleted") is True
    stream = [json.loads(line) for line in (run_dir / "stream.jsonl").read_text().splitlines()]
    assert stream[0]["model_id"] == 7 and stream[0]["box_x2"] == 200.0
    period = json.loads((run_dir / "periods.jsonl").read_text().splitlines()[0])
    assert period["period_id"] == 1 and period["coordinated_cat_id"] == 25
    assert period["stats"]["detect_length"] == [10, 0]
    assert store.counters["deleted"] == 1 and store.counters["periods"] == 1


def _synth_rows(n, cat_ids):
    rows = []
    for i in range(n):
        cid = cat_ids[i % len(cat_ids)]
        rows.append(
            {
                "create_time": (T0 + datetime.timedelta(seconds=i * 5)).isoformat(),
                "box_x1": 100 + i * 10,
                "box_y1": 50,
                "box_x2": 220 + i * 10,
                "box_y2": 170,
                "cat_id": cid,
                "conf": 0.9,
            }
        )
    return rows


def test_charts_render_two_nonempty_pngs(tmp_path):
    cat_ids = [25, 26]
    escalations = [
        Escalation(time=T0 + datetime.timedelta(seconds=20), scope="frame", code="T5", reason="t"),
        Escalation(
            time=T0 + datetime.timedelta(seconds=10),
            end_time=T0 + datetime.timedelta(seconds=40),
            scope="period",
            code="P1",
            reason="p",
        ),
    ]
    out = tmp_path / "charts"
    orig, rev = plot_detection_charts(
        _synth_rows(20, cat_ids),
        _synth_rows(20, cat_ids),
        out,
        cat_ids,
        cat_names={25: "Qb", 26: "Qbo"},
        escalations=escalations,
    )
    assert orig.name == "original.png" and rev.name == "revised.png"
    assert orig.stat().st_size > 0 and rev.stat().st_size > 0


def test_charts_handle_empty_input(tmp_path):
    out = tmp_path / "empty"
    orig, rev = plot_detection_charts([], [], out, [25, 26])
    assert orig.stat().st_size > 0 and rev.stat().st_size > 0


def test_resolve_cat_ids_real_view():
    cat_ids, id_to_name = resolve_cat_ids(3)
    assert cat_ids == [25, 26]
    assert id_to_name == {25: "Qb", 26: "Qbo"}
    cat_ids_1, _ = resolve_cat_ids(1)
    assert cat_ids_1 == [0, 1]


def test_resolve_cat_ids_unknown_view():
    with pytest.raises(ValueError, match="known views"):
        resolve_cat_ids(999)
