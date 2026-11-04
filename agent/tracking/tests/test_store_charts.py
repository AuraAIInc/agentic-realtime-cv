"""Phase 3 tests: RunStore/FileSink, strip charts, view->cat resolution."""

import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.escalation import Escalation
from agent.tracking.cat_config import resolve_cat_ids, resolve_view_size
from agent.tracking.charts import plot_detection_charts, plot_strip_chart, plot_validated_chart
from agent.tracking.store import DEFAULT_RUNS_DIR, FileSink, RunStore

T0 = datetime.datetime(2026, 10, 30, 12, 0, 0)


def test_default_runs_dir_is_repo_root_output():
    root = Path(__file__).resolve().parent.parent.parent.parent
    assert DEFAULT_RUNS_DIR == root / "output" / "tracking_runs"


def _rev_row(i, cat_id=25, period_id=1, view_id=3):
    t = T0 + datetime.timedelta(seconds=i * 5)
    return [t, 100 + i * 10, 50.0, 200 + i * 10, 150.0, cat_id, 0.9, cat_id, period_id, view_id]


def test_runstore_creates_run_dir_and_jsonl(tmp_path):
    store = RunStore("r1", base_dir=tmp_path)
    store.append_stream({"create_time": T0.isoformat(), "cat_id": 25})
    store.append_revised({"create_time": T0.isoformat(), "cat_id": 25})
    store.append_period({"period_id": 1})
    store.append_image({"create_time": T0.isoformat()})
    store.append_escalation(Escalation(time=T0, scope="frame", code="T5", reason="test", seq=4))
    meta_path = store.write_run_json({"run": "r1", "view_id": 3})
    store.close()

    run_dir = tmp_path / "r1"
    for fname in ("stream.jsonl", "revised.jsonl", "periods.jsonl", "escalations.jsonl", "run.json"):
        assert (run_dir / fname).is_file(), fname
    assert json.loads(meta_path.read_text())["view_id"] == 3
    assert store.counters == {"stream": 1, "revised": 1, "periods": 1, "images": 1, "escalations": 1, "deleted": 0, "validations": 0}
    esc_line = json.loads((run_dir / "escalations.jsonl").read_text().splitlines()[0])
    assert esc_line["code"] == "T5" and esc_line["escalation"] is True
    assert (esc_line["seq"], esc_line["name"]) == (4, "E4-T5")


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
        Escalation(time=T0 + datetime.timedelta(seconds=20), scope="frame", code="T5", reason="t", seq=1),
        Escalation(
            time=T0 + datetime.timedelta(seconds=10),
            end_time=T0 + datetime.timedelta(seconds=40),
            scope="period",
            code="P1",
            reason="p",
            seq=2,
        ),
        {"time": (T0 + datetime.timedelta(seconds=30)).isoformat(), "scope": "frame", "code": "T3"},
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


def test_charts_y_axis_zero_to_n_with_lane_separators(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    real_subplots = plt.subplots
    axes = []

    def spy_subplots(*a, **k):
        fig, ax = real_subplots(*a, **k)
        axes.append(ax)
        return fig, ax

    monkeypatch.setattr(plt, "subplots", spy_subplots)
    cat_ids = [25, 26]
    out = tmp_path / "charts"
    plot_detection_charts(
        _synth_rows(10, cat_ids),
        _synth_rows(10, cat_ids),
        out,
        cat_ids,
        cat_names={25: "Qb", 26: "Qbo"},
        escalations=[Escalation(time=T0 + datetime.timedelta(seconds=20), scope="frame", code="T5", reason="t", seq=1)],
    )
    plot_validated_chart(_synth_rows(10, cat_ids), out, cat_ids, cat_names={25: "Qb", 26: "Qbo"})

    assert len(axes) == 3  # original, revised, validated share _plot_strip
    for ax in axes:
        assert ax.get_ylim() == (0, 2)
        assert list(ax.get_yticks()) == [0.5, 1.5]
        assert [t.get_text() for t in ax.get_yticklabels()] == ["Qb", "Qbo"]
        seps = [line for line in ax.lines if line.get_linestyle() == ":"]
        assert len(seps) == 1
        assert seps[0].get_ydata()[0] == 1
        assert seps[0].get_color() == "grey"
        assert seps[0].get_linewidth() == 0.8
    red_labels = [t for t in axes[0].texts if t.get_color() == "red"]
    assert [t.get_text() for t in red_labels] == ["E1-T5"]
    assert {t.get_position()[1] for t in red_labels} == {0}


def test_process_frame_labels_escalations_below_the_line(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    from agent.tracking.process_viz import render_process_frame

    calls = []
    monkeypatch.setattr(plt, "text", lambda *a, **k: calls.append((a, k)))
    fake_track = SimpleNamespace(
        buffer={0: [(T0, 10, 10, 100, 100)]},
        revised_detection={0: []},
        base={0: None},
        target={0: None},
        escalations=[Escalation(time=T0, scope="frame", code="T5", reason="t", seq=1)],
    )
    period_escs = [Escalation(time=T0, end_time=T0 + datetime.timedelta(seconds=5), scope="period", code="P1", reason="p", seq=2)]
    out = render_process_frame(fake_track, period_escs, 0, str(tmp_path / "f.jpg"))
    assert out is not None and Path(out).stat().st_size > 0
    labels = [a[2] for a, _ in calls]
    assert labels == ["E1-T5", "E2-P1"]
    assert all(k.get("color") == "red" for _, k in calls)


def test_validated_chart_writes_nonempty_png(tmp_path):
    cat_ids = [25, 26]
    rows = _synth_rows(10, cat_ids)
    for r in rows:
        r["validated_by"] = "E1-T2"
        r["verdict"] = "confirm"
    out = tmp_path / "charts"
    out.mkdir()
    path = plot_validated_chart(rows, out, cat_ids, cat_names={25: "Qb", 26: "Qbo"})
    assert path.name == "validated.png"
    assert path.is_file() and path.stat().st_size > 0


def test_validated_chart_keeps_nulled_rows_on_unknown_lane(tmp_path, monkeypatch):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    real_subplots = plt.subplots
    axes = []

    def spy_subplots(*a, **k):
        fig, ax = real_subplots(*a, **k)
        axes.append(ax)
        return fig, ax

    monkeypatch.setattr(plt, "subplots", spy_subplots)
    # known rows cluster near T0; the nulled (mark_unknown) row sits far later:
    # dropping it would shrink the chart to the cluster.
    rows = _synth_rows(4, [25])
    rows.append(
        {
            "create_time": (T0 + datetime.timedelta(seconds=300)).isoformat(),
            "box_x1": 100,
            "box_y1": 50,
            "box_x2": 200,
            "box_y2": 150,
            "cat_id": None,
            "verdict": "mark_unknown",
        }
    )
    out = tmp_path / "charts"
    path = plot_validated_chart(rows, out, [25, 26], cat_names={25: "Qb", 26: "Qbo"})
    assert path.stat().st_size > 0
    (ax,) = axes
    assert ax.get_ylim() == (0, 3)
    assert [t.get_text() for t in ax.get_yticklabels()] == ["Qb", "Qbo", "unknown"]
    assert len(ax.collections[0].get_segments()) == 5  # the nulled row is drawn, not skipped
    lo, hi = ax.get_xlim()
    assert lo < mdates.date2num(T0)
    assert hi > mdates.date2num(T0 + datetime.timedelta(seconds=290))


def test_charts_handle_empty_input(tmp_path):
    out = tmp_path / "empty"
    orig, rev = plot_detection_charts([], [], out, [25, 26])
    assert orig.stat().st_size > 0 and rev.stat().st_size > 0


def test_plot_strip_chart_writes_custom_path_and_title(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    real_subplots = plt.subplots
    axes = []

    def spy_subplots(*a, **k):
        fig, ax = real_subplots(*a, **k)
        axes.append(ax)
        return fig, ax

    monkeypatch.setattr(plt, "subplots", spy_subplots)
    out = tmp_path / "custom" / "gt.png"
    path = plot_strip_chart(
        _synth_rows(10, [25, 26]), out, "Ground truth detections", [25, 26], cat_names={25: "Qb", 26: "Qbo"}
    )
    assert path == out and out.stat().st_size > 0
    assert axes[0].get_title() == "Ground truth detections"


def test_resolve_cat_ids_real_view():
    cat_ids, id_to_name = resolve_cat_ids(3)
    assert cat_ids == [25, 26]
    assert id_to_name == {25: "Qb", 26: "Qbo"}
    cat_ids_1, _ = resolve_cat_ids(1)
    assert cat_ids_1 == [0, 1]


def test_resolve_cat_ids_unknown_view():
    with pytest.raises(ValueError, match="known views"):
        resolve_cat_ids(999)


def test_resolve_view_size_real_view():
    assert resolve_view_size(3) == (800, 600)


def test_resolve_view_size_falls_back_to_legacy(tmp_path):
    assert resolve_view_size(3, config_dir=tmp_path) == (640, 480)  # no zones.yaml
    (tmp_path / "zones.yaml").write_text("version: 1\nviews: {}\n", encoding="utf-8")
    assert resolve_view_size(999, config_dir=tmp_path) == (640, 480)  # view missing
    (tmp_path / "zones.yaml").write_text(
        "version: 1\nviews:\n  '3':\n    size: [0, -1]\n", encoding="utf-8"
    )
    assert resolve_view_size(3, config_dir=tmp_path) == (640, 480)  # malformed size
    (tmp_path / "zones.yaml").write_text(
        "version: 1\nviews:\n  '3':\n    size: [800, 600]\n", encoding="utf-8"
    )
    assert resolve_view_size(3, config_dir=tmp_path) == (800, 600)
