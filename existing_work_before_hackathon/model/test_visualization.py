"""plot_detection_details works against the current schema: detection tables have
no image_name column (join key is create_time), recent images rows have
NULL light, and tracker cat ids (e.g. 25/26 on view 3) map to chart lanes
instead of indexing the fixed palette.

Run from the existing_work_before_hackathon/ folder:
    python -m pytest model/test_visualization.py -q
"""

import datetime
import os
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model.visualization as viz  # noqa: E402


class _SchemaFaithfulDB:
    """Fake MysqlDB: detection tables raise 1054 for image_name, like production."""

    def __init__(self, with_data=True):
        self.with_data = with_data
        self.seen_sql = []

    def execute_query(self, query, args=None):
        self.seen_sql.append(query)
        if ("revised_detection" in query or "FROM detection" in query or "aura_cat.detection" in query) and "image_name" in query:
            raise Exception('(1054, "Unknown column \'image_name\' in \'field list\'")')
        t0 = datetime.datetime(2026, 10, 27, 18, 0, 1)
        if "FROM aura_cat.images" in query:
            return [
                ("2026-10-27-18-00-01-000001.jpg", t0, None, 1, 0),  # light NULL on new rows
                ("2026-10-27-18-00-02-000002.jpg", t0 + datetime.timedelta(seconds=1), None, 0, 0),
            ]
        if not self.with_data:
            return []
        if "revised_detection" in query or "aura_cat.detection" in query:
            return [
                (t0, 25, 10.0, 10.0, 50.0, 50.0, 0.9),
                (t0 + datetime.timedelta(seconds=1), 26, 20.0, 20.0, 60.0, 60.0, 0.8),
            ]
        if "detected_period" in query:
            return [(25, t0, t0 + datetime.timedelta(minutes=1), 5, 10, 60.0, 0.8)]
        if "cat_states" in query:
            return [(26, t0, t0 + datetime.timedelta(minutes=1), 5, 10, 60.0, 0.8, 1,
                     0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 50.0, 50.0)]
        return []


def _patched(monkeypatch, **kwargs):
    db = _SchemaFaithfulDB(**kwargs)
    monkeypatch.setattr(viz, "MysqlDB", lambda: db)
    return db


def _plot(start, dest):
    import matplotlib.pyplot as plt
    try:
        ok = viz.plot_detection_details(start, start + datetime.timedelta(hours=1),
                                        revised=True, save_path=dest, show=False)
        ylim = plt.gca().get_ylim()
    finally:
        plt.close("all")
    return ok, ylim


def test_detection_query_avoids_image_name(monkeypatch):
    db = _patched(monkeypatch)
    start = datetime.datetime(2026, 10, 27, 18, 0, 0)
    images, detections, _, _ = viz._get_data_from_sql_for_plot(start, start + datetime.timedelta(hours=1), True)
    det_sql = [q for q in db.seen_sql if "revised_detection" in q]
    assert det_sql and all("image_name" not in q for q in det_sql)
    assert len(images) == 2 and len(detections) == 2


def test_plot_renders_high_cat_ids_as_lanes(monkeypatch, tmp_path):
    _patched(monkeypatch)
    start = datetime.datetime(2026, 10, 27, 18, 0, 0)
    dest = str(tmp_path / "detections.png")
    ok, ylim = _plot(start, dest)
    assert ok is True
    assert os.path.isfile(dest)
    assert ylim == (-0.5, 2.5)  # one lane per cat present (25, 26)


def test_plot_renders_without_detections(monkeypatch, tmp_path):
    _patched(monkeypatch, with_data=False)
    start = datetime.datetime(2026, 10, 27, 18, 0, 0)
    dest = str(tmp_path / "detections.png")
    ok, ylim = _plot(start, dest)
    assert ok is True
    assert os.path.isfile(dest)
    assert ylim == (-0.5, 3.5)
