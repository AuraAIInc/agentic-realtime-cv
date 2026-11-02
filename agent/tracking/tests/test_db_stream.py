"""db_stream groups DB rows into camera messages (fake DB, no MySQL).

Run from the repo root:
    .venv/bin/python -m pytest agent/tracking/tests/test_db_stream.py -q
"""

from __future__ import annotations

import datetime

from agent.tracking.db_stream import fetch_messages

T0 = datetime.datetime(2026, 10, 28, 16, 33, 0)


class FakeDB:
    def __init__(self, rows):
        self.rows = rows
        self.seen = None

    def execute_query(self, query, args=None):
        self.seen = (query, args)
        return self.rows


def test_groups_rows_and_keeps_empty_frames():
    rows = [
        (T0, 25, 10, 10, 50, 50, 0.7),
        (T0, 26, 60, 60, 100, 100, 0.6),
        (T0 + datetime.timedelta(seconds=1), None, None, None, None, None, None),
    ]
    msgs = list(fetch_messages(FakeDB(rows), 3, T0, T0 + datetime.timedelta(minutes=2)))
    assert len(msgs) == 2
    assert msgs[0]["create_time"] == "2026-10-28-16-33-00-000000"
    assert msgs[0]["view_id"] == 3
    assert msgs[0]["detections"] == [
        {"cat_id": 25, "confidence": 0.7, "box": [10.0, 10.0, 50.0, 50.0]},
        {"cat_id": 26, "confidence": 0.6, "box": [60.0, 60.0, 100.0, 100.0]},
    ]
    assert msgs[1]["detections"] == []  # empty frame kept: advances timeouts


def test_window_passed_to_sql():
    db = FakeDB([])
    list(fetch_messages(db, 3, T0, T0 + datetime.timedelta(minutes=2)))
    query, args = db.seen
    assert "LEFT JOIN detection" in query and args[0] == 3
    assert args[1] == T0 and args[2] == T0 + datetime.timedelta(minutes=2)
    assert query.strip().upper().startswith("SELECT")  # read-only: no writes, ever


def test_replay_db_fails_cleanly_when_mysql_is_unreachable(monkeypatch):
    from types import SimpleNamespace

    import pytest

    import agent.tracking.db_stream as db_stream
    from agent.main import run_replay_db

    def boom():
        raise OSError("connection refused")

    monkeypatch.setattr(db_stream, "connect", boom)
    run = SimpleNamespace(view_id=3, start=T0, end_exclusive=T0)
    with pytest.raises(SystemExit, match="cannot connect to MySQL"):
        run_replay_db(run)
