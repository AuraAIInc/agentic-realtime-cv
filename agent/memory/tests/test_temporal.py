"""Bucket boundaries, history windows, child ids, expiry."""

from __future__ import annotations

import datetime

import pytest

from agent.memory import MemoryConfig, TemporalIndex

T = datetime.datetime(2026, 10, 31, 14, 3, 0)  # a Saturday


@pytest.fixture()
def index():
    return TemporalIndex(MemoryConfig.from_yaml())


def test_slot_boundaries(index):
    assert index.slot_id(datetime.datetime(2026, 10, 31, 14, 0, 0)) == "2026-10-31T14:00"
    assert index.slot_id(datetime.datetime(2026, 10, 31, 14, 4, 59)) == "2026-10-31T14:00"
    assert index.slot_id(datetime.datetime(2026, 10, 31, 14, 5, 0)) == "2026-10-31T14:05"
    assert index.slot_id(T) == "2026-10-31T14:00"


def test_hour_day_boundaries(index):
    assert index.hour_id(T) == "2026-10-31T14:00"
    assert index.day_id(T) == "2026-10-31"
    assert index.day_id(datetime.datetime(2026, 10, 31, 23, 59)) == "2026-10-31"
    assert index.day_id(datetime.datetime(2026, 11, 1, 0, 0)) == "2026-11-01"


def test_week_monday_anchored(index):
    monday = datetime.datetime(2026, 10, 26, 9, 0)
    sunday = datetime.datetime(2026, 11, 1, 23, 0)
    next_monday = datetime.datetime(2026, 11, 2, 0, 0)
    assert monday.weekday() == 0
    assert index.week_id(monday) == index.week_id(sunday) == index.week_id(T)
    assert index.week_id(next_monday) != index.week_id(T)


def test_bucket_id_dispatch_and_start_roundtrip(index):
    assert index.bucket_id("slot_5min", T) == "2026-10-31T14:00"
    assert index.bucket_id("hour", T) == "2026-10-31T14:00"
    assert index.bucket_id("day", T) == "2026-10-31"
    assert index.bucket_id("week", T) == index.week_id(T)
    assert index.bucket_start("slot_5min", "2026-10-31T14:00") == datetime.datetime(2026, 10, 31, 14, 0)
    assert index.bucket_start("day", "2026-10-31") == datetime.datetime(2026, 10, 31)
    assert index.bucket_start("week", index.week_id(T)) == datetime.datetime(2026, 10, 26)
    with pytest.raises(KeyError):
        index.bucket_id("month", T)
    with pytest.raises(ValueError):
        index.bucket_start("week", "2026-10-31")


def test_history_windows(index):
    assert index.same_slot_history(T, 2) == [
        "2026-10-29T14:00",
        "2026-10-30T14:00",
        "2026-10-31T14:00",
    ]
    assert index.history_ids("hour", T, 1) == ["2026-10-30T14:00", "2026-10-31T14:00"]
    assert index.history_ids("day", T, 2) == ["2026-10-29", "2026-10-30", "2026-10-31"]


def test_child_ids(index):
    slots = index.child_ids("hour", "2026-10-31T14:00")
    assert len(slots) == 12 and slots[0] == "2026-10-31T14:00"
    assert slots[-1] == "2026-10-31T14:55"
    assert len(index.child_ids("day", "2026-10-31")) == 24
    assert len(index.child_ids("week", index.week_id(T))) == 7
    assert index.child_ids("slot_5min", "2026-10-31T14:00") == []


def test_expiry(index):
    now = datetime.datetime(2026, 11, 10, 12, 0)
    assert index.is_expired("slot_5min", "2026-10-31T14:00", now, {"slot_5min": 7}) is True
    assert index.is_expired("slot_5min", "2026-11-10T12:00", now, {"slot_5min": 7}) is False
    assert index.is_expired("week", "2026-W40", now, {"week": 0}) is False  # 0 = keep
