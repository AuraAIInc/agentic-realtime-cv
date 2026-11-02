"""Daily short-term memory: record, render, roll over, merge into long-term."""

from __future__ import annotations

import datetime
import json

import pytest

from agent.memory import (
    DailyShortTermMemory,
    LongTermMemory,
    MemoryConfig,
    StubLLMClient,
)

T0 = datetime.datetime(2026, 10, 28, 16, 34, 0)
SOFA_BOX = (500.0, 300.0, 600.0, 400.0)


@pytest.fixture()
def deps(tmp_path):
    cfg = MemoryConfig.from_yaml()
    cfg.store_dir = str(tmp_path)
    mem = LongTermMemory(
        config=cfg, store_path=tmp_path / "mem.yaml", llm_client=StubLLMClient()
    )
    short = DailyShortTermMemory(cfg, view_id=3, long_term=mem)
    yield cfg, mem, short
    mem.shutdown()


def _period(short, pid=2, start=T0, cat_id=25, cat_name="Qb", conf=0.71, n=772):
    return short.record_period(
        period_id=pid, cat_id=cat_id, cat_name=cat_name,
        start=start, end=start + datetime.timedelta(minutes=6),
        detections=n, mean_conf=conf, box=SOFA_BOX,
        keyframes=["a.jpg", "b.jpg", "c.jpg"],
    )


def test_record_renders_plain_language_and_persists(deps):
    cfg, mem, short = deps
    short.start_session(T0, "run1")
    _period(short)
    short.record_episode(name="E6-T2", code="T2", time=T0, reason="relabeled box.", cat_id=25)
    assert short.record_verdict("E6-T2", "keep_original") is True
    assert short.record_verdict("E9-X", "merge") is False
    md = cfg.short_term_path(T0.date(), 3).read_text(encoding="utf-8")
    assert "16:34–16:40" in md and "Qb (cat 25)" in md and "772 sightings" in md
    assert "a.jpg" in md and "[E6-T2]" in md and "keep_original" in md
    state = json.loads(cfg.short_term_path(T0.date(), 3).with_suffix(".json").read_text())
    assert state["sessions"]["run1"]["periods"][0]["period_id"] == 2


def test_duplicate_period_skipped_and_session_sections_accumulate(deps):
    cfg, mem, short = deps
    short.start_session(T0, "run1")
    assert _period(short) is not None
    assert _period(short) is None  # same (period, start): replayed twice
    short.start_session(T0 + datetime.timedelta(hours=1), "run2")
    _period(short, pid=4, start=T0 + datetime.timedelta(hours=1), n=1, conf=0.82)
    md = cfg.short_term_path(T0.date(), 3).read_text(encoding="utf-8")
    assert "## Session run1" in md and "## Session run2" in md


def test_rollover_merges_day_and_marks_sessions(deps):
    cfg, mem, short = deps
    short.start_session(T0, "run1")
    _period(short)
    short.record_episode(name="E6-T2", code="T2", time=T0, reason="relabeled box.")
    assert short.rollover(T0 + datetime.timedelta(hours=1)) is None
    merged = short.rollover(T0 + datetime.timedelta(days=1))
    assert merged == T0.date()
    mem.drain()
    mem.promote(force=True)
    mem.drain()
    day = mem.mem_store.get_temporal("day", mem.temporal.day_id(T0))
    assert day is not None and not day.is_empty and "Qb visited" in day.summary
    l1 = mem.mem_store.get_spatial("L1", "r1c2")
    assert l1 is not None and not l1.is_empty
    # already merged: second merge is a no-op
    assert short.merge_into(mem) == []
    md = cfg.short_term_path(T0.date(), 3).read_text(encoding="utf-8")
    assert "Merged into long-term memory: run1" in md


def test_merge_partial_session_skips_merged_ones(deps):
    cfg, mem, short = deps
    short.start_session(T0, "run1")
    _period(short)
    short.start_session(T0 + datetime.timedelta(hours=1), "run2")
    receipts = short.merge_into(mem)
    assert len(receipts) == 2  # day narrative + one zone summary (run1 only has periods)
    assert short.merge_into(mem) == []
