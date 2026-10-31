"""Compression: dominant patterns kept, one-offs dropped, word caps held."""

from __future__ import annotations

import pytest

from agent.memory import (
    LongTermMemory,
    MemoryConfig,
    StubLLMClient,
)
from agent.memory.llm_client import WORD_CAPS
from agent.memory.store import SpatialUnit, TemporalUnit

DOMINANT = "Qb naps on the sofa every afternoon."
ONE_OFF = "Qbo chased a housefly once."
LONG_WORDS = [f"sofa{i}" for i in range(200)]


class VerbatimStub(StubLLMClient):
    """Summarize without truncating, so the test pins long_term's own cap."""

    def summarize(self, level: str, children: list[dict], stats: dict) -> dict:
        texts = [str(c.get("text", "")) for c in children if c.get("text")]
        return {"text": " ".join(texts), "keywords": ["sofa"]}


@pytest.fixture()
def mem(tmp_path):
    cfg = MemoryConfig.from_yaml()
    m = LongTermMemory(
        config=cfg, store_path=tmp_path / "mem.yaml", llm_client=StubLLMClient()
    )
    yield m
    m.shutdown()


def _spatial(uid, text, visits):
    return SpatialUnit(
        id=uid,
        bounds=[0.0, 0.0, 100.0, 100.0],
        activity=text,
        keywords=["sofa"],
        weight=min(visits, 50),
        stats={"visits": visits, "by_cat": {"1": visits}},
        updated="2026-10-31T14:00:00",
    )


def _temporal(uid, text, visits):
    return TemporalUnit(
        id=uid,
        summary=text,
        keywords=["nap"],
        weight=min(visits, 50),
        stats={"visits": visits, "by_cat": {"1": visits}},
        updated="2026-10-31T14:00:00",
    )


def test_dominant_kept_one_off_dropped_spatial(mem):
    # r3c5 (dominant) and r2c4 (one-off) share L1 parent r1c2.
    with mem.worker.lock:
        mem.mem_store.put_spatial("L0", _spatial("r3c5", DOMINANT, 5))
        mem.mem_store.put_spatial("L0", _spatial("r2c4", ONE_OFF, 1))
    mem.promote(force=True)
    mem.drain()
    parent = mem.mem_store.get_spatial("L1", "r1c2")
    assert parent is not None
    assert "sofa" in parent.activity
    assert "housefly" not in parent.activity
    assert parent.stats["visits"] == 6  # stats roll up all children


def test_dominant_kept_one_off_dropped_temporal(mem):
    with mem.worker.lock:
        mem.mem_store.put_temporal(
            "slot_5min", _temporal("2026-10-31T14:00", "Qb napping on the sofa.", 5)
        )
        mem.mem_store.put_temporal(
            "slot_5min", _temporal("2026-10-31T14:05", "Qbo sneezed once.", 1)
        )
    mem.promote(force=True)
    mem.drain()
    hour = mem.mem_store.get_temporal("hour", "2026-10-31T14:00")
    assert hour is not None
    assert "napping" in hour.summary
    assert "sneezed" not in hour.summary
    day = mem.mem_store.get_temporal("day", "2026-10-31")
    assert day is not None and "napping" in day.summary  # bottom-up chaining


def test_word_caps_held_by_long_term(tmp_path):
    cfg = MemoryConfig.from_yaml()
    m = LongTermMemory(
        config=cfg, store_path=tmp_path / "mem.yaml", llm_client=VerbatimStub()
    )
    try:
        with m.worker.lock:
            m.mem_store.put_spatial("L0", _spatial("r3c5", " ".join(LONG_WORDS), 5))
        m.promote(force=True)
        m.drain()
        parent = m.mem_store.get_spatial("L1", "r1c2")
        assert parent is not None
        assert len(parent.activity.split()) == WORD_CAPS["L1"] == 100
    finally:
        m.shutdown()


def test_min_pattern_support_gates_parent_text(mem):
    # Every child below support: stats roll up but no LLM text is written.
    with mem.worker.lock:
        mem.mem_store.put_spatial("L0", _spatial("r3c5", ONE_OFF, 1))
        mem.mem_store.put_spatial("L0", _spatial("r2c4", "Qb blinked once.", 2))
    mem.promote(force=True)
    mem.drain()
    parent = mem.mem_store.get_spatial("L1", "r1c2")
    assert parent is not None
    assert parent.activity == ""
    assert parent.stats["visits"] == 3


def test_stats_only_rollup_still_drives_upper_levels(tmp_path):
    # Non-force promote: below-support L0 rolls stats into L1, and the fresh
    # L1 stats roll onward into L2 even though no LLM text is written.
    from agent.memory import MemoryEvent
    import datetime

    cfg = MemoryConfig.from_yaml()
    m = LongTermMemory(
        config=cfg, store_path=tmp_path / "mem.yaml", llm_client=StubLLMClient()
    )
    try:
        m.store(MemoryEvent(
            time=datetime.datetime(2026, 10, 31, 14, 2), view_id=3,
            box=(500.0, 300.0, 600.0, 400.0), cat_id=1, text="Qb naps once.",
        ))
        m.drain()
        m.promote()  # non-force: dirty-subtree walk
        m.drain()
        l1 = m.mem_store.get_spatial("L1", "r1c2")
        assert l1 is not None and l1.stats["visits"] == 1 and l1.activity == ""
        l2 = m.mem_store.get_spatial("L2", "r0c0")
        assert l2 is not None and l2.stats["visits"] == 1
    finally:
        m.shutdown()


def test_promote_budget_defers_and_resumes(tmp_path):
    cfg = MemoryConfig.from_yaml()
    cfg.max_units_per_call = 1  # one LLM call per promotion
    m = LongTermMemory(
        config=cfg, store_path=tmp_path / "mem.yaml", llm_client=StubLLMClient()
    )
    try:
        with m.worker.lock:
            m.mem_store.put_spatial("L0", _spatial("r3c5", DOMINANT, 5))  # L1 r1c2
            m.mem_store.put_spatial("L0", _spatial("r0c0", DOMINANT, 5))  # L1 r0c0
        first = m.promote(force=True)
        m.drain()
        assert first.result.promoted >= 1
        assert first.result.deferred >= 1
        cfg.max_units_per_call = 8
        second = m.promote(force=True)
        m.drain()
        assert second.result.deferred == 0
        assert m.mem_store.get_spatial("L1", "r1c2") is not None
        assert m.mem_store.get_spatial("L1", "r0c0") is not None
    finally:
        m.shutdown()
