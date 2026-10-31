"""Weighting: contradiction counting vs flip_threshold, caps, bounded recent."""

from __future__ import annotations

import datetime

import pytest

from agent.memory import (
    LongTermMemory,
    MemoryConfig,
    MemoryEvent,
    StubLLMClient,
)

T0 = datetime.datetime(2026, 10, 31, 14, 2, 0)
SOFA_BOX = (500.0, 300.0, 600.0, 400.0)  # L0 r3c5
ROUTINE = "Qb naps on the sofa every afternoon."
FLIP = "Qb now naps on the CHAIR by the window, never the sofa."


class SignallingStub(StubLLMClient):
    """Stub that reports a contradiction when the observation has the marker."""

    def __init__(self, marker: str = "CHAIR"):
        self.marker = marker

    def store_update(self, unit: dict, observation: str) -> dict:
        out = super().store_update(unit, observation)
        out["contradicts_stable"] = self.marker in observation
        return out


class FailingStub(StubLLMClient):
    def store_update(self, unit: dict, observation: str) -> dict:
        raise ConnectionError("model endpoint down")


@pytest.fixture()
def mem(tmp_path):
    cfg = MemoryConfig.from_yaml()
    m = LongTermMemory(
        config=cfg, store_path=tmp_path / "mem.yaml",
        llm_client=SignallingStub(),
    )
    yield m
    m.shutdown()


def _event(text, when=T0, box=SOFA_BOX, cat_id=1):
    return MemoryEvent(
        time=when, view_id=3, box=box, cat_id=cat_id,
        kind="period", text=text,
    )


def test_one_contradiction_keeps_routine(mem):
    mem.store(_event(ROUTINE))
    mem.store(_event("Qb still napping on the sofa this afternoon."))
    mem.store(_event(FLIP))
    mem.drain()
    unit = mem.mem_store.get_spatial("L0", "r3c5")
    assert unit.activity == ROUTINE
    assert unit.contradictions == 1
    assert any("CHAIR" in line for line in unit.recent)


def test_three_contradictions_flip_routine(mem):
    mem.store(_event(ROUTINE))
    for i in range(3):
        mem.store(_event(FLIP, when=T0 + datetime.timedelta(minutes=10 * (i + 1))))
    mem.drain()
    unit = mem.mem_store.get_spatial("L0", "r3c5")
    assert "CHAIR" in unit.activity
    assert unit.contradictions == 0  # counter resets after the flip


def test_agreement_resets_contradiction_counter(mem):
    mem.store(_event(ROUTINE))
    mem.store(_event(FLIP))
    mem.store(_event("Qb back on the sofa for another nap."))
    mem.drain()
    unit = mem.mem_store.get_spatial("L0", "r3c5")
    assert unit.activity == ROUTINE
    assert unit.contradictions == 0


def test_weight_cap(tmp_path):
    cfg = MemoryConfig.from_yaml()
    cfg.stable_weight_cap = 5
    m = LongTermMemory(
        config=cfg, store_path=tmp_path / "mem.yaml",
        llm_client=SignallingStub(),
    )
    try:
        for i in range(7):
            m.store(_event(f"sofa observation {i}."))
        m.drain()
        unit = m.mem_store.get_spatial("L0", "r3c5")
        assert unit.stats["visits"] == 7
        assert unit.weight == 5  # capped, while raw visits keep counting
    finally:
        m.shutdown()


def test_recent_buffer_bounded(mem):
    for i in range(7):
        mem.store(_event(f"sofa observation {i}."))
    mem.drain()
    unit = mem.mem_store.get_spatial("L0", "r3c5")
    assert len(unit.recent) == mem.config.recent_kept == 5
    assert "observation 0." not in unit.recent[0]
    assert "observation 6." in unit.recent[-1]


def test_llm_error_degrades_to_pending_mode(tmp_path):
    cfg = MemoryConfig.from_yaml()
    m = LongTermMemory(
        config=cfg, store_path=tmp_path / "mem.yaml", llm_client=FailingStub()
    )
    try:
        receipt = m.store(_event(ROUTINE))
        m.drain()
        assert receipt.error is None  # never raises into callers
        assert m.worker.is_alive
        unit = m.mem_store.get_spatial("L0", "r3c5")
        assert unit.activity == ""  # stable text untouched
        assert unit.recent == [f"{T0.isoformat(timespec='minutes')} {ROUTINE}"]
        assert unit.stats["visits"] == 1  # counters still advance
    finally:
        m.shutdown()
