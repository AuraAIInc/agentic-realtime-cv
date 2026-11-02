"""Trust gate: long-term memory keeps only high-confidence knowledge."""

from __future__ import annotations

import datetime

import pytest

from agent.memory import (
    LongTermMemory,
    MemoryConfig,
    MemoryEvent,
    MemoryQuery,
    StubLLMClient,
)

T0 = datetime.datetime(2026, 10, 31, 14, 2, 0)
SOFA_BOX = (500.0, 300.0, 600.0, 400.0)


@pytest.fixture()
def mem(tmp_path):
    cfg = MemoryConfig.from_yaml()
    m = LongTermMemory(
        config=cfg, store_path=tmp_path / "mem.yaml", llm_client=StubLLMClient()
    )
    yield m
    m.shutdown()


def _event(text, *, kind="period", confidence=0.95, **kw):
    args = {"time": T0, "view_id": 3, "box": SOFA_BOX, "cat_id": 1,
            "kind": kind, "text": text, "confidence": confidence}
    args.update(kw)
    return MemoryEvent(**args)


def test_gate_drops_low_confidence_and_untrusted(mem):
    low = mem.store(_event("Qb naps maybe.", confidence=0.5))
    missing = mem.store(_event("Qb naps.", confidence=None))
    mem.drain()
    assert low.result.persisted is False
    assert missing.result.persisted is False
    assert mem.mem_store.get_temporal("slot_5min", mem.temporal.slot_id(T0)) is None
    ctx = mem.invoke(MemoryQuery(time=T0, box=SOFA_BOX))
    assert all(h.unobserved for h in ctx.spatial) and not ctx.temporal


def test_gate_drops_escalation_occurrences_but_keeps_verdicts_and_summaries(mem):
    esc = mem.store(_event("[E6-T2] relabeled.", kind="escalation", code="T2", confidence=0.95))
    verdict = mem.store(_event("[E6-T2] keep_original.", kind="label", code="T2", confidence=0.9))
    summary = mem.store(_event("Today Qb napped.", kind="summary", confidence=0.5))
    mem.drain()
    assert esc.result.persisted is False
    assert verdict.result.persisted is True
    assert summary.result.persisted is True  # curated day summaries always qualify
    assert mem.mem_store.get_temporal("slot_5min", mem.temporal.slot_id(T0)) is not None


def test_evidence_attached_to_units_and_hits(mem):
    mem.store(_event(
        "Qb naps on the sofa.", confidence=0.95,
        image_names=["2026-10-31-14-02-00-000000.jpg"],
        detection={"box": [500, 300, 600, 400], "cat_id": 1, "conf": 0.9},
        appearance="Qb: short-haired white cat.",
    ))
    mem.drain()
    unit = mem.mem_store.get_spatial("L0", "r3c5")
    assert unit.images == ["2026-10-31-14-02-00-000000.jpg"]
    assert unit.detections[0]["cat_id"] == 1
    assert "short-haired white" in unit.appearance
    ctx = mem.invoke(MemoryQuery(time=T0, box=SOFA_BOX))
    hit = next(h for h in ctx.spatial if h.unit_id == "r3c5")
    assert hit.extra["images"] == ["2026-10-31-14-02-00-000000.jpg"]
    assert hit.extra["detections"][0]["conf"] == 0.9


def test_spatial_promote_writes_appearance_and_behavior(mem):
    for i in range(3):
        mem.store(_event(
            f"Qb naps on the sofa, visit {i}.", confidence=0.95,
            appearance="Qb: short-haired white cat.",
        ))
    mem.drain()
    mem.promote(force=True)
    mem.drain()
    l1 = mem.mem_store.get_spatial("L1", "r1c2")
    assert l1 is not None and not l1.is_empty
    assert "short-haired white" in l1.appearance
    assert l1.behavior  # stub compresses child activity into behavior
