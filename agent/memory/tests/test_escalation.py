"""invoke_for_escalation: every code family returns relevant context."""

from __future__ import annotations

import datetime

import pytest

from agent.escalation.escalation import Escalation
from agent.memory import LongTermMemory, MemoryConfig, MemoryEvent, StubLLMClient

T0 = datetime.datetime(2026, 10, 31, 14, 2, 0)
SOFA_BOX = (500.0, 300.0, 600.0, 400.0)  # L0 r3c5
BOWL_BOX = (50.0, 450.0, 150.0, 550.0)  # L0 r5c1

FAMILIES = {
    "T1": "identity", "T2": "identity", "T5": "identity",
    "T4": "motion", "T6": "motion",
    "T3": "continuity", "T7": "continuity",
    "P1": "routine", "P2": "routine", "P5": "routine",
    "P3": "conflict", "P4": "occlusion",
    "P6": "merge", "P7": "merge",
}


@pytest.fixture()
def mem(tmp_path):
    cfg = MemoryConfig.from_yaml()
    m = LongTermMemory(
        config=cfg, store_path=tmp_path / "mem.yaml", llm_client=StubLLMClient()
    )
    for day in (29, 30, 31):
        m.store(
            MemoryEvent(
                time=datetime.datetime(2026, 10, day, 14, 2), view_id=3,
                box=SOFA_BOX, cat_id=1, kind="period",
                text="Qb naps on the sofa every afternoon.",
            )
        )
    m.store(
        MemoryEvent(
            time=T0, view_id=3, box=BOWL_BOX, cat_id=2, kind="period",
            text="Qbo eats from the food bowl by the wall.",
        )
    )
    m.drain()
    m.promote(force=True)
    m.drain()
    yield m
    m.shutdown()


def _esc(code, scope="frame", end_time=None):
    return Escalation(
        time=T0, scope=scope, code=code, reason=f"test {code}",
        end_time=end_time, cat_id=1, period_id=7, metrics={},
    )


def test_t_code_returns_spatial_context(mem):
    ctx = mem.invoke_for_escalation(_esc("T1"), box=SOFA_BOX)
    assert ctx.brief.startswith("[T1 identity]")
    assert len(ctx.spatial) >= 9  # hit cell + neighbors (+L1/L2)
    assert any("sofa" in h.text for h in ctx.spatial)


def test_p_code_returns_temporal_context(mem):
    ctx = mem.invoke_for_escalation(_esc("P1", scope="period"), box=SOFA_BOX)
    assert ctx.brief.startswith("[P1 routine]")
    slot_ids = [h.unit_id for h in ctx.temporal if h.level == "slot_5min"]
    assert slot_ids == ["2026-10-29T14:00", "2026-10-30T14:00", "2026-10-31T14:00"]


def test_every_family_answers_with_and_without_box(mem):
    for code, family in FAMILIES.items():
        scope = "frame" if code.startswith("T") else "period"
        end = T0 + datetime.timedelta(hours=2) if family == "merge" else None
        for box in (SOFA_BOX, None):
            ctx = mem.invoke_for_escalation(_esc(code, scope, end_time=end), box=box)
            assert ctx.brief.startswith(f"[{code} {family}]"), code
            assert (ctx.spatial or ctx.temporal or ctx.brief), code


def test_merge_family_covers_gap_span(mem):
    mem.store(
        MemoryEvent(
            time=T0 + datetime.timedelta(hours=1), view_id=3, box=SOFA_BOX,
            cat_id=1, kind="period", text="Qb still asleep on the sofa.",
        )
    )
    mem.drain()
    ctx = mem.invoke_for_escalation(
        _esc("P7", scope="period", end_time=T0 + datetime.timedelta(hours=2)),
        box=SOFA_BOX,
    )
    assert ctx.brief.startswith("[P7 merge]")
    assert any(h.level == "slot_5min" for h in ctx.temporal)


def test_unknown_code_falls_back_to_generic(mem):
    ctx = mem.invoke_for_escalation(_esc("Z9"), box=SOFA_BOX)
    assert ctx.brief.startswith("[Z9 generic]")
