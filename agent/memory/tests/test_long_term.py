"""Long-term memory: store/invoke, history, keyword search, smoke test, reset."""

from __future__ import annotations

import datetime
import threading

import pytest

from agent.memory import (
    LongTermMemory,
    MemoryConfig,
    MemoryEvent,
    MemoryQuery,
    StubLLMClient,
)

T0 = datetime.datetime(2026, 10, 31, 14, 2, 0)
SOFA_BOX = (500.0, 300.0, 600.0, 400.0)  # L0 r3c5
BOWL_BOX = (50.0, 450.0, 150.0, 550.0)  # L0 r5c1


@pytest.fixture()
def mem(tmp_path):
    cfg = MemoryConfig.from_yaml()
    m = LongTermMemory(
        config=cfg, store_path=tmp_path / "mem.yaml", llm_client=StubLLMClient()
    )
    yield m
    m.shutdown()


def _event(text, when=T0, box=SOFA_BOX, cat_id=1, period_id=None, code=None):
    return MemoryEvent(
        time=when, view_id=3, box=box, cat_id=cat_id, period_id=period_id,
        kind="period", text=text, code=code,
    )


def test_store_then_invoke_spatial(mem):
    mem.store(_event("Qb naps on the sofa every afternoon.", period_id=12))
    mem.drain()
    ctx = mem.invoke(MemoryQuery(time=T0, box=SOFA_BOX, spatial_radius=1))
    assert len(ctx.spatial) == 9
    center = next(h for h in ctx.spatial if h.unit_id == "r3c5")
    assert "sofa" in center.text and center.unobserved is False
    assert "9 spatial cells" in ctx.brief


def test_neighbor_radius_zero(mem):
    mem.store(_event("Qb naps on the sofa."))
    mem.drain()
    ctx = mem.invoke(MemoryQuery(time=T0, box=SOFA_BOX, spatial_radius=0))
    assert [h.unit_id for h in ctx.spatial] == ["r3c5"]


def test_unit_counters(mem):
    mem.store(_event("Qb naps on the sofa.", cat_id=1))
    mem.store(_event("Qb still napping on the sofa.", cat_id=1))
    mem.store(_event("Qbo walks past the sofa.", cat_id=2))
    mem.drain()
    unit = mem.mem_store.get_spatial("L0", "r3c5")
    assert unit.weight == 3
    assert unit.stats["visits"] == 3
    assert unit.stats["by_cat"] == {"1": 2, "2": 1}
    assert len(unit.recent) == 3
    assert "sofa" in unit.keywords


def test_history_windows(mem):
    for day in (29, 30, 31):
        mem.store(_event("Qb afternoon nap.", when=datetime.datetime(2026, 10, day, 14, 2)))
    mem.store(_event("Qb evening zoomies.", when=datetime.datetime(2026, 10, 31, 15, 30)))
    mem.drain()
    ctx = mem.invoke(MemoryQuery(time=datetime.datetime(2026, 10, 31, 14, 3), history_days=7))
    slot_ids = [h.unit_id for h in ctx.temporal if h.level == "slot_5min"]
    assert slot_ids == ["2026-10-29T14:00", "2026-10-30T14:00", "2026-10-31T14:00"]
    ctx1 = mem.invoke(MemoryQuery(time=datetime.datetime(2026, 10, 31, 14, 3), history_days=1))
    assert [h.unit_id for h in ctx1.temporal if h.level == "slot_5min"] == [
        "2026-10-30T14:00",
        "2026-10-31T14:00",
    ]


def test_keyword_ranking(mem):
    mem.store(_event("Qb naps on the sofa every afternoon.", box=SOFA_BOX, cat_id=1))
    mem.store(_event("Qbo eats from the food bowl by the wall.", box=BOWL_BOX, cat_id=2))
    mem.drain()
    sofa_hits = mem.semantic_search("sofa naps")
    assert sofa_hits and sofa_hits[0].unit_id == "r3c5"
    bowl_hits = mem.semantic_search("food bowl")
    assert bowl_hits and bowl_hits[0].unit_id == "r5c1"
    assert len(mem.semantic_search("sofa", top_k=1)) == 1
    assert mem.semantic_search("the and of") == []  # stopwords only
    ctx = mem.invoke(MemoryQuery(time=T0, box=SOFA_BOX, text="sofa"))
    assert ctx.semantic and ctx.semantic[0].unit_id == "r3c5"


def test_promote_and_tick(mem):
    mem.store(_event("Qb naps."))
    mem.drain()
    r1 = mem.promote()
    r2 = mem.promote()
    mem.drain()
    assert r1.done and r2.done  # coalesced or sequential: both complete
    assert mem.tick(mem.last_store + datetime.timedelta(seconds=60)) is None
    receipt = mem.tick(mem.last_store + datetime.timedelta(seconds=301))
    assert receipt is not None and receipt.kind == "promote"
    mem.drain()


def test_tick_without_stores(tmp_path):
    cfg = MemoryConfig.from_yaml()
    m = LongTermMemory(
        config=cfg, store_path=tmp_path / "mem.yaml", llm_client=StubLLMClient()
    )
    try:
        assert m.tick(datetime.datetime(2026, 10, 31, 14, 0)) is None
    finally:
        m.shutdown()


def test_invoke_during_store_smoke(mem):
    errors: list = []

    def hammer():
        try:
            for _ in range(30):
                mem.invoke(MemoryQuery(time=T0, box=SOFA_BOX))
        except Exception as exc:  # noqa: BLE001 - collected, asserted below
            errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    for i in range(20):
        mem.store(_event(f"observation {i}."))
    for t in threads:
        t.join(timeout=30)
    mem.drain(timeout=30)
    assert errors == []
    ctx = mem.invoke(MemoryQuery(time=T0, box=SOFA_BOX))
    assert any("observation" in h.text for h in ctx.spatial)


def test_reset_with_backup(mem, tmp_path):
    mem.store(_event("Qb naps on the sofa."))
    mem.drain()
    assert (tmp_path / "mem.yaml").exists()
    mem.reset(keep_backup=True)
    backups = list(tmp_path.glob("mem.yaml.*.bak"))
    assert len(backups) == 1 and "r3c5" in backups[0].read_text()
    assert mem.mem_store.get_spatial("L0", "r3c5") is None
    assert mem.mem_store.bootstrapped is False
    ctx = mem.invoke(MemoryQuery(time=T0, box=SOFA_BOX, spatial_radius=0))
    assert ctx.spatial[0].unobserved is True
    mem.store(_event("post-reset observation."))  # worker still running
    mem.drain()


def test_reset_without_backup(mem, tmp_path):
    mem.store(_event("Qb naps on the sofa."))
    mem.drain()
    mem.reset(keep_backup=False)
    assert list(tmp_path.glob("*.bak")) == []
    assert (tmp_path / "mem.yaml").exists()
    assert mem.mem_store.get_spatial("L0", "r3c5") is None


def test_bootstrap_idempotent(mem):
    mem.bootstrap(b"fake-image-bytes")
    mem.drain()
    assert mem.mem_store.bootstrapped is True
    l2 = mem.mem_store.get_spatial("L2", "r0c0")
    assert l2 is not None and l2.environment != ""
    mem.bootstrap(b"fake-image-bytes")
    mem.drain()
    assert mem.mem_store.bootstrapped is True
