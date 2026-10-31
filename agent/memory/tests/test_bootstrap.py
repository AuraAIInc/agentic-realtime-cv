"""Bootstrap: vision job fills L2+L1, L0 inherits unobserved until refined."""

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
SOFA_BOX = (500.0, 300.0, 600.0, 400.0)  # L0 r3c5, L1 r1c2


class ZoneStub(StubLLMClient):
    def bootstrap(self, image: bytes) -> dict:
        out = super().bootstrap(image)
        out["zones"] = [
            {
                "id": "r1c2",
                "environment": "Dark fabric sofa zone.",
                "background": "Dark grey fabric.",
                "cat_contrast": "High contrast for a white cat.",
            },
            {"id": "bogus", "environment": "x", "background": "y", "cat_contrast": "z"},
        ]
        return out


class FailingVisionStub(StubLLMClient):
    def bootstrap(self, image: bytes) -> dict:
        raise ConnectionError("vision endpoint down")


@pytest.fixture()
def mem(tmp_path):
    cfg = MemoryConfig.from_yaml()
    m = LongTermMemory(
        config=cfg, store_path=tmp_path / "mem.yaml", llm_client=StubLLMClient()
    )
    yield m
    m.shutdown()


def test_bootstrap_marks_l0_unobserved_until_refined(mem):
    mem.bootstrap(b"fake-image-bytes")
    mem.drain()
    assert mem.mem_store.bootstrapped is True
    l2 = mem.mem_store.get_spatial("L2", "r0c0")
    assert l2 is not None and "Stub full-view" in l2.environment
    ctx = mem.invoke(MemoryQuery(time=T0, box=SOFA_BOX, spatial_radius=0))
    assert ctx.spatial[0].unobserved is True
    assert "Stub" in ctx.spatial[0].text  # inherited from L2 at read time
    mem.store(
        MemoryEvent(time=T0, view_id=3, box=SOFA_BOX, cat_id=1, text="Qb naps.")
    )
    mem.drain()
    ctx = mem.invoke(MemoryQuery(time=T0, box=SOFA_BOX, spatial_radius=0))
    assert ctx.spatial[0].unobserved is False


def test_bootstrap_zones_fill_l1(tmp_path):
    cfg = MemoryConfig.from_yaml()
    m = LongTermMemory(
        config=cfg, store_path=tmp_path / "mem.yaml", llm_client=ZoneStub()
    )
    try:
        receipt = m.bootstrap(b"fake-image-bytes")
        m.drain()
        assert receipt.result["zones"] == 1  # bogus id skipped
        zone = m.mem_store.get_spatial("L1", "r1c2")
        assert zone is not None and "sofa zone" in zone.environment
        # L0 inherits the nearest stored ancestor (L1), still marked unobserved.
        view, unobserved = m.mem_store.effective_l0("r3c5", m.spatial)
        assert unobserved is True
        assert view["environment"] == "Dark fabric sofa zone."
    finally:
        m.shutdown()


def test_bootstrap_failure_degrades_to_text_only(tmp_path):
    cfg = MemoryConfig.from_yaml()
    m = LongTermMemory(
        config=cfg, store_path=tmp_path / "mem.yaml", llm_client=FailingVisionStub()
    )
    try:
        receipt = m.bootstrap(b"fake-image-bytes")
        m.drain()
        assert receipt.error is None
        assert receipt.result["degraded"] is True
        assert m.mem_store.bootstrapped is True
        l2 = m.mem_store.get_spatial("L2", "r0c0")
        assert l2 is not None and "Unknown" in l2.environment
        assert m.worker.is_alive
    finally:
        m.shutdown()


def test_text_only_store_marks_vision_unknown(mem):
    mem.store(
        MemoryEvent(time=T0, view_id=3, box=SOFA_BOX, cat_id=1, text="Qb naps.")
    )
    mem.drain()
    assert mem.mem_store.bootstrapped is False
    unit = mem.mem_store.get_spatial("L0", "r3c5")
    assert unit is not None and "Unknown" in unit.environment
    assert unit.cat_contrast == "Unknown contrast."
    mem.bootstrap(b"fake-image-bytes")  # later bootstrap fills L2, keeps L0 text
    mem.drain()
    assert mem.mem_store.get_spatial("L2", "r0c0") is not None
    assert mem.mem_store.get_spatial("L0", "r3c5").activity == "Qb naps."


def test_first_store_with_image_auto_bootstraps(mem):
    assert mem.mem_store.bootstrapped is False
    mem.store(
        MemoryEvent(time=T0, view_id=3, box=SOFA_BOX, cat_id=1, text="Qb naps."),
        image=b"first-frame-bytes",
    )
    mem.drain()
    assert mem.mem_store.bootstrapped is True
    assert mem.mem_store.get_spatial("L2", "r0c0") is not None
    assert mem.mem_store.get_spatial("L0", "r3c5") is not None
