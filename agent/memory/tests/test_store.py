"""YAML round-trip, sparseness, retention pruning, L0 inheritance."""

from __future__ import annotations

import datetime

from agent.memory import MemoryConfig, MemoryStore, SpatialIndex, SpatialUnit, TemporalUnit


def _config():
    return MemoryConfig.from_yaml()


def test_yaml_round_trip(tmp_path):
    store = MemoryStore(_config(), path=tmp_path / "mem.yaml")
    store.put_spatial(
        "L0",
        SpatialUnit(
            id="r3c5",
            bounds=[500, 300, 600, 400],
            environment="sofa",
            activity="Qb naps here.",
            keywords=["sofa", "qb"],
            weight=3,
            recent=["a", "b"],
            stats={"visits": 3, "by_cat": {"1": 3}},
            updated="2026-10-31T14:05:00",
            sources=["period 12"],
        ),
    )
    store.put_temporal("day", TemporalUnit(id="2026-10-31", summary="quiet day", weight=1))
    store.bootstrapped = True
    store.save()
    assert list(tmp_path.glob("*.tmp")) == []  # atomic write leaves no temp file

    loaded = MemoryStore(_config(), path=tmp_path / "mem.yaml").load()
    assert loaded.bootstrapped is True
    assert loaded.get_spatial("L0", "r3c5").to_dict() == store.get_spatial("L0", "r3c5").to_dict()
    assert loaded.get_temporal("day", "2026-10-31").summary == "quiet day"


def test_sparse_and_missing_file(tmp_path):
    store = MemoryStore(_config(), path=tmp_path / "mem.yaml")
    store.put_spatial("L0", SpatialUnit(id="r0c0"))  # empty: never persisted
    store.save()
    assert "r0c0" not in (tmp_path / "mem.yaml").read_text()
    fresh = MemoryStore(_config(), path=tmp_path / "absent.yaml").load()
    assert fresh.get_spatial("L0", "r0c0") is None and fresh.bootstrapped is False


def test_retention_pruning(tmp_path):
    store = MemoryStore(_config(), path=tmp_path / "mem.yaml")
    store.put_temporal("slot_5min", TemporalUnit(id="2026-10-20T14:00", summary="old"))
    store.put_temporal("slot_5min", TemporalUnit(id="2026-11-09T14:00", summary="fresh"))
    store.put_temporal("week", TemporalUnit(id="2026-W30", summary="kept forever"))
    removed = store.prune(datetime.datetime(2026, 11, 10, 12, 0))
    assert removed == 1
    assert store.get_temporal("slot_5min", "2026-10-20T14:00") is None
    assert store.get_temporal("slot_5min", "2026-11-09T14:00") is not None
    assert store.get_temporal("week", "2026-W30") is not None


def test_effective_l0_inheritance(tmp_path):
    cfg = _config()
    index = SpatialIndex(cfg)
    store = MemoryStore(cfg, path=tmp_path / "mem.yaml")
    store.put_spatial(
        "L1",
        SpatialUnit(
            id="r1c2", environment="dark fabric sofa", background="grey fabric", weight=5
        ),
    )
    view, unobserved = store.effective_l0("r3c5", index)
    assert unobserved is True
    assert view["environment"] == "dark fabric sofa"  # inherited from L1 parent
    assert view["bounds"] == [500, 300, 600, 400]  # own L0 bounds

    store.put_spatial(
        "L0", SpatialUnit(id="r3c5", bounds=[500, 300, 600, 400], activity="Qb naps.")
    )
    view, unobserved = store.effective_l0("r3c5", index)
    assert unobserved is False and view["activity"] == "Qb naps."

    view, unobserved = store.effective_l0("r0c0", index)  # nothing stored anywhere
    assert unobserved is True and view["environment"] == ""
