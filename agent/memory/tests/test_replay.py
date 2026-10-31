"""Replay integration: store -> tick -> promote -> invoke over periods/escalations.

Feeds a small synthetic fixture (with boxes, so spatial levels fill) plus, when
a real tracking run exists on disk (read-only), its periods.jsonl /
escalations.jsonl through the same TrackingRun event builders, then asserts the
YAML store exists, L2/L1/day units are non-empty, and invoke_for_escalation
returns context for a T-code and a P-code.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest
import yaml

from agent.escalation.escalation import Escalation
from agent.main import (
    escalation_dict_to_memory_event,
    period_dict_to_memory_event,
)
from agent.memory import (
    LongTermMemory,
    MemoryConfig,
    MemoryEvent,
    MemoryQuery,
    StubLLMClient,
)

SOFA_BOX = (500.0, 300.0, 600.0, 400.0)  # L0 r3c5 -> L1 r1c2
BOWL_BOX = (450.0, 350.0, 550.0, 450.0)  # L0 r3c4-ish, same L1 zone as sofa
REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO_ROOT / "agent" / "output" / "tracking_runs"


def _mem(tmp_path, name="mem.yaml"):
    cfg = MemoryConfig.from_yaml()
    return LongTermMemory(
        config=cfg, store_path=tmp_path / name, llm_client=StubLLMClient()
    )


def _synthetic_events():
    """3+ observations per cell so min_pattern_support=3 promotes text upward."""
    base = datetime.datetime(2026, 10, 31, 14, 2, 0)
    events = []
    for i in range(4):
        events.append(
            MemoryEvent(
                time=base + datetime.timedelta(minutes=i), view_id=3,
                box=SOFA_BOX, cat_id=1, period_id=12 + i, kind="period",
                text="Qb naps on the dark sofa in the afternoon.",
            )
        )
    for i in range(3):
        events.append(
            MemoryEvent(
                time=base + datetime.timedelta(minutes=i), view_id=3,
                box=BOWL_BOX, cat_id=2, period_id=20 + i, kind="period",
                text="Qbo eats from the food bowl near the sofa.",
            )
        )
    events.append(
        MemoryEvent(
            time=base, view_id=3, box=None, cat_id=2, period_id=20,
            kind="escalation", code="P2",
            text="[P2] period 20 winner cat 2 is weak: score 2.4.",
        )
    )
    return events


def test_synthetic_replay_store_tick_promote_invoke(tmp_path):
    mem = _mem(tmp_path)
    try:
        for ev in _synthetic_events():
            mem.store(ev)
        mem.drain()
        # Idle tick enqueues a promote once last_store is old enough.
        tick_at = datetime.datetime(2026, 10, 31, 15, 30, 0)
        assert mem.tick(tick_at) is not None
        mem.promote(force=True)
        mem.drain()

        store_file = tmp_path / "mem.yaml"
        assert store_file.exists()
        raw = yaml.safe_load(store_file.read_text(encoding="utf-8"))
        assert raw.get("spatial", {}).get("L2"), "L2 must be non-empty"
        assert raw.get("spatial", {}).get("L1"), "L1 must be non-empty"
        assert raw.get("temporal", {}).get("day"), "day must be non-empty"

        l2 = mem.mem_store.get_spatial("L2", "r0c0")
        assert l2 is not None and not l2.is_empty and l2.activity
        day_id = mem.temporal.day_id(datetime.datetime(2026, 10, 31, 14, 2))
        day = mem.mem_store.get_temporal("day", day_id)
        assert day is not None and not day.is_empty and day.summary

        ctx = mem.invoke(MemoryQuery(time=tick_at, box=SOFA_BOX))
        assert ctx.spatial and ctx.temporal

        t_ctx = mem.invoke_for_escalation(
            Escalation(
                time=tick_at, scope="frame", code="T1",
                reason="detection fits 2 cats; first match won",
                cat_id=1, metrics={},
            ),
            box=SOFA_BOX,
        )
        assert t_ctx.spatial and "[T1 identity]" in t_ctx.brief
        p_ctx = mem.invoke_for_escalation(
            Escalation(
                time=tick_at, scope="period", code="P2",
                reason="period 20 winner is weak",
                end_time=tick_at, cat_id=2, period_id=20, metrics={},
            )
        )
        assert p_ctx.temporal and "[P2 routine]" in p_ctx.brief
    finally:
        mem.shutdown()


def _find_real_run():
    if not RUNS_DIR.is_dir():
        return None
    for run_dir in sorted(RUNS_DIR.iterdir()):
        periods = run_dir / "periods.jsonl"
        if periods.is_file() and periods.stat().st_size:
            return run_dir
    return None


def _read_jsonl(path, cap_bytes=1_000_000):
    if path.stat().st_size > cap_bytes:
        pytest.skip(f"{path.name} too large for the replay test")
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_real_run_replay_read_only(tmp_path):
    run_dir = _find_real_run()
    if run_dir is None:
        pytest.skip("no tracking run on disk")
    periods = _read_jsonl(run_dir / "periods.jsonl")
    if not periods:
        pytest.skip("run has no periods")
    escs = []
    esc_file = run_dir / "escalations.jsonl"
    if esc_file.is_file() and esc_file.stat().st_size:
        escs = _read_jsonl(esc_file)

    mem = _mem(tmp_path, "replay.yaml")
    try:
        view_id = int(periods[0].get("view_id") or 3)
        for row in periods:
            mem.store(period_dict_to_memory_event(row, view_id=view_id))
        for row in escs:
            mem.store(escalation_dict_to_memory_event(row, view_id=view_id))
        mem.drain()
        last = max(
            datetime.datetime.fromisoformat(p["end"].replace(" ", "T"))
            for p in periods
        )
        mem.tick(last + datetime.timedelta(hours=2))
        mem.promote(force=True)
        mem.drain()

        store_file = tmp_path / "replay.yaml"
        assert store_file.exists()
        day_id = mem.temporal.day_id(last)
        day = mem.mem_store.get_temporal("day", day_id)
        assert day is not None and not day.is_empty, "day must summarize replayed periods"

        probe = Escalation(
            time=last, scope="period", code="P2", reason="replay probe",
            end_time=last, metrics={},
        )
        ctx = mem.invoke_for_escalation(probe)
        assert ctx.temporal, "replayed periods must be retrievable"
    finally:
        mem.shutdown()
