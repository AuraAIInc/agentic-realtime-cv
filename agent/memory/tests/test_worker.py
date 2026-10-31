"""Worker: FIFO ordering, promote coalescing, bootstrap priority, drain/shutdown."""

from __future__ import annotations

import logging
import threading

import pytest

from agent.memory import MemoryConfig, MemoryWorker


def _config(**overrides):
    cfg = MemoryConfig.from_yaml()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _recorder(calls, gate=None):
    def run(payload):
        if gate is not None and payload.get("block"):
            assert gate.wait(timeout=10)
        calls.append(payload.get("name", payload))
        return payload.get("name")

    return run


def _handlers(calls, gate=None):
    return {
        "store": _recorder(calls, gate),
        "promote": _recorder(calls),
        "bootstrap": _recorder(calls),
    }


def test_fifo_ordering_and_receipts():
    calls = []
    worker = MemoryWorker(_config(), _handlers(calls))
    try:
        receipts = [worker.submit("store", {"name": n}) for n in ("a", "b", "c")]
        assert len({r.job_id for r in receipts}) == 3
        worker.drain(timeout=10)
        assert calls == ["a", "b", "c"]
        assert all(r.done and r.error is None for r in receipts)
        assert receipts[0].wait(timeout=1) is True
    finally:
        worker.shutdown()


def test_promote_coalescing():
    calls = []
    gate = threading.Event()
    worker = MemoryWorker(_config(), _handlers(calls, gate))
    try:
        worker.submit("store", {"name": "blocker", "block": True})
        first = worker.submit("promote", {"name": "p1"})
        second = worker.submit("promote", {"name": "p2"})
        assert second.job_id == first.job_id  # coalesced into the queued promote
        gate.set()
        worker.drain(timeout=10)
        assert calls.count("p1") + calls.count("p2") == 1
    finally:
        gate.set()
        worker.shutdown()


def test_promote_no_coalesce_when_disabled():
    calls = []
    gate = threading.Event()
    worker = MemoryWorker(_config(promote_coalesce=False), _handlers(calls, gate))
    try:
        worker.submit("store", {"name": "blocker", "block": True})
        r1 = worker.submit("promote", {"name": "p1"})
        r2 = worker.submit("promote", {"name": "p2"})
        assert r1.job_id != r2.job_id
        gate.set()
        worker.drain(timeout=10)
        assert sorted(calls) == ["blocker", "p1", "p2"]
    finally:
        gate.set()
        worker.shutdown()


def test_promote_force_bypasses_coalescing():
    calls = []
    gate = threading.Event()
    worker = MemoryWorker(_config(), _handlers(calls, gate))
    try:
        worker.submit("store", {"name": "blocker", "block": True})
        worker.submit("promote", {"name": "p1"})
        worker.submit("promote", {"name": "p2", "force": True})
        gate.set()
        worker.drain(timeout=10)
        assert sorted(calls) == ["blocker", "p1", "p2"]
    finally:
        gate.set()
        worker.shutdown()


def test_bootstrap_jumps_ahead():
    calls = []
    gate = threading.Event()
    started = threading.Event()

    def run(payload):
        if payload.get("block"):
            started.set()
            assert gate.wait(timeout=10)
        calls.append(payload.get("name", payload))
        return payload.get("name")

    worker = MemoryWorker(
        _config(), {"store": run, "promote": run, "bootstrap": run}
    )
    try:
        worker.submit("store", {"name": "blocker", "block": True})
        assert started.wait(timeout=10)  # blocker running, so a/b/boot queue up
        worker.submit("store", {"name": "a"})
        worker.submit("store", {"name": "b"})
        worker.submit("bootstrap", {"name": "boot"})
        gate.set()
        worker.drain(timeout=10)
        assert calls[0] == "blocker"
        assert calls.index("boot") < calls.index("a") < calls.index("b")
    finally:
        gate.set()
        worker.shutdown()


def test_failure_isolation():
    calls = []

    def flaky(payload):
        if payload.get("fail"):
            raise RuntimeError("boom")
        calls.append(payload["name"])
        return "ok"

    worker = MemoryWorker(
        _config(), {"store": flaky, "promote": flaky, "bootstrap": flaky}
    )
    try:
        bad = worker.submit("store", {"fail": True})
        good = worker.submit("store", {"name": "after"})
        worker.drain(timeout=10)
        assert isinstance(bad.error, RuntimeError) and bad.done
        assert good.error is None and good.result == "ok"
        assert calls == ["after"] and worker.is_alive
    finally:
        worker.shutdown()


def test_queue_warn_depth_logs(caplog):
    gate = threading.Event()
    calls = []
    worker = MemoryWorker(_config(queue_warn_depth=1), _handlers(calls, gate))
    try:
        worker.submit("store", {"name": "blocker", "block": True})
        with caplog.at_level(logging.WARNING, logger="agent.memory.worker"):
            worker.submit("store", {"name": "a"})
            worker.submit("store", {"name": "b"})
            worker.submit("store", {"name": "c"})
        assert any("queue depth" in rec.message for rec in caplog.records)
    finally:
        gate.set()
        worker.shutdown()


def test_shutdown_and_errors():
    worker = MemoryWorker(
        _config(),
        {"store": lambda p: None, "promote": lambda p: None, "bootstrap": lambda p: None},
    )
    with pytest.raises(ValueError):
        worker.submit("nope", {})
    worker.submit("store", {})
    worker.shutdown(timeout=10)
    assert not worker.is_alive
    worker.shutdown()  # idempotent
    with pytest.raises(RuntimeError):
        worker.submit("store", {})
