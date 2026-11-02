"""Worker consumes escalations in order, parses verdicts, journals ops, uses memory."""

import datetime
import json

from agent.escalation import Escalation
from agent.validation.worker import ValidationWorker, build_validation_query, ensure_supervisor_importable, parse_review

T0 = datetime.datetime(2026, 10, 28, 16, 35, 21, 976102)


def _esc(code="T2", **kw):
    args = {"time": T0, "scope": "frame", "code": code, "reason": f"{code} reason",
            "cat_id": 25, "metrics": {"original_cat_id": 26, "rule": "distance"}, "seq": 6}
    args.update(kw)
    return Escalation(**args)


def _review_json(**kw):
    payload = {"verdict": "confirm"}
    payload.update(kw)
    return json.dumps(payload)


def test_parse_review_accepts_and_rejects():
    assert parse_review("T2", _review_json())["verdict"] == "confirm"
    assert parse_review("T2", "```json\n" + _review_json() + "\n```")["verdict"] == "confirm"
    for bad in ["not json", "[1, 2]", _review_json(verdict="maybe"), json.dumps({}), _review_json(confidence=0.9)]:
        try:
            parse_review("T2", bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"should reject: {bad[:40]}")


def test_query_carries_prompt_schema_and_memory():
    query, schema = build_validation_query(_esc())
    assert "Validate:" in query and "ONLY a JSON object" in query
    assert schema["properties"]["verdict"]["enum"] == ["confirm", "keep_original", "reassign_other", "drop"]
    query2, _ = build_validation_query(_esc(), memory_brief="cat 25 naps here daily")
    assert "cat 25 naps here daily" in query2


def test_worker_validates_in_order_and_journals():
    seen = []

    async def backend(query, schema, code):
        seen.append(code)
        verdict = "confirm" if code == "T2" else "confirm_survivor"
        return _review_json(verdict=verdict)

    journaled = []
    worker = ValidationWorker(backend=backend, journal=journaled.append)
    worker.start()
    worker.enqueue(_esc("T2"))
    worker.enqueue(_esc("T3", cat_id=25, metrics={"killed_cat_id": 26, "iou": 0.8, "conf_margin": 0.04}))
    ops = worker.stop(timeout=30)
    assert seen == ["T2", "T3"]
    assert [op["review"]["verdict"] for op in ops] == ["confirm", "confirm_survivor"]
    assert journaled == ops
    assert ops[0]["escalation"]["name"] == "E6-T2"


def test_ensure_supervisor_importable_puts_agent_on_path(monkeypatch):
    import sys
    from pathlib import Path

    agent_dir = str(Path("agent/validation/worker.py").resolve().parent.parent)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != agent_dir])
    ensure_supervisor_importable()
    assert sys.path[0] == agent_dir
    ensure_supervisor_importable()
    assert sys.path.count(agent_dir) == 1


def test_worker_failure_journals_error_op_and_continues():
    async def backend(query, schema, code):
        if code == "T2":
            raise RuntimeError("agent down")
        return _review_json(verdict="confirm_survivor")

    worker = ValidationWorker(backend=backend)
    worker.start()
    worker.enqueue(_esc("T2"))
    worker.enqueue(_esc("T3", cat_id=25, metrics={"killed_cat_id": 26, "iou": 0.8, "conf_margin": 0.04}, seq=7))
    ops = worker.stop(timeout=30)
    assert ops[0]["review"]["verdict"] == "error"
    assert "agent down" in ops[0]["review"]["error"]
    assert ops[1]["review"]["verdict"] == "confirm_survivor"


class _StubCtx:
    brief = "cat 25 naps here daily"


class _StubMemory:
    def __init__(self):
        self.stored = []
        self.config = type("Cfg", (), {"view_id": 3})()

    def invoke_for_escalation(self, esc, box=None):
        assert box == (1.0, 2.0, 3.0, 4.0)
        return _StubCtx()

    def store(self, event):
        self.stored.append(event)


def test_worker_uses_memory_for_context_and_verdict():
    queries = []

    async def backend(query, schema, code):
        queries.append(query)
        return _review_json()

    memory = _StubMemory()
    worker = ValidationWorker(backend=backend, memory=memory, box_lookup=lambda esc: (1.0, 2.0, 3.0, 4.0))
    worker.start()
    worker.enqueue(_esc())
    worker.stop(timeout=30)
    assert "cat 25 naps here daily" in queries[0]
    assert len(memory.stored) == 1
    assert memory.stored[0].kind == "label"
    assert memory.stored[0].code == "T2"
    assert "confirm" in memory.stored[0].text
