"""Worker consumes escalations in order, parses verdicts, journals ops, uses memory."""

import datetime
import json

from agent.escalation import Escalation
from agent.validation.worker import (
    ValidationWorker,
    build_merged_query,
    build_validation_query,
    ensure_supervisor_importable,
    merged_review_schema,
    parse_merged_review,
    parse_review,
)

T0 = datetime.datetime(2026, 10, 28, 16, 35, 21, 976102)


def _esc(code="T2", **kw):
    args = {"time": T0, "scope": "frame", "code": code, "reason": f"{code} reason",
            "cat_id": 25, "metrics": {"original_cat_id": 26, "rule": "distance"}, "seq": 6}
    args.update(kw)
    return Escalation(**args)


def _review_json(**kw):
    payload = {"verdict": "confirm", "rationale": "coat matches cat 25"}
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
    from agent.validation.worker import TOOL_CAVEATS

    query, schema = build_validation_query(_esc())
    assert "Validate:" in query and "ONLY a JSON object" in query
    assert TOOL_CAVEATS in query  # no crop retries, stale period detail ignored
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
    # Far-apart times: separate batches, one backend call each, in order.
    worker.enqueue(_esc("T2"))
    worker.enqueue(
        _esc("T3", time=T0 + datetime.timedelta(hours=2), cat_id=25, metrics={"killed_cat_id": 26, "iou": 0.8, "conf_margin": 0.04})
    )
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


def test_backend_receives_full_prompt_and_schema():
    from agent.escalation import build_prompt, review_schema_json

    seen = {}

    async def backend(query, schema, code):
        seen["query"], seen["schema"], seen["code"] = query, schema, code
        return _review_json()

    worker = ValidationWorker(backend=backend)
    worker.start()
    esc = _esc()
    worker.enqueue(esc)
    worker.stop(timeout=30)
    expected, _ = build_validation_query(esc)
    assert seen["query"] == expected
    assert "Validate:" in seen["query"] and "Resources" in seen["query"]
    assert seen["schema"] == json.loads(review_schema_json("T2"))
    assert seen["code"] == "T2"
    assert build_prompt(esc) in seen["query"]  # full escalation prompt, nothing truncated


def test_agent_event_log_covers_all_kinds(caplog):
    import logging
    from types import SimpleNamespace

    from agent.validation.worker import _agent_event_log

    with caplog.at_level(logging.INFO, logger="agent.validation.worker"):
        _agent_event_log("tool_calls", [("images__get_image", '{"a": 1}')])
        _agent_event_log("tool_result", SimpleNamespace(tool="images__get_image", is_error=False, round_trip_ms=12.3, error=""))
        _agent_event_log("tool_result", SimpleNamespace(tool="t", is_error=True, round_trip_ms=1.0, error="boom"))
        _agent_event_log("llm_text", (2, "looking at frames"))
    assert "tool call: images__get_image" in caplog.text
    assert "tool result: images__get_image: ok (round trip 12.3 ms)" in caplog.text
    assert "tool error: boom" in caplog.text
    assert "turn 2 text:\nlooking at frames" in caplog.text


def test_worker_failure_journals_error_op_and_continues():
    async def backend(query, schema, code):
        if code == "T2":
            raise RuntimeError("agent down")
        return _review_json(verdict="confirm_survivor")

    worker = ValidationWorker(backend=backend)
    worker.start()
    # Far-apart times: separate batches, so the T2 failure cannot touch T3.
    worker.enqueue(_esc("T2"))
    worker.enqueue(
        _esc("T3", time=T0 + datetime.timedelta(hours=2), cat_id=25, metrics={"killed_cat_id": 26, "iou": 0.8, "conf_margin": 0.04}, seq=7)
    )
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


def test_worker_uses_memory_for_context_only():
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
    assert memory.stored == []  # verdicts never reach long-term memory directly


class _StubShortTerm:
    def __init__(self):
        self.verdicts = []

    def record_verdict(self, name, verdict, rationale=None):
        self.verdicts.append((name, verdict, rationale))
        return True


def test_worker_verdict_goes_to_short_term_only():
    async def backend(query, schema, code):
        return _review_json()

    async def failing(query, schema, code):
        raise RuntimeError("boom")

    memory = _StubMemory()
    short_term = _StubShortTerm()
    worker = ValidationWorker(backend=backend, memory=memory, short_term=short_term)
    worker.start()
    worker.enqueue(_esc())
    worker.stop(timeout=30)
    assert memory.stored == []
    assert short_term.verdicts == [("E6-T2", "confirm", "coat matches cat 25")]

    memory2 = _StubMemory()
    short_term2 = _StubShortTerm()
    worker2 = ValidationWorker(backend=failing, memory=memory2, short_term=short_term2)
    worker2.start()
    worker2.enqueue(_esc())
    worker2.stop(timeout=30)
    assert memory2.stored == []  # errors carry no knowledge anywhere durable
    assert short_term2.verdicts == [("E6-T2", "error", None)]


def _merged_pair():
    first = _esc("T2", seq=6)
    second = _esc("T3", cat_id=25, metrics={"killed_cat_id": 26, "iou": 0.8, "conf_margin": 0.04}, seq=7)
    return [first, second]


def _merged_answer(**overrides):
    payload = {
        "reviews": [
            {"name": "E6-T2", "verdict": "confirm", "rationale": "coat matches", "chosen_cat_id": None, "cut_time": None},
            {"name": "E7-T3", "verdict": "confirm_survivor", "rationale": "survivor box fits", "chosen_cat_id": None, "cut_time": None},
        ]
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_validation_tools_include_reference_lookup():
    ensure_supervisor_importable()
    from agent.validation.worker import VALIDATION_TOOLS
    from supervisor_agent.local_tools.server import TOOLS

    assert "local__find_references" in VALIDATION_TOOLS
    assert "local__get_reference_image" in VALIDATION_TOOLS
    assert "local__get_frame" in VALIDATION_TOOLS
    assert "local__get_track_box" in VALIDATION_TOOLS
    assert {"find_references", "get_reference_image", "get_frame", "get_track_box"} <= {t["name"] for t in TOOLS}


def test_tool_caveats_scope_no_crop_to_images_server():
    from agent.validation.worker import TOOL_CAVEATS

    assert "do NOT retry crops on that server" in TOOL_CAVEATS
    assert "local__get_frame crops locally" in TOOL_CAVEATS
    assert "local__get_frame using the same timestamp" in TOOL_CAVEATS


def _op(name, verdict, when, end=None, rationale="coat matches", images=6, metrics=None):
    return {
        "escalation": {
            "name": name,
            "code": name.split("-")[1] if "-" in name else "?",
            "time": when.isoformat() if isinstance(when, datetime.datetime) else when,
            "end_time": (end.isoformat() if isinstance(end, datetime.datetime) else end) if end else None,
            "cat_id": 26,
            "period_id": 2,
            "reason": f"{name} reason",
            "metrics": metrics or {},
        },
        "review": {"verdict": verdict, "rationale": rationale},
        "agent_run": {"images_used": images},
    }


def test_prior_verdict_block_cites_overlapping_and_nearby():
    from agent.validation.worker import prior_verdict_block

    target = _esc("P8", time=T0 - datetime.timedelta(seconds=100), end_time=T0)
    ops = [
        _op("E3-T2", "confirm", T0 - datetime.timedelta(seconds=50), rationale="rug box is Qbo"),
        _op("E1-T6", "confirm_split", T0 - datetime.timedelta(seconds=160), rationale="two cats at once"),
        _op("E0-H1", "flag_issue", T0 - datetime.timedelta(seconds=10), rationale="servers down"),
        _op("E9-P2", "error", T0 - datetime.timedelta(seconds=5)),
        _op("E7-P6", "different_cats", T0 + datetime.timedelta(seconds=30), rationale="visits differ"),
    ]
    block = prior_verdict_block(ops, target)
    assert "E3-T2 -> confirm" in block and "rug box is Qbo" in block  # in span
    assert "E7-P6 -> different_cats" in block  # 30 s after end: nearby
    assert "E1-T6" in block  # exactly 60 s before start: nearby
    assert "E0-H1" not in block and "E9-P2" not in block  # undecided never cited
    assert "stay consistent" in block and "unanimous box-cropped evidence" in block


def test_prior_verdict_block_skips_far_and_empty():
    from agent.validation.worker import prior_verdict_block

    target = _esc("P2", time=T0, end_time=T0 + datetime.timedelta(seconds=2))
    far = _op("E1-T6", "confirm_split", T0 - datetime.timedelta(seconds=61))
    assert prior_verdict_block([far], target) == ""
    assert prior_verdict_block([], target) == ""
    assert prior_verdict_block([_op("E3-T2", "confirm", T0)], _esc("P2", time=None)) == ""


def test_prior_verdict_block_orders_by_time_and_caps():
    from agent.validation.worker import PRIOR_MAX_LINES, prior_verdict_block

    target = _esc("P8", time=T0 - datetime.timedelta(seconds=300), end_time=T0)
    ops = [_op(f"E{i}-T2", "confirm", T0 - datetime.timedelta(seconds=10 * (10 - i))) for i in range(10)]
    block = prior_verdict_block(ops, target)
    assert block.count("-> confirm") == PRIOR_MAX_LINES == 8
    assert block.index("E0-T2") < block.index("E1-T2")  # time order, earliest first


def test_prior_verdict_lines_carry_ruled_boxes():
    from agent.validation.worker import prior_verdict_block

    target = _esc("P8", time=T0 - datetime.timedelta(seconds=100), end_time=T0)
    ops = [
        _op("E3-T2", "confirm", T0 - datetime.timedelta(seconds=50),
            metrics={"box": [296.0, 369.0, 371.0, 531.0]}),
        _op("E6-T4", "drop_box_as_outlier", T0 - datetime.timedelta(seconds=40),
            metrics={"base_box": [289.0, 380.0, 376.0, 541.0], "box": [145.0, 357.0, 234.0, 462.0]}),
        _op("E10-P2", "confirm", T0 - datetime.timedelta(seconds=30),
            metrics={"key_frames": [{"t": T0.isoformat(), "box": [145.2, 357.4, 234.1, 462.0], "real": True}]}),
        _op("E8-H1", "confirm", T0 - datetime.timedelta(seconds=20)),
    ]
    block = prior_verdict_block(ops, target)
    assert "box=[296, 369, 371, 531]" in block  # T2 disputed box
    assert "base=[289, 380, 376, 541] new=[145, 357, 234, 462]" in block  # T4 anchor -> new
    assert "box=[145, 357, 234, 462]" in block  # P2 falls back to first real key frame
    h1_line = next(line for line in block.splitlines() if "E8-H1" in line)
    assert "box=" not in h1_line  # boxless codes cite no box


def test_prior_p8_split_cites_cut_time():
    from agent.validation.worker import prior_verdict_block

    target = _esc("P8", time=T0 - datetime.timedelta(seconds=100), end_time=T0)
    op = _op("E11-P8", "split_mixed", T0 - datetime.timedelta(seconds=90),
             T0 - datetime.timedelta(seconds=10))
    op["review"]["cut_time"] = (T0 - datetime.timedelta(seconds=50)).isoformat()
    op["review"]["chosen_cat_id"] = 25
    block = prior_verdict_block([op], target)
    assert "split@16:34:31->25" in block


def test_queries_carry_prior_block_when_given():
    from agent.validation.worker import MERGED_PREAMBLE

    query, _ = build_validation_query(_esc(), prior_block="PRIOR-RULINGS")
    assert "PRIOR-RULINGS" in query
    query2, _ = build_validation_query(_esc())
    assert "Already decided in this run" not in query2
    merged, _ = build_merged_query([_esc("T2"), _esc("T3")], [None, None], "PRIOR-RULINGS")
    assert "PRIOR-RULINGS" in merged
    assert "one continuous track cannot be two cats" in MERGED_PREAMBLE


def test_merged_query_carries_full_sections_and_envelope_schema():
    from agent.escalation import build_prompt

    from agent.validation.worker import TOOL_CAVEATS

    members = _merged_pair()
    query, schema = build_merged_query(members, ["brief one", None])
    assert "reviewing 2 related escalation cases" in query
    assert TOOL_CAVEATS in query
    assert "minimum total tool" in query and "fetch each image once" in query
    assert "=== Case 1 of 2: E6-T2 ===" in query
    assert "=== Case 2 of 2: E7-T3 ===" in query
    for esc in members:
        assert build_prompt(esc) in query  # full per-case prompt, nothing truncated
    assert "brief one" in query
    assert "one review per case, keyed by case name" in query
    items = schema["properties"]["reviews"]["items"]["properties"]
    assert set(items["verdict"]["enum"]) == {
        "confirm",
        "keep_original",
        "reassign_other",
        "drop",
        "confirm_survivor",
        "keep_killed",
        "keep_both",
    }
    assert schema["properties"]["reviews"]["minItems"] == 2
    assert schema["properties"]["reviews"]["maxItems"] == 2


def test_merged_schema_survives_strict_walk():
    from agent.validation.worker import _strict

    strict = _strict(merged_review_schema(_merged_pair()))
    item = strict["properties"]["reviews"]["items"]
    assert item["required"] == ["name", "verdict", "rationale", "chosen_cat_id", "cut_time"]
    assert item["additionalProperties"] is False


def test_parse_merged_review_happy_path_filters_to_code_fields():
    reviews = parse_merged_review(_merged_pair(), _merged_answer())
    # Envelope extras (name/cut_time) are stripped: each review fits its own model.
    assert reviews == {
        "E6-T2": {"verdict": "confirm", "rationale": "coat matches", "chosen_cat_id": None},
        "E7-T3": {"verdict": "confirm_survivor", "rationale": "survivor box fits"},
    }


def test_parse_merged_review_isolates_item_failures():
    members = _merged_pair()
    answer = json.dumps(
        {
            "reviews": [
                {"name": "E6-T2", "verdict": "not_a_verdict", "rationale": "x", "chosen_cat_id": None, "cut_time": None},
                {"name": "E7-T3", "verdict": "confirm_survivor", "rationale": "fits", "chosen_cat_id": None, "cut_time": None},
                {"name": "E stranger", "verdict": "confirm", "rationale": "x", "chosen_cat_id": None, "cut_time": None},
            ]
        }
    )
    reviews = parse_merged_review(members, answer)
    assert reviews["E6-T2"]["verdict"] == "error"
    assert "fails T2 review schema" in reviews["E6-T2"]["error"]
    assert reviews["E7-T3"] == {"verdict": "confirm_survivor", "rationale": "fits"}
    assert set(reviews) == {"E6-T2", "E7-T3"}  # unknown names ignored


def test_parse_merged_review_missing_member_errors_only_it():
    members = _merged_pair()
    answer = json.dumps({"reviews": [{"name": "E6-T2", "verdict": "confirm", "rationale": "coat matches", "chosen_cat_id": None, "cut_time": None}]})
    reviews = parse_merged_review(members, answer)
    assert reviews["E6-T2"]["verdict"] == "confirm"
    assert reviews["E7-T3"]["verdict"] == "error"
    assert "no review for case E7-T3" in reviews["E7-T3"]["error"]


def test_parse_merged_review_rejects_unusable_answers():
    members = _merged_pair()
    for bad in ["not json", "[1, 2]", json.dumps({"reviews": "nope"}), json.dumps({})]:
        try:
            parse_merged_review(members, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"should reject: {bad[:40]}")


def test_worker_merges_close_escalations_into_one_call():
    seen = []

    async def backend(query, schema, code):
        seen.append((code, schema["properties"]["reviews"]["maxItems"]))
        assert "=== Case 1 of 2: E6-T2 ===" in query
        assert "=== Case 2 of 2: E7-T3 ===" in query
        return _merged_answer()

    journaled = []
    memory = _StubMemory()
    short_term = _StubShortTerm()
    worker = ValidationWorker(backend=backend, journal=journaled.append, memory=memory, short_term=short_term)
    for esc in _merged_pair():  # same timestamp: one batch, one backend call
        worker.enqueue(esc)
    worker.start()  # start after enqueueing so the merge is deterministic
    ops = worker.stop(timeout=30)
    assert seen == [("E6-T2+E7-T3", 2)]
    assert [op["review"]["verdict"] for op in ops] == ["confirm", "confirm_survivor"]
    assert [op["escalation"]["name"] for op in ops] == ["E6-T2", "E7-T3"]
    assert journaled == ops
    assert memory.stored == []  # merged verdicts stay out of long-term memory too
    assert short_term.verdicts == [("E6-T2", "confirm", "coat matches"), ("E7-T3", "confirm_survivor", "survivor box fits")]


def test_worker_merged_batch_failure_errors_every_member_and_continues():
    calls = []

    async def backend(query, schema, code):
        calls.append(code)
        if "+" in code:
            raise RuntimeError("agent down")
        return _review_json()

    worker = ValidationWorker(backend=backend)
    for esc in _merged_pair():
        worker.enqueue(esc)
    worker.enqueue(_esc("T2", time=T0 + datetime.timedelta(hours=2), seq=8))
    worker.start()  # start after enqueueing so the merge is deterministic
    ops = worker.stop(timeout=30)
    assert calls == ["E6-T2+E7-T3", "T2"]
    assert [op["review"]["verdict"] for op in ops] == ["error", "error", "confirm"]
    assert "agent down" in ops[0]["review"]["error"]


def test_worker_max_batch_one_disables_merging():
    seen = []

    async def backend(query, schema, code):
        seen.append(code)
        return _review_json()

    worker = ValidationWorker(backend=backend, merge_max_batch=1)
    for esc in _merged_pair():  # same timestamp, but merging is off
        worker.enqueue(esc)
    worker.start()
    ops = worker.stop(timeout=30)
    assert seen == ["T2", "T3"]
    assert len(ops) == 2


def test_worker_records_backend_run_stats_on_ops():
    async def backend(query, schema, code):
        backend.last_run_stats = {
            "run_dir": "/tmp/r-validation",
            "trace": "/tmp/r-validation/trace.md",
            "images_used": 3,
        }
        return _review_json()

    worker = ValidationWorker(backend=backend)
    worker.start()
    worker.enqueue(_esc())
    ops = worker.stop(timeout=30)
    assert ops[0]["agent_run"]["images_used"] == 3
    assert ops[0]["agent_run"]["trace"] == "/tmp/r-validation/trace.md"


def test_worker_agent_run_is_none_when_backend_reports_nothing():
    async def backend(query, schema, code):
        return _review_json()

    worker = ValidationWorker(backend=backend)
    worker.start()
    worker.enqueue(_esc())
    ops = worker.stop(timeout=30)
    assert ops[0]["agent_run"] is None


def test_merged_query_pins_each_case_to_its_own_verdicts():
    escs = [_esc("T7", seq=6), _esc("P2", seq=7, scope="period")]
    query, schema = build_merged_query(escs)
    assert "Allowed verdicts for E6-T7: keep_synthesis | cut_at_gap_start | shorten_gap." in query
    assert "Allowed verdicts for E7-P2: confirm | drop | reassign." in query
    # The envelope itself still carries the union (fixed-shape strict schema).
    enum = schema["properties"]["reviews"]["items"]["properties"]["verdict"]["enum"]
    assert "confirm" in enum and "keep_synthesis" in enum


def test_batch_item_failure_retries_solo_with_exact_verdict():
    calls = []

    async def backend(query, schema, code):
        calls.append(code)
        if "+" in code:  # merged batch: T7 answers a sibling-only verdict
            return json.dumps({"reviews": [
                {"name": "E6-T7", "verdict": "confirm", "rationale": "looks fine",
                 "chosen_cat_id": None, "cut_time": None},
                {"name": "E7-P2", "verdict": "confirm", "rationale": "real visit",
                 "chosen_cat_id": None, "cut_time": None},
            ]})
        return json.dumps({"verdict": "keep_synthesis", "rationale": "cat stayed put"})

    journaled = []
    worker = ValidationWorker(backend=backend, journal=journaled.append)
    worker.start()
    worker.enqueue(_esc("T7", seq=6))  # same timestamp: one merged batch
    worker.enqueue(_esc("P2", seq=7, scope="period"))
    ops = worker.stop(timeout=30)
    assert calls == ["E6-T7+E7-P2", "T7"]
    assert [op["review"]["verdict"] for op in ops] == ["keep_synthesis", "confirm"]
    assert journaled == ops
