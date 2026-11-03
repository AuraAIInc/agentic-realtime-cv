"""Memory LLM traces: one markdown file per store/promote call.

Mirrors the supervisor trace layout at single-turn scale: table of contents,
#1 system prompt, #2 user input, #3 assistant output, parsed answer, stats.
"""

from __future__ import annotations

import json
import logging

from agent.memory import MetaModelClient, render_memory_trace, write_memory_trace


def _render(**kw):
    args = {
        "model": "muse-spark-1.3",
        "kind": "store",
        "label": "period 2",
        "started_at": "2026-11-02 01:21:00",
        "instructions": "You maintain one unit of memory.",
        "user_text": "STABLE (weight 1): Qb naps here.\nOBSERVATION: Qb naps.",
        "user_images": [],
        "text_format": {"type": "json_schema", "name": "memory_store_update", "schema": {}},
        "output_text": '{"contradicts_stable": false, "note": "Qb naps."}',
        "parsed": {"contradicts_stable": False, "note": "Qb naps."},
        "error": None,
        "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        "pricing": {"input": 1.25, "cached_input": 0.15, "output": 4.25},
        "est_cost_usd": 0.00021,
        "llm_ms": 1234.5,
        "response_id": "resp_abc",
    }
    args.update(kw)
    return render_memory_trace(**args)


def test_render_has_toc_messages_answer_and_stats():
    md = _render()
    assert md.startswith("# Memory LLM — store — period 2 — ")
    assert "| [#1](#msg-1) | system |" in md
    assert "| [#2](#msg-2) | user |" in md
    assert "| [#3](#msg-3) | assistant |" in md
    assert "### #1. system" in md and "You maintain one unit of memory." in md
    assert "### #2. user" in md and "OBSERVATION: Qb naps." in md
    assert "### #3. assistant" in md and '"note": "Qb naps."' in md
    assert "## Answer" in md
    assert "- model: muse-spark-1.3" in md
    assert "- kind: store" in md and "- label: period 2" in md
    assert "- input_tokens: 100" in md and "- output_tokens: 20" in md
    assert "- est_cost_usd: 0.000210" in md
    assert "- response_id: resp_abc" in md
    assert "text_format" in md and "memory_store_update" in md


def test_render_without_usage_or_pricing():
    md = _render(usage={}, pricing=None, est_cost_usd=None, response_id=None)
    assert "ctx n/a" in md
    assert "- input_tokens: n/a" in md
    assert "- pricing_usd_per_1m: n/a (no pricing for muse-spark-1.3)" in md
    assert "- est_cost_usd: n/a" in md


def test_render_error_and_unparseable():
    md = _render(output_text="not json", parsed=None, error="non-JSON output")
    assert "**Error:** `non-JSON output`" in md
    assert "(unparseable)" in md


def test_write_creates_file_with_kind_and_safe_label(tmp_path):
    path = write_memory_trace(
        tmp_path,
        model="m",
        kind="promote",
        label="L1:r2c1",
        instructions="sys",
        user_text="in",
        text_format=None,
        output_text="out",
        parsed={"text": "t"},
        error=None,
        usage={},
        pricing=None,
        llm_ms=1.0,
        response_id=None,
    )
    assert path.parent == tmp_path and path.suffix == ".md"
    assert "-promote-L1-r2c1.md" in path.name  # colon sanitized
    assert "### #2. user" in path.read_text()


def test_write_saves_and_embeds_images(tmp_path):
    path = write_memory_trace(
        tmp_path,
        model="m",
        kind="bootstrap",
        label="first-frame",
        instructions="sys",
        user_text="Describe this frame.",
        text_format=None,
        output_text="{}",
        parsed={},
        error=None,
        usage={},
        pricing=None,
        llm_ms=1.0,
        response_id=None,
        images=[(".jpg", b"\xff\xd8fake-jpeg")],
    )
    img = path.with_name(path.stem + ".jpg")
    assert img.read_bytes() == b"\xff\xd8fake-jpeg"
    assert f"![input image]({img.name})" in path.read_text()


class _UsageResp:
    def __init__(self, text):
        self.output_text = text
        self.id = "resp_1"
        self.usage = {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "input_tokens_details": {"cached_tokens": 10},
        }


class _UsageResponses:
    def __init__(self, script):
        self.script = list(script)

    def create(self, **kwargs):
        text = self.script.pop(0)
        if isinstance(text, Exception):
            raise text
        return _UsageResp(text)


class _UsageClient:
    def __init__(self, script):
        self.responses = _UsageResponses(script)


def test_client_writes_one_trace_per_call(tmp_path):
    c = MetaModelClient(
        client=_UsageClient([
            '{"contradicts_stable": false, "note": "n"}',
            '{"text": "t"}',
        ]),
        trace_dir=tmp_path,
    )
    c.store_update({"stable": "", "weight": 1}, "obs", label="period 2")
    c.summarize("hour", [], {}, label="hour:2026-10-28T16:00")
    files = sorted(tmp_path.glob("*.md"))
    assert len(files) == 2
    assert "store-period-2" in files[0].name
    assert "promote-hour-2026-10-28T16-00" in files[1].name
    body = files[0].read_text()
    assert "OBSERVATION: obs" in body  # full input
    assert '"note": "n"' in body  # raw output
    assert "- cached_tokens: 10" in body
    assert "- est_cost_usd: 0.000199" in body  # (90*1.25 + 10*0.15 + 20*4.25)/1e6


def test_object_shape_usage_is_read(tmp_path):
    from types import SimpleNamespace

    from agent.memory.llm_client import _usage_of

    resp = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=50,
            output_tokens=5,
            total_tokens=55,
            input_tokens_details=SimpleNamespace(cached_tokens=7),
        )
    )
    assert _usage_of(resp) == {
        "input_tokens": 50,
        "output_tokens": 5,
        "total_tokens": 55,
        "cached_tokens": 7,
    }
    assert _usage_of(SimpleNamespace()) == {}
    assert _usage_of(SimpleNamespace(usage=None)) == {}


def test_client_without_trace_dir_writes_nothing(tmp_path, monkeypatch):
    import os

    monkeypatch.chdir(tmp_path)
    c = MetaModelClient(client=_UsageClient(['{"contradicts_stable": false, "note": "n"}']))
    assert c.trace_dir is None
    c.store_update({}, "o")
    assert list(tmp_path.iterdir()) == []


def test_trace_write_failure_never_breaks_call(tmp_path, caplog):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir")
    c = MetaModelClient(
        client=_UsageClient(['{"contradicts_stable": false, "note": "n"}']),
        trace_dir=blocker / "sub",  # mkdir fails: parent is a file
    )
    with caplog.at_level(logging.WARNING, logger="agent.memory.llm_client"):
        out = c.store_update({}, "o", label="period 1")
    assert out == {"contradicts_stable": False, "note": "n"}
    assert "trace write failed" in caplog.text


def test_failed_call_still_traces_and_raises(tmp_path):
    from agent.memory import LLMClientError

    c = MetaModelClient(client=_UsageClient(["not json"]), trace_dir=tmp_path)
    try:
        c.store_update({}, "o", label="period 1")
    except LLMClientError:
        pass
    else:
        raise AssertionError("should raise")
    (path,) = tmp_path.glob("*.md")
    assert "not json" in path.read_text()  # raw output preserved
    assert "(unparseable)" in path.read_text()
