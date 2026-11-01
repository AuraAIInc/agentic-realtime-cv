"""Offline tests for the markdown run trace (trace.md).

Run from the agent/ folder:
    python -m pytest supervisor_agent/tests/test_trace.py -q
"""

from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace

from supervisor_agent.agent import MCPAgent, RunOptions
from supervisor_agent.config import AgentConfig
from supervisor_agent.mcp_hub import MCPHub
from supervisor_agent.tests.test_tasks import FakeServer, jpeg

MARKER = "TRACE-SYSTEM-MARKER-9c1e"
LONG_TEXT = "full-output-line-" * 60  # >300 chars: proves the trace is not preview-truncated


class UsageModel:
    """Scripted Responses client with per-turn usage. Each script step is
    (payload, usage): payload is a str (final text), a list of (name, args)
    tool calls, or {"calls": [...], "text": "..."} for both."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.responses = SimpleNamespace(create=self._create)

    async def _create(self, **kw):
        self.requests.append(kw)
        payload, usage = self.script.pop(0)
        if isinstance(payload, dict):
            calls, text = payload.get("calls", []), payload.get("text", "")
        elif isinstance(payload, list):
            calls, text = payload, ""
        else:
            calls, text = [], payload
        items = [SimpleNamespace(type="function_call", name=n, arguments=json.dumps(a), call_id=f"c{i}") for i, (n, a) in enumerate(calls)]
        ns = None
        if usage:
            ns = SimpleNamespace(
                input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                input_tokens_details=SimpleNamespace(cached_tokens=usage.get("cached_tokens", 0)),
            )
        return SimpleNamespace(id=f"r{len(self.requests)}", output=items, output_text=text, usage=ns)


def make_agent(tmp_path, script):
    hub = MCPHub([])
    hub.add_local(FakeServer("echo", {"ping": lambda **a: {"ok": True, "echo": LONG_TEXT}}))
    hub.add_local(FakeServer("cam", {"shot": lambda **a: jpeg(64, 48)}))
    cfg = AgentConfig(model="fake", save_dir=str(tmp_path))
    cfg.pricing = {"fake": {"input": 1.0, "cached_input": 0.1, "output": 2.0}}
    return MCPAgent(cfg, hub, client=UsageModel(script))


def test_trace_md_has_full_messages_timing_tokens_and_cost(tmp_path):
    script = [
        ([("echo__ping", {"q": 1}), ("cam__shot", {})], {"input_tokens": 100, "output_tokens": 10, "cached_tokens": 20}),
        ("the answer", {"input_tokens": 200, "output_tokens": 30, "cached_tokens": 40}),
    ]
    agent = make_agent(tmp_path, script)
    result = asyncio.run(agent.run("is any cat visible?", options=RunOptions(instructions=MARKER, tag="tracetest")))
    run_dir = result.run_dir
    assert run_dir is not None
    trace = f"{run_dir}/trace.md"
    with open(trace) as f:
        md = f.read()

    # trace.md replaces run.json
    import os
    assert os.path.isfile(trace) and not os.path.exists(f"{run_dir}/run.json")

    # table of contents over all messages: system, user, 2 assistant turns, 2 tool results
    assert "## Table of Contents" in md
    assert "| [#1](#msg-1) | system |" in md
    assert "| [#2](#msg-2) | user |" in md
    assert "assistant (turn 1)" in md and "assistant (turn 2)" in md
    assert md.count("| tool_result (turn 1) |") == 2  # TOC rows
    assert md.count(". tool_result (turn 1) (") == 2  # section headings

    # full system prompt, query, LLM output and answer
    assert MARKER in md
    assert "is any cat visible?" in md
    assert "the answer" in md and "## Answer" in md

    # full tool input + full (untruncated) tool output, not the 300-char preview
    assert '"q": 1' in md
    assert LONG_TEXT in md
    assert "echo__ping" in md and "cam__shot" in md

    # timing per turn / per tool + totals
    assert "round trip" in md and "llm_ms" in md and "tools_ms" in md and "total_ms" in md

    # tokens per turn + totals, pricing and cost
    assert "- input_tokens: 300" in md
    assert "- output_tokens: 40" in md
    assert "- cached_tokens: 60" in md
    assert "pricing_usd_per_1m: input=1.0 cached_input=0.1 output=2.0" in md
    assert "est_cost_usd: 0.000326" in md  # (240*1.0 + 60*0.1 + 40*2.0)/1e6

    # tool image saved next to the trace and embedded in it
    imgs = [f for f in os.listdir(run_dir) if f.endswith(".jpg")]
    assert len(imgs) == 1 and imgs[0].startswith("step")
    assert f"]({imgs[0]})" in md
    assert "data:image" not in md  # no base64 blobs inlined


def test_trace_embeds_query_input_images(tmp_path):
    data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg(32, 24)).decode()
    query = [{"role": "user", "content": [
        {"type": "input_text", "text": "what is in this crop?"},
        {"type": "input_image", "image_url": data_url, "detail": "auto"},
    ]}]
    agent = make_agent(tmp_path, [("a cat", {"input_tokens": 50, "output_tokens": 5})])
    result = asyncio.run(agent.run(query, options=RunOptions(instructions=MARKER)))
    with open(f"{result.run_dir}/trace.md") as f:
        md = f.read()
    import os
    imgs = [f for f in os.listdir(result.run_dir) if f.startswith("input_img")]
    assert len(imgs) == 1
    assert f"]({imgs[0]})" in md
    assert "what is in this crop?" in md
    assert "data:image" not in md
