"""The gateway 400s on chained image tool-outputs when the chain root carried
the `instructions` parameter (see documents/model-api-chaining-probe.md), so
MCPAgent.run folds the instructions into the turn-1 input and never sends the
parameter. These tests pin that request shape with a scripted model.

Run from the agent/ folder:
    python -m pytest supervisor_agent/tests/test_agent_chaining.py -q
"""

from __future__ import annotations

import copy

from supervisor_agent.agent import MCPAgent, RunOptions
from supervisor_agent.config import AgentConfig
from supervisor_agent.mcp_hub import MCPHub
from supervisor_agent.tests.test_tasks import FakeServer, ScriptedModel

MARKER = "SYSTEM-PROMPT-MARKER-7f3a"


def make_agent(model_script):
    hub = MCPHub([])
    hub.add_local(FakeServer("echo", {"ping": lambda **a: {"ok": True}}))
    cfg = AgentConfig(model="fake", save_dir=None)
    return MCPAgent(cfg, hub, client=ScriptedModel(model_script)), hub


def test_instructions_folded_into_string_query():
    agent, _ = make_agent([[("echo__ping", {})], "done"])
    opts = RunOptions(instructions=MARKER, tool_names=["echo__ping"])
    import asyncio
    asyncio.run(agent.run("is any cat visible?", options=opts))
    reqs = agent.client.requests
    assert len(reqs) == 2
    assert all("instructions" not in r for r in reqs)
    first_input = reqs[0]["input"]
    assert isinstance(first_input, str)
    assert MARKER in first_input and "is any cat visible?" in first_input
    # turn 2 still sends the tool output as its own input (chained, unchanged)
    assert reqs[1]["input"][0]["type"] == "function_call_output"
    assert reqs[1]["previous_response_id"] == "r1"


def test_instructions_folded_into_message_list_query():
    agent, _ = make_agent(["done"])
    query = [{"role": "user", "content": [{"type": "input_text", "text": "PAYLOAD"}]}]
    snapshot = copy.deepcopy(query)
    opts = RunOptions(instructions=MARKER, tool_names=["echo__ping"])
    import asyncio
    asyncio.run(agent.run(query, options=opts))
    reqs = agent.client.requests
    assert len(reqs) == 1
    assert "instructions" not in reqs[0]
    content = reqs[0]["input"][0]["content"]
    assert content[0]["type"] == "input_text" and MARKER in content[0]["text"]
    assert content[1:] == snapshot[0]["content"]
    assert query == snapshot  # caller-owned list untouched


def test_empty_instructions_means_none_for_chained_retries():
    agent, _ = make_agent(["done"])
    import asyncio
    asyncio.run(agent.run("retry me", previous_response_id="r0", options=RunOptions(instructions="", tool_names=["echo__ping"])))
    reqs = agent.client.requests
    assert reqs[0]["input"] == "retry me"  # no system prompt re-sent into the chain
    assert reqs[0]["previous_response_id"] == "r0"


def test_errored_image_calls_refund_the_budget():
    import asyncio

    def explode(**a):
        raise RuntimeError("shutter stuck")

    hub = MCPHub([])
    hub.add_local(FakeServer("camera", {"capture_frame": explode}))
    cfg = AgentConfig(model="fake", save_dir=None)
    agent = MCPAgent(cfg, hub, client=ScriptedModel([[("camera__capture_frame", {})], "done"]))
    result = asyncio.run(agent.run("look", options=RunOptions(image_budget=1, tool_names=["camera__capture_frame"])))
    assert result.images_used == 0  # error delivered no pixels: reservation refunded
    assert result.tool_steps[0].is_error


def test_successful_image_calls_consume_the_budget():
    import asyncio

    from supervisor_agent.tests.test_tasks import jpeg

    hub = MCPHub([])
    hub.add_local(FakeServer("camera", {"capture_frame": lambda **a: jpeg(16, 12)}))
    cfg = AgentConfig(model="fake", save_dir=None)
    agent = MCPAgent(cfg, hub, client=ScriptedModel([[("camera__capture_frame", {})], "done"]))
    result = asyncio.run(agent.run("look", options=RunOptions(image_budget=1, tool_names=["camera__capture_frame"])))
    assert result.images_used == 1 and not result.tool_steps[0].is_error
