"""The agent loop (OpenAI Responses API + MCP tools).

    user query
      -> model decides which tools to call (0..n per turn, run concurrently)
      -> MCP hub executes them on the camera / image servers
      -> results (text + images + timing) go back to the model
      -> repeat until the model answers without calling a tool, or max_steps
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field

from supervisor_agent.config import AgentConfig
from supervisor_agent.mcp_hub import MCPHub
from supervisor_agent.prompts import build_instructions
from supervisor_agent.tool_adapter import server_timing, to_function_output, to_openai_tool

logger = logging.getLogger(__name__)

# Tools that return pixels. Each call that returns an image counts against a run's image
# budget; when the budget is spent, further calls are refused so the model must decide or
# escalate to a human instead of looking again.
IMAGE_TOOLS = {
    "camera__capture_frame", "camera__capture_roi", "camera__capture_high_res", "camera__get_track_frame",
    "images__get_image", "tracking__get_arrival_path",
}
HIGH_RES_TOOLS = {"camera__capture_high_res"}


@dataclass
class RunOptions:
    """Per-run overrides used by the task runner (tasks.py). Defaults = plain chat behaviour."""
    instructions: str | None = None  # replaces the default system instructions
    tool_names: list[str] | None = None  # only expose these tools (None = all)
    text_format: dict | None = None  # Responses API text.format, e.g. a strict json_schema
    image_budget: int | None = None  # image-returning calls allowed (None = unlimited)
    high_res_budget: int | None = None
    max_steps: int | None = None
    reasoning_effort: str | None = None
    tag: str | None = None  # appended to the run directory name


@dataclass
class ToolStep:
    turn: int
    tool: str
    arguments: dict
    round_trip_ms: float
    server_elapsed_ms: float | None
    server_steps_ms: dict | None
    is_error: bool
    error: str | None
    images_saved: list[str]
    preview: str  # first part of the text result


@dataclass
class RunResult:
    query: str
    answer: str
    model: str
    turns: int
    tool_steps: list[ToolStep] = field(default_factory=list)
    llm_ms: float = 0.0
    tools_ms: float = 0.0
    total_ms: float = 0.0
    usage: dict = field(default_factory=dict)
    images_used: int = 0
    high_res_used: int = 0
    stopped_reason: str = "answered"
    run_dir: str | None = None
    last_response_id: str | None = None  # pass back as previous_response_id to continue the conversation

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


class MCPAgent:
    def __init__(self, cfg: AgentConfig, hub: MCPHub, client=None, on_event=None):
        self.cfg = cfg
        self.hub = hub
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()
        self.client = client
        self.on_event = on_event or (lambda kind, data: None)

    def _tools(self, names: list[str] | None = None) -> list[dict]:
        if names is None:
            return [to_openai_tool(t) for t in self.hub.tools.values()]
        missing = [n for n in names if n not in self.hub.tools]
        if missing:
            logger.warning(f"tools not available (server down?): {missing}")
        return [to_openai_tool(self.hub.tools[n]) for n in names if n in self.hub.tools]

    async def run(self, query: str | list, previous_response_id: str | None = None, options: RunOptions | None = None) -> RunResult:
        opts = options or RunOptions()
        t_start = time.perf_counter()
        run_dir = None
        if self.cfg.save_dir:
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            run_dir = os.path.join(self.cfg.save_dir, f"{stamp}-{opts.tag}" if opts.tag else stamp)
        result = RunResult(query=query if isinstance(query, str) else json.dumps(query, default=str)[:2000], answer="", model=self.cfg.model, turns=0, run_dir=run_dir)
        tools = self._tools(opts.tool_names)
        instructions = opts.instructions or build_instructions(self.hub.server_info, self.hub.failed)
        self._opts = opts

        extra = {}
        effort = opts.reasoning_effort or self.cfg.reasoning_effort
        if effort:
            extra["reasoning"] = {"effort": effort}
        if opts.text_format:
            extra["text"] = {"format": opts.text_format}

        next_input: list | str = query
        prev_id = previous_response_id
        response = None
        max_steps = opts.max_steps or self.cfg.max_steps
        for turn in range(1, max_steps + 1):
            result.turns = turn
            request = dict(
                model=self.cfg.model,
                instructions=instructions,
                input=next_input,
                tools=tools,
                previous_response_id=prev_id,
                **extra,
            )
            self.on_event("llm_request", (turn, request))
            t0 = time.perf_counter()
            response = await self.client.responses.create(**request)
            result.llm_ms += (time.perf_counter() - t0) * 1000
            _add_usage(result.usage, getattr(response, "usage", None))
            prev_id = response.id

            calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
            if not calls:
                result.answer = response.output_text or ""
                break

            self.on_event("tool_calls", [(c.name, c.arguments) for c in calls])
            t1 = time.perf_counter()
            outputs = await asyncio.gather(*[self._execute(c, turn, run_dir, result) for c in calls])
            result.tools_ms += (time.perf_counter() - t1) * 1000
            next_input = list(outputs)
        else:
            result.stopped_reason = f"max_steps ({max_steps}) reached"
            result.answer = (response.output_text if response is not None else "") or "(stopped before the model gave a final answer)"

        result.last_response_id = prev_id
        result.total_ms = (time.perf_counter() - t_start) * 1000
        result.llm_ms, result.tools_ms, result.total_ms = (round(v, 1) for v in (result.llm_ms, result.tools_ms, result.total_ms))
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "run.json"), "w") as f:
                f.write(result.to_json())
        return result

    async def _execute(self, call, turn: int, run_dir: str | None, result: RunResult) -> dict:
        try:
            args = json.loads(call.arguments or "{}")
        except json.JSONDecodeError as e:
            args, bad = {}, f"invalid JSON arguments: {e}"
        else:
            bad = None

        refused = None if bad else self._budget_refusal(call.name, args, result)
        if refused:
            items = [{"type": "input_text", "text": json.dumps({"tool": call.name, "error": refused})}]
            step = ToolStep(turn, call.name, args, 0.0, None, None, True, refused, [], "")
            result.tool_steps.append(step)
            self.on_event("tool_result", step)
            return {"type": "function_call_output", "call_id": call.call_id, "output": items}
        if bad:
            outcome = None
            items = [{"type": "input_text", "text": json.dumps({"error": bad})}]
            step = ToolStep(turn, call.name, {}, 0.0, None, None, True, bad, [], "")
        else:
            outcome = await self.hub.call(call.name, args)
            items, saved = to_function_output(outcome, self.cfg.image_detail, run_dir, step=len(result.tool_steps) + 1)

            timing = server_timing(outcome)
            texts = [it["text"] for it in items[1:] if it["type"] == "input_text"]
            step = ToolStep(
                turn=turn,
                tool=call.name,
                arguments=args,
                round_trip_ms=outcome.round_trip_ms,
                server_elapsed_ms=(timing or {}).get("elapsed_ms"),
                server_steps_ms=(timing or {}).get("steps_ms"),
                is_error=bool(outcome.error or getattr(outcome.result, "is_error", False)),
                error=outcome.error,
                images_saved=saved,
                preview=(texts[0] if texts else "")[:300],
            )
        result.tool_steps.append(step)
        self.on_event("tool_result", step)
        return {"type": "function_call_output", "call_id": call.call_id, "output": items}


    def _budget_refusal(self, name: str, args: dict, result: RunResult) -> str | None:
        opts = getattr(self, "_opts", None) or RunOptions()
        tool = self.hub.tools.get(name)
        returns_image = name in IMAGE_TOOLS or bool(tool and (tool.meta or {}).get("returns_image"))
        if name == "tracking__get_arrival_path" and not args.get("include_images"):
            returns_image = False
        if returns_image and opts.image_budget is not None and result.images_used >= opts.image_budget:
            return (f"image budget reached ({opts.image_budget} image calls for this task). Decide from what you have, "
                    "or answer with the human-review action.")
        if name in HIGH_RES_TOOLS and opts.high_res_budget is not None and result.high_res_used >= opts.high_res_budget:
            return f"capture_high_res budget reached ({opts.high_res_budget}); use the stored frame or best_frame instead."
        # Reserve before running: parallel calls in one turn must not overshoot the budget.
        result.images_used += returns_image
        result.high_res_used += name in HIGH_RES_TOOLS
        return None


def _add_usage(acc: dict, usage) -> None:
    if usage is None:
        return
    for k in ("input_tokens", "output_tokens", "total_tokens"):
        v = getattr(usage, k, None)
        if isinstance(v, int):
            acc[k] = acc.get(k, 0) + v
    cached = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", None)
    if isinstance(cached, int):
        acc["cached_tokens"] = acc.get("cached_tokens", 0) + cached
