"""The agent loop (Meta Model API Responses endpoint + MCP tools).

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
import uuid
from dataclasses import asdict, dataclass, field

from supervisor_agent.config import AgentConfig
from supervisor_agent.invoke_log import get_invoke_log
from supervisor_agent.mcp_hub import MCPHub
from supervisor_agent.prompts import build_instructions
from supervisor_agent.tool_adapter import server_timing, to_function_output, to_function_tool
from supervisor_agent.trace import TurnTrace, render_trace, save_data_url_images

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
    instructions: str | None = None  # replaces the default system instructions (folded into the turn-1 input, never sent as a parameter)
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
    call_id: str = ""
    output_texts: list[str] = field(default_factory=list)  # full text result (for trace.md)


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
            # The Meta Model API is OpenAI-compatible, so the openai SDK is only the HTTP client here.
            from openai import AsyncOpenAI

            api_key = os.getenv(cfg.api_key_env)
            if not api_key:
                raise RuntimeError(f"{cfg.api_key_env} is not set (Meta Model API key)")
            client = AsyncOpenAI(base_url=cfg.base_url, api_key=api_key)
        self.client = client
        self.on_event = on_event or (lambda kind, data: None)
        self.invoke_log = get_invoke_log()  # None unless AGENT_INVOKE_DUMP=true

    def _tools(self, names: list[str] | None = None) -> list[dict]:
        if names is None:
            return [to_function_tool(t) for t in self.hub.tools.values()]
        missing = [n for n in names if n not in self.hub.tools]
        if missing:
            logger.warning(f"tools not available (server down?): {missing}")
        return [to_function_tool(self.hub.tools[n]) for n in names if n in self.hub.tools]

    async def run(self, query: str | list, previous_response_id: str | None = None, options: RunOptions | None = None) -> RunResult:
        opts = options or RunOptions()
        t_start = time.perf_counter()
        invoke_time = datetime.datetime.now()
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

        # Never send the `instructions` parameter: the gateway 400s on chained
        # image tool-outputs when the chain root carried it (see
        # documents/model-api-chaining-probe.md). The text rides in the turn-1
        # input instead and persists via chaining like any other context.
        next_input: list | str = _fold_instructions(query, instructions)
        prev_id = previous_response_id
        response = None
        turns_log = []  # (turn, started_at, calls, outputs) for the invoke dump
        trace_turns: list[TurnTrace] = []
        max_steps = opts.max_steps or self.cfg.max_steps
        for turn in range(1, max_steps + 1):
            result.turns = turn
            request = dict(
                model=self.cfg.model,
                input=next_input,
                tools=tools,
                previous_response_id=prev_id,
                **extra,
            )
            self.on_event("llm_request", (turn, request))
            t0 = time.perf_counter()
            response = await self.client.responses.create(**request)
            turn_llm_ms = (time.perf_counter() - t0) * 1000
            result.llm_ms += turn_llm_ms
            turn_usage = _usage_dict(getattr(response, "usage", None))
            for k, v in turn_usage.items():
                result.usage[k] = result.usage.get(k, 0) + v
            prev_id = response.id

            calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
            output_text = getattr(response, "output_text", "") or ""
            trace_turns.append(TurnTrace(
                turn=turn,
                previous_response_id=request.get("previous_response_id"),
                response_id=getattr(response, "id", None),
                llm_ms=round(turn_llm_ms, 1),
                output_text=output_text,
                output_item_types=[getattr(item, "type", type(item).__name__) for item in response.output],
                usage=turn_usage,
                calls=[{"name": c.name, "arguments": c.arguments, "call_id": c.call_id} for c in calls],
            ))
            if not calls:
                result.answer = output_text
                break

            self.on_event("tool_calls", [(c.name, c.arguments) for c in calls])
            t1 = time.perf_counter()
            started_at = datetime.datetime.now()
            outputs = await asyncio.gather(*[self._execute(c, turn, run_dir, result) for c in calls])
            result.tools_ms += (time.perf_counter() - t1) * 1000
            turns_log.append((turn, started_at, calls, outputs))
            next_input = list(outputs)
        else:
            result.stopped_reason = f"max_steps ({max_steps}) reached"
            result.answer = (response.output_text if response is not None else "") or "(stopped before the model gave a final answer)"

        result.last_response_id = prev_id
        result.total_ms = (time.perf_counter() - t_start) * 1000
        result.llm_ms, result.tools_ms, result.total_ms = (round(v, 1) for v in (result.llm_ms, result.tools_ms, result.total_ms))
        if self.invoke_log:
            # Snapshot only (no serialisation here); the writer thread builds and inserts the rows.
            self.invoke_log.submit(dict(
                invoke_id=str(uuid.uuid4()), invoke_time=invoke_time, model=self.cfg.model, tag=opts.tag, query=query,
                answer=result.answer, usage=dict(result.usage), est_cost_usd=self.cfg.estimate_cost_usd(result.usage),
                pricing=self.cfg.pricing.get(self.cfg.model),
                tool_steps=list(result.tool_steps), turns=turns_log, tools_ms=result.tools_ms, llm_ms=result.llm_ms,
                total_ms=result.total_ms, turns_count=result.turns, stopped_reason=result.stopped_reason, reasoning_effort=effort,
                previous_response_id=previous_response_id, last_response_id=prev_id, run_dir=run_dir,
            ))
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)
            query_images = save_data_url_images(query, run_dir, "input")
            md = render_trace(
                model=self.cfg.model, tag=opts.tag, started_at=invoke_time.strftime("%Y-%m-%d %H:%M:%S"),
                instructions=instructions, query=query, query_images=query_images,
                tools_exposed=[t["name"] for t in tools], text_format=opts.text_format, reasoning_effort=effort,
                turns=trace_turns, tool_steps=result.tool_steps, answer=result.answer, usage=result.usage,
                pricing=self.cfg.pricing.get(self.cfg.model), est_cost_usd=self.cfg.estimate_cost_usd(result.usage),
                llm_ms=result.llm_ms, tools_ms=result.tools_ms, total_ms=result.total_ms,
                stopped_reason=result.stopped_reason, images_used=result.images_used, high_res_used=result.high_res_used,
                previous_response_id=previous_response_id, last_response_id=prev_id,
            )
            with open(os.path.join(run_dir, "trace.md"), "w") as f:
                f.write(md)
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
            step = ToolStep(turn, call.name, args, 0.0, None, None, True, refused, [], "", call.call_id, [items[0]["text"]])
            result.tool_steps.append(step)
            self.on_event("tool_result", step)
            return {"type": "function_call_output", "call_id": call.call_id, "output": items}
        if bad:
            outcome = None
            items = [{"type": "input_text", "text": json.dumps({"error": bad})}]
            step = ToolStep(turn, call.name, {}, 0.0, None, None, True, bad, [], "", call.call_id, [items[0]["text"]])
        else:
            outcome = await self.hub.call(call.name, args)
            items, saved = to_function_output(outcome, self.cfg.image_detail, run_dir, step=len(result.tool_steps) + 1)

            timing = server_timing(outcome)
            texts = [it["text"] for it in items if it["type"] == "input_text"]
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
                preview=(texts[1] if len(texts) > 1 else (texts[0] if texts else ""))[:300],
                call_id=call.call_id,
                output_texts=texts,
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


def _fold_instructions(query: str | list, instructions: str) -> str | list:
    """Prepend instructions to the turn-1 input (never sent as a parameter).

    String queries become one text block; message lists (tasks.py) get the
    text inserted at the front of the first message's content, using only the
    role/content shapes the gateway already accepts. The caller's list is
    never mutated.
    """
    if not instructions:
        return query
    if isinstance(query, str):
        return f"{instructions}\n\n{query}" if query else instructions
    items = list(query)
    first = items[0] if items else None
    if isinstance(first, dict) and isinstance(first.get("content"), list):
        items[0] = {**first, "content": [{"type": "input_text", "text": instructions}, *first["content"]]}
    else:
        items.insert(0, {"role": "user", "content": [{"type": "input_text", "text": instructions}]})
    return items


def _usage_dict(usage) -> dict:
    """This response's token usage as a plain dict ({} when not reported)."""
    out: dict = {}
    if usage is None:
        return out
    for k in ("input_tokens", "output_tokens", "total_tokens"):
        v = getattr(usage, k, None)
        if isinstance(v, int):
            out[k] = v
    cached = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", None)
    if isinstance(cached, int):
        out["cached_tokens"] = cached
    return out
