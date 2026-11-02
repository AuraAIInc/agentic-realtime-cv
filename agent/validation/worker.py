"""Live escalation validation loop: queue -> supervisor agent -> verdict ops.

The tracking run enqueues every escalation as it is emitted; this worker
consumes them one after another in a background thread. Per escalation it
builds the reviewer prompt (``prompting``) plus the expected JSON output
(``review_schema``), optionally prepends spatial-temporal memory context,
sends it all to the supervisor agent, parses the answer back through the
pydantic model, and journals the resulting op for ``apply``. Memory also
receives every verdict as a ``label`` event. Failures never stop the loop:
a failed validation journals an error op (rows pass through unchanged).
"""

from __future__ import annotations

import asyncio
import copy
import datetime
import json
import logging
import queue
import threading

from agent.escalation import build_prompt
from agent.escalation.review_schema import review_model_for, review_schema_json

logger = logging.getLogger(__name__)

#: Tools the validator may use: look at frames, read visit verdicts. Nothing else.
VALIDATION_TOOLS = ["images__get_image", "images__list_images", "tracking__get_period_detail"]

_SENTINEL = object()


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[: -len("```")]
    return text.strip()


def parse_review(code: str, answer: str) -> dict:
    """Parse an LLM answer into the code's review dict; raises ValueError when unusable."""
    try:
        payload = json.loads(_strip_fences(answer))
    except (json.JSONDecodeError, IndexError) as exc:
        raise ValueError(f"answer is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"answer JSON is {type(payload).__name__}, want an object")
    try:
        return review_model_for(code).model_validate(payload).model_dump(mode="json")
    except Exception as exc:
        raise ValueError(f"answer fails {code} review schema: {exc}") from exc


def build_validation_query(esc, memory_brief: str | None = None) -> tuple[str, dict]:
    """Return (query, json_schema) to send the supervisor agent for one escalation."""
    prompt = build_prompt(esc)
    if memory_brief:
        prompt += f"\nPast context from memory (may help, may be stale): {memory_brief}"
    prompt += "\nReturn ONLY a JSON object following the output schema — no prose, no fences."
    return prompt, json.loads(review_schema_json(esc.code))


class ValidationWorker:
    """Background thread consuming escalations sequentially.

    ``backend`` is ``"supervisor"`` (default: drive the real supervisor agent
    over one persistent MCP connection) or an async callable
    ``(query, schema, code) -> answer str`` (tests, dry runs). ``journal`` is
    called with every op dict; ``memory`` enables context + verdict events.
    """

    def __init__(self, backend="supervisor", *, journal=None, memory=None, box_lookup=None):
        self._backend = backend
        self._journal = journal
        self._memory = memory
        self._box_lookup = box_lookup
        self._inbox: queue.Queue = queue.Queue()
        self._pending: list = []
        self._started = threading.Event()
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None
        self.ops: list = []

    # -- control (called from the tracking thread) --

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="validation-worker", daemon=True)
        self._thread.start()

    def enqueue(self, esc) -> None:
        """Queue one escalation for validation (thread-safe, non-blocking)."""
        self._inbox.put(esc)

    def stop(self, timeout: float | None = None) -> list:
        """Finish the queued items, stop the thread, return all ops."""
        self._inbox.put(_SENTINEL)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._stopped.set()
        return self.ops

    # -- loop (runs in the worker thread) --

    def _run(self) -> None:
        try:
            asyncio.run(self._serve())
        except Exception:
            logger.exception("validation worker crashed; queued escalations are dropped")
        finally:
            self._started.set()

    async def _serve(self) -> None:
        if callable(self._backend):
            await self._consume(self._backend)
            return
        from supervisor_agent.agent import MCPAgent, RunOptions
        from supervisor_agent.config import AgentConfig
        from supervisor_agent.mcp_hub import MCPHub

        cfg = AgentConfig.from_env()
        try:
            async with MCPHub(cfg.servers) as hub:
                agent = MCPAgent(cfg, hub)
                logger.info("validation worker connected to supervisor backends")

                async def run_agent(query: str, schema: dict, code: str) -> str:
                    fmt = {"type": "json_schema", "name": f"{code}_review", "schema": _strict(schema)}
                    result = await agent.run(
                        query,
                        options=RunOptions(text_format=fmt, tool_names=list(VALIDATION_TOOLS), tag="validation"),
                    )
                    return result.answer

                await self._consume(run_agent)
        except Exception:
            logger.exception("supervisor backend unavailable; validations will error out")
            await self._consume(_failing_backend)

    async def _consume(self, run_agent) -> None:
        loop = asyncio.get_running_loop()
        while True:
            esc = await loop.run_in_executor(None, self._inbox.get)
            try:
                if esc is _SENTINEL:
                    return
                op = await self._validate_one(esc, run_agent)
                self.ops.append(op)
                if self._journal is not None:
                    try:
                        self._journal(op)
                    except Exception:
                        logger.exception("validation journal failed; op kept in memory")
            finally:
                self._inbox.task_done()

    async def _validate_one(self, esc, run_agent) -> dict:
        name = getattr(esc, "name", None) or getattr(esc, "code", "?")
        logger.info("validating escalation %s", name)
        brief = self._memory_brief(esc)
        query, schema = build_validation_query(esc, memory_brief=brief)
        try:
            answer = await run_agent(query, schema, esc.code)
            review = parse_review(esc.code, answer)
            logger.info("escalation %s verdict: %s", name, review.get("verdict"))
        except Exception as exc:
            logger.warning("escalation %s validation failed: %s", name, exc)
            review = {"verdict": "error", "error": str(exc)}
        self._memory_verdict(esc, review)
        return {"escalation": json.loads(esc.to_json()), "review": review}

    # -- memory hooks (best-effort; never break validation) --

    def _memory_brief(self, esc) -> str | None:
        if self._memory is None:
            return None
        try:
            box = self._box_lookup(esc) if self._box_lookup is not None else None
            ctx = self._memory.invoke_for_escalation(esc, box)
            brief = (getattr(ctx, "brief", "") or "").strip()
            return brief[:600] if brief else None
        except Exception:
            logger.exception("memory context failed; validating without it")
            return None

    def _memory_verdict(self, esc, review: dict) -> None:
        if self._memory is None:
            return
        try:
            from agent.memory import MemoryEvent

            self._memory.store(
                MemoryEvent(
                    time=getattr(esc, "time", None) or datetime.datetime.now(),
                    end_time=getattr(esc, "end_time", None),
                    view_id=int(getattr(self._memory.config, "view_id", 0) or 0),
                    cat_id=getattr(esc, "cat_id", None),
                    period_id=getattr(esc, "period_id", None),
                    kind="label",
                    text=f"[{getattr(esc, 'name', None) or esc.code}] {review.get('verdict')}"
                    + (f": {review.get('error')}" if review.get("verdict") == "error" else ""),
                    code=esc.code,
                )
            )
        except Exception:
            logger.exception("memory verdict store failed; dropping event")


async def _failing_backend(query: str, schema: dict, code: str) -> str:
    raise RuntimeError("supervisor backend unavailable")


def _strict(obj: dict) -> dict:
    """Mark every property required and forbid extras (Responses API strict json_schema)."""
    obj = copy.deepcopy(obj)

    def walk(s):
        if isinstance(s, dict):
            if s.get("type") == "object" and "properties" in s:
                s["required"] = list(s["properties"])
                s["additionalProperties"] = False
            for v in s.values():
                walk(v)
        elif isinstance(s, list):
            for v in s:
                walk(v)

    walk(obj)
    return obj
