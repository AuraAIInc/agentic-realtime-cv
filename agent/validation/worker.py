"""Live escalation validation loop: queue -> supervisor agent -> verdict ops.

The tracking run enqueues every escalation as it is emitted; this worker
consumes them batch after batch in a background thread (temporally close
escalations merge into one agent call; see ``batcher``). Per batch it builds
the reviewer prompt (``prompting``) plus the expected JSON output
(``review_schema``), optionally prepends spatial-temporal memory context,
sends it all to the supervisor agent, parses the answer back through the
pydantic models, and journals one op per escalation for ``apply``. Memory
also receives every verdict as a ``label`` event. Failures never stop the
loop: a failed validation journals an error op (rows pass through unchanged).
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import sys
import threading
from pathlib import Path
from typing import get_args

from agent.escalation import build_prompt
from agent.escalation.review_schema import review_model_for, review_schema_json
from agent.validation.batcher import EscalationBatcher

logger = logging.getLogger(__name__)


def ensure_supervisor_importable() -> None:
    """Put ``agent/`` on ``sys.path`` so top-level ``supervisor_agent`` imports resolve.

    ``python -m agent.main`` from the repo root leaves ``agent/`` off the path
    (unlike ``cd agent`` runs); without this the real backend crashes on import.
    Same bootstrap pattern as ``db_stream.connect()``.
    """
    agent_dir = str(Path(__file__).resolve().parent.parent)
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)

#: Tools the validator may use: look at frames, read visit verdicts. Nothing else.
VALIDATION_TOOLS = [
    "images__get_image",
    "images__list_images",
    "tracking__get_period_detail",
    "local__find_references",
    "local__get_reference_image",
    "local__get_frame",
    "local__get_track_box",
]

#: Preamble for a joint multi-case review: share evidence, minimize total work.
MERGED_PREAMBLE = (
    "You are reviewing {n} related escalation cases from the same camera window.\n"
    "They may share frames and context — fetch each image once and use it for\n"
    "every case it belongs to. Leverage evidence across the whole time window\n"
    "for all of the cases, and resolve every case with the minimum total tool\n"
    "calls and reasoning steps. Judge each case on its own evidence. When two\n"
    "cases in this batch cover the same cat or adjacent frames, keep their\n"
    "identities consistent — one continuous track cannot be two cats."
)

#: Tool-failure rules appended to every validation query (single and merged).
#: The image server fails fast when it cannot crop, and the period-detail tool
#: can serve a stale window in replays — both have burned whole batches before.
TOOL_CAVEATS = (
    "Tool rules: if an images__get_image call with box/max_side errors, that server cannot crop — "
    "do NOT retry crops on that server; fetch the frame once without box/max_side and judge from the full image. "
    "(local__get_frame crops locally, so box=... is always safe there — use crops when judging which cat is inside "
    "a box.) "
    "If an images__* call errors (the images server is down), fetch the same frame from local disk with "
    "local__get_frame using the same timestamp — same pixels, no server needed. "
    "If tracking__get_period_detail returns timestamps outside this review window, it is stale — "
    "ignore it and judge from the key frames plus images."
)


def _agent_event_log(kind: str, data) -> None:
    """Terminal log for one supervisor-agent event during validation."""
    if kind == "tool_calls":
        for name, args in data:
            logger.info("validation agent tool call: %s(%s)", name, args)
    elif kind == "tool_result":
        status = "ERROR" if data.is_error else "ok"
        logger.info(
            "validation agent tool result: %s: %s (round trip %.1f ms)",
            data.tool,
            status,
            data.round_trip_ms,
        )
        if data.error:
            logger.info("validation agent tool error: %s", data.error)
    elif kind == "llm_text":
        turn, text = data
        logger.info("validation agent turn %d text:\n%s", turn, text)


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


def build_validation_query(esc, memory_brief: str | None = None, prior_block: str | None = None) -> tuple[str, dict]:
    """Return (query, json_schema) to send the supervisor agent for one escalation."""
    prompt = build_prompt(esc)
    if memory_brief:
        prompt += f"\nPast context from memory (may help, may be stale): {memory_brief}"
    prompt += f"\n{TOOL_CAVEATS}"
    if prior_block:
        prompt += f"\n{prior_block}"
    prompt += "\nReturn ONLY a JSON object following the output schema — no prose, no fences."
    return prompt, json.loads(review_schema_json(esc.code))


#: Verdicts that settle nothing: never cited as prior rulings.
_UNDECIDED = frozenset({"error", "flag_issue"})

#: Nearby-frame radius (s) for prior rulings: spans overlapping or this close count.
PRIOR_NEAR_S = 60.0

#: Most prior rulings cited in one query (each ~1 line).
PRIOR_MAX_LINES = 8


def _span_of(esc) -> tuple:
    """(start, end) datetimes for an Escalation or a journaled escalation dict."""
    import datetime as _dt

    def _one(value):
        if value is None:
            return None
        if isinstance(value, _dt.datetime):
            return value
        try:
            return _dt.datetime.fromisoformat(value)
        except (TypeError, ValueError):
            return None

    if isinstance(esc, dict):
        start, end = _one(esc.get("time")), _one(esc.get("end_time"))
    else:
        start, end = _one(getattr(esc, "time", None)), _one(getattr(esc, "end_time", None))
    if start is None:
        return None, None
    return start, end or start


def _box4(value) -> list | None:
    """[x1, y1, x2, y2] as ints when value is a 4-number box, else None."""
    try:
        nums = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if len(nums) != 4 or not all(v == v and v not in (float("inf"), float("-inf")) for v in nums):
        return None
    return [int(round(v)) for v in nums]


def _prior_box_txt(code: str | None, metrics: dict, review: dict) -> str:
    """Box/cut evidence for a prior line, so the reviewer can match objects.

    A timestamp alone lets a later case discount a prior ruling as "another
    object"; with the ruled box it can verify same-track overlap instead.
    "" when the escalation carries no box.
    """
    pairs: list = []
    if code == "T4":
        pairs = [("base", metrics.get("base_box")), ("new", metrics.get("box"))]
    elif code == "T6":
        pairs = [("pre", metrics.get("pre_box")), ("post", metrics.get("post_box"))]
    elif code == "P6":
        pairs = [("prev_end", metrics.get("prev_end_box")), ("new_start", metrics.get("new_start_box"))]
    else:
        for key in ("box", "winner_box", "new_box", "post_box", "new_start_box"):
            if metrics.get(key) is not None:
                pairs = [("", metrics.get(key))]
                break
        if not pairs:
            for kf in metrics.get("key_frames") or []:
                if isinstance(kf, dict) and kf.get("real") and kf.get("box") is not None:
                    pairs = [("", kf["box"])]
                    break
    shown = []
    for label, raw in pairs:
        box = _box4(raw)
        if box is not None:
            shown.append(f"{label + '=' if label else 'box='}{box}")
    txt = " ".join(shown)
    if code == "P8" and (review or {}).get("verdict") == "split_mixed" and (review or {}).get("cut_time"):
        try:
            import datetime as _dt
            cut = _dt.datetime.fromisoformat(review.get("cut_time"))
            txt += f" split@{cut:%H:%M:%S}->{review.get('chosen_cat_id')}"
        except (TypeError, ValueError):
            pass
    return (", " + txt.strip()) if txt.strip() else ""


def prior_verdict_block(ops: list, targets) -> str:
    """Already-decided rulings overlapping the target span(s), as a query block.

    Later cases see earlier rulings on the same scene, so a reviewer judging
    one ambiguous frame weighs the track identity siblings already settled
    (appearance flips frame to frame under mixed lighting; the track does
    not). Undecided verdicts (error/flag_issue) are never cited. "" when
    nothing decided nearby yet.
    """
    if isinstance(targets, dict) or not isinstance(targets, (list, tuple)):
        targets = [targets]
    spans = [_span_of(t) for t in targets]
    spans = [(s, e) for s, e in spans if s is not None]
    if not spans:
        return ""
    lines = []
    for op in ops or []:
        review = (op or {}).get("review") or {}
        if review.get("verdict") in _UNDECIDED or not review.get("verdict"):
            continue
        o_esc = (op or {}).get("escalation") or {}
        start, end = _span_of(o_esc)
        if start is None:
            continue
        near = False
        for s, e in spans:
            if start <= e and s <= end:
                near = True
                break
            gap = min(abs((start - e).total_seconds()), abs((s - end).total_seconds()))
            if gap <= PRIOR_NEAR_S:
                near = True
                break
        if not near:
            continue
        name = o_esc.get("name") or o_esc.get("code") or "?"
        when = f"{start:%H:%M:%S}" + (f"–{end:%H:%M:%S}" if end != start else "")
        imgs = ((op.get("agent_run") or {}).get("images_used") or 0)
        why = (review.get("rationale") or o_esc.get("reason") or "").replace("\n", " ").strip()
        if len(why) > 160:
            why = why[:157] + "..."
        box_txt = _prior_box_txt(o_esc.get("code"), o_esc.get("metrics") or {}, review)
        lines.append((start, f"- {name} -> {review.get('verdict')} @{when} "
                             f"(cat {o_esc.get('cat_id')}, visit {o_esc.get('period_id')}{box_txt}, "
                             f"{imgs} images): {why}"))
    if not lines:
        return ""
    lines.sort(key=lambda item: item[0])
    body = "\n".join(text for _, text in lines[:PRIOR_MAX_LINES])
    return (
        "Already decided in this run on overlapping/nearby frames (weigh heavily):\n"
        f"{body}\n"
        "A continuous track does not change cats without a visible handover. When a prior ruling covers nearby "
        "frames of the same scene, reconcile explicitly: same track (stay consistent) or a different object (say "
        "why). Contradict a prior ruling only with unanimous box-cropped evidence."
    )


def _verdict_literals(code: str) -> tuple | None:
    """Allowed verdict values for a code, or None when unconstrained (base model)."""
    try:
        return get_args(review_model_for(code).model_fields["verdict"].annotation) or None
    except (KeyError, AttributeError):
        return None


def merged_review_schema(escalations: list) -> dict:
    """Fixed-shape envelope schema for a joint review: one item per escalation.

    Strict ``json_schema`` needs fixed properties, but member codes have
    different verdict enums — so the envelope carries the union of member
    literals and each item is re-validated against its own code's model in
    :func:`parse_merged_review`.
    """
    literals: list[str] = []
    for esc in escalations:
        for lit in _verdict_literals(getattr(esc, "code", "")) or ():
            if lit not in literals:
                literals.append(lit)
    verdict_schema: dict = {"type": "string"}
    if literals:
        verdict_schema["enum"] = literals
    return {
        "type": "object",
        "properties": {
            "reviews": {
                "type": "array",
                "minItems": len(escalations),
                "maxItems": len(escalations),
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "verdict": verdict_schema,
                        "rationale": {"type": "string"},
                        "chosen_cat_id": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                        "cut_time": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    },
                    "required": ["name", "verdict", "rationale", "chosen_cat_id", "cut_time"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["reviews"],
        "additionalProperties": False,
    }


def build_merged_query(escalations: list, memory_briefs: list | None = None, prior_block: str | None = None) -> tuple[str, dict]:
    """Return (query, json_schema) for one joint review of several escalations.

    Each member keeps its own full prompt section plus its own memory brief;
    the answer is one review per case keyed by escalation name.
    """
    briefs = list(memory_briefs or []) + [None] * len(escalations)
    parts = [MERGED_PREAMBLE.format(n=len(escalations)), TOOL_CAVEATS]
    if prior_block:
        parts.append(prior_block)
    for i, esc in enumerate(escalations):
        name = getattr(esc, "name", None) or getattr(esc, "code", "?")
        section = f"\n=== Case {i + 1} of {len(escalations)}: {name} ===\n{build_prompt(esc)}"
        allowed = _verdict_literals(getattr(esc, "code", "")) or ()
        if allowed:
            # The envelope enum is the union over members; pin each case to
            # its own verdicts so a sibling's option can't leak in.
            section += f"\nAllowed verdicts for {name}: {' | '.join(allowed)}."
        if briefs[i]:
            section += f"\nPast context from memory (may help, may be stale): {briefs[i]}"
        parts.append(section)
    parts.append("Return ONLY a JSON object following the output schema — one review per case, keyed by case name. No prose, no fences.")
    return "\n".join(parts), merged_review_schema(escalations)


def parse_merged_review(escalations: list, answer: str) -> dict[str, dict]:
    """Parse a joint answer into {escalation name: review}; per-item failures isolated.

    Each item is matched by ``name`` and validated against its own code's
    model (filtered to that model's fields first — the envelope carries
    ``name``/``chosen_cat_id``/``cut_time`` for every item, but each code
    model accepts only its own subset and forbids extras). A missing item
    or a per-item schema failure becomes an ``error`` review for that
    escalation only; siblings still journal. Raises ValueError only when
    the whole answer is unusable.
    """
    try:
        payload = json.loads(_strip_fences(answer))
    except (json.JSONDecodeError, IndexError) as exc:
        raise ValueError(f"answer is not JSON: {exc}") from exc
    reviews = payload.get("reviews") if isinstance(payload, dict) else None
    if not isinstance(reviews, list):
        raise ValueError(f"answer JSON has no reviews list, got: {str(payload)[:80]}")
    by_name: dict[str, dict] = {}
    for item in reviews:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            by_name.setdefault(item["name"], item)
    result: dict[str, dict] = {}
    for esc in escalations:
        name = getattr(esc, "name", None) or getattr(esc, "code", "?")
        item = by_name.get(name)
        if item is None:
            result[name] = {"verdict": "error", "error": f"no review for case {name}"}
            continue
        try:
            model = review_model_for(esc.code)
            fields = {k: v for k, v in item.items() if k in model.model_fields}
            result[name] = model.model_validate(fields).model_dump(mode="json")
        except Exception as exc:
            result[name] = {"verdict": "error", "error": f"case {name} fails {esc.code} review schema: {exc}"}
    return result


class ValidationWorker:
    """Background thread consuming escalations batch after batch.

    ``backend`` is ``"supervisor"`` (default: drive the real supervisor agent
    over one persistent MCP connection) or an async callable
    ``(query, schema, code) -> answer str`` (tests, dry runs); ``code`` is
    the escalation code, or ``+``-joined member names for a merged batch.
    ``journal`` is called with every op dict; ``memory`` enables context +
    verdict events. ``merge_window_s`` / ``merge_max_batch`` /
    ``merge_max_chars`` control queue merging (see ``batcher``);
    ``merge_max_batch=1`` disables merging (every escalation validates solo).

    Every op also carries ``agent_run``: per-call stats (run dir, trace.md
    path, turns, tool calls, images used, timing, tokens, cost) when the
    backend reports them (the supervisor backend always does), else None.
    The session report uses these to link each escalation to its LLM trace.
    """

    def __init__(
        self,
        backend="supervisor",
        *,
        journal=None,
        memory=None,
        box_lookup=None,
        short_term=None,
        merge_window_s: float = 60.0,
        merge_max_batch: int = 4,
        merge_max_chars: int = 12_000,
    ):
        self._backend = backend
        self._journal = journal
        self._memory = memory
        self._box_lookup = box_lookup
        self._short_term = short_term
        self._batcher = EscalationBatcher(
            window_s=merge_window_s,
            max_batch=merge_max_batch,
            max_chars=merge_max_chars,
            size_of=lambda esc: len(build_prompt(esc)),
        )
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
        """Queue one escalation for validation (thread-safe, non-blocking).

        May merge into the tail waiting batch when temporally close to it.
        """
        self._batcher.put(esc)

    def stop(self, timeout: float | None = None) -> list:
        """Finish the queued items, stop the thread, return all ops."""
        self._batcher.close()
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
        ensure_supervisor_importable()
        from supervisor_agent.agent import MCPAgent, RunOptions
        from supervisor_agent.config import AgentConfig
        from supervisor_agent.local_tools import LocalTools
        from supervisor_agent.mcp_hub import MCPHub

        cfg = AgentConfig.from_env()
        try:
            async with MCPHub(cfg.servers) as hub:
                hub.add_local(LocalTools(), "Cat profiles, reference images (agent_data/) and local frames.")
                agent = MCPAgent(cfg, hub, on_event=_agent_event_log)
                logger.info("validation worker connected to supervisor backends")

                async def run_agent(query: str, schema: dict, code: str) -> str:
                    logger.info("validation agent full prompt for %s:\n%s", code, query)
                    # Merged batches arrive with "+"-joined member names (solo codes never contain "+").
                    fmt_name = "merged_review" if "+" in code else f"{code}_review"
                    fmt = {"type": "json_schema", "name": fmt_name, "schema": _strict(schema)}
                    result = await agent.run(
                        query,
                        options=RunOptions(text_format=fmt, tool_names=list(VALIDATION_TOOLS), tag="validation"),
                    )
                    logger.info("validation agent final answer:\n%s", result.answer)
                    run_agent.last_run_stats = {
                        "run_dir": result.run_dir,
                        "trace": str(Path(result.run_dir) / "trace.md") if result.run_dir else None,
                        "model": result.model,
                        "turns": result.turns,
                        "tool_calls": len(result.tool_steps),
                        "images_used": result.images_used,
                        "high_res_used": result.high_res_used,
                        "llm_ms": result.llm_ms,
                        "tools_ms": result.tools_ms,
                        "total_ms": result.total_ms,
                        "usage": dict(result.usage),
                        "est_cost_usd": cfg.estimate_cost_usd(result.usage),
                        "stopped_reason": result.stopped_reason,
                    }
                    return result.answer

                run_agent.last_run_stats = None

                await self._consume(run_agent)
        except Exception:
            logger.exception("supervisor backend unavailable; validations will error out")
            await self._consume(_failing_backend)

    async def _consume(self, run_agent) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = await loop.run_in_executor(None, self._batcher.take)
            if batch is None:
                return
            for op in await self._validate_batch(batch, run_agent):
                self.ops.append(op)
                if self._journal is not None:
                    try:
                        self._journal(op)
                    except Exception:
                        logger.exception("validation journal failed; op kept in memory")

    async def _validate_batch(self, batch, run_agent) -> list:
        """Validate one batch: one agent call, one op per member in arrival order."""
        members = list(batch.members)
        if len(members) == 1:
            return [await self._validate_one(members[0], run_agent)]
        names = [getattr(esc, "name", None) or getattr(esc, "code", "?") for esc in members]
        label = "+".join(names)
        span = ""
        if batch.start is not None and batch.end is not None:
            span = f", span {batch.start:%H:%M:%S}–{batch.end:%H:%M:%S}"
        logger.info("validating batch [%s] (%d cases%s)", ", ".join(names), len(members), span)
        briefs = [self._memory_brief(esc) for esc in members]
        query, schema = build_merged_query(members, briefs, prior_verdict_block(self.ops, members))
        agent_run = None
        retry_ops: dict[str, dict] = {}
        try:
            answer = await run_agent(query, schema, label)
            agent_run = getattr(run_agent, "last_run_stats", None)
            reviews = parse_merged_review(members, answer)
            for name in names:
                logger.info("escalation %s verdict: %s", name, reviews[name].get("verdict"))
                if reviews[name].get("rationale"):
                    logger.info("escalation %s rationale: %s", name, reviews[name]["rationale"])
            for esc, name in zip(members, names):
                if reviews[name].get("verdict") == "error":
                    # Salvage with one single-case retry under the exact enum
                    # (a verdict can pass the merged union yet fail its own
                    # code's schema); _validate_one journals memory itself.
                    logger.info("escalation %s failed in batch, retrying solo: %s", name, reviews[name].get("error"))
                    retry_ops[name] = await self._validate_one(esc, run_agent)
        except Exception as exc:
            logger.warning("batch [%s] validation failed: %s", label, exc)
            reviews = {name: {"verdict": "error", "error": str(exc)} for name in names}
        ops = []
        for esc, name in zip(members, names):
            if name in retry_ops:
                ops.append(retry_ops[name])
                continue
            review = reviews[name]
            self._memory_verdict(esc, review)
            ops.append({"escalation": json.loads(esc.to_json()), "review": review, "agent_run": agent_run})
        return ops

    async def _validate_one(self, esc, run_agent) -> dict:
        name = getattr(esc, "name", None) or getattr(esc, "code", "?")
        logger.info("validating escalation %s", name)
        brief = self._memory_brief(esc)
        query, schema = build_validation_query(
            esc, memory_brief=brief, prior_block=prior_verdict_block(self.ops, esc)
        )
        agent_run = None
        try:
            answer = await run_agent(query, schema, esc.code)
            agent_run = getattr(run_agent, "last_run_stats", None)
            review = parse_review(esc.code, answer)
            logger.info("escalation %s verdict: %s", name, review.get("verdict"))
            if review.get("rationale"):
                logger.info("escalation %s rationale: %s", name, review["rationale"])
        except Exception as exc:
            logger.warning("escalation %s validation failed: %s", name, exc)
            review = {"verdict": "error", "error": str(exc)}
        self._memory_verdict(esc, review)
        return {"escalation": json.loads(esc.to_json()), "review": review, "agent_run": agent_run}

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
        # Verdicts attach to the day's short-term episodes only. Long-term
        # memory never stores escalations or verdicts directly — it learns
        # from labeled period activity, which the verdicts confirm or correct
        # through the validated rows.
        name = getattr(esc, "name", None) or getattr(esc, "code", "?")
        if self._short_term is None:
            return
        try:
            self._short_term.record_verdict(
                name, str(review.get("verdict")), rationale=review.get("rationale")
            )
        except Exception:
            logger.exception("short-term verdict store failed; dropping event")


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
