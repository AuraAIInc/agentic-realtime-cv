"""Connects to several MCP servers at once and routes tool calls to them.

Tools are exposed to the LLM as "<server>__<tool>" so names from different
servers can't collide. A server that can't be reached is skipped with a
warning; the agent still runs with the others.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from dataclasses import dataclass, field

from mcp import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

from agentic_realtime_cv_agent.config import ServerConfig

logger = logging.getLogger(__name__)

SEP = "__"


@dataclass
class HubTool:
    server: str
    name: str  # name on the MCP server
    qualified_name: str  # name given to the LLM
    description: str
    input_schema: dict
    title: str | None = None
    output_schema: dict | None = None
    annotations: dict = field(default_factory=dict)  # readOnlyHint, destructiveHint, idempotentHint, openWorldHint, ...
    execution: dict | None = None  # e.g. task support
    meta: dict | None = None


@dataclass
class CallOutcome:
    tool: HubTool
    arguments: dict
    result: object | None  # mcp CallToolResult
    error: str | None
    round_trip_ms: float


class MCPHub:
    def __init__(self, servers: list[ServerConfig]):
        self.servers = servers
        self.sessions: dict[str, ClientSession] = {}
        self.tools: dict[str, HubTool] = {}
        self.server_info: dict[str, str] = {}  # server name -> instructions
        self.failed: dict[str, str] = {}
        self.local: dict[str, object] = {}  # server name -> in-process toolset (see add_local)
        self._stack = contextlib.AsyncExitStack()

    async def __aenter__(self) -> "MCPHub":
        for s in self.servers:
            try:
                await self._connect(s)
            except BaseException as e:  # anyio raises ExceptionGroups on connect failure
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                msg = _short_error(e)
                if "Cancel" in msg or "ConnectError" in msg:
                    msg = "could not connect (server down, wrong URL, or not reachable from here)"
                elif "error response" in msg:
                    msg = "server rejected the connection (check the token)"
                self.failed[s.name] = msg
                logger.warning(f"MCP server '{s.name}' at {s.url} unavailable: {msg}")
        return self

    async def __aexit__(self, *exc):
        with contextlib.suppress(BaseException):
            await self._stack.aclose()

    async def _connect(self, s: ServerConfig) -> None:
        # Each server gets its own exit stack so a failed connect can be unwound cleanly.
        stack = contextlib.AsyncExitStack()
        try:
            headers = {"Authorization": f"Bearer {s.token}"} if s.token else None
            http = await stack.enter_async_context(create_mcp_http_client(headers=headers))
            streams = await stack.enter_async_context(streamable_http_client(s.url, http_client=http))
            session = await stack.enter_async_context(ClientSession(streams[0], streams[1], read_timeout_seconds=s.timeout_s))
            init = await session.initialize()
            listed = await session.list_tools()
        except BaseException:
            with contextlib.suppress(BaseException):
                await stack.aclose()
            raise
        self._stack.push_async_callback(stack.aclose)
        self.sessions[s.name] = session
        self.server_info[s.name] = getattr(init, "instructions", None) or ""
        for t in listed.tools:
            q = f"{s.name}{SEP}{t.name}"
            self.tools[q] = HubTool(
                s.name,
                t.name,
                q,
                t.description or "",
                t.input_schema or {"type": "object", "properties": {}},
                title=t.title or (t.annotations.title if t.annotations else None),
                output_schema=t.output_schema,
                annotations=_dump(t.annotations) or {},
                execution=_dump(t.execution),
                meta=t.meta,
            )
        info = init.server_info
        logger.info(
            f"Connected to '{s.name}' ({s.url}): server {info.name} {info.version or ''}, "
            f"protocol {init.protocol_version}, {len(listed.tools)} tools"
        )
        for t in listed.tools:
            _log_tool(self.tools[f"{s.name}{SEP}{t.name}"])

    def add_local(self, toolset, info: str = "") -> None:
        """Register an in-process toolset (an object with server_name, specs() -> [{name,
        description, input_schema}], and async call(name, args) -> CallToolResult). Its tools are
        exposed as "<server_name>__<tool>" exactly like a remote MCP server's."""
        name = toolset.server_name
        self.local[name] = toolset
        if hasattr(toolset, "hub"):
            toolset.hub = self
        self.server_info[name] = info
        for spec in toolset.specs():
            q = f"{name}{SEP}{spec['name']}"
            self.tools[q] = HubTool(name, spec["name"], q, spec.get("description", ""), spec.get("input_schema") or {"type": "object", "properties": {}},
                                    meta={"returns_image": True} if spec.get("returns_image") else None)
        logger.info(f"Registered in-process '{name}' tools: {len(toolset.specs())}")

    async def call(self, qualified_name: str, arguments: dict) -> CallOutcome:
        tool = self.tools.get(qualified_name)
        t0 = time.perf_counter()
        if tool is None:
            return CallOutcome(HubTool("?", qualified_name, qualified_name, "", {}), arguments, None, f"unknown tool {qualified_name}", 0.0)
        if tool.server in self.local:
            try:
                result, err = await self.local[tool.server].call(tool.name, arguments), None
            except Exception as e:
                result, err = None, _short_error(e)
            return CallOutcome(tool, arguments, result, err, round((time.perf_counter() - t0) * 1000, 2))
        try:
            result = await self.sessions[tool.server].call_tool(tool.name, arguments)
            err = None
        except Exception as e:
            result, err = None, _short_error(e)
        return CallOutcome(tool, arguments, result, err, round((time.perf_counter() - t0) * 1000, 2))


def _dump(model) -> dict | None:
    # MCP pydantic models -> plain dict with the protocol's camelCase keys, unset fields dropped.
    return model.model_dump(by_alias=True, exclude_none=True) if model is not None else None


def _log_tool(t: HubTool) -> None:
    lines = [f"Tool {t.qualified_name}" + (f" ({t.title})" if t.title else "")]
    lines.append("  description: " + (t.description.strip().replace("\n", "\n    ") or "(none)"))
    lines.append("  input_schema: " + json.dumps(t.input_schema, indent=2).replace("\n", "\n  "))
    if t.output_schema:
        lines.append("  output_schema: " + json.dumps(t.output_schema, indent=2).replace("\n", "\n  "))
    if t.annotations:
        lines.append("  annotations: " + json.dumps(t.annotations))
    if t.execution:
        lines.append("  execution: " + json.dumps(t.execution))
    if t.meta:
        lines.append("  _meta: " + json.dumps(t.meta, default=str))
    logger.info("\n".join(lines))


def _short_error(e: BaseException) -> str:
    # Unwrap anyio ExceptionGroups to the first real cause.
    while isinstance(e, BaseExceptionGroup) and e.exceptions:
        e = e.exceptions[0]
    return f"{type(e).__name__}: {e}"
