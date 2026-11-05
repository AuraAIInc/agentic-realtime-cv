"""Command line: one query, or an interactive session.

python -m supervisor_agent "Is any cat on camera right now?"
python -m supervisor_agent            # interactive, keeps context
python -m supervisor_agent --list-tools

Tasks from the CV pipeline (see tasks.py for payloads; @file reads JSON from a file):
python -m supervisor_agent --task check_trajectory --payload '{"period_id": 8812}'
python -m supervisor_agent --task validate_detection --payload @detection.json
python -m supervisor_agent --labels     # chat to add labels / zones
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from supervisor_agent.agent import MCPAgent, RunOptions, RunResult, ToolStep
from supervisor_agent.config import AgentConfig
from supervisor_agent.local_tools import LocalTools
from supervisor_agent.local_tools.skills import SkillTools
from supervisor_agent.mcp_hub import MCPHub
from supervisor_agent.tasks import TaskRunner, chat_options

logger = logging.getLogger(__name__)


def _out(text: str = "", *, flush: bool = False) -> None:
    """Write one stdout line: the CLI's output channel (logging goes to stderr)."""
    sys.stdout.write(str(text) + "\n")
    if flush:
        sys.stdout.flush()


_COLOR = sys.stdout.isatty() and not os.getenv("NO_COLOR")


def _color(code: str):
    return (lambda text: f"\033[{code}m{text}\033[0m") if _COLOR else (lambda text: text)


green, blue, grey = _color("32"), _color("34"), _color("90")

# Tools hidden from default chat: write operations, rare reads, and raw DB dumps.
# Every hidden schema would otherwise be re-sent on each turn; --full-tools restores them.
# (Pipeline-only skill__* tools are already gated by their frontmatter `surface`.)
CHAT_HIDDEN_TOOLS = frozenset({
    "camera__set_schedule", "camera__reset_schedule",
    "camera__list_models", "camera__get_camera_modes",
    "tracking__get_raw_detections",
    "local__add_label", "local__save_zone",
})


def chat_tool_names(hub) -> list[str]:
    """All registered tools minus the ones default chat hides."""
    return [n for n in hub.tools if n not in CHAT_HIDDEN_TOOLS]


def _secs(ms: float) -> str:
    """Milliseconds -> seconds, at most 3 decimals: 718.96 -> '0.719 s'."""
    return f"{ms / 1000:.3f}".rstrip("0").rstrip(".") + " s"


def _redact_images(obj):
    """Shorten base64 data URLs so the prompt log stays readable."""
    if isinstance(obj, dict):
        return {k: _redact_images(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact_images(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("data:") and ";base64," in obj:
        head, data = obj.split(",", 1)
        return f"{head},<{len(data)} base64 chars>"
    return obj


def _print_event(kind: str, data) -> None:
    if kind == "llm_request":
        turn, req = data
        rest = {k: v for k, v in req.items() if k != "instructions"}
        rest["tools"] = [t["name"] for t in req["tools"]]  # full tool schemas are logged at startup
        # _out(grey(
        #     f"── LLM request (turn {turn}) ──\n[instructions]\n{req['instructions']}\n[request]\n"
        #     + json.dumps(_redact_images(rest), indent=2, ensure_ascii=False)
        # ), flush=True)
    elif kind == "tool_calls":
        for name, args in data:
            _out(green(f"  → {name}({args})"), flush=True)
    elif kind == "tool_result":
        s: ToolStep = data
        status = "ERROR" if s.is_error else "ok"
        server = f", server {_secs(s.server_elapsed_ms)}" if s.server_elapsed_ms is not None else ""
        imgs = f", {len(s.images_saved)} image(s) saved" if s.images_saved else ""
        _out(green(f"  ← {s.tool}: {status} (round trip {_secs(s.round_trip_ms)}{server}{imgs})"), flush=True)
        if s.error:
            _out(green(f"     {s.error}"), flush=True)


def _print_result(r: RunResult, cfg: AgentConfig) -> None:
    _out("\n" + blue(r.answer.strip()) + "\n")
    u = r.usage
    cost = cfg.estimate_cost_usd(u, r.model)
    cost_s = f"${cost:.4f}" if cost is not None else f"? (no pricing for {r.model})"
    _out(
        grey(
            f"[{r.model} | turns {r.turns} | tool calls {len(r.tool_steps)} | LLM {_secs(r.llm_ms)} | "
            f"tools {_secs(r.tools_ms)} | total {_secs(r.total_ms)} | "
            f"tokens in {u.get('input_tokens', '?')} / out {u.get('output_tokens', '?')} / total {u.get('total_tokens', '?')} | "
            f"est. cost {cost_s}" + (f" | {r.stopped_reason}" if r.stopped_reason != "answered" else "") + "]"
        )
    )
    if r.run_dir:
        _out(grey(f"[trace + images: {r.run_dir}]"))


async def amain(args) -> int:
    cfg = AgentConfig.from_env(args.servers, args.model_config)
    if args.model:
        cfg.model = args.model
    if args.max_steps:
        cfg.max_steps = args.max_steps
    if args.no_save:
        cfg.save_dir = None
    cfg.image_detail = args.image_detail

    async with MCPHub(cfg.servers) as hub:
        local = LocalTools()
        hub.add_local(local, "Cat profiles, reference images, trajectory checks, contact sheets, label set and zones (agent_data/).")
        hub.add_local(SkillTools(), "Task skills (skills/): invoking one loads its full instructions.")
        for name, err in hub.failed.items():
            logger.warning(f"{name} server unavailable: {err}")
        if not any(t.server not in hub.local for t in hub.tools.values()):
            logger.error("No MCP tools available; check CAMERA_MCP_URL / IMAGE_MCP_URL and tokens.")
            return 2
        if args.list_tools:
            for t in hub.tools.values():
                _out(f"{t.qualified_name}: {t.description.splitlines()[0] if t.description else ''}")
            return 0

        agent = MCPAgent(cfg, hub, on_event=_print_event)
        if args.task:
            payload = _load_payload(args.payload)
            out = await TaskRunner(agent, local).run(args.task, payload)
            _out(json.dumps(out, indent=2, default=str))
            _out(grey(f"[model called: {out.get('model_called')} | images {out.get('images_used', 0)} | {out.get('elapsed_s')} s | log {out.get('log_id', '-')}]"))
            return 0
        chat_opts = chat_options(local.store) if args.labels else None
        if chat_opts is None and not args.full_tools:
            chat_opts = RunOptions(tool_names=chat_tool_names(hub))
        if args.query:
            r = await agent.run(" ".join(args.query), options=chat_opts)
            _print_result(r, cfg)
            if args.json:
                _out(r.to_json())
            return 0

        mode = "label manager" if args.labels else "MCP agent"
        _out(f"{mode} ({cfg.model}) with {len(hub.tools)} tools. Empty line or Ctrl-D to quit.")
        prev = None
        while True:
            try:
                q = input("\n> ").strip()
            except EOFError:
                break
            if not q:
                break
            r = await agent.run(q, previous_response_id=prev, options=chat_opts)
            prev = r.last_response_id
            _print_result(r, cfg)
    return 0


def _load_payload(arg: str | None) -> dict:
    if not arg:
        raise SystemExit("--task needs --payload (JSON or @file.json)")
    if arg.startswith("@"):
        with open(arg[1:]) as f:
            return json.load(f)
    return json.loads(arg)


def main() -> None:
    from dotenv import load_dotenv

    # Always the repo-root .env (agent/supervisor_agent/cli.py -> parents[2]),
    # never whatever the current working directory happens to be.
    # AGENT_ENV_FILE (process env) swaps in another file, e.g. the demo's isolated config.
    load_dotenv(os.environ.get("AGENT_ENV_FILE") or Path(__file__).resolve().parents[2] / ".env", override=True)
    p = argparse.ArgumentParser(description="Muse Spark agent over the RCAT MCP servers")
    p.add_argument("query", nargs="*", help="question to answer (omit for interactive mode)")
    p.add_argument("--servers", help="JSON file: list of {name, url, token}")
    p.add_argument("--model-config", help="models.json with the provider, per-role models and token pricing (default: next to config.py)")
    p.add_argument("--model", help="override the agent model (roles.agent.model) set in the model config")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--image-detail", choices=["low", "high", "auto"], default="auto")
    p.add_argument("--list-tools", action="store_true")
    p.add_argument("--json", action="store_true", help="also print the full run trace as JSON")
    p.add_argument("--no-save", action="store_true", help="don't write trace.md / images to <repo-root>/output/agent_runs")
    p.add_argument("--task", choices=["validate_detection", "check_trajectory", "assign_label", "label_visit"],
                   help="run one pipeline task (skill + strict JSON answer) instead of a chat query")
    p.add_argument("--payload", help="task payload: JSON string or @file.json")
    p.add_argument("--labels", action="store_true", help="chat with the manage-labels skill to add labels and zones")
    p.add_argument("--full-tools", action="store_true", help="expose every tool in chat (default hides schedules, zones, labels, raw DB reads)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s - [%(levelname)s] %(filename)s:%(lineno)d: %(message)s")
    # Always show the servers and tools loaded at startup.
    logging.getLogger("supervisor_agent.mcp_hub").setLevel(logging.INFO)
    sys.exit(asyncio.run(amain(args)))


if __name__ == "__main__":
    main()
