"""Full LLM trace of one memory call, written as one markdown file per call.

Mirrors ``supervisor_agent.trace`` at single-turn scale: a table of contents
over the three messages (#1 system prompt, #2 user input, #3 assistant
output), then the full text of each message, the parsed answer, and a stats
block with timing, tokens, pricing and cost. One file per store/promote
LLM call under ``<trace_dir>/`` (flat; bootstrap input images sit next to
their markdown as ``<stem>.jpg``).

Tracing must never break memory: callers wrap :func:`write_memory_trace` in
try/except (or rely on the client, which already does).
"""

from __future__ import annotations

import datetime
import itertools
import json
import re
import threading
from pathlib import Path

_counter = itertools.count(1)
_counter_lock = threading.Lock()


def estimate_tokens(text: str) -> int:
    """Rough token count (chars / 4), same scale as the supervisor traces."""
    return max(0, len(text or "") // 4)


def estimate_cost_usd(usage: dict, pricing: dict | None) -> float | None:
    """Short-context Standard-tier estimate; None without a pricing entry."""
    if not pricing:
        return None
    cached = usage.get("cached_tokens", 0) or 0
    uncached = (usage.get("input_tokens", 0) or 0) - cached
    return (
        uncached * pricing["input"]
        + cached * pricing.get("cached_input", pricing["input"])
        + (usage.get("output_tokens", 0) or 0) * pricing["output"]
    ) / 1_000_000


def _preview(text: str, n: int = 40) -> str:
    return " ".join((text or "").split())[:n].replace("|", "\\|")


def _fence(text: str) -> str:
    longest = 0
    run = 0
    for ch in text or "":
        if ch == "`":
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return "`" * max(3, longest + 1)


def _block(text: str) -> str:
    fence = _fence(text or "")
    return f"{fence}\n{text or ''}\n{fence}"


def _safe_label(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", label or "call").strip("-")
    return cleaned or "call"


def render_memory_trace(
    *,
    model: str,
    kind: str,
    label: str,
    started_at: str,
    instructions: str,
    user_text: str,
    user_images: list[str],
    text_format: dict | None,
    output_text: str,
    parsed: dict | None,
    error: str | None,
    usage: dict,
    pricing: dict | None,
    est_cost_usd: float | None,
    llm_ms: float,
    response_id: str | None,
) -> str:
    """Build the full trace markdown for one memory LLM call."""
    messages: list[tuple[str, str, int, int, str]] = []
    messages.append(("system", _block(instructions or "(none)"), estimate_tokens(instructions), len(instructions or ""), instructions or ""))

    user_body = _block(user_text or "(empty input)")
    for img in user_images:
        user_body += f"\n\n![input image]({img})"
    messages.append(("user", user_body, estimate_tokens(user_text), len(user_text or ""), user_text or ""))

    asst_head = f"LLM {llm_ms:.1f} ms"
    if usage:
        asst_head += f" · ctx {usage.get('input_tokens', '?')} in / {usage.get('output_tokens', '?')} out"
        if usage.get("cached_tokens") is not None:
            asst_head += f" (cached {usage['cached_tokens']})"
    else:
        asst_head += " · ctx n/a"
    if response_id:
        asst_head += f" · response_id={response_id}"
    asst_body = _block(output_text or "(no output)")
    if error:
        asst_body += f"\n\n**Error:** `{error}`"
    if text_format:
        asst_body += f"\n\n*text.format={json.dumps(text_format, default=str)}*"
    messages.append(("assistant", f"*{asst_head}*\n\n{asst_body}", estimate_tokens(output_text), len(output_text or ""), output_text or ""))

    stamp = started_at.replace(":", "").replace("-", "").replace(" ", "T")
    lines = [f"# Memory LLM — {kind} — {label} — {stamp}", ""]
    lines += ["## Table of Contents", "", "| # | Type | Tokens | Preview |", "|---|---|---|---|"]
    for i, (msg_kind, _body, toks, _chars, prev) in enumerate(messages, 1):
        lines.append(f"| [#{i}](#msg-{i}) | {msg_kind} | {toks} | {_preview(prev)} |")
    lines += ["", "## Messages", ""]
    for i, (msg_kind, body, toks, chars, _prev) in enumerate(messages, 1):
        lines += [f'<a id="msg-{i}"></a>', f"### #{i}. {msg_kind} ({toks} tokens, {chars} chars)", "", body, ""]
    lines += ["## Answer", "", _block(json.dumps(parsed, indent=2, ensure_ascii=False) if parsed is not None else "(unparseable)"), ""]
    lines += ["## Stats", "", f"- model: {model}", f"- kind: {kind}", f"- label: {label}", f"- started_at: {started_at}"]
    lines.append(f"- llm_ms: {llm_ms:.1f}")
    lines.append(f"- input_tokens: {usage.get('input_tokens', 'n/a')}")
    lines.append(f"- output_tokens: {usage.get('output_tokens', 'n/a')}")
    lines.append(f"- cached_tokens: {usage.get('cached_tokens', 'n/a')}")
    lines.append(f"- total_tokens: {usage.get('total_tokens', 'n/a')}")
    if pricing:
        lines.append(
            f"- pricing_usd_per_1m: input={pricing.get('input')} cached_input={pricing.get('cached_input', pricing.get('input'))} output={pricing.get('output')}"
        )
    else:
        lines.append(f"- pricing_usd_per_1m: n/a (no pricing for {model})")
    lines.append(f"- est_cost_usd: {est_cost_usd:.6f}" if est_cost_usd is not None else "- est_cost_usd: n/a")
    if text_format:
        lines.append(f"- text_format: {json.dumps(text_format, default=str)}")
    lines.append(f"- response_id: {response_id or '-'}")
    lines.append("")
    return "\n".join(lines)


def write_memory_trace(
    trace_dir: str | Path,
    *,
    model: str,
    kind: str,
    label: str,
    instructions: str,
    user_text: str,
    text_format: dict | None,
    output_text: str,
    parsed: dict | None,
    error: str | None,
    usage: dict,
    pricing: dict | None,
    llm_ms: float,
    response_id: str | None,
    images: list[tuple[str, bytes]] | None = None,
) -> Path:
    """Render one memory-call trace and write it; returns the markdown path.

    ``images`` are ``(suffix, raw bytes)`` pairs saved next to the markdown
    as ``<stem><suffix>`` and embedded in the user message (bootstrap use).
    """
    started = datetime.datetime.now()
    with _counter_lock:
        seq = next(_counter)
    stem = f"{started:%Y%m%d-%H%M%S-%f}-{seq:03d}-{kind}-{_safe_label(label)}"
    out_dir = Path(trace_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for suffix, raw in images or []:
        name = f"{stem}{suffix}"
        (out_dir / name).write_bytes(bytes(raw))
        saved.append(name)
    md = render_memory_trace(
        model=model,
        kind=kind,
        label=label,
        started_at=started.strftime("%Y-%m-%d %H:%M:%S"),
        instructions=instructions,
        user_text=user_text,
        user_images=saved,
        text_format=text_format,
        output_text=output_text,
        parsed=parsed,
        error=error,
        usage=usage or {},
        pricing=pricing,
        est_cost_usd=estimate_cost_usd(usage or {}, pricing),
        llm_ms=llm_ms,
        response_id=response_id,
    )
    path = out_dir / f"{stem}.md"
    path.write_text(md, encoding="utf-8")
    return path


__all__ = ["estimate_cost_usd", "estimate_tokens", "render_memory_trace", "write_memory_trace"]
