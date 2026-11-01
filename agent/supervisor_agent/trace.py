"""Full LLM trace of an agent run, written as one markdown file per run.

Each run directory (``output/agent_runs/<timestamp>[-tag]/``) gets a ``trace.md``
that mimics the message-dump format: a table of contents over every message,
then the full text of each message -- system prompt, user input, every LLM
output, every tool invocation (arguments + full result text) -- with per-turn
timing, per-turn token usage, embedded images, and a final stats block with
tokens, pricing and cost.

Message layout (``#N`` rows in the table of contents)::

    #1  system                     the instructions (folded into the turn-1 input)
    #2  user                       the query text (+ input images, if any)
    #3  assistant (turn 1)         LLM output text + tool calls of turn 1
    #4  tool_result ...            one entry per tool call, in call order
    ...

Tool and input images live next to ``trace.md`` in the run directory and are
embedded with relative links; no base64 blobs are inlined in the markdown.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field


@dataclass
class TurnTrace:
    """Everything the model sent / received on one turn."""

    turn: int
    previous_response_id: str | None  # chaining id sent with this turn's request
    response_id: str | None  # response.id of this turn
    llm_ms: float  # this turn's Responses API round trip
    output_text: str  # response.output_text (the model's text, if any)
    output_item_types: list[str]  # e.g. ["message", "function_call", "function_call"]
    usage: dict = field(default_factory=dict)  # this turn's usage, {} when not reported
    calls: list[dict] = field(default_factory=list)  # [{name, arguments, call_id}]


def estimate_tokens(text: str) -> int:
    """Rough token count (chars / 4), the same scale the reference dump uses."""
    return max(0, len(text) // 4)


def extract_texts(query) -> list[str]:
    """All input_text strings of a ``run(query)`` argument, in order."""
    if isinstance(query, str):
        return [query] if query else []
    texts: list[str] = []

    def walk(obj) -> None:
        if isinstance(obj, dict):
            if obj.get("type") == "input_text" and isinstance(obj.get("text"), str):
                texts.append(obj["text"])
            else:
                for v in obj.values():
                    walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(query)
    return texts


def count_images(obj) -> int:
    """input_image items anywhere in a Responses API input structure."""
    if isinstance(obj, dict):
        return (obj.get("type") == "input_image") + sum(count_images(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(count_images(v) for v in obj)
    return 0


def save_data_url_images(obj, run_dir: str, prefix: str) -> list[str]:
    """Save every base64 data-URL image under ``obj`` into ``run_dir``.

    Returns the saved file basenames in document order (for markdown embeds).
    """
    urls: list[str] = []

    def walk(o) -> None:
        if isinstance(o, dict):
            if o.get("type") == "input_image" and isinstance(o.get("image_url"), str) and o["image_url"].startswith("data:"):
                urls.append(o["image_url"])
            else:
                for v in o.values():
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    saved: list[str] = []
    for i, url in enumerate(urls):
        try:
            head, data = url.split(",", 1)
            mime = head.split(";")[0].split(":", 1)[1] if ":" in head else "image/jpeg"
            ext = mime.split("/")[-1].replace("jpeg", "jpg")
            name = f"{prefix}_img{i:02d}.{ext}"
            with open(os.path.join(run_dir, name), "wb") as f:
                f.write(base64.b64decode(data))
            saved.append(name)
        except (ValueError, IndexError):
            continue
    return saved


def _preview(text: str, n: int = 40) -> str:
    one_line = " ".join((text or "").split())
    return one_line[:n].replace("|", "\\|")


def _fence(text: str) -> str:
    """A backtick fence longer than any backtick run inside ``text``."""
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


def _fmt_ms(ms) -> str:
    return "-" if ms is None else f"{ms:.1f} ms"


def _usage_str(usage: dict) -> str:
    if not usage:
        return "n/a"
    parts = [f"+{usage.get('input_tokens', '?')}/-{usage.get('output_tokens', '?')}"]
    if usage.get("cached_tokens") is not None:
        parts.append(f"cached {usage['cached_tokens']}")
    return " ".join(parts)


def render_trace(
    *,
    model: str,
    tag: str | None,
    started_at: str,
    instructions: str,
    query,
    query_images: list[str],
    tools_exposed: list[str],
    text_format: dict | None,
    reasoning_effort: str | None,
    turns: list[TurnTrace],
    tool_steps,  # list[ToolStep]: turn, tool, arguments, timings, error, images_saved, output_texts
    answer: str,
    usage: dict,
    pricing: dict | None,
    est_cost_usd: float | None,
    llm_ms: float,
    tools_ms: float,
    total_ms: float,
    stopped_reason: str,
    images_used: int,
    high_res_used: int,
    previous_response_id: str | None,
    last_response_id: str | None,
) -> str:
    """Build the full ``trace.md`` content for one agent run."""
    query_texts = extract_texts(query)
    query_text = "\n".join(query_texts)
    steps_by_turn: dict[int, list] = {}
    for st in tool_steps:
        steps_by_turn.setdefault(st.turn, []).append(st)

    # ------------------------------------------------------------------ messages
    # Each entry: (type label, body markdown, tokens, chars, preview source).
    messages: list[tuple[str, str, int, int, str]] = []

    sys_text = instructions or ""
    messages.append((
        "system",
        _block(sys_text or "(none)") + "\n\n*Folded into the turn-1 input (never sent as the `instructions` parameter); persists via chaining like any other context.*",
        estimate_tokens(sys_text),
        len(sys_text),
        sys_text,
    ))

    user_body = _block(query_text or "(empty query)")
    for img in query_images:
        user_body += f"\n\n![query image]({img})"
    if not query_images and count_images(query):
        user_body += "\n\n*(input images were already consumed before the run dir existed and could not be embedded)*"
    messages.append(("user", user_body, estimate_tokens(query_text), len(query_text), query_text))

    for t in turns:
        label = f"assistant (turn {t.turn})"
        n_calls = len(t.calls)
        head = f"LLM {_fmt_ms(t.llm_ms)} · tokens {_usage_str(t.usage)} · {n_calls} tool call{'s' if n_calls != 1 else ''}"
        if t.previous_response_id:
            head += f" · previous_response_id={t.previous_response_id}"
        if t.response_id:
            head += f" · response_id={t.response_id}"
        body = _block(t.output_text or "(no text — tool calls only)")
        if t.calls:
            lines = ["", "**Tool calls:**"]
            for c in t.calls:
                lines.append(f"- `{c['name']}` (id={c.get('call_id', '-')})" + (f": `{c.get('arguments', '{}')}`" if c.get("arguments") not in (None, "", "{}") else ": `{}`"))
            body += "\n" + "\n".join(lines)
        if t.turn == 1:
            extra = []
            if reasoning_effort:
                extra.append(f"reasoning_effort={reasoning_effort}")
            if text_format:
                extra.append(f"text.format={json.dumps(text_format, default=str)}")
            body += f"\n\n*Turn-1 input = system prompt (#1) + user query (#2). Tools exposed ({len(tools_exposed)}): {', '.join(tools_exposed)}.*"
            if extra:
                body += f"\n\n*{' · '.join(extra)}*"
        else:
            body += f"\n\n*Input = tool results of turn {t.turn - 1} (chained on {t.previous_response_id or 'nothing'}). Tools exposed: {len(tools_exposed)}.*"
        call_text = " ".join(f"{c['name']} {c.get('arguments', '')}" for c in t.calls)
        asst_text = (t.output_text or "") + call_text
        messages.append((label, f"*{head}*\n\n{body}", estimate_tokens(asst_text), len(asst_text), t.output_text or call_text))

        for st in steps_by_turn.get(t.turn, []):
            out_text = "\n".join(getattr(st, "output_texts", None) or [getattr(st, "preview", "") or ""])
            timing = f"round trip {_fmt_ms(st.round_trip_ms)}"
            if st.server_elapsed_ms is not None:
                timing += f" · server {_fmt_ms(st.server_elapsed_ms)}"
            status = "ERROR" if st.is_error else "ok"
            body = f"*`{st.tool}` · {status} · {timing}*\n\n"
            body += "**Arguments:**\n" + _block(json.dumps(st.arguments, indent=2, ensure_ascii=False, default=str))
            body += "\n\n**Output:**\n" + _block(out_text or "(empty)")
            for path in st.images_saved or []:
                body += f"\n\n![{st.tool}]({os.path.basename(path)})"
            if st.server_steps_ms:
                body += f"\n\n*server_steps_ms={json.dumps(st.server_steps_ms, default=str)}*"
            if st.error:
                body += f"\n\n**Error:** `{st.error}`"
            if getattr(st, "call_id", ""):
                body += f"\n\n*call_id={st.call_id}*"
            tool_text = out_text + json.dumps(st.arguments, default=str)
            messages.append((
                f"tool_result (turn {t.turn})",
                body,
                estimate_tokens(tool_text),
                len(tool_text),
                out_text,
            ))

    # ------------------------------------------------------------------ document
    stamp = started_at.replace(":", "").replace("-", "").replace(" ", "T")
    lines = [f"# Agent run — {model} — {tag or 'chat'} — {stamp}", ""]
    lines.append(f"model={model} turns={len(turns)} tool_calls={len(tool_steps)} answer_chars={len(answer or '')}")
    lines.append("")
    lines.append("## Table of Contents")
    lines.append("")
    lines.append("| # | Type | Tokens | Preview |")
    lines.append("|---|---|---|---|")
    for i, (kind, _body, toks, _chars, prev) in enumerate(messages, 1):
        lines.append(f"| [#{i}](#msg-{i}) | {kind} | {toks} | {_preview(prev)} |")
    lines.append("")
    lines.append("## Messages")
    lines.append("")
    for i, (kind, body, toks, chars, _prev) in enumerate(messages, 1):
        lines.append(f'<a id="msg-{i}"></a>')
        lines.append(f"### #{i}. {kind} ({toks} tokens, {chars} chars)")
        lines.append("")
        lines.append(body)
        lines.append("")
    lines.append("## Answer")
    lines.append("")
    lines.append(_block(answer or "(no answer)"))
    lines.append("")
    lines.append("## Stats")
    lines.append("")
    lines.append(f"- model: {model}")
    if tag:
        lines.append(f"- tag: {tag}")
    lines.append(f"- started_at: {started_at}")
    lines.append(f"- turns: {len(turns)}")
    lines.append(f"- tool_calls: {len(tool_steps)}")
    lines.append(f"- llm_ms: {llm_ms:.1f}")
    lines.append(f"- tools_ms: {tools_ms:.1f}")
    lines.append(f"- total_ms: {total_ms:.1f}")
    lines.append(f"- input_tokens: {usage.get('input_tokens', 'n/a')}")
    lines.append(f"- output_tokens: {usage.get('output_tokens', 'n/a')}")
    lines.append(f"- cached_tokens: {usage.get('cached_tokens', 'n/a')}")
    lines.append(f"- total_tokens: {usage.get('total_tokens', 'n/a')}")
    if pricing:
        lines.append(f"- pricing_usd_per_1m: input={pricing.get('input')} cached_input={pricing.get('cached_input', pricing.get('input'))} output={pricing.get('output')}")
    else:
        lines.append(f"- pricing_usd_per_1m: n/a (no pricing for {model})")
    lines.append(f"- est_cost_usd: {est_cost_usd:.6f}" if est_cost_usd is not None else "- est_cost_usd: n/a")
    lines.append(f"- images_used: {images_used} (high_res: {high_res_used})")
    if reasoning_effort:
        lines.append(f"- reasoning_effort: {reasoning_effort}")
    if text_format:
        lines.append(f"- text_format: {json.dumps(text_format, default=str)}")
    lines.append(f"- stopped_reason: {stopped_reason}")
    lines.append(f"- previous_response_id: {previous_response_id or '-'}")
    lines.append(f"- last_response_id: {last_response_id or '-'}")
    lines.append("")
    return "\n".join(lines)
