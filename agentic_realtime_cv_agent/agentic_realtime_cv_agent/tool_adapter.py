"""Translation between MCP and the Responses API (Meta Model API).

* MCP tool definitions  -> function tools
* MCP tool results      -> function_call_output items (text + images), so the
  model can actually look at frames returned by capture_frame / get_image.
"""

from __future__ import annotations

import base64
import json
import os

from agentic_realtime_cv_agent.mcp_hub import CallOutcome, HubTool


def to_function_tool(tool: HubTool) -> dict:
    schema = dict(tool.input_schema or {})
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return {
        "type": "function",
        "name": tool.qualified_name,
        "description": (tool.description or "")[:1024],
        "parameters": schema,
        "strict": False,  # MCP schemas have optional params with defaults; strict mode requires all fields
    }


def server_timing(outcome: CallOutcome) -> dict | None:
    """Pull the `timing` object the RCAT MCP servers put in their JSON text (if any)."""
    res = outcome.result
    if res is None:
        return None
    sc = getattr(res, "structured_content", None)
    if isinstance(sc, dict) and isinstance(sc.get("timing"), dict):
        return sc["timing"]
    for block in getattr(res, "content", []) or []:
        if getattr(block, "type", None) == "text":
            try:
                obj = json.loads(block.text)
            except (ValueError, TypeError):
                continue
            if isinstance(obj, dict) and isinstance(obj.get("timing"), dict):
                return obj["timing"]
    return None


def to_function_output(outcome: CallOutcome, image_detail: str = "auto", save_dir: str | None = None, step: int = 0) -> tuple[list[dict], list[str]]:
    """Return (output items for the model, paths of images saved to disk)."""
    header = {"tool": outcome.tool.qualified_name, "round_trip_ms": outcome.round_trip_ms}
    if outcome.error:
        return [{"type": "input_text", "text": json.dumps({**header, "error": outcome.error})}], []

    res = outcome.result
    items: list[dict] = []
    saved: list[str] = []
    if getattr(res, "is_error", False):
        header["is_error"] = True
    items.append({"type": "input_text", "text": json.dumps(header)})
    for i, block in enumerate(getattr(res, "content", []) or []):
        kind = getattr(block, "type", None)
        if kind == "text":
            items.append({"type": "input_text", "text": block.text})
        elif kind == "image":
            mime = block.mime_type or "image/jpeg"
            items.append({"type": "input_image", "image_url": f"data:{mime};base64,{block.data}", "detail": image_detail})
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                ext = mime.split("/")[-1].replace("jpeg", "jpg")
                path = os.path.join(save_dir, f"step{step:02d}_{outcome.tool.name}_{i}.{ext}")
                with open(path, "wb") as f:
                    f.write(base64.b64decode(block.data))
                saved.append(path)
        else:
            items.append({"type": "input_text", "text": f"[{kind} content omitted]"})
    return items, saved
