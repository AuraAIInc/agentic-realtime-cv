"""LLM clients: sync Meta Model API client + deterministic offline stub.

``MetaModelClient`` mirrors the ``supervisor_agent`` construction
(``OpenAI(base_url, api_key)`` from ``models.json`` roles, sync flavor since
callers and the memory worker are sync). Any transport or schema failure
raises ``LLMClientError``; ``LongTermMemory`` catches it and degrades the
affected unit to pending-observation mode, so unit tests and the ingest loop
never see an exception.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path

from agent.memory.prompts import (
    BOOTSTRAP_VISION_SCHEMA,
    PROMOTE_SUMMARY_SCHEMA,
    SPATIAL_SUMMARY_SCHEMA,
    STORE_UPDATE_SCHEMA,
    bootstrap_vision_input,
    bootstrap_vision_instructions,
    promote_summary_input,
    promote_summary_instructions,
    store_update_input,
    store_update_instructions,
)

logger = logging.getLogger(__name__)

# Stable-text word caps per level (design doc); the prompt states the cap and
# truncate_words() enforces it programmatically on the way back in.
WORD_CAPS = {
    "L0": 60,
    "L1": 100,
    "L2": 150,
    "slot_5min": 40,
    "hour": 80,
    "day": 120,
    "week": 150,
}

DEFAULT_MODEL_CONFIG = (
    Path(__file__).resolve().parent.parent / "supervisor_agent" / "models.json"
)
MEMORY_MODEL_FALLBACK = "muse-spark-1.3"

_WORD_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for",
        "with", "from", "by", "is", "are", "was", "were", "be", "been",
        "it", "its", "this", "that", "these", "those", "as", "so", "but",
        "if", "then", "than", "too", "very", "here", "there", "when",
        "while", "during", "over", "under", "into", "onto", "out", "up",
        "down", "again", "most", "more", "each", "other", "such", "no",
        "not", "only", "just", "also", "per", "still", "every",
    }
)


def extract_keywords(text: str) -> list[str]:
    """Deterministic keyword tokens: lowercase alphanumerics, stopwords dropped,
    single characters dropped, first-seen order, no duplicates."""
    seen: set[str] = set()
    out: list[str] = []
    for tok in _WORD_RE.findall(text.lower()):
        if len(tok) < 2 or tok in _STOPWORDS or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def truncate_words(text: str, cap: int) -> str:
    """Keep at most ``cap`` whitespace-separated words (programmatic word cap)."""
    if cap <= 0:
        return ""
    return " ".join(str(text or "").split()[:cap])


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def load_model_settings(
    role: str = "memory", path: str | Path | None = None
) -> tuple[str, str, str]:
    """Return ``(model, base_url, api_key_env)`` for a ``models.json`` role.

    Falls back to ``roles.agent`` when the role is absent and to
    ``muse-spark-1.3`` when the role names no model.
    """
    cfg_path = Path(path or os.getenv("MODEL_CONFIG") or DEFAULT_MODEL_CONFIG)
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    roles = data.get("roles", {})
    role_cfg = dict(roles.get(role) or roles.get("agent") or {})
    model = role_cfg.get("model") or MEMORY_MODEL_FALLBACK
    base_url = role_cfg.get("base_url") or data.get("base_url", "")
    api_key_env = role_cfg.get("api_key_env") or data.get(
        "api_key_env", "MODEL_API_KEY"
    )
    return str(model), str(base_url), str(api_key_env)


def default_zone_ids(model_config: str | Path | None = None) -> list[str]:
    """L1 zone ids (``r<row>c<col>``) from the memory config's L1 grid."""
    del model_config  # reserved for future per-deployment grids
    from agent.memory.memory_config import MemoryConfig

    levels = MemoryConfig.from_yaml().spatial_levels
    cols, rows = (levels[1].cols, levels[1].rows) if len(levels) > 1 else (1, 1)
    return [f"r{row}c{col}" for row in range(rows) for col in range(cols)]


def _image_mime(image: bytes) -> str:
    if image[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    return "image/jpeg"


def _input_texts(obj) -> list[str]:
    """All input_text strings of a Responses API input structure, in order."""
    if isinstance(obj, str):
        return [obj] if obj else []
    texts: list[str] = []

    def walk(o) -> None:
        if isinstance(o, dict):
            if o.get("type") == "input_text" and isinstance(o.get("text"), str):
                texts.append(o["text"])
            else:
                for v in o.values():
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return texts


_TRACE_KIND = {
    "memory_store_update": "store",
    "memory_promote_summary": "promote",
    "memory_bootstrap_vision": "bootstrap",
}


class LLMClientError(Exception):
    """A memory LLM call failed (transport error or unparseable output)."""


class LLMClient(ABC):
    """Store merge, parent summarization, vision bootstrap interface.

    ``label`` names the unit for logs and LLM traces (e.g. ``"period 2"``,
    ``"L1:r2c1"``); it never reaches the model.
    """

    @abstractmethod
    def store_update(self, unit: dict, observation: str, label: str | None = None) -> dict:
        """Return ``{contradicts_stable, note}`` for one observation."""

    @abstractmethod
    def summarize(self, level: str, children: list[dict], stats: dict, label: str | None = None) -> dict:
        """Return ``{text}`` compressing children into a parent.

        Spatial levels (L1/L2) additionally return ``appearance`` and
        ``behavior`` zone descriptions; ``text`` carries the activity.
        """

    @abstractmethod
    def bootstrap(self, image: bytes, label: str | None = None) -> dict:
        """Return ``{full_view, zones}`` environment descriptions for one frame."""


def _usage_of(resp) -> dict:
    """Best-effort usage dict from a Responses API response ({} when absent)."""
    raw = getattr(resp, "usage", None)
    if raw is None:
        return {}
    if isinstance(raw, dict):
        get = raw.get
        details = raw.get("input_tokens_details") or {}
        dget = details.get if isinstance(details, dict) else (lambda _k: None)
    else:
        get = lambda k: getattr(raw, k, None)  # noqa: E731 - tiny local adapter
        details = getattr(raw, "input_tokens_details", None)
        dget = lambda k: getattr(details, k, None) if details is not None else None
    usage = {
        "input_tokens": get("input_tokens"),
        "output_tokens": get("output_tokens"),
        "total_tokens": get("total_tokens"),
    }
    cached = dget("cached_tokens")
    if cached is not None:
        usage["cached_tokens"] = cached
    return {k: v for k, v in usage.items() if v is not None}


def _pricing_for(model: str, model_config: str | Path | None) -> dict | None:
    """Per-model USD-per-1M-tokens pricing from models.json (None when absent)."""
    try:
        cfg_path = Path(model_config or os.getenv("MODEL_CONFIG") or DEFAULT_MODEL_CONFIG)
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        entry = (data.get("pricing_usd_per_1m_tokens") or {}).get(model)
        return dict(entry) if isinstance(entry, dict) else None
    except (OSError, ValueError):
        return None


class MetaModelClient(LLMClient):
    """Sync Meta Model API client using the Responses endpoint, strict JSON.

    ``trace_dir`` enables one markdown LLM trace per call (system prompt,
    input, raw output, stats); ``None`` disables tracing. Trace writes are
    best-effort and never raise.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        role: str = "memory",
        model_config: str | Path | None = None,
        zone_ids: list[str] | None = None,
        client=None,
        timeout: float = 60.0,
        trace_dir: str | Path | None = None,
    ):
        from openai import OpenAI

        resolved, base_url, api_key_env = load_model_settings(role, model_config)
        self.model = model or resolved
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.zone_ids = list(zone_ids) if zone_ids is not None else default_zone_ids()
        self.trace_dir = Path(trace_dir) if trace_dir is not None else None
        self.pricing = _pricing_for(self.model, model_config)
        if client is None:
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise RuntimeError(f"{api_key_env} is not set (Meta Model API key)")
            client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._client = client

    def store_update(self, unit: dict, observation: str, label: str | None = None) -> dict:
        data = self._json_call(
            store_update_instructions(),
            store_update_input(unit, observation),
            "memory_store_update",
            STORE_UPDATE_SCHEMA,
            label=label,
        )
        return {
            "contradicts_stable": bool(data.get("contradicts_stable", False)),
            "note": str(data.get("note", "") or ""),
        }

    def summarize(
        self, level: str, children: list[dict], stats: dict, label: str | None = None
    ) -> dict:
        cap = WORD_CAPS.get(level, 100)
        spatial = level in ("L1", "L2")
        data = self._json_call(
            promote_summary_instructions(level, cap),
            promote_summary_input(level, children, stats, cap),
            "memory_promote_summary",
            SPATIAL_SUMMARY_SCHEMA if spatial else PROMOTE_SUMMARY_SCHEMA,
            label=label,
        )
        out = {"text": str(data.get("activity" if spatial else "text", "") or "")}
        if spatial:
            out["appearance"] = str(data.get("appearance", "") or "")
            out["behavior"] = str(data.get("behavior", "") or "")
        return out

    def bootstrap(self, image: bytes, label: str | None = None) -> dict:
        payload = bytes(image)
        b64 = base64.b64encode(payload).decode("ascii")
        vision_input = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": bootstrap_vision_input()},
                    {
                        "type": "input_image",
                        "image_url": f"data:{_image_mime(payload)};base64,{b64}",
                        "detail": "auto",
                    },
                ],
            }
        ]
        data = self._json_call(
            bootstrap_vision_instructions(self.zone_ids),
            vision_input,
            "memory_bootstrap_vision",
            BOOTSTRAP_VISION_SCHEMA,
            label=label,
            trace_images=[(".jpg" if _image_mime(payload) == "image/jpeg" else ".png", payload)],
        )
        full_view = data.get("full_view", {}) or {}
        zones: list[dict] = []
        for zone in data.get("zones", []) or []:
            if not isinstance(zone, dict) or not zone.get("id"):
                continue
            zones.append(
                {
                    "id": str(zone["id"]),
                    "environment": str(zone.get("environment", "") or ""),
                    "background": str(zone.get("background", "") or ""),
                    "cat_contrast": str(zone.get("cat_contrast", "") or ""),
                }
            )
        return {
            "full_view": {
                "environment": str(full_view.get("environment", "") or ""),
                "background": str(full_view.get("background", "") or ""),
                "cat_contrast": str(full_view.get("cat_contrast", "") or ""),
            },
            "zones": zones,
        }

    def _json_call(
        self,
        instructions: str,
        user_input,
        schema_name: str,
        schema: dict,
        label: str | None = None,
        trace_images: list[tuple[str, bytes]] | None = None,
    ) -> dict:
        import time

        started = time.perf_counter()
        try:
            resp = self._client.responses.create(
                model=self.model,
                instructions=instructions,
                input=user_input,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    }
                },
            )
            text = (getattr(resp, "output_text", "") or "").strip()
            usage, response_id = _usage_of(resp), getattr(resp, "id", None)
        except LLMClientError:
            raise
        except Exception as exc:
            self._write_trace(
                instructions, user_input, schema_name, schema, label,
                output_text="", parsed=None, error=str(exc), usage={},
                llm_ms=(time.perf_counter() - started) * 1000,
                response_id=None, trace_images=trace_images,
            )
            raise LLMClientError(f"Meta Model API call failed: {exc}") from exc
        llm_ms = (time.perf_counter() - started) * 1000
        try:
            data = json.loads(_strip_fences(text))
        except (json.JSONDecodeError, ValueError) as exc:
            self._write_trace(
                instructions, user_input, schema_name, schema, label,
                output_text=text, parsed=None, error=f"non-JSON output: {text[:200]!r}",
                usage=usage, llm_ms=llm_ms, response_id=response_id,
                trace_images=trace_images,
            )
            raise LLMClientError(
                f"LLM returned non-JSON output: {text[:200]!r}"
            ) from exc
        if not isinstance(data, dict):
            self._write_trace(
                instructions, user_input, schema_name, schema, label,
                output_text=text, parsed=None, error=f"non-object JSON: {text[:200]!r}",
                usage=usage, llm_ms=llm_ms, response_id=response_id,
                trace_images=trace_images,
            )
            raise LLMClientError(f"LLM returned non-object JSON: {text[:200]!r}")
        self._write_trace(
            instructions, user_input, schema_name, schema, label,
            output_text=text, parsed=data, error=None, usage=usage,
            llm_ms=llm_ms, response_id=response_id, trace_images=trace_images,
        )
        return data

    def _write_trace(
        self,
        instructions: str,
        user_input,
        schema_name: str,
        schema: dict,
        label: str | None,
        output_text: str,
        parsed: dict | None,
        error: str | None,
        usage: dict,
        llm_ms: float,
        response_id: str | None,
        trace_images: list[tuple[str, bytes]] | None,
    ) -> None:
        """One markdown trace per call; silent no-op unless ``trace_dir`` is set."""
        if self.trace_dir is None:
            return
        try:
            from agent.memory.trace import write_memory_trace

            if isinstance(user_input, str):
                user_text = user_input
            else:
                user_text = "\n".join(
                    _input_texts(user_input)
                    or [json.dumps(user_input, ensure_ascii=False)[:2000]]
                )
            path = write_memory_trace(
                self.trace_dir,
                model=self.model,
                kind=_TRACE_KIND.get(schema_name, schema_name),
                label=label or schema_name,
                instructions=instructions,
                user_text=user_text,
                text_format={"type": "json_schema", "name": schema_name, "schema": schema},
                output_text=output_text,
                parsed=parsed,
                error=error,
                usage=usage,
                pricing=self.pricing,
                llm_ms=llm_ms,
                response_id=response_id,
                images=trace_images,
            )
            logger.info("memory LLM trace: %s", path)
        except Exception as exc:  # noqa: BLE001 - tracing must never break memory
            logger.warning("memory LLM trace write failed: %s", exc)


class StubLLMClient(LLMClient):
    """Deterministic offline stand-in with the same interface; no network."""

    def store_update(self, unit: dict, observation: str, label: str | None = None) -> dict:
        return {
            "contradicts_stable": False,
            "note": observation[:240],
        }

    def summarize(
        self, level: str, children: list[dict], stats: dict, label: str | None = None
    ) -> dict:
        cap = WORD_CAPS.get(level, 100)
        texts = [str(c.get("text", "")) for c in children if c.get("text")]
        words = " ".join(texts).split()
        out = {"text": " ".join(words[:cap])}
        if level in ("L1", "L2"):
            apps = [str(c.get("appearance", "")) for c in children if c.get("appearance")]
            behavs = [str(c.get("behavior", "")) for c in children if c.get("behavior")]
            if not behavs:  # mirror the prompt: behavior derives from activity lines
                behavs = texts
            out["appearance"] = " ".join(" ".join(apps).split()[:cap])
            out["behavior"] = " ".join(" ".join(behavs).split()[:cap])
        return out

    def bootstrap(self, image: bytes, label: str | None = None) -> dict:
        return {
            "full_view": {
                "environment": "Stub full-view environment.",
                "background": "Stub background.",
                "cat_contrast": "Stub contrast.",
            },
            "zones": [],
        }
