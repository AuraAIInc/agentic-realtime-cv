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


class LLMClientError(Exception):
    """A memory LLM call failed (transport error or unparseable output)."""


class LLMClient(ABC):
    """Store merge, parent summarization, vision bootstrap interface."""

    @abstractmethod
    def store_update(self, unit: dict, observation: str) -> dict:
        """Return ``{keywords, contradicts_stable, note}`` for one observation."""

    @abstractmethod
    def summarize(self, level: str, children: list[dict], stats: dict) -> dict:
        """Return ``{text, keywords}`` compressing children into a parent."""

    @abstractmethod
    def bootstrap(self, image: bytes) -> dict:
        """Return ``{full_view, zones}`` environment descriptions for one frame."""


class MetaModelClient(LLMClient):
    """Sync Meta Model API client using the Responses endpoint, strict JSON."""

    def __init__(
        self,
        model: str | None = None,
        *,
        role: str = "memory",
        model_config: str | Path | None = None,
        zone_ids: list[str] | None = None,
        client=None,
        timeout: float = 60.0,
    ):
        from openai import OpenAI

        resolved, base_url, api_key_env = load_model_settings(role, model_config)
        self.model = model or resolved
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.zone_ids = list(zone_ids) if zone_ids is not None else default_zone_ids()
        if client is None:
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise RuntimeError(f"{api_key_env} is not set (Meta Model API key)")
            client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._client = client

    def store_update(self, unit: dict, observation: str) -> dict:
        data = self._json_call(
            store_update_instructions(),
            store_update_input(unit, observation),
            "memory_store_update",
            STORE_UPDATE_SCHEMA,
        )
        return {
            "keywords": [str(k) for k in data.get("keywords", []) or []],
            "contradicts_stable": bool(data.get("contradicts_stable", False)),
            "note": str(data.get("note", "") or ""),
        }

    def summarize(self, level: str, children: list[dict], stats: dict) -> dict:
        cap = WORD_CAPS.get(level, 100)
        data = self._json_call(
            promote_summary_instructions(level, cap),
            promote_summary_input(level, children, stats, cap),
            "memory_promote_summary",
            PROMOTE_SUMMARY_SCHEMA,
        )
        return {
            "text": str(data.get("text", "") or ""),
            "keywords": [str(k) for k in data.get("keywords", []) or []],
        }

    def bootstrap(self, image: bytes) -> dict:
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
        self, instructions: str, user_input, schema_name: str, schema: dict
    ) -> dict:
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
        except LLMClientError:
            raise
        except Exception as exc:
            raise LLMClientError(f"Meta Model API call failed: {exc}") from exc
        try:
            data = json.loads(_strip_fences(text))
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMClientError(
                f"LLM returned non-JSON output: {text[:200]!r}"
            ) from exc
        if not isinstance(data, dict):
            raise LLMClientError(f"LLM returned non-object JSON: {text[:200]!r}")
        return data


class StubLLMClient(LLMClient):
    """Deterministic offline stand-in with the same interface; no network."""

    def store_update(self, unit: dict, observation: str) -> dict:
        return {
            "keywords": extract_keywords(observation),
            "contradicts_stable": False,
            "note": observation[:240],
        }

    def summarize(self, level: str, children: list[dict], stats: dict) -> dict:
        cap = WORD_CAPS.get(level, 100)
        texts = [str(c.get("text", "")) for c in children if c.get("text")]
        words = " ".join(texts).split()
        keywords: list[str] = []
        for child in children:
            for kw in child.get("keywords", []):
                if kw not in keywords:
                    keywords.append(kw)
        return {"text": " ".join(words[:cap]), "keywords": keywords[:8]}

    def bootstrap(self, image: bytes) -> dict:
        return {
            "full_view": {
                "environment": "Stub full-view environment.",
                "background": "Stub background.",
                "cat_contrast": "Stub contrast.",
            },
            "zones": [],
        }
