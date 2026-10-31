"""Memory configuration: YAML defaults plus ``MEMORY_*`` env overrides.

Mirrors the ``agent.escalation.escalation_config`` pattern: checked-in
defaults in ``memory_config.yaml`` next to this file, with per-knob
environment overrides so tuning needs no code change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "memory_config.yaml"


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw is not None else default


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw is not None else default


def _str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw if raw is not None else default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class SpatialLevel:
    name: str
    cols: int
    rows: int


@dataclass(frozen=True)
class TemporalLevel:
    name: str
    minutes: int


def _default_spatial_levels() -> list[SpatialLevel]:
    return [
        SpatialLevel("cells", 8, 6),
        SpatialLevel("zones", 4, 3),
        SpatialLevel("full_view", 1, 1),
    ]


def _default_temporal_levels() -> list[TemporalLevel]:
    return [
        TemporalLevel("slot_5min", 5),
        TemporalLevel("hour", 60),
        TemporalLevel("day", 1440),
        TemporalLevel("week", 10080),
    ]


def _default_retention_days() -> dict[str, int]:
    return {"slot_5min": 7, "hour": 30, "day": 180, "week": 0}


@dataclass
class MemoryConfig:
    version: int = 1
    view_id: int = 3
    frame_width: int = 800
    frame_height: int = 600
    spatial_levels: list[SpatialLevel] = field(default_factory=_default_spatial_levels)
    temporal_levels: list[TemporalLevel] = field(default_factory=_default_temporal_levels)
    retention_days: dict[str, int] = field(default_factory=_default_retention_days)
    spatial_chebyshev_radius: int = 1
    temporal_history_days: int = 7
    semantic_top_k: int = 5
    idle_seconds: float = 300
    min_pattern_support: int = 3
    stable_weight_cap: int = 50
    flip_threshold: int = 3
    recent_kept: int = 5
    queue_warn_depth: int = 100
    promote_coalesce: bool = True
    store_dir: str = "agent/memory/data"
    llm_model_role: str = "memory"
    max_units_per_call: int = 8

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "MemoryConfig":
        cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        raw: dict = {}
        if cfg_path.exists():
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        elif path is not None:
            raise FileNotFoundError(f"memory config not found: {cfg_path}")
        cfg = cls()
        if "version" in raw:
            cfg.version = int(raw["version"])
        if "view_id" in raw:
            cfg.view_id = int(raw["view_id"])
        if "frame_size" in raw:
            w, h = raw["frame_size"]
            cfg.frame_width, cfg.frame_height = int(w), int(h)
        if "spatial_levels" in raw:
            cfg.spatial_levels = [
                SpatialLevel(str(lv["name"]), int(lv["grid"][0]), int(lv["grid"][1]))
                for lv in raw["spatial_levels"]
            ]
        if "temporal_levels" in raw:
            cfg.temporal_levels = [
                TemporalLevel(str(lv["name"]), int(lv["minutes"]))
                for lv in raw["temporal_levels"]
            ]
        if "retention_days" in raw:
            cfg.retention_days = {str(k): int(v) for k, v in raw["retention_days"].items()}
        retrieval = raw.get("retrieval") or {}
        if "spatial_chebyshev_radius" in retrieval:
            cfg.spatial_chebyshev_radius = int(retrieval["spatial_chebyshev_radius"])
        if "temporal_history_days" in retrieval:
            cfg.temporal_history_days = int(retrieval["temporal_history_days"])
        if "semantic_top_k" in retrieval:
            cfg.semantic_top_k = int(retrieval["semantic_top_k"])
        promotion = raw.get("promotion") or {}
        if "idle_seconds" in promotion:
            cfg.idle_seconds = float(promotion["idle_seconds"])
        for key in ("min_pattern_support", "stable_weight_cap", "flip_threshold", "recent_kept"):
            if key in promotion:
                setattr(cfg, key, int(promotion[key]))
        worker = raw.get("worker") or {}
        if "queue_warn_depth" in worker:
            cfg.queue_warn_depth = int(worker["queue_warn_depth"])
        if "promote_coalesce" in worker:
            cfg.promote_coalesce = bool(worker["promote_coalesce"])
        if "store_dir" in raw:
            cfg.store_dir = str(raw["store_dir"])
        llm = raw.get("llm") or {}
        if "model_role" in llm:
            cfg.llm_model_role = str(llm["model_role"])
        if "max_units_per_call" in llm:
            cfg.max_units_per_call = int(llm["max_units_per_call"])
        return cfg.with_env_overrides()

    def with_env_overrides(self) -> "MemoryConfig":
        self.view_id = _int("MEMORY_VIEW_ID", self.view_id)
        self.frame_width = _int("MEMORY_FRAME_WIDTH", self.frame_width)
        self.frame_height = _int("MEMORY_FRAME_HEIGHT", self.frame_height)
        self.spatial_chebyshev_radius = _int(
            "MEMORY_SPATIAL_CHEBYSHEV_RADIUS", self.spatial_chebyshev_radius
        )
        self.temporal_history_days = _int(
            "MEMORY_TEMPORAL_HISTORY_DAYS", self.temporal_history_days
        )
        self.semantic_top_k = _int("MEMORY_SEMANTIC_TOP_K", self.semantic_top_k)
        self.idle_seconds = _float("MEMORY_IDLE_SECONDS", self.idle_seconds)
        self.min_pattern_support = _int("MEMORY_MIN_PATTERN_SUPPORT", self.min_pattern_support)
        self.stable_weight_cap = _int("MEMORY_STABLE_WEIGHT_CAP", self.stable_weight_cap)
        self.flip_threshold = _int("MEMORY_FLIP_THRESHOLD", self.flip_threshold)
        self.recent_kept = _int("MEMORY_RECENT_KEPT", self.recent_kept)
        self.queue_warn_depth = _int("MEMORY_QUEUE_WARN_DEPTH", self.queue_warn_depth)
        self.promote_coalesce = _bool("MEMORY_PROMOTE_COALESCE", self.promote_coalesce)
        self.store_dir = _str("MEMORY_STORE_DIR", self.store_dir)
        self.llm_model_role = _str("MEMORY_LLM_MODEL_ROLE", self.llm_model_role)
        self.max_units_per_call = _int("MEMORY_MAX_UNITS_PER_CALL", self.max_units_per_call)
        return self

    def store_path(self, view_id: int | None = None) -> Path:
        vid = self.view_id if view_id is None else view_id
        return Path(self.store_dir) / f"view_{vid}_long_term.yaml"
