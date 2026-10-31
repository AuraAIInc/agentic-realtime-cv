"""YAML long-term store: atomic round-trip, sparse units, retention, inheritance."""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from agent.memory.memory_config import MemoryConfig
from agent.memory.temporal import TemporalIndex


def _norm_stats(stats: dict | None) -> dict:
    stats = dict(stats or {})
    by_cat = {str(k): int(v) for k, v in (stats.get("by_cat") or {}).items()}
    return {"visits": int(stats.get("visits", 0)), "by_cat": by_cat}


@dataclass
class SpatialUnit:
    id: str
    bounds: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    environment: str = ""
    background: str = ""
    cat_contrast: str = ""
    activity: str = ""
    keywords: list[str] = field(default_factory=list)
    weight: int = 0
    contradictions: int = 0
    recent: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=lambda: {"visits": 0, "by_cat": {}})
    updated: str | None = None
    sources: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return (
            not self.environment
            and not self.background
            and not self.cat_contrast
            and not self.activity
            and int(self.stats.get("visits", 0)) == 0
            and not self.recent
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bounds": [float(v) for v in self.bounds],
            "environment": self.environment,
            "background": self.background,
            "cat_contrast": self.cat_contrast,
            "activity": self.activity,
            "keywords": list(self.keywords),
            "weight": int(self.weight),
            "contradictions": int(self.contradictions),
            "recent": list(self.recent),
            "stats": _norm_stats(self.stats),
            "updated": self.updated,
            "sources": list(self.sources),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpatialUnit":
        return cls(
            id=str(data["id"]),
            bounds=[float(v) for v in data.get("bounds", [0.0, 0.0, 0.0, 0.0])],
            environment=str(data.get("environment") or ""),
            background=str(data.get("background") or ""),
            cat_contrast=str(data.get("cat_contrast") or ""),
            activity=str(data.get("activity") or ""),
            keywords=[str(k) for k in data.get("keywords", []) or []],
            weight=int(data.get("weight", 0)),
            contradictions=int(data.get("contradictions", 0)),
            recent=[str(r) for r in data.get("recent", []) or []],
            stats=_norm_stats(data.get("stats")),
            updated=data.get("updated"),
            sources=[str(s) for s in data.get("sources", []) or []],
        )


@dataclass
class TemporalUnit:
    id: str
    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    weight: int = 0
    contradictions: int = 0
    recent: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=lambda: {"visits": 0, "by_cat": {}})
    updated: str | None = None
    sources: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return (
            not self.summary
            and int(self.stats.get("visits", 0)) == 0
            and not self.recent
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "summary": self.summary,
            "keywords": list(self.keywords),
            "weight": int(self.weight),
            "contradictions": int(self.contradictions),
            "recent": list(self.recent),
            "stats": _norm_stats(self.stats),
            "updated": self.updated,
            "sources": list(self.sources),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TemporalUnit":
        return cls(
            id=str(data["id"]),
            summary=str(data.get("summary") or ""),
            keywords=[str(k) for k in data.get("keywords", []) or []],
            weight=int(data.get("weight", 0)),
            contradictions=int(data.get("contradictions", 0)),
            recent=[str(r) for r in data.get("recent", []) or []],
            stats=_norm_stats(data.get("stats")),
            updated=data.get("updated"),
            sources=[str(s) for s in data.get("sources", []) or []],
        )


class MemoryStore:
    """In-memory spatial/temporal trees with atomic YAML persistence."""

    def __init__(self, config: MemoryConfig, path: str | Path | None = None):
        self.config = config
        self.path = Path(path) if path is not None else config.store_path()
        self.version = config.version
        self.view_id = config.view_id
        self.spatial: dict[str, dict[str, SpatialUnit]] = {}
        self.temporal: dict[str, dict[str, TemporalUnit]] = {}
        self.bootstrapped = False
        self.clear()

    @property
    def spatial_labels(self) -> list[str]:
        return [f"L{i}" for i in range(len(self.config.spatial_levels))]

    @property
    def temporal_names(self) -> list[str]:
        return [lv.name for lv in self.config.temporal_levels]

    def clear(self) -> None:
        self.spatial = {label: {} for label in self.spatial_labels}
        self.temporal = {name: {} for name in self.temporal_names}
        self.bootstrapped = False

    def load(self) -> "MemoryStore":
        self.clear()
        self.version = self.config.version
        self.view_id = self.config.view_id
        if not self.path.exists():
            return self
        data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        self.version = int(data.get("version", self.version))
        self.view_id = int(data.get("view_id", self.view_id))
        self.bootstrapped = bool(data.get("bootstrapped", False))
        for label, items in (data.get("spatial") or {}).items():
            if label not in self.spatial:
                continue
            for item in items or []:
                unit = SpatialUnit.from_dict(item)
                self.spatial[label][unit.id] = unit
        for name, items in (data.get("temporal") or {}).items():
            if name not in self.temporal:
                continue
            for item in items or []:
                unit = TemporalUnit.from_dict(item)
                self.temporal[name][unit.id] = unit
        return self

    def save(self) -> Path:
        spatial = {
            label: [u.to_dict() for u in units.values() if not u.is_empty]
            for label, units in self.spatial.items()
        }
        temporal = {
            name: [u.to_dict() for u in units.values() if not u.is_empty]
            for name, units in self.temporal.items()
        }
        data = {
            "version": self.version,
            "view_id": self.view_id,
            "bootstrapped": self.bootstrapped,
            "spatial": {k: v for k, v in spatial.items() if v},
            "temporal": {k: v for k, v in temporal.items() if v},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.parent / (self.path.name + ".tmp")
        tmp.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        os.replace(tmp, self.path)  # atomic: readers never see a torn file
        return self.path

    def get_spatial(self, label: str, unit_id: str) -> SpatialUnit | None:
        return self.spatial.get(label, {}).get(unit_id)

    def put_spatial(self, label: str, unit: SpatialUnit) -> None:
        if label not in self.spatial:
            raise KeyError(f"unknown spatial level: {label!r}")
        self.spatial[label][unit.id] = unit

    def get_temporal(self, level_name: str, bucket_id: str) -> TemporalUnit | None:
        return self.temporal.get(level_name, {}).get(bucket_id)

    def put_temporal(self, level_name: str, unit: TemporalUnit) -> None:
        if level_name not in self.temporal:
            raise KeyError(f"unknown temporal level: {level_name!r}")
        self.temporal[level_name][unit.id] = unit

    def prune(self, now: datetime.datetime) -> int:
        """Drop temporal buckets past their retention; returns the count removed."""
        tindex = TemporalIndex(self.config)
        removed = 0
        for name, units in self.temporal.items():
            retention = self.config.retention_days.get(name, 0)
            if retention is None or retention <= 0:
                continue
            for bucket_id in list(units):
                if tindex.is_expired(name, bucket_id, now, retention):
                    del units[bucket_id]
                    removed += 1
        return removed

    def effective_l0(self, cell_id: str, spatial_index) -> tuple[dict, bool]:
        """Read-time L0 view: stored text, else the nearest stored ancestor's
        environment/background/contrast with an ``unobserved`` marker."""
        stored = self.spatial.get("L0", {}).get(cell_id)
        if stored is not None and not stored.is_empty:
            view = stored.to_dict()
            view["unobserved"] = False
            return view, False
        col, row = spatial_index.parse_cell_id(cell_id)
        bounds = spatial_index.cell_bounds(0, col, row)
        level, c, r = 0, col, row
        while True:
            parent = spatial_index.parent(level, c, r)
            if parent is None:
                break
            level, c, r, pid = parent
            ancestor = self.spatial.get(spatial_index.level_label(level), {}).get(pid)
            if ancestor is not None and not ancestor.is_empty:
                return (
                    {
                        "id": cell_id,
                        "bounds": bounds,
                        "environment": ancestor.environment,
                        "background": ancestor.background,
                        "cat_contrast": ancestor.cat_contrast,
                        "activity": "",
                        "keywords": list(ancestor.keywords),
                        "unobserved": True,
                    },
                    True,
                )
        return (
            {
                "id": cell_id,
                "bounds": bounds,
                "environment": "",
                "background": "",
                "cat_contrast": "",
                "activity": "",
                "keywords": [],
                "unobserved": True,
            },
            True,
        )
