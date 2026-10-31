"""Plain dataclasses for the spatial-temporal memory API surface."""

from __future__ import annotations

import datetime
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryEvent:
    time: datetime.datetime
    end_time: datetime.datetime | None = None
    view_id: int = 3
    box: tuple[float, float, float, float] | None = None
    cat_id: int | None = None
    period_id: int | None = None
    kind: str = "period"  # "period" | "escalation" | "label"
    text: str = ""
    code: str | None = None  # T1..T7 / P1..P7 when from escalation


@dataclass
class MemoryQuery:
    time: datetime.datetime
    box: tuple[float, float, float, float] | None = None
    cat_id: int | None = None
    spatial_radius: int = 1  # Chebyshev neighbor rings around the hit cell
    history_days: int = 7  # same-slot history across days
    text: str | None = None  # when set, blend in semantic_search hits


@dataclass
class MemoryHit:
    level: str  # "L0".."L2" or a temporal level name
    unit_id: str
    text: str
    score: float = 0.0
    unobserved: bool = False
    extra: dict = field(default_factory=dict)


@dataclass
class MemoryContext:
    spatial: list[MemoryHit] = field(default_factory=list)
    temporal: list[MemoryHit] = field(default_factory=list)
    semantic: list[MemoryHit] = field(default_factory=list)
    brief: str = ""


@dataclass
class JobReceipt:
    job_id: int
    kind: str  # "store" | "promote" | "bootstrap"
    enqueued_at: datetime.datetime
    error: Exception | None = field(default=None, compare=False, repr=False)
    result: Any = field(default=None, compare=False, repr=False)
    _done: threading.Event = field(default_factory=threading.Event, compare=False, repr=False)

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    def mark_done(self, error: Exception | None = None, result: Any = None) -> None:
        self.error = error
        self.result = result
        self._done.set()


@dataclass
class StoreResult:
    spatial_ids: list[str] = field(default_factory=list)
    temporal_ids: list[str] = field(default_factory=list)
    persisted: bool = True


@dataclass
class PromoteResult:
    promoted: int = 0
    deferred: int = 0
    errors: list[str] = field(default_factory=list)
