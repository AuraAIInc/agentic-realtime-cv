"""Short-term memory: interface + in-memory stub (full build is a later phase).

Same ``store``/``invoke`` shape as long-term memory, but with no persistence,
no LLM, and no worker: direct calls over a bounded deque of recent events
plus live per-cat box state.
"""

from __future__ import annotations

import datetime
import math
from abc import ABC, abstractmethod
from collections import deque

from agent.memory.models import MemoryContext, MemoryEvent, MemoryQuery


class ShortTermMemory(ABC):
    @abstractmethod
    def store(self, event: MemoryEvent) -> None:
        """Record one event (direct call, no queue)."""

    @abstractmethod
    def invoke(self, query: MemoryQuery) -> MemoryContext:
        """Answer from live state inside the horizon."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all live state."""


class InMemoryShortTermMemory(ShortTermMemory):
    def __init__(self, horizon_minutes: float = 15.0):
        self.horizon = datetime.timedelta(minutes=horizon_minutes)
        self._events: deque[MemoryEvent] = deque()
        self._last_box: dict[str, tuple] = {}
        self._prev_box: dict[str, tuple] = {}

    def store(self, event: MemoryEvent) -> None:
        key = str(event.cat_id)
        if event.box is not None:
            if key in self._last_box:
                self._prev_box[key] = self._last_box[key]
            self._last_box[key] = tuple(event.box)
        self._events.append(event)
        self._prune(event.time)

    def invoke(self, query: MemoryQuery) -> MemoryContext:
        self._prune(query.time)
        brief = f"{len(self._events)} recent events in short-term horizon"
        states = []
        for key in sorted(self._last_box):
            box = self._last_box[key]
            prev = self._prev_box.get(key)
            state = "resting"
            if prev is not None and _center_dist(prev, box) >= 15.0:
                state = "moving"
            states.append(f"cat {key}: {state}")
        if states:
            brief += "; " + ", ".join(states)
        return MemoryContext(brief=brief + ".")

    def reset(self) -> None:
        self._events.clear()
        self._last_box.clear()
        self._prev_box.clear()

    def _prune(self, now: datetime.datetime) -> None:
        while self._events and now - self._events[0].time > self.horizon:
            self._events.popleft()


def _center_dist(a: tuple, b: tuple) -> float:
    return math.dist(
        ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2),
        ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2),
    )
