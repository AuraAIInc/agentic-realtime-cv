"""Per-request timing attached to every tool response."""

from __future__ import annotations

import time


class Timer:
    """Records the duration of each named step plus the total, in milliseconds."""

    def __init__(self):
        self._t0 = time.perf_counter()
        self._last = self._t0
        self.steps: dict[str, float] = {}

    def mark(self, step: str) -> None:
        now = time.perf_counter()
        self.steps[step] = round((now - self._last) * 1000, 3)
        self._last = now

    def report(self) -> dict:
        return {
            "elapsed_ms": round((time.perf_counter() - self._t0) * 1000, 3),
            "steps_ms": dict(self.steps),
        }
