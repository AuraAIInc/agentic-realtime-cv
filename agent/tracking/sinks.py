"""Persistence sinks for the tracking correctors (Phase 2 of 5).

The legacy correctors wrote straight to MySQL (or skipped writes with
``save_db=False``). Here persistence is a small injected ``Sink``: production
code passes a DB-backed sink, tests and the visualizer use ``InMemorySink``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class Sink(Protocol):
    """Where corrector rows go. All methods accept plain row lists/tuples."""

    def save_revised(self, row: list) -> None:
        """One revised_detection row (9 detection fields + view_id)."""
        ...

    def save_original(self, rows: list) -> None:
        """Raw detection rows for one frame (each: create_time, cat_id, box..., conf, model_id, view_id)."""
        ...

    def save_image(self, row: tuple) -> None:
        """One images row: (create_time, detected, is_grey, view_id, device_id)."""
        ...

    def delete_revised(self, row: list) -> None:
        """Retract a previously saved revised row (T6 teleport split)."""
        ...

    def save_period(self, row: list) -> None:
        """One fully-computed period row (UPDATE payload + period_id last)."""
        ...

    def flush(self) -> None:
        """Persist any buffered rows; no-op for in-memory sinks."""
        ...


@dataclass
class InMemorySink:
    """Collects rows in lists. Replaces ``save_db=False`` in-memory semantics."""

    revised: list = field(default_factory=list)
    original: list = field(default_factory=list)
    images: list = field(default_factory=list)
    periods: list = field(default_factory=list)
    deleted_revised: list = field(default_factory=list)

    def save_revised(self, row: list) -> None:
        self.revised.append(row)

    def save_original(self, rows: list) -> None:
        self.original.extend(rows)

    def save_image(self, row: tuple) -> None:
        self.images.append(row)

    def delete_revised(self, row: list) -> None:
        self.deleted_revised.append(row)
        for k, saved in enumerate(self.revised):
            if list(saved[:9]) == list(row[:9]):
                self.revised.pop(k)
                return

    def save_period(self, row: list) -> None:
        self.periods.append(row)

    def flush(self) -> None:
        return None


__all__ = ["Sink", "InMemorySink"]
