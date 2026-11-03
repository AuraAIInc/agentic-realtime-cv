"""Merge temporally close escalations into joint validation batches.

The tracking run produces escalations one at a time; the validation worker
consumes them one batch at a time. When a new escalation arrives within
``window_s`` of the tail waiting batch's span — and the batch is under its
member and prompt-char caps — it joins that batch instead of starting a new
one, so one supervisor-agent call can resolve several related cases. Batches
the worker already took are in flight and never gain members.

Merging is purely a queue concern: downstream (journal, apply, memory) still
sees one op per escalation, in arrival order.
"""

from __future__ import annotations

import datetime
import logging
import threading
from collections import deque
from dataclasses import dataclass, field

from agent.escalation import esc_label

logger = logging.getLogger(__name__)


@dataclass
class EscalationBatch:
    """Waiting batch: members in arrival order plus their joint span/size."""

    members: list = field(default_factory=list)
    start: datetime.datetime | None = None
    end: datetime.datetime | None = None
    chars: int = 0


def _span_of(esc) -> tuple[datetime.datetime | None, datetime.datetime | None]:
    start = getattr(esc, "time", None)
    end = getattr(esc, "end_time", None) or start
    return start, end


def _span_gap(a, b) -> datetime.timedelta | None:
    """Gap between two spans; zero when they overlap, None when unspanned."""
    (s1, e1), (s2, e2) = a, b
    if s1 is None or e1 is None or s2 is None or e2 is None:
        return None
    return max(datetime.timedelta(0), max(s1, s2) - min(e1, e2))


class EscalationBatcher:
    """Thread-safe batching queue: ``put`` (producer), ``take`` (consumer), ``close``.

    ``window_s`` is the max span gap for joining the tail batch,
    ``max_batch`` caps members per batch, ``max_chars`` caps the summed
    ``size_of`` prompt sizes (plus per-case section overhead). ``size_of``
    defaults to a cheap ``len(reason)`` estimate; the worker passes the real
    prompt length. A batch that the worker took is in flight — later puts
    can only join batches still waiting.
    """

    _SECTION_OVERHEAD = 120  # "=== Case i of n: ... ===" framing per member

    def __init__(self, window_s: float = 60.0, max_batch: int = 4, max_chars: int = 12_000, size_of=None):
        self._window = datetime.timedelta(seconds=window_s)
        self._max_batch = max(1, max_batch)
        self._max_chars = max(1, max_chars)
        self._size_of = size_of or (lambda esc: len(getattr(esc, "reason", "") or ""))
        self._batches: deque[EscalationBatch] = deque()
        self._closed = False
        self._cond = threading.Condition()

    def put(self, esc) -> None:
        """Queue one escalation, merging into the tail batch when close enough."""
        try:
            size = self._size_of(esc)
        except Exception:
            size = 0  # sizing must never break tracking; the count cap still bounds
        label = esc_label(esc)
        with self._cond:
            if self._closed:
                logger.warning("escalation %s arrived after queue close; dropping it", label)
                return
            tail = self._batches[-1] if self._batches else None
            gap = _span_gap((tail.start, tail.end), _span_of(esc)) if tail is not None else None
            if (
                tail is not None
                and gap is not None
                and gap <= self._window
                and len(tail.members) < self._max_batch
                and tail.chars + size + self._SECTION_OVERHEAD <= self._max_chars
            ):
                tail.members.append(esc)
                start, end = _span_of(esc)
                tail.start = min(t for t in (tail.start, start) if t is not None)
                tail.end = max(t for t in (tail.end, end) if t is not None)
                tail.chars += size + self._SECTION_OVERHEAD
                logger.info(
                    "merged %s into batch [%s] (gap %.1f s)",
                    label,
                    ", ".join(esc_label(m) for m in tail.members[:-1]),
                    gap.total_seconds(),
                )
            else:
                start, end = _span_of(esc)
                self._batches.append(EscalationBatch([esc], start, end, size + self._SECTION_OVERHEAD))
            self._cond.notify_all()

    def take(self) -> EscalationBatch | None:
        """Block for the head batch; None once closed and drained."""
        with self._cond:
            while not self._batches and not self._closed:
                self._cond.wait()
            if not self._batches:
                return None
            return self._batches.popleft()

    def close(self) -> None:
        """Stop accepting; queued batches still drain, then ``take`` gives None."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def queued_batches(self) -> int:
        with self._cond:
            return len(self._batches)


__all__ = ["EscalationBatch", "EscalationBatcher"]
