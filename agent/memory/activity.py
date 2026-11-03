"""Period activity aggregation for memory: combine, label, narrate.

Long-term memory attaches to *period activity*, never to individual
escalations. Consecutive periods of the same cat whose gap is under
``activity_combine_gap_minutes`` (default 5) merge into one
:class:`ActivityRecord`; lone short periods keep their own record (nothing
high-confidence is ever dropped for being short). A record is emitted once
it is closed (by a far-enough next period, or by :meth:`flush`) *and* every
member period has an activity label — :meth:`flush` backstops anything
still unlabeled with an unknown-activity record so nothing is lost.

Each member period's activity comes from one LLM look at its key frames
(see :class:`agent.validation.PeriodLabeler`), assigned to the activity set
in ``agent/agent_data/config/labels.yaml``. :meth:`ActivityRecord.narrative`
renders the shared shape for both memories: which cat, from what time to
what time, at which area (bounding box), doing what activity, what the cat
looks like, and what the background looks like.
"""

from __future__ import annotations

import datetime
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_LABELS_PATH = Path(__file__).resolve().parent.parent / "agent_data" / "config" / "labels.yaml"


def load_activity_labels(path: str | Path | None = None) -> dict:
    """Load the activity label set: ``{version, activity: [{name, description}]}``."""
    import yaml

    raw = yaml.safe_load(Path(path or DEFAULT_LABELS_PATH).read_text(encoding="utf-8")) or {}
    return {
        "version": raw.get("version"),
        "activity": [
            {"name": str(a.get("name")), "description": str(a.get("description") or "")}
            for a in raw.get("activity", []) or []
            if a.get("name")
        ],
    }


@dataclass
class ActivityLabel:
    """One period's LLM-judged activity (or a degraded unknown)."""

    activity: str | None  # labels.yaml activity name, or None when unknown
    confidence: float | None = None
    appearance: str = ""  # what the cat looks like in this period
    background: str = ""  # what the background looks like
    degraded: bool = False  # True when labeling failed/skipped (frames missing, LLM down)


@dataclass
class PeriodMember:
    period_id: int | None
    cat_id: int | None
    cat_name: str | None
    start: datetime.datetime
    end: datetime.datetime
    detections: int = 0
    mean_conf: float | None = None
    box: tuple[float, float, float, float] | None = None
    keyframes: list[str] = field(default_factory=list)
    label: ActivityLabel | None = None


def _median_box(boxes: list) -> tuple[float, float, float, float] | None:
    vals = [[float(v) for v in b] for b in boxes if b and len(b) == 4]
    if not vals:
        return None
    cols = sorted(vals, key=lambda b: (b[0], b[1], b[2], b[3]))
    mid = cols[len(cols) // 2]
    return (mid[0], mid[1], mid[2], mid[3])


def _spread(items: list, bound: int) -> list:
    if len(items) <= bound:
        return list(items)
    idx = [round(i * (len(items) - 1) / (bound - 1)) for i in range(bound)]
    return [items[i] for i in idx]


@dataclass
class ActivityRecord:
    """One combined memory record: same cat, contiguous activity."""

    members: list[PeriodMember] = field(default_factory=list)
    max_keyframes: int = 5

    @property
    def ready(self) -> bool:
        return bool(self.members) and all(m.label is not None for m in self.members)

    @property
    def cat_id(self) -> int | None:
        return self.members[0].cat_id if self.members else None

    @property
    def cat_name(self) -> str | None:
        return self.members[0].cat_name if self.members else None

    @property
    def start(self) -> datetime.datetime | None:
        return min((m.start for m in self.members), default=None)

    @property
    def end(self) -> datetime.datetime | None:
        return max((m.end for m in self.members), default=None)

    @property
    def period_ids(self) -> list:
        return [m.period_id for m in self.members]

    @property
    def detections(self) -> int:
        return sum(int(m.detections or 0) for m in self.members)

    @property
    def mean_conf(self) -> float | None:
        num = sum((m.mean_conf or 0.0) * (m.detections or 0) for m in self.members)
        den = sum(m.detections or 0 for m in self.members)
        if den <= 0:
            vals = [m.mean_conf for m in self.members if m.mean_conf is not None]
            return sum(vals) / len(vals) if vals else None
        return num / den

    @property
    def box(self):
        return _median_box([m.box for m in self.members])

    @property
    def keyframes(self) -> list[str]:
        seen: list[str] = []
        for m in self.members:
            for k in m.keyframes:
                if k not in seen:
                    seen.append(k)
        return _spread(seen, self.max_keyframes)

    def _activity_clause(self) -> str:
        acts = [(m.label.activity if m.label else None,
                 m.label.confidence if m.label else None, m.start, m.end)
                for m in self.members]
        names = {a for a, _, _, _ in acts}
        if len(names) == 1:
            name, conf, _, _ = acts[0]
            if name is None:
                return "doing unknown activity"
            if conf is None:
                return f"doing {name}"
            return f"doing {name} (confidence {conf:.2f})"
        parts = []
        for name, _conf, start, end in acts:
            span = f"{start.strftime('%H:%M')}–{end.strftime('%H:%M')}"
            parts.append(f"{name or 'unknown activity'} {span}")
        return "doing " + ", then ".join(parts)

    def _first_nonempty(self, attr: str) -> str:
        seen: list[str] = []
        for m in self.members:
            val = (getattr(m.label, attr, "") if m.label else "") or ""
            if val and val not in seen:
                seen.append(val)
        return "; ".join(seen)

    def narrative(self) -> str:
        """Which cat, what span, which area, what activity, looks, background."""
        who = (
            f"{self.cat_name} (cat {self.cat_id})"
            if self.cat_id is not None
            else "no cat assigned"
        )
        span = f"{self.start.strftime('%H:%M')} to {self.end.strftime('%H:%M')}"
        at = f" at [{', '.join(f'{v:.0f}' for v in self.box)}]" if self.box else ""
        pids = "+".join(str(p) for p in self.period_ids if p is not None)
        prove = f" (period{'s' if len(self.members) > 1 else ''} {pids}, {self.detections} sightings" if pids else f" ({self.detections} sightings"
        conf = self.mean_conf
        prove += f", mean detector confidence {conf:.2f}" if conf is not None else ""
        prove += ")"
        text = f"{who} from {span}{at}{prove}: {self._activity_clause()}."
        looks = self._first_nonempty("appearance")
        if looks:
            text += f" Looks like: {looks}."
        bg = self._first_nonempty("background")
        if bg:
            text += f" Background: {bg}."
        return text


class PeriodActivityAggregator:
    """Combine same-cat periods across small gaps; emit when labeled.

    ``add`` feeds one closed period and returns newly completed records;
    ``set_label`` attaches a period's activity label and returns records it
    completes; ``flush`` closes everything and emits all (unlabeled members
    degrade to unknown activity so nothing is lost). Returned records are
    always fully labeled.
    """

    def __init__(self, gap_minutes: float = 5.0, max_keyframes: int = 5):
        self._gap_s = float(gap_minutes) * 60.0
        self._max_keyframes = max(1, int(max_keyframes))
        self._open: dict = {}  # cat key -> ActivityRecord still growing
        self._waiting: list[ActivityRecord] = []  # closed, pending labels
        # add() runs on the tracking thread, set_label() on the labeler thread.
        self._mu = threading.RLock()

    @staticmethod
    def _key(cat_id) -> tuple:
        return ("cat", cat_id)

    def add(self, member: PeriodMember) -> list[ActivityRecord]:
        """Feed one closed period (chronological order); emit completed records."""
        with self._mu:
            return self._add_locked(member)

    def _add_locked(self, member: PeriodMember) -> list[ActivityRecord]:
        if member.label is None and not member.detections:
            member.label = ActivityLabel(activity=None, degraded=True)  # nothing to look at
        emitted: list[ActivityRecord] = []
        key = self._key(member.cat_id)
        cur = self._open.get(key)
        if cur is not None and cur.members:
            if any(m.period_id == member.period_id for m in cur.members):
                return emitted  # duplicate feed; ignore
            gap = (member.start - cur.end).total_seconds()
            if gap < self._gap_s:
                cur.members.append(member)
                return emitted
            del self._open[key]
            self._close(cur, emitted)
        self._open[key] = ActivityRecord([member], max_keyframes=self._max_keyframes)
        return emitted

    def set_label(self, period_id: int | None, label: ActivityLabel) -> list[ActivityRecord]:
        """Attach a period's activity label; emit records it completes."""
        with self._mu:
            return self._set_label_locked(period_id, label)

    def _set_label_locked(self, period_id: int | None, label: ActivityLabel) -> list[ActivityRecord]:
        emitted: list[ActivityRecord] = []
        for record in list(self._open.values()) + self._waiting:
            for member in record.members:
                if member.period_id == period_id and member.label is None:
                    member.label = label
        still_waiting: list[ActivityRecord] = []
        for record in self._waiting:
            if record.ready:
                emitted.append(record)
            else:
                still_waiting.append(record)
        self._waiting = still_waiting
        return emitted

    def flush(self) -> list[ActivityRecord]:
        """Close all open records and emit everything (unknown when unlabeled)."""
        with self._mu:
            return self._flush_locked()

    def _flush_locked(self) -> list[ActivityRecord]:
        emitted: list[ActivityRecord] = []
        for record in self._open.values():
            self._close(record, emitted)
        self._open = {}
        for record in self._waiting:
            for member in record.members:
                if member.label is None:
                    member.label = ActivityLabel(activity=None, degraded=True)
            emitted.append(record)
        self._waiting = []
        return emitted

    def _close(self, record: ActivityRecord, emitted: list[ActivityRecord]) -> None:
        if record.ready:
            emitted.append(record)
        else:
            self._waiting.append(record)

    @property
    def pending_periods(self) -> int:
        """Members held (open or waiting) that flush() would emit."""
        return sum(len(r.members) for r in list(self._open.values()) + self._waiting)


__all__ = [
    "ActivityLabel",
    "ActivityRecord",
    "PeriodActivityAggregator",
    "PeriodMember",
    "load_activity_labels",
]
