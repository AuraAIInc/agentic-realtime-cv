"""Daily short-term memory: every period activity across one day, in plain language.

Where long-term memory keeps only high-confidence knowledge, short-term memory
remembers the full day: every closed period (all confidences) with its
detections and a few key frames, plus tracker episodes (escalations) and their
validation verdicts. Entries read like "Qb stayed in the rug area from 16:34
to 16:40 (772 sightings, mean confidence 0.71)".

Persistence is two files per view/day (see ``MemoryConfig.short_term_path``):

- ``view_<id>_short_term_<YYYY-MM-DD>.md`` — the plain-language day narrative,
  re-rendered on every change;
- ``view_<id>_short_term_<YYYY-MM-DD>.json`` — machine state (sessions,
  entries, merge markers) the Markdown is rendered from.

Lifecycle: a new session starts a new ``## Session`` section (live entries
reset; the day file accumulates). When event time crosses midnight, the old
day is merged into long-term memory (curated ``summary`` events that promote
into the top spatial levels and the day bucket) and a new day starts. Session
end merges the session's entries the same way; merged sessions are marked so
nothing merges twice.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


def _fmt_span(start: datetime.datetime, end: datetime.datetime) -> str:
    return f"{start.strftime('%H:%M')}–{end.strftime('%H:%M')}"


@dataclass
class PeriodEntry:
    period_id: int | None
    cat_id: int | None
    cat_name: str | None
    start: datetime.datetime
    end: datetime.datetime
    detections: int
    mean_conf: float | None
    box: tuple[float, float, float, float] | None = None
    keyframes: list[str] = field(default_factory=list)
    # Combined activity record: which member periods merged here, what the cat
    # was doing (labels.yaml activity judged from the key frames), what it
    # looked like, and what the background looked like.
    period_ids: list[int] = field(default_factory=list)
    activity: str | None = None
    activity_confidence: float | None = None
    appearance: str = ""
    background: str = ""

    @property
    def key(self) -> str:
        return f"{self.period_id}@{self.start.strftime('%Y-%m-%dT%H:%M')}"

    def narrative(self) -> str:
        span = _fmt_span(self.start, self.end)
        who = f"{self.cat_name} (cat {self.cat_id})" if self.cat_id is not None else "no cat assigned"
        conf = f", mean confidence {self.mean_conf:.2f}" if self.mean_conf is not None else ""
        area = f" at {list(self.box)}" if self.box else ""
        if self.activity:
            act_conf = f" (confidence {self.activity_confidence:.2f})" if self.activity_confidence is not None else ""
            doing = f" Activity: {self.activity}{act_conf}."
        else:
            doing = ""
        looks = f" Looks: {self.appearance}." if self.appearance else ""
        scene = f" Background: {self.background}." if self.background else ""
        frames = f" Key frames: {', '.join(self.keyframes)}." if self.keyframes else ""
        return (
            f"{span}: {who}{area} — {self.detections} sightings{conf}."
            f"{doing}{looks}{scene}"
            f"{frames}"
        )

    def to_dict(self) -> dict:
        return {
            "period_id": self.period_id,
            "cat_id": self.cat_id,
            "cat_name": self.cat_name,
            "start": self.start.isoformat(timespec="seconds"),
            "end": self.end.isoformat(timespec="seconds"),
            "detections": self.detections,
            "mean_conf": self.mean_conf,
            "box": list(self.box) if self.box else None,
            "keyframes": list(self.keyframes),
            "period_ids": list(self.period_ids),
            "activity": self.activity,
            "activity_confidence": self.activity_confidence,
            "appearance": self.appearance,
            "background": self.background,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PeriodEntry":
        box = data.get("box")
        pids = data.get("period_ids") or []
        if not pids and data.get("period_id") is not None:
            pids = [data["period_id"]]
        return cls(
            period_id=data.get("period_id"),
            cat_id=data.get("cat_id"),
            cat_name=data.get("cat_name"),
            start=datetime.datetime.fromisoformat(data["start"]),
            end=datetime.datetime.fromisoformat(data["end"]),
            detections=int(data.get("detections", 0)),
            mean_conf=data.get("mean_conf"),
            box=tuple(box) if box else None,
            keyframes=[str(k) for k in data.get("keyframes", []) or []],
            period_ids=[int(p) for p in pids],
            activity=data.get("activity"),
            activity_confidence=data.get("activity_confidence"),
            appearance=str(data.get("appearance") or ""),
            background=str(data.get("background") or ""),
        )


@dataclass
class EpisodeEntry:
    name: str  # E<seq>-<code>
    code: str
    time: datetime.datetime
    reason: str
    cat_id: int | None = None
    period_id: int | None = None
    image_names: list[str] = field(default_factory=list)
    verdict: str | None = None
    rationale: str | None = None

    def narrative(self) -> str:
        base = f"{self.time.strftime('%H:%M:%S')} [{self.name}]: {self.reason}"
        if self.verdict:
            base += f" Validation verdict: {self.verdict}."
            if self.rationale:
                base += f" {self.rationale}"
        return base

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "code": self.code,
            "time": self.time.isoformat(timespec="seconds"),
            "reason": self.reason,
            "cat_id": self.cat_id,
            "period_id": self.period_id,
            "image_names": list(self.image_names),
            "verdict": self.verdict,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodeEntry":
        return cls(
            name=str(data["name"]),
            code=str(data.get("code") or ""),
            time=datetime.datetime.fromisoformat(data["time"]),
            reason=str(data.get("reason") or ""),
            cat_id=data.get("cat_id"),
            period_id=data.get("period_id"),
            image_names=[str(k) for k in data.get("image_names", []) or []],
            verdict=data.get("verdict"),
            rationale=data.get("rationale"),
        )


class DailyShortTermMemory:
    """One day of full-fidelity activity; merges into long-term at day end."""

    def __init__(self, config, view_id: int, long_term=None):
        self.config = config
        self.view_id = int(view_id)
        self.long_term = long_term
        self.day: datetime.date | None = None
        self.session_name: str = "session"
        self.session_started: datetime.datetime | None = None
        # day state (persisted to the .json sidecar, rendered to .md)
        self.sessions: dict[str, dict] = {}
        self.merged_sessions: set[str] = set()
        # Periods arrive from the labeler thread while episodes/verdicts arrive
        # from the tracking/validation threads; every mutation + render holds this.
        self._mu = threading.RLock()

    # -- session lifecycle --

    def start_session(self, now: datetime.datetime, session_name: str) -> None:
        """Begin a session: roll the day over at midnight, else open a section."""
        with self._mu:
            if self.day is not None and now.date() != self.day and self.long_term is not None:
                self.rollover(now)
            if self.day != now.date():
                self.day = now.date()
                self.sessions = {}
                self.merged_sessions = set()
                self._load_day()
            self.session_name = session_name
            self.session_started = now
            self.sessions.setdefault(session_name, {"started": now.isoformat(timespec="seconds"), "periods": [], "episodes": []})
            self._save()

    def rollover(self, now: datetime.datetime) -> datetime.date | None:
        """Merge the open day into long-term memory when midnight passed.

        Returns the merged day, or None when no rollover happened.
        """
        with self._mu:
            if self.day is None or now.date() == self.day:
                return None
            merged = self.day
            if self.long_term is not None:
                self.merge_into(self.long_term)
            self.day = now.date()
            self.sessions = {}
            self.merged_sessions = set()
            self._load_day()
            self._save()
            logger.info("short-term memory rolled over: merged %s, starting %s", merged, self.day)
            return merged

    # -- ingest (period detections + key frames; episodes; verdicts) --

    def record_period(
        self,
        *,
        period_id: int | None,
        cat_id: int | None,
        cat_name: str | None,
        start: datetime.datetime,
        end: datetime.datetime,
        detections: int,
        mean_conf: float | None,
        box=None,
        keyframes: list[str] | None = None,
        period_ids: list[int] | None = None,
        activity: str | None = None,
        activity_confidence: float | None = None,
        appearance: str = "",
        background: str = "",
    ) -> PeriodEntry | None:
        """Record one closed (possibly combined) period. Duplicates are skipped."""
        with self._mu:
            self._ensure_session(start)
            entry = PeriodEntry(
                period_id=period_id, cat_id=cat_id, cat_name=cat_name,
                start=start, end=end, detections=detections, mean_conf=mean_conf,
                box=tuple(box) if box else None, keyframes=list(keyframes or []),
                period_ids=list(period_ids) if period_ids else ([period_id] if period_id is not None else []),
                activity=activity, activity_confidence=activity_confidence,
                appearance=appearance or "", background=background or "",
            )
            periods = self.sessions[self.session_name]["periods"]
            if any(p["period_id"] == entry.period_id and p["start"] == entry.to_dict()["start"] for p in periods):
                return None
            periods.append(entry.to_dict())
            self._save()
            logger.info("short-term memory recorded period %s: %s", period_id, entry.narrative())
            return entry

    def record_episode(
        self,
        *,
        name: str,
        code: str,
        time: datetime.datetime,
        reason: str,
        cat_id: int | None = None,
        period_id: int | None = None,
        image_names: list[str] | None = None,
    ) -> EpisodeEntry | None:
        """Record one tracker episode (escalation occurrence)."""
        with self._mu:
            self._ensure_session(time)
            episodes = self.sessions[self.session_name]["episodes"]
            if any(e["name"] == name for e in episodes):
                return None
            entry = EpisodeEntry(
                name=name, code=code, time=time, reason=reason,
                cat_id=cat_id, period_id=period_id, image_names=list(image_names or []),
            )
            episodes.append(entry.to_dict())
            self._save()
            return entry

    def record_verdict(self, name: str, verdict: str, rationale: str | None = None) -> bool:
        """Attach a validation verdict (+ optional rationale) to an episode. False if unknown."""
        with self._mu:
            for session in self.sessions.values():
                for raw in session["episodes"]:
                    if raw["name"] == name:
                        raw["verdict"] = verdict
                        if rationale:
                            raw["rationale"] = rationale
                        self._save()
                        return True
            return False

    # -- day-end merge into long-term memory --

    def merge_into(self, long_term) -> list:
        """Merge unmerged sessions as curated day summaries; mark them merged.

        One full-day narrative (temporal path: promotes into the day bucket)
        plus one summary per active zone (spatial path: promotes into the top
        spatial levels). Returns the store receipts.
        """
        from agent.memory.models import MemoryEvent

        with self._mu:
            return self._merge_into_locked(long_term)

    def _merge_into_locked(self, long_term) -> list:
        from agent.memory.models import MemoryEvent

        receipts = []
        pending = [s for s in self.sessions if s not in self.merged_sessions]
        if not pending or self.day is None:
            return receipts
        day_start = datetime.datetime(self.day.year, self.day.month, self.day.day)
        for session_name in pending:
            session = self.sessions[session_name]
            periods = [PeriodEntry.from_dict(p) for p in session["periods"]]
            episodes = [EpisodeEntry.from_dict(e) for e in session["episodes"]]
            if not periods and not episodes:
                self.merged_sessions.add(session_name)
                continue
            confs = [p.mean_conf for p in periods if p.mean_conf is not None]
            confidence = sum(confs) / len(confs) if confs else 0.5
            keyframes: list[str] = []
            for p in periods:
                for kf in p.keyframes:
                    if kf not in keyframes:
                        keyframes.append(kf)
            # Full-day narrative -> temporal day bucket (via promote).
            receipts.append(
                long_term.store(
                    MemoryEvent(
                        time=day_start,
                        end_time=day_start + datetime.timedelta(days=1),
                        view_id=self.view_id,
                        kind="summary",
                        text=self._day_narrative(periods, episodes),
                        confidence=confidence,
                        image_names=keyframes[:5],
                        detection={"periods": len(periods), "episodes": len(episodes)},
                    )
                )
            )
            # Per-zone summaries -> top spatial levels (via promote).
            for zone_box, zone_periods in self._group_by_zone(long_term, periods).items():
                zconfs = [p.mean_conf for p in zone_periods if p.mean_conf is not None]
                zframes: list[str] = []
                for p in zone_periods:
                    for kf in p.keyframes:
                        if kf not in zframes:
                            zframes.append(kf)
                receipts.append(
                    long_term.store(
                        MemoryEvent(
                            time=zone_periods[0].start,
                            end_time=zone_periods[-1].end,
                            view_id=self.view_id,
                            box=zone_box,
                            cat_id=zone_periods[0].cat_id,
                            kind="summary",
                            text=self._zone_narrative(zone_periods),
                            confidence=sum(zconfs) / len(zconfs) if zconfs else 0.5,
                            image_names=zframes[:5],
                            detection={"periods": len(zone_periods)},
                        )
                    )
                )
            self.merged_sessions.add(session_name)
        self._save()
        logger.info(
            "short-term memory merged %d session(s) of %s into long-term memory",
            len(pending), self.day,
        )
        return receipts

    # -- rendering + persistence --

    def render(self) -> str:
        """The full day narrative as plain-language Markdown."""
        with self._mu:
            return self._render_locked()

    def _render_locked(self) -> str:
        if self.day is None:
            return "# Short-term memory — no day open\n"
        lines = [
            f"# Short-term memory — view {self.view_id} — {self.day.isoformat()}",
            "",
            f"Sessions: {', '.join(self.sessions) or '(none)'}",
            f"Merged into long-term memory: {', '.join(sorted(self.merged_sessions)) or '(none yet)'}",
            "",
        ]
        for name, session in self.sessions.items():
            lines.append(f"## Session {name} (started {session['started']})")
            lines.append("")
            lines.append("### Periods")
            periods = session["periods"]
            if not periods:
                lines.append("- (none)")
            for raw in periods:
                lines.append(f"- {PeriodEntry.from_dict(raw).narrative()}")
            lines.append("")
            lines.append("### Episodes")
            episodes = session["episodes"]
            if not episodes:
                lines.append("- (none)")
            for raw in episodes:
                lines.append(f"- {EpisodeEntry.from_dict(raw).narrative()}")
            lines.append("")
        return "\n".join(lines)

    @property
    def _paths(self) -> tuple[Path, Path]:
        md = self.config.short_term_path(self.day, self.view_id)
        return md, md.with_suffix(".json")

    def _save(self) -> None:
        if self.day is None:
            return
        md_path, json_path = self._paths
        md_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "view_id": self.view_id,
            "day": self.day.isoformat(),
            "sessions": self.sessions,
            "merged_sessions": sorted(self.merged_sessions),
        }
        tmp = json_path.parent / (json_path.name + ".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(tmp, json_path)
        md_path.write_text(self.render(), encoding="utf-8")

    def _load_day(self) -> None:
        _, json_path = self._paths
        if not json_path.exists():
            return
        try:
            state = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("short-term state for %s unreadable; starting fresh", self.day)
            return
        if state.get("day") != self.day.isoformat():
            return
        self.sessions = state.get("sessions") or {}
        self.merged_sessions = set(state.get("merged_sessions") or [])

    def _ensure_session(self, when: datetime.datetime) -> None:
        if self.day is None:
            self.start_session(when, self.session_name)
        elif when.date() != self.day:
            self.rollover(when)
            self.start_session(when, self.session_name)

    # -- narrative builders --

    def _day_narrative(self, periods: list[PeriodEntry], episodes: list[EpisodeEntry]) -> str:
        parts = [f"On {self.day.isoformat()}, "]
        if periods:
            by_cat: dict[str, list[PeriodEntry]] = {}
            for p in periods:
                by_cat.setdefault(p.cat_name or "unassigned", []).append(p)
            segs = []
            for cat, ps in sorted(by_cat.items()):
                first, last = ps[0].start, ps[-1].end
                total = sum(p.detections for p in ps)
                acts = sorted({p.activity for p in ps if p.activity})
                doing = f", seen {', '.join(acts)}" if acts else ""
                segs.append(
                    f"{cat} visited {len(ps)} time(s) from {first.strftime('%H:%M')} "
                    f"to {last.strftime('%H:%M')} ({total} sightings{doing})"
                )
            parts.append("; ".join(segs) + ". ")
        else:
            parts.append("no visits were recorded. ")
        if episodes:
            resolved = sum(1 for e in episodes if e.verdict)
            parts.append(
                f"The tracker raised {len(episodes)} episode(s), "
                f"{resolved} resolved by validation. "
            )
            for e in episodes[:3]:
                parts.append(e.narrative() + " ")
        return "".join(parts).strip()

    def _zone_narrative(self, periods: list[PeriodEntry]) -> str:
        first, last = periods[0].start, periods[-1].end
        cats = sorted({p.cat_name or "unassigned" for p in periods})
        total = sum(p.detections for p in periods)
        acts = sorted({p.activity for p in periods if p.activity})
        doing = f" Cats here were seen {', '.join(acts)}." if acts else ""
        return (
            f"Zone active {_fmt_span(first, last)}: {', '.join(cats)} seen here "
            f"in {len(periods)} visit(s), {total} sightings in total.{doing}"
        )

    def _group_by_zone(self, long_term, periods: list[PeriodEntry]) -> dict:
        """Group periods by L0 cell; key is a representative box per cell."""
        groups: dict[str, list[PeriodEntry]] = {}
        boxes: dict[str, tuple] = {}
        for p in periods:
            if p.box is None:
                groups.setdefault("nobox", []).append(p)
                continue
            try:
                hit = long_term.spatial.box_to_cell(p.box, 0)
            except Exception:
                hit = None
            cid = hit[2] if hit else "nobox"
            groups.setdefault(cid, []).append(p)
            boxes.setdefault(cid, tuple(p.box) if p.box else None)
        out = {}
        for cid, ps in groups.items():
            if boxes.get(cid) is None:
                continue  # temporal-only day narrative already covers boxless periods
            out[boxes[cid]] = ps
        return out
