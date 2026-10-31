"""1D hierarchical time-bucket math: slot/hour/day/week ids, history, expiry."""

from __future__ import annotations

import datetime
import re

from agent.memory.memory_config import MemoryConfig

_WEEK_RE = re.compile(r"^(\d{4})-W(\d{2})$")


class TemporalIndex:
    """Pure bucket math over the configured temporal levels (finest first)."""

    def __init__(self, config: MemoryConfig):
        self._levels = [(lv.name, lv.minutes) for lv in config.temporal_levels]

    @property
    def level_names(self) -> list[str]:
        return [name for name, _ in self._levels]

    def minutes(self, level_name: str) -> int:
        for name, minutes in self._levels:
            if name == level_name:
                return minutes
        raise KeyError(f"unknown temporal level: {level_name!r}")

    def _level_index(self, level_name: str) -> int:
        try:
            return self.level_names.index(level_name)
        except ValueError:
            raise KeyError(f"unknown temporal level: {level_name!r}") from None

    def _slot_minutes(self) -> int:
        return self._levels[0][1]

    def slot_id(self, dt: datetime.datetime) -> str:
        floored = (dt.minute // self._slot_minutes()) * self._slot_minutes()
        return dt.replace(minute=floored, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M")

    @staticmethod
    def hour_id(dt: datetime.datetime) -> str:
        return dt.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:00")

    @staticmethod
    def day_id(dt: datetime.datetime) -> str:
        return dt.strftime("%Y-%m-%d")

    @staticmethod
    def week_id(dt: datetime.datetime) -> str:
        iso = dt.isocalendar()  # Monday-anchored ISO week
        return f"{iso.year}-W{iso.week:02d}"

    def bucket_id(self, level_name: str, dt: datetime.datetime) -> str:
        idx = self._level_index(level_name)
        fns = (self.slot_id, self.hour_id, self.day_id, self.week_id)
        if idx >= len(fns):
            raise KeyError(f"no bucketing rule for level: {level_name!r}")
        return fns[idx](dt)

    def bucket_start(self, level_name: str, bucket_id: str) -> datetime.datetime:
        idx = self._level_index(level_name)
        if idx in (0, 1):
            return datetime.datetime.strptime(bucket_id, "%Y-%m-%dT%H:%M")
        if idx == 2:
            return datetime.datetime.strptime(bucket_id, "%Y-%m-%d")
        if idx == 3:
            m = _WEEK_RE.match(bucket_id)
            if not m:
                raise ValueError(f"bad week id: {bucket_id!r}")
            monday = datetime.date.fromisocalendar(int(m.group(1)), int(m.group(2)), 1)
            return datetime.datetime(monday.year, monday.month, monday.day)
        raise KeyError(f"no bucketing rule for level: {level_name!r}")

    def child_ids(self, level_name: str, bucket_id: str) -> list[str]:
        """Bucket ids one level down (hour -> its slots, day -> its hours, ...)."""
        idx = self._level_index(level_name)
        start = self.bucket_start(level_name, bucket_id)
        if idx == 0:
            return []
        if idx == 1:
            n = 60 // self._slot_minutes()
            step = datetime.timedelta(minutes=self._slot_minutes())
            return [self.slot_id(start + k * step) for k in range(n)]
        if idx == 2:
            return [self.hour_id(start + datetime.timedelta(hours=k)) for k in range(24)]
        if idx == 3:
            return [self.day_id(start + datetime.timedelta(days=k)) for k in range(7)]
        raise KeyError(f"no bucketing rule for level: {level_name!r}")

    def same_slot_history(self, dt: datetime.datetime, days: int) -> list[str]:
        """Same time-of-day slot for the current day plus the previous ``days``.

        Returns ``days + 1`` ids, oldest first, ending with ``slot_id(dt)``.
        """
        return [
            self.slot_id(dt - datetime.timedelta(days=d)) for d in range(days, -1, -1)
        ]

    def history_ids(self, level_name: str, dt: datetime.datetime, days: int) -> list[str]:
        """Same-bucket history across days (weeks: current + previous ``days // 7``)."""
        idx = self._level_index(level_name)
        if idx == 0:
            return self.same_slot_history(dt, days)
        if idx == 1:
            return [
                self.hour_id(dt - datetime.timedelta(days=d))
                for d in range(days, -1, -1)
            ]
        if idx == 2:
            return [
                self.day_id(dt - datetime.timedelta(days=d)) for d in range(days, -1, -1)
            ]
        n_prev = max(1, days // 7)
        return [
            self.week_id(dt - datetime.timedelta(weeks=w))
            for w in range(n_prev, -1, -1)
        ]

    def is_expired(
        self,
        level_name: str,
        bucket_id: str,
        now: datetime.datetime,
        retention: dict[str, int] | int,
    ) -> bool:
        """True when a bucket outlives its retention; retention <= 0 keeps forever."""
        ret = retention.get(level_name, 0) if isinstance(retention, dict) else int(retention)
        if ret <= 0:
            return False
        age = (now - self.bucket_start(level_name, bucket_id)).total_seconds()
        return age > ret * 86400
