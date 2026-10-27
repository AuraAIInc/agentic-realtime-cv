"""First/last sighting of each tracker ID, so the supervisor can re-check the
frame where a track first appeared.

Tracker IDs restart whenever the tracker is reset (model or ROI change), so each
reset starts a new `epoch` and entries are keyed by (epoch, track_id).
"""

from __future__ import annotations

import datetime
import threading
from collections import OrderedDict


class TrackHistory:
    def __init__(self, max_tracks: int = 500):
        self.max_tracks = max_tracks
        self.epoch = 0
        self._tracks: "OrderedDict[tuple[int, int], dict]" = OrderedDict()
        self._lock = threading.Lock()

    def new_epoch(self) -> int:
        with self._lock:
            self.epoch += 1
            return self.epoch

    def update(self, t: datetime.datetime, detections: list[dict]) -> None:
        ts = t.isoformat(timespec="milliseconds")
        with self._lock:
            for d in detections:
                tid = d.get("track_id")
                if tid is None:
                    continue
                key = (self.epoch, tid)
                rec = self._tracks.get(key)
                if rec is None:
                    rec = {
                        "epoch": self.epoch,
                        "track_id": tid,
                        "first_seen": ts,
                        "first_box": d["box"],
                        "first_cat_id": d["cat_id"],
                        "hits": 0,
                        "cat_id_counts": {},
                        "max_confidence": 0.0,
                    }
                    self._tracks[key] = rec
                rec["last_seen"] = ts
                rec["last_box"] = d["box"]
                rec["hits"] += 1
                rec["max_confidence"] = max(rec["max_confidence"], d["confidence"])
                c = str(d["cat_id"])
                rec["cat_id_counts"][c] = rec["cat_id_counts"].get(c, 0) + 1
                self._tracks.move_to_end(key)
            while len(self._tracks) > self.max_tracks:
                self._tracks.popitem(last=False)

    def get(self, track_id: int, epoch: int | None = None) -> dict | None:
        with self._lock:
            rec = self._tracks.get((self.epoch if epoch is None else epoch, track_id))
            return dict(rec) if rec else None

    def snapshot(
        self, limit: int = 50, current_epoch_only: bool = True, at: datetime.datetime | None = None, window_seconds: float = 2.0
    ) -> list[dict]:
        """Most recent `limit` tracks. With `at`, only tracks seen within window_seconds of it (any epoch)."""
        with self._lock:
            recs = [dict(r, cat_id_counts=dict(r["cat_id_counts"])) for r in self._tracks.values()]
        if at is not None:
            lo, hi = at - datetime.timedelta(seconds=window_seconds), at + datetime.timedelta(seconds=window_seconds)
            recs = [
                r for r in recs
                if datetime.datetime.fromisoformat(r["first_seen"]) <= hi and datetime.datetime.fromisoformat(r["last_seen"]) >= lo
            ]
        elif current_epoch_only:
            recs = [r for r in recs if r["epoch"] == self.epoch]
        out = []
        for r in recs[-limit:]:
            # More than one cat_id on one track = possible identity switch, worth validating.
            r["identity_consistent"] = len(r["cat_id_counts"]) <= 1
            out.append(r)
        return out
