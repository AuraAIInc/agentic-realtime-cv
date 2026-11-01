"""Ring buffer of recently processed frames, stored as JPEG to keep memory small.

Lets the supervisor look back at the exact frame a detection or a track's first
appearance came from. Frames are full capture resolution.
"""

from __future__ import annotations

import bisect
import datetime
import threading

import numpy as np

from camera_v3 import imaging


class FrameBuffer:
    def __init__(self, maxlen: int, jpeg_quality: int = 85):
        self.maxlen = maxlen
        self.jpeg_quality = jpeg_quality
        self._times: list[datetime.datetime] = []
        self._data: list[bytes] = []
        self._dets: list[list[dict]] = []  # loop detections per frame, same order as _times
        self._lock = threading.Lock()

    def add(self, t: datetime.datetime, frame: np.ndarray, detections: list[dict] | None = None) -> None:
        if self.maxlen <= 0:
            return
        data = imaging.encode(frame, self.jpeg_quality)
        with self._lock:
            self._times.append(t)
            self._data.append(data)
            self._dets.append(list(detections or []))
            if len(self._times) > self.maxlen:
                del self._times[0], self._data[0], self._dets[0]

    def nearest(self, t: datetime.datetime) -> tuple[datetime.datetime, np.ndarray, list[dict]] | None:
        """Closest buffered frame to t, with the loop detections recorded for it."""
        with self._lock:
            if not self._times:
                return None
            i = bisect.bisect_left(self._times, t)
            candidates = [j for j in (i - 1, i) if 0 <= j < len(self._times)]
            j = min(candidates, key=lambda k: abs((self._times[k] - t).total_seconds()))
            found_t, data, dets = self._times[j], self._data[j], self._dets[j]
        return found_t, imaging.decode(data), [dict(d) for d in dets]

    def span(self) -> dict:
        with self._lock:
            if not self._times:
                return {"count": 0}
            return {
                "count": len(self._times),
                "oldest": self._times[0].isoformat(timespec="milliseconds"),
                "newest": self._times[-1].isoformat(timespec="milliseconds"),
                "bytes": sum(len(d) for d in self._data),
            }

    def times(self, limit: int = 50) -> list[str]:
        with self._lock:
            if limit <= 0:
                return []
            return [t.isoformat(timespec="milliseconds") for t in self._times[-limit:]]
