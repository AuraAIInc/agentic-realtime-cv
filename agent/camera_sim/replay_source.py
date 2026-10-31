"""Replay recorded images as if they were captured by a realtime camera.

Implements the FrameSource consumer API (get_latest / capture_still /
list_modes / device_path / actual_size / read_fps), so the real
camera_v3 detection loop, service and MCP server run unchanged on top of it.

Image file names are capture timestamps (YYYY-MM-DD-HH-MM-SS-ffffff.jpg);
the stamped time is preserved, so streamed create_time values look exactly
like the real camera's.
"""

from __future__ import annotations

import datetime
import logging
import os
import re
import threading
import time
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)

NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{6})\.jpg$")
STAMP_FMT = "%Y-%m-%d-%H-%M-%S-%f"


def parse_stamp(name: str) -> datetime.datetime | None:
    m = NAME_RE.match(os.path.basename(name))
    if not m:
        return None
    try:
        return datetime.datetime.strptime(m.group(1), STAMP_FMT)
    except ValueError:
        return None


def scan_images(image_dir: str, start: str = "", end: str = "", limit: int = 0) -> list[tuple[datetime.datetime, str]]:
    """Sorted (capture_time, path) within [start, end]. Bounds accept an hour
    dir (2026-10-26-02), a name prefix, or a full name, with or without .jpg."""
    start = os.path.basename(start).removesuffix(".jpg")
    end = os.path.basename(end).removesuffix(".jpg")
    found: list[tuple[datetime.datetime, str]] = []
    for root, _, files in os.walk(image_dir):
        for fn in files:
            t = parse_stamp(fn)
            if t is None:
                continue
            stem = fn.removesuffix(".jpg")
            if start and stem < start:
                continue
            if end and stem > end and not stem.startswith(end):
                continue
            found.append((t, os.path.join(root, fn)))
    found.sort()
    if limit and limit > 0:
        found = found[:limit]
    return found


class ReplaySource:
    def __init__(
        self,
        image_dir: str,
        capture_size: tuple[int, int] | None,
        start: str = "",
        end: str = "",
        speed: float = 1.0,
        max_gap_sleep: float = 2.0,
        loop: bool = False,
        limit: int = 0,
    ):
        self.files = scan_images(image_dir, start, end, limit)
        if not self.files:
            raise ValueError(f"no replay images in {image_dir} for range [{start or '...'} .. {end or '...'}]")
        self.capture_size = tuple(capture_size) if capture_size else None
        self.speed = speed
        self.max_gap_sleep = max_gap_sleep
        self.loop = loop
        self.frames_total = len(self.files)
        logger.info(
            f"Replay: {self.frames_total} images "
            f"({os.path.basename(self.files[0][1])} .. {os.path.basename(self.files[-1][1])}), "
            f"speed={speed if speed > 0 else 'uncapped'}, loop={loop}"
        )

        self._cond = threading.Condition()
        # Every image is delivered exactly once, in order: the producer blocks
        # when the consumer falls behind (bounded backlog) instead of dropping
        # frames like the realtime camera's newest-frame handoff does.
        self._pending: deque[tuple[np.ndarray, datetime.datetime, int]] = deque()
        self._backlog = 8
        self._newest: tuple[np.ndarray, datetime.datetime, int] | None = None
        self._seq = 0
        self._served = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="replay_source", daemon=True)
        self.actual_size: tuple[int, int] | None = None
        self.read_fps: float = 0.0
        self.finished = threading.Event()

    # ---- lifecycle ----
    def start(self) -> "ReplaySource":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()

    # ---- consumer API (same as camera_v3.FrameSource) ----
    def get_latest(self, after_seq: int | None = None, timeout: float = 2.0):
        """Return (frame, capture_time, seq).

        With `after_seq` (the detection loop): the next frame in order, so no
        image is ever skipped. Without it (MCP captures): the newest frame.
        Returns None on timeout. Frames are shared: copy before modifying.
        """
        deadline = time.monotonic() + timeout
        with self._cond:
            if after_seq is None:
                while self._newest is None:
                    if self._wait_until(deadline):
                        return None
                return self._newest
            while True:
                while self._pending and self._pending[0][2] <= after_seq:
                    self._pending.popleft()
                if self._pending:
                    self._cond.notify_all()  # free the producer if it was blocked
                    return self._pending[0]
                if self._wait_until(deadline):
                    return None

    def _wait_until(self, deadline: float) -> bool:
        """Wait for news; True = timed out (or stopped with nothing to give)."""
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or self._stop.is_set():
                return True
            self._cond.wait(remaining)
            return False

    @property
    def device_path(self) -> None:
        return None  # fixed resolution, like an RTSP / file source

    def capture_still(self, size=None, fourcc=None, warmup: int = 5, timeout: float = 30.0):
        """The newest replay frame at native size (resolution is fixed by the files)."""
        got = self.get_latest(timeout=timeout)
        if got is None:
            raise RuntimeError("no replay frame available yet")
        frame, t, _ = got
        return frame.copy(), t

    def list_modes(self) -> list[dict]:
        return []

    # ---- sim status ----
    def progress(self) -> dict:
        with self._cond:
            seq, served = self._seq, self._served
        return {
            "frames_total": self.frames_total,
            "frames_served": served,
            "seq": seq,
            "finished": self.finished.is_set(),
            "loop": self.loop,
            "first": self.files[0][0].isoformat(timespec="milliseconds"),
            "last": self.files[-1][0].isoformat(timespec="milliseconds"),
        }

    # ---- internals ----
    def _run(self) -> None:
        idx, prev_t = 0, None
        t_window, n_window = time.monotonic(), 0
        while not self._stop.is_set():
            t, path = self.files[idx]
            if prev_t is not None and self.speed > 0:
                dt = (t - prev_t).total_seconds() / self.speed
                if dt > 0:
                    time.sleep(min(dt, self.max_gap_sleep))
            frame = cv2.imread(path)
            if frame is None:
                logger.warning(f"Unreadable replay image, skipping: {path}")
            else:
                if self.capture_size and (frame.shape[1], frame.shape[0]) != self.capture_size:
                    frame = cv2.resize(frame, self.capture_size)
                with self._cond:
                    while len(self._pending) >= self._backlog and not self._stop.is_set():
                        self._cond.wait(0.5)  # consumer is behind; don't read ahead
                    if self._stop.is_set():
                        return
                    self._seq += 1
                    self._served += 1
                    self._pending.append((frame, t, self._seq))
                    self._newest = (frame, t, self._seq)
                    if self.actual_size is None:
                        self.actual_size = (frame.shape[1], frame.shape[0])
                        logger.info(f"Replay frame size: {self.actual_size}")
                    self._cond.notify_all()
                n_window += 1
                if time.monotonic() - t_window >= 5:
                    self.read_fps = n_window / (time.monotonic() - t_window)
                    t_window, n_window = time.monotonic(), 0
            prev_t = t
            idx += 1
            if idx >= len(self.files):
                if self.loop:
                    idx, prev_t = 0, None
                    logger.info("Replay looping back to the first image")
                else:
                    logger.info(f"Replay exhausted ({self._served} frames served)")
                    self.finished.set()
                    return
