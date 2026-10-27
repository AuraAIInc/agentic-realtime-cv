"""Control plane between MCP tools and the detection loop.

Two mechanisms:

* Overrides (set_override): replace the loop's settings (fps, roi, imgsz, model)
  for a limited time. When the TTL runs out the loop returns to defaults by
  itself, so a crashed or disconnected supervisor can never leave the camera
  stuck in a degraded mode.

* One-shot jobs (submit): a callable that must run on the loop thread (anything
  that uses a YOLO model). The caller blocks on a Future; the loop runs pending
  jobs at the top of every tick, and wakes early from its rate-limit sleep when
  a job arrives.

Priority each tick: pending jobs > active override > defaults.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import asdict, dataclass, replace

logger = logging.getLogger(__name__)

MAX_DURATION_S = 24 * 3600
MAX_FPS = 30.0


@dataclass(frozen=True)
class Settings:
    fps: float | None = None  # None = as fast as inference allows (v2 behaviour)
    roi: tuple[int, int, int, int] | None = None  # display coords; None = full frame
    imgsz: tuple[int, int] | None = None  # model input size; None = config.model_img_size
    model: str = "default"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["roi"] = list(self.roi) if self.roi else None
        d["imgsz"] = list(self.imgsz) if self.imgsz else None
        return d


class Controller:
    def __init__(self, defaults: Settings, model_names: list[str]):
        self.defaults = defaults
        self.model_names = model_names
        self._override: Settings | None = None
        self._expires_at = 0.0  # time.monotonic()
        self._override_reason = ""
        self._lock = threading.Lock()
        self._jobs: queue.Queue = queue.Queue()
        self._wake = threading.Event()

    # ---------------- overrides ----------------
    def set_override(self, duration_seconds: float, reason: str = "", merge: bool = True, **fields) -> dict:
        """Fields left as None keep their value from the current override (merge=True) or the defaults."""
        duration_seconds = float(duration_seconds)
        if not 0 < duration_seconds <= MAX_DURATION_S:
            raise ValueError(f"duration_seconds must be in (0, {MAX_DURATION_S}]")
        fps = fields.get("fps")
        if fps is not None and not 0 < float(fps) <= MAX_FPS:
            raise ValueError(f"fps must be in (0, {MAX_FPS}] (use 1/30 = 0.0333 for one frame per 30 s)")
        model = fields.get("model")
        if model is not None and model not in self.model_names:
            raise ValueError(f"unknown model {model!r}; available: {self.model_names}")
        roi = fields.get("roi")
        if roi is not None:
            if len(roi) != 4 or roi[2] <= roi[0] or roi[3] <= roi[1]:
                raise ValueError("roi must be [x1, y1, x2, y2] with x2>x1, y2>y1 (display coords)")
            fields["roi"] = tuple(int(v) for v in roi)
        imgsz = fields.get("imgsz")
        if imgsz is not None:
            if len(imgsz) != 2:
                raise ValueError("imgsz must be [width, height]")
            fields["imgsz"] = tuple(max(32, int(v) // 32 * 32) for v in imgsz)
        with self._lock:
            base = (self._current_override_locked() if merge else None) or self.defaults
            self._override = replace(base, **{k: v for k, v in fields.items() if v is not None})
            self._expires_at = time.monotonic() + duration_seconds
            self._override_reason = reason
        logger.info(f"Override set for {duration_seconds:.0f}s: {self._override.as_dict()} ({reason})")
        self._wake.set()
        return self.status()

    def clear_override(self) -> dict:
        with self._lock:
            self._override = None
            self._expires_at = 0.0
            self._override_reason = ""
        logger.info("Override cleared, back to defaults")
        self._wake.set()
        return self.status()

    def _current_override_locked(self) -> Settings | None:
        if self._override is not None and time.monotonic() >= self._expires_at:
            logger.info("Override expired, back to defaults")
            self._override = None
        return self._override

    def effective(self) -> tuple[Settings, str]:
        with self._lock:
            ov = self._current_override_locked()
            return (ov, "override") if ov else (self.defaults, "default")

    def status(self) -> dict:
        with self._lock:
            ov = self._current_override_locked()
            return {
                "mode": "override" if ov else "default",
                "settings": (ov or self.defaults).as_dict(),
                "defaults": self.defaults.as_dict(),
                "override_expires_in_s": round(self._expires_at - time.monotonic(), 1) if ov else None,
                "override_reason": self._override_reason if ov else None,
                "pending_jobs": self._jobs.qsize(),
            }

    # ---------------- one-shot jobs ----------------
    def submit(self, fn, timeout: float = 60.0):
        """Run `fn()` on the detection loop thread and return its result (called from tool threads)."""
        fut: Future = Future()
        self._jobs.put((fn, fut))
        self._wake.set()
        return fut.result(timeout=timeout)

    def run_pending(self, max_jobs: int = 8) -> int:
        """Called by the loop thread."""
        n = 0
        while n < max_jobs:
            try:
                fn, fut = self._jobs.get_nowait()
            except queue.Empty:
                break
            if fut.set_running_or_notify_cancel():
                try:
                    fut.set_result(fn())
                except Exception as e:
                    logger.exception("One-shot job failed")
                    fut.set_exception(e)
            n += 1
        return n

    def wait(self, timeout: float) -> None:
        """Sleep up to `timeout`, waking early on a new job or override change."""
        if timeout > 0:
            self._wake.wait(timeout)
        self._wake.clear()
