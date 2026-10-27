"""Capture thread: the only code that touches cv2.VideoCapture.

It reads continuously and keeps just the newest frame, so a slow consumer
(e.g. the detection loop at 1 frame / 30 s) always gets a current frame
instead of one that has been sitting in the RTSP / V4L2 buffer.

It also serves one-off stills at a different resolution (capture_still): the
capture thread briefly reopens a local camera at e.g. 4K, grabs a frame and
goes back to the normal capture size.
"""

from __future__ import annotations

import datetime
import logging
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import Future, TimeoutError as FutureTimeout

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Pixel format for every capture from a local camera. In YUYV this camera scales the full sensor
# at every size; in MJPG it centre-crops below 1920x1080, so a still would show a different view.
PIXEL_FORMAT = "YUYV"


class FrameSource:
    def __init__(self, source: str | int, capture_size: tuple[int, int] | None, reopen_after_failures: int = 50):
        self.source = source
        self.capture_size = capture_size
        self.reopen_after_failures = reopen_after_failures
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._time: datetime.datetime | None = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="frame_source", daemon=True)
        self.actual_size: tuple[int, int] | None = None
        self.read_fps: float = 0.0
        self._still_requests: queue.Queue = queue.Queue()
        self._still_lock = threading.Lock()  # one still at a time
        self._modes: list[dict] | None = None

    # ---- lifecycle ----
    def start(self) -> "FrameSource":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    # ---- consumer API ----
    def get_latest(self, after_seq: int | None = None, timeout: float = 2.0):
        """Return (frame, capture_time, seq), waiting for a frame newer than `after_seq`.

        Returns None on timeout. The frame is shared: copy it before modifying in place.
        """
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._frame is None or (after_seq is not None and self._seq <= after_seq):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            return self._frame, self._time, self._seq

    # ---- one-off stills ----
    @property
    def device_path(self) -> str | None:
        """/dev/videoN for a local camera, None for RTSP / files."""
        if isinstance(self.source, int):
            return f"/dev/video{self.source}"
        if isinstance(self.source, str) and self.source.startswith("/dev/video"):
            return self.source
        return None

    def capture_still(self, size: tuple[int, int] | None, fourcc: str | None = PIXEL_FORMAT, warmup: int = 5, timeout: float = 30.0):
        """Grab one frame at `size` (w, h) and return (frame, capture_time).

        Runs on the capture thread, which pauses normal reads meanwhile (a few seconds at 4K).
        A local camera is reopened at `size` and `fourcc` (None = driver default) and the
        first `warmup` frames are
        dropped so exposure can settle; the driver may pick the nearest size it supports.
        RTSP / file sources have a fixed resolution, so the next frame is returned at its
        native size (not resized to capture_size) and `size` is ignored.
        """
        fut: Future = Future()
        if self._still_lock.locked():
            logger.info("Still request waiting: another still is in progress")
        with self._still_lock:
            self._still_requests.put((tuple(size) if size else None, fourcc, int(warmup), fut))
            logger.info(f"Still request queued for the capture thread: {size} {fourcc}, warmup {warmup}, timeout {timeout}s")
            try:
                return fut.result(timeout=timeout)
            except FutureTimeout:
                fut.cancel()
                logger.error(f"Still request timed out after {timeout}s: {size} {fourcc}")
                raise

    def list_modes(self) -> list[dict]:
        """Formats and sizes a local camera advertises, from `v4l2-ctl --list-formats-ext` (cached).

        Empty for RTSP / files, or when v4l2-ctl (package v4l-utils) is not installed.
        """
        if self._modes is None:
            dev = self.device_path
            exe = shutil.which("v4l2-ctl")
            if not dev or not exe:
                logger.info(f"Camera modes unknown: {'not a local camera' if not dev else 'v4l2-ctl not installed (v4l-utils)'}")
                return []
            try:
                out = subprocess.run([exe, "-d", dev, "--list-formats-ext"], capture_output=True, text=True, timeout=5).stdout
            except (OSError, subprocess.SubprocessError) as e:
                logger.warning(f"v4l2-ctl failed: {e}")
                return []
            self._modes = parse_v4l2_formats(out)
            logger.info(f"Camera modes from `{exe} -d {dev} --list-formats-ext` (cached until restart):")
            for m in self._modes:
                logger.info(f"  {m['format']:<5} {m['width']}x{m['height']} max {m['max_fps'] or '?'} fps")
            best = largest_mode(self._modes)
            logger.info(
                f"Camera modes for {dev}: {len(self._modes)} listed, max "
                + (f"{best['width']}x{best['height']} {best['format']}" if best else "none")
            )
        return self._modes

    def _serve_stills(self, vid: cv2.VideoCapture) -> cv2.VideoCapture:
        """Called on the capture thread; returns the (possibly reopened) normal capture."""
        while True:
            try:
                size, fourcc, warmup, fut = self._still_requests.get_nowait()
            except queue.Empty:
                return vid
            if not fut.set_running_or_notify_cancel():
                logger.info(f"Still request {size} {fourcc} was cancelled before it ran")
                continue
            if self.device_path is None:
                logger.info("Still from RTSP / file source: returning the next frame at native size")
                try:
                    fut.set_result(self._read_native(vid))
                except Exception as e:
                    logger.error(f"Still failed: {e}")
                    fut.set_exception(e)
                continue
            t0 = time.monotonic()
            logger.info(f"Still: pausing the stream and reopening {self.device_path} at {size} {fourcc}")
            vid.release()  # a V4L2 device can only be opened once
            try:
                fut.set_result(self._grab_still(size, fourcc, warmup))
            except Exception as e:
                logger.error(f"Still failed: {e}")
                fut.set_exception(e)
            finally:
                vid = self._open("after still")
                logger.info(f"Still: stream resumed, paused for {time.monotonic() - t0:.2f}s")

    def _grab_still(self, size, fourcc, warmup) -> tuple[np.ndarray, datetime.datetime]:
        t0 = time.monotonic()
        vid = cv2.VideoCapture(self.source)
        try:
            if fourcc:  # set before the size: the sizes a camera offers depend on the format
                vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            if size:
                vid.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
                vid.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            log_negotiated_size(vid, size, f"still open ({fourcc})")
            frame, good, bad = None, 0, 0
            for _ in range(warmup + 10):  # up to 10 failed reads on top of the warm-up frames
                ret, f = vid.read()
                if ret and f is not None and f.size:
                    frame, good = f, good + 1
                    if good > warmup:
                        break
                else:
                    bad += 1
            logger.info(f"Still warm-up: {good} good frame(s), {bad} failed read(s)")
            if frame is None:
                raise RuntimeError(f"camera returned no frame at {size} ({fourcc})")
            logger.info(f"Still captured at {frame.shape[1]}x{frame.shape[0]} (requested {size}, {fourcc}) in {time.monotonic() - t0:.2f}s")
            return frame, datetime.datetime.now()
        finally:
            vid.release()

    @staticmethod
    def _read_native(vid: cv2.VideoCapture) -> tuple[np.ndarray, datetime.datetime]:
        for _ in range(10):
            ret, frame = vid.read()
            if ret and frame is not None and frame.size:
                return frame, datetime.datetime.now()
        raise RuntimeError("no frame from video source")

    # ---- internals ----
    def _open(self, reason: str = "startup") -> cv2.VideoCapture:
        vid = cv2.VideoCapture(self.source)
        if self.device_path:  # set before the size: the sizes a camera offers depend on the format
            vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*PIXEL_FORMAT))
        if self.capture_size:
            vid.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_size[0])
            vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_size[1])
        log_negotiated_size(vid, self.capture_size, f"stream open ({reason})")
        return vid

    def _run(self) -> None:
        # A video file (used for testing) is played back at its own frame rate and looped.
        is_file = isinstance(self.source, str) and os.path.isfile(self.source)
        vid = self._open()
        file_dt = 1.0 / (vid.get(cv2.CAP_PROP_FPS) or 15.0) if is_file else 0.0
        failures = 0
        t_window, n_window = time.monotonic(), 0
        logger.info(f"Video source opened: {self.source} (file={is_file})")
        while not self._stop.is_set():
            if not self._still_requests.empty():
                vid = self._serve_stills(vid)
            ret, frame = vid.read()
            if not ret or frame is None or frame.size == 0:
                failures += 1
                if is_file:
                    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                if failures % 10 == 1:
                    logger.warning(f"Empty frame from camera ({failures} in a row)")
                if failures >= self.reopen_after_failures:
                    logger.warning("Reopening video source ...")
                    vid.release()
                    time.sleep(1.0)
                    vid = self._open("reopen after failures")
                    failures = 0
                else:
                    time.sleep(0.05)
                continue
            failures = 0
            if self.capture_size and (frame.shape[1], frame.shape[0]) != tuple(self.capture_size):
                # Camera ignored the requested size (common with RTSP): normalise so coordinates are stable.
                frame = cv2.resize(frame, tuple(self.capture_size))
            with self._cond:
                self._frame = frame
                self._time = datetime.datetime.now()
                self._seq += 1
                if self.actual_size is None:
                    self.actual_size = (frame.shape[1], frame.shape[0])
                    logger.info(f"Capture frame size: {self.actual_size}")
                self._cond.notify_all()
            n_window += 1
            if time.monotonic() - t_window >= 5:
                self.read_fps = n_window / (time.monotonic() - t_window)
                t_window, n_window = time.monotonic(), 0
            if file_dt:
                time.sleep(file_dt)
        vid.release()


def fourcc_of(vid: cv2.VideoCapture) -> str | None:
    """Pixel format the driver is delivering, e.g. "YUYV" or "MJPG" (None if unknown)."""
    code = int(vid.get(cv2.CAP_PROP_FOURCC))
    s = "".join(chr((code >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ")
    return s if s.isprintable() and s else None


def log_negotiated_size(vid: cv2.VideoCapture, requested: tuple[int, int] | None, what: str) -> None:
    """Log the size and format the driver actually picked; it silently falls back to the nearest supported size."""
    actual = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fmt = fourcc_of(vid) or "?"
    if requested and actual != tuple(requested):
        logger.warning(f"Camera size set, {what}: requested {requested[0]}x{requested[1]}, driver gave {actual[0]}x{actual[1]} {fmt}")
    else:
        logger.info(f"Camera size set, {what}: {actual[0]}x{actual[1]} {fmt} (requested {requested or 'driver default'})")


def parse_v4l2_formats(text: str) -> list[dict]:
    """Parse `v4l2-ctl --list-formats-ext` into [{"format", "width", "height", "max_fps"}]."""
    modes: list[dict] = []
    fmt = None
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"\[\d+\]: '(\w+)'", line)
        if m:
            fmt = m.group(1)
            continue
        if line.startswith("Size:") and fmt:
            sizes = re.findall(r"(\d+)x(\d+)", line)  # "Stepwise 32x32 - 3840x2160 ..." -> take the max
            if sizes:
                w, h = sizes[-1]
                modes.append({"format": fmt, "width": int(w), "height": int(h), "max_fps": None})
            continue
        m = re.search(r"\(([\d.]+) fps\)", line)
        if m and modes:
            fps = float(m.group(1))
            modes[-1]["max_fps"] = max(modes[-1]["max_fps"] or 0.0, fps)
    return modes


def largest_mode(modes: list[dict], aspect: float | None = None) -> dict | None:
    """Biggest advertised size; among equal sizes prefer MJPG (fast enough at 4K), then higher fps.

    With `aspect` (w / h), modes with that aspect ratio (within 2%) win if there are any.
    """
    if aspect:
        same = [m for m in modes if abs(m["width"] / m["height"] / aspect - 1) < 0.02]
        modes = same or modes
    if not modes:
        return None
    return max(modes, key=lambda m: (m["width"] * m["height"], m["format"] == "MJPG", m["max_fps"] or 0))
