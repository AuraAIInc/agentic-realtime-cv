"""The always-on detection loop.

Each tick:
  1. run any one-shot jobs queued by MCP tools
  2. pick settings: active override, else defaults
  3. rate-limit to settings.fps (sleep is interrupted by new jobs / overrides)
  4. take the newest frame, track, publish JSON over WebSocket, save images

With no MCP activity this does what camera_main_v2.run() did.
"""

from __future__ import annotations

import datetime
import json
import logging
import threading
import time

from camera_v3 import imaging
from camera_v3.controller import Controller, Settings
from camera_v3.frame_buffer import FrameBuffer
from camera_v3.frame_source import FrameSource
from camera_v3.image_store import ImageStore
from camera_v3.inference import Inference
from camera_v3.track_history import TrackHistory

logger = logging.getLogger(__name__)


class DetectionLoop:
    def __init__(
        self,
        cfg,
        source: FrameSource,
        inference: Inference,
        controller: Controller,
        buffer: FrameBuffer,
        tracks: TrackHistory,
        publish,  # callable(str) -> None, e.g. WebSocketSender.enqueue
        image_store: ImageStore,
    ):
        self.cfg = cfg
        self.source = source
        self.inference = inference
        self.controller = controller
        self.buffer = buffer
        self.tracks = tracks
        self.publish = publish
        self.image_store = image_store
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.last_message: dict | None = None
        self.frames_processed = 0
        self.loop_fps = 0.0

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self.run, name="detection_loop", daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "frames_processed": self.frames_processed,
                "loop_fps": round(self.loop_fps, 3),
                "last_message": self.last_message,
            }

    def run(self) -> None:
        last_seq = None
        last_key = None
        next_due = 0.0
        fps_t, fps_n = time.monotonic(), 0
        while not self._stop.is_set():
            self.controller.run_pending()
            settings, mode = self.controller.effective()

            # Tracker state is only valid for one model + crop + input size; reset when they change.
            key = (settings.model, settings.roi, settings.imgsz)
            if key != last_key:
                if last_key is not None:
                    self.inference.registry.reset_tracker(settings.model)
                    epoch = self.tracks.new_epoch()
                    logger.info(f"Settings changed -> {settings.as_dict()} (mode={mode}, track epoch {epoch})")
                last_key = key

            now = time.monotonic()
            if settings.fps and now < next_due:
                self.controller.wait(min(next_due - now, 1.0))
                continue

            got = self.source.get_latest(after_seq=last_seq, timeout=1.0)
            if got is None:
                continue
            full, captured_at, last_seq = got
            next_due = now + (1.0 / settings.fps if settings.fps else 0.0)

            try:
                self._process(full, captured_at, settings, mode)
            except Exception:
                logger.exception("Frame processing failed")
                time.sleep(0.5)
                continue

            fps_n += 1
            if time.monotonic() - fps_t >= 10:
                self.loop_fps = fps_n / (time.monotonic() - fps_t)
                fps_t, fps_n = time.monotonic(), 0

    def _process(self, full, captured_at: datetime.datetime, settings: Settings, mode: str) -> None:
        t1 = time.monotonic()
        display = imaging.resize_to(full, self.cfg.camera_img_size)
        self.image_store.save(display, captured_at)
        grey = imaging.is_grey(display)
        t2 = time.monotonic()

        detections = self.inference.run(full, settings.model, settings.roi, settings.imgsz, track=True)
        t3 = time.monotonic()

        msg = {
            "detections": detections,
            "is_grey": int(grey),
            "create_time": captured_at.strftime("%Y-%m-%d-%H-%M-%S-%f"),
            "model_id": self.cfg.model_id,
            "view_id": self.cfg.view_id,
            "device_id": self.cfg.device_id,
            # v3 additions (backend can ignore them):
            "source": mode,  # "default" or "override"
            "settings": settings.as_dict() if mode == "override" else None,
        }
        self.publish(json.dumps(msg))
        self.buffer.add(captured_at, full)
        self.tracks.update(captured_at, detections)
        t4 = time.monotonic()

        with self._lock:
            self.last_message = msg
            self.frames_processed += 1

        names = [(d["label"], d["confidence"]) for d in detections]
        logger.info(
            f"{captured_at:%Y-%m-%d %H:%M:%S} [{mode}] PRE:{t2 - t1:.3f} IF:{t3 - t2:.3f} POST:{t4 - t3:.3f} "
            f"Total:{t4 - t1:.3f} {names}"
        )
