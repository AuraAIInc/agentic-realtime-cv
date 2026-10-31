"""Sim wiring: the real camera_v3 pipeline on top of a replay source.

Threads (same as camera_v3.main):
  replay_source       - serves recorded images with their original timestamps
  detection_loop      - default detection + one-shot MCP jobs (sole YOLO user)
  ws_run_forever/send - WebSocket client streaming the exact camera_v3 JSON
  mcp_server          - the camera_v3 MCP tools over HTTP

No view-info API: identity and settings come from sim_config.yaml.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time

import camera_sim  # noqa: F401  (adds camera_server/ to sys.path)
from camera_v3.camera_service import CameraService

from camera_sim.config import DEFAULT_CONFIG, SimConfig, from_yaml
from camera_sim.replay_source import ReplaySource
from camera_sim.weights import LazyS3Store, ensure_hub_weights

LOG_FORMAT = "%(asctime)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s"
logger = logging.getLogger("camera_sim")


class NullSender:
    """WebSocketSender-compatible sink when ws_enabled is false."""

    def __init__(self):
        self.sent = 0

    def start(self) -> "NullSender":
        return self

    def enqueue(self, message: str) -> None:
        self.sent += 1

    def qsize(self) -> int:
        return 0

    @property
    def connected(self) -> bool:
        return False


class JsonlTee:
    """Wraps publish() to also append every message to a JSONL file."""

    def __init__(self, path: str, downstream):
        self._f = open(path, "a")
        self._lock = threading.Lock()
        self._downstream = downstream
        logger.info(f"Teeing streamed messages to {path}")

    def __call__(self, message: str) -> None:
        with self._lock:
            self._f.write(message + "\n")
            self._f.flush()
        self._downstream(message)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay recorded images through the camera_v3 pipeline.")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="sim config file (default: sim_config.yaml)")
    p.add_argument("--start", default=None, help="replay start (hour/prefix/full name), overrides config")
    p.add_argument("--end", default=None, help="replay end (hour/prefix/full name), overrides config")
    p.add_argument("--speed", type=float, default=None, help="replay speed, 0 = uncapped")
    p.add_argument("--limit", type=int, default=None, help="max frames to serve")
    p.add_argument("--loop", action="store_true", default=None, help="loop the range")
    p.add_argument("--model", default=None, help="model the always-on loop runs")
    p.add_argument("--ws-url", default=None, help="WebSocket URL to stream to")
    p.add_argument("--no-ws", action="store_true", help="don't open a WebSocket connection")
    p.add_argument("--no-mcp", action="store_true", help="don't start the MCP server")
    p.add_argument("--mcp-port", type=int, default=None, help="MCP HTTP port")
    p.add_argument("--jsonl", default=None, help="tee streamed messages to this JSONL file")
    p.add_argument("--stay", action="store_true", help="stay up after the range is exhausted")
    return p.parse_args(argv)


def apply_args(cfg: SimConfig, a: argparse.Namespace) -> SimConfig:
    if a.start is not None:
        cfg.replay_start = a.start
    if a.end is not None:
        cfg.replay_end = a.end
    if a.speed is not None:
        cfg.replay_speed = a.speed
    if a.limit is not None:
        cfg.limit = a.limit
    if a.loop:
        cfg.loop = True
    if a.model is not None:
        cfg.loop_model = a.model
    if a.ws_url is not None:
        cfg.ws_url = a.ws_url
    if a.no_ws:
        cfg.ws_enabled = False
    if a.no_mcp:
        cfg.mcp_enabled = False
    if a.mcp_port is not None:
        cfg.mcp_port = a.mcp_port
    if a.jsonl is not None:
        cfg.output_jsonl = a.jsonl
    if a.stay:
        cfg.exit_when_done = False
    return cfg


def build(cfg: SimConfig):
    """Wire the components together. Returns (source, loop, service, sender)."""
    from camera_v3.camera_service import CameraService
    from camera_v3.controller import Controller, Settings
    from camera_v3.detection_loop import DetectionLoop
    from camera_v3.frame_buffer import FrameBuffer
    from camera_v3.image_store import ImageStore
    from camera_v3.inference import Inference
    from camera_v3.models import ModelRegistry
    from camera_v3.track_history import TrackHistory

    from ultralytics import settings as yolo_settings

    runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    if os.path.abspath(yolo_settings["runs_dir"]) != runs_dir:
        yolo_settings.update({"runs_dir": runs_dir})  # saved in .ultralytics/, not global

    s3_store = LazyS3Store()
    registry = ModelRegistry.from_config(cfg, s3_store)

    # Weights live in the sim project: resolve relative paths against models.json,
    # and fetch hub models (yolov8*.pt) when the file is missing.
    models_dir = os.path.dirname(os.path.abspath(cfg.models_config))
    for name, spec in registry.specs.items():
        if not os.path.isabs(spec.weights):
            spec.weights = os.path.normpath(os.path.join(models_dir, spec.weights))
        hub = spec.info.get("hub")
        if hub and not os.path.exists(spec.weights) and not spec.s3_key:
            ensure_hub_weights(spec.weights, hub)
    if cfg.loop_model not in registry.names():
        raise ValueError(f"loop_model {cfg.loop_model!r} not in models file; available: {registry.names()}")
    registry.get(cfg.loop_model, "loop")  # download + load before starting, as camera_v3 does

    source = ReplaySource(
        cfg.image_dir,
        cfg.effective_capture_size,
        start=cfg.replay_start,
        end=cfg.replay_end,
        speed=cfg.replay_speed,
        max_gap_sleep=cfg.max_gap_sleep,
        loop=cfg.loop,
        limit=cfg.limit,
    )
    controller = Controller(Settings(model=cfg.loop_model), registry.names())
    logger.info(f"Default loop settings: {controller.defaults.as_dict()}")
    inference = Inference(registry, cfg.cat_names, cfg.camera_img_size, cfg.model_img_size)
    buffer = FrameBuffer(cfg.frame_buffer_size, cfg.frame_buffer_jpeg_quality)
    tracks = TrackHistory()
    image_store = ImageStore(cfg, s3_store)  # image_save_place "None": save() is a no-op

    if cfg.ws_enabled:
        from camera_v3.ws_sender import WebSocketSender

        sender = WebSocketSender(cfg.ws_url, cfg.q_max)
    else:
        sender = NullSender()
    ws_status = lambda: {
        "enabled": cfg.ws_enabled,
        "url": cfg.ws_url if cfg.ws_enabled else None,
        "connected": sender.connected,
        "queue": sender.qsize(),
        "sent": sender.sent,
    }
    publish = sender.enqueue
    if cfg.output_jsonl:
        publish = JsonlTee(cfg.output_jsonl, sender.enqueue)

    loop = DetectionLoop(cfg, source, inference, controller, buffer, tracks, publish, image_store)
    service = SimCameraService(cfg, source, controller, inference, buffer, tracks, loop, image_store, ws_status)
    return source, loop, service, sender


class SimCameraService(CameraService):
    """The camera_v3 service, plus replay progress in get_status."""

    def status(self) -> dict:
        d = super().status()
        d["sim"] = True
        d["replay"] = self.source.progress()
        return d

    def camera_modes(self) -> dict:
        d = super().camera_modes()
        d["note"] = f"replay source: resolution is fixed by the files in {self.cfg.image_dir}"
        return d


def main(argv=None) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    a = parse_args(argv)
    cfg = apply_args(from_yaml(a.config), a)
    logging.getLogger().setLevel(cfg.log_level)
    cfg.log_summary()

    source, loop, service, sender = build(cfg)

    source.start()
    sender.start()
    loop_thread = loop.start()
    if cfg.mcp_enabled:
        from camera_v3 import mcp_server

        mcp_server.start_in_thread(service, cfg.mcp_host, cfg.mcp_port, cfg.mcp_auth_token)

    exit_when_done = cfg.exit_when_done and not cfg.loop
    try:
        while loop_thread.is_alive():
            loop_thread.join(timeout=2)
            if exit_when_done and source.finished.is_set() and sender.qsize() == 0:
                if loop.frames_processed >= source.frames_total:
                    logger.info("Replay fully streamed, shutting down")
                    break
    except KeyboardInterrupt:
        logger.info("Stopping ...")
    finally:
        loop.stop()
        source.stop()
        time.sleep(0.5)


if __name__ == "__main__":
    main()
