"""Entry point: load config, build every component, start the threads.

Threads:
  frame_source        - reads the camera, keeps the newest frame
  detection_loop      - default detection + one-shot MCP jobs (sole YOLO user)
  ws_run_forever/send - WebSocket to the Aura server
  upload_image_to_s3  - history image uploads
  mcp_server          - MCP tools over HTTP
"""

from __future__ import annotations

import logging
import os
import time

from dotenv import load_dotenv

LOG_FORMAT = "%(asctime)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s"
logger = logging.getLogger("camera_v3")


def build(cfg, s3_store, publish, ws_status=None, yolo_factory=None):
    """Wire the components together. Separated from main() so tests can inject fakes."""
    from camera_v3.camera_service import CameraService
    from camera_v3.controller import Controller, Settings
    from camera_v3.detection_loop import DetectionLoop
    from camera_v3.frame_buffer import FrameBuffer
    from camera_v3.frame_source import FrameSource
    from camera_v3.image_store import ImageStore
    from camera_v3.inference import Inference
    from camera_v3.models import DEFAULT_MODEL, ModelRegistry
    from camera_v3.track_history import TrackHistory

    registry = ModelRegistry.from_config(cfg, s3_store, yolo_factory)
    registry.get(DEFAULT_MODEL, "loop")  # download + load before starting, as v2 did

    video_source = 0 if cfg.rtsp_stream_url == "" else cfg.rtsp_stream_url
    source = FrameSource(video_source, _resolve_capture_size(cfg, video_source))
    controller = Controller(Settings(), registry.names())
    logger.info(f"Default loop settings: {controller.defaults.as_dict()}")
    logger.info(f"Registered models: {registry.names()}")
    logger.info(f"Video source: {video_source}")
    inference = Inference(registry, cfg.cat_names, cfg.camera_img_size, cfg.model_img_size)
    buffer = FrameBuffer(cfg.frame_buffer_size, cfg.frame_buffer_jpeg_quality)
    tracks = TrackHistory()
    image_store = ImageStore(cfg, s3_store)
    loop = DetectionLoop(cfg, source, inference, controller, buffer, tracks, publish, image_store)
    service = CameraService(cfg, source, controller, inference, buffer, tracks, loop, image_store, ws_status)
    return source, loop, image_store, service


def _resolve_capture_size(cfg, video_source):
    """Decide the resolution the camera is opened at.

    Explicit view-info capture_width/height wins; capture_max probes the camera's
    advertised modes and takes the largest (preferring PIXEL_FORMAT, which keeps the
    full-sensor field of view at every size); otherwise the display size is used.
    """
    if cfg.capture_size:
        return cfg.capture_size
    if not cfg.capture_max:
        return cfg.effective_capture_size
    from camera_v3.frame_source import PIXEL_FORMAT, FrameSource

    probe = FrameSource(video_source, None)
    modes = probe.list_modes()
    if not modes:
        logger.warning("capture_max is set but the camera lists no modes (RTSP / file, or v4l2-ctl missing); using display size")
        return cfg.effective_capture_size
    same_format = [m for m in modes if m["format"] == PIXEL_FORMAT]
    best = max(same_format or modes, key=lambda m: m["width"] * m["height"])
    size = (best["width"], best["height"])
    logger.info(f"capture_max: opening the camera at {size[0]}x{size[1]} ({best['format']})")
    return size


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    from camera_v3 import mcp_server
    from camera_v3.config import load_config
    from camera_v3.image_store import S3Store
    from camera_v3.ws_sender import WebSocketSender

    cfg = load_config()
    logging.getLogger().setLevel(cfg.log_level)
    cfg.log_summary()

    s3_store = S3Store()
    ws = WebSocketSender(f"ws://{cfg.aura_server_ip}:{cfg.ws_port}", cfg.q_max)
    ws_status = lambda: {"connected": ws.connected, "queue": ws.qsize(), "sent": ws.sent}

    source, loop, image_store, service = build(cfg, s3_store, ws.enqueue, ws_status)

    source.start()
    ws.start()
    image_store.start()
    loop_thread = loop.start()
    if cfg.mcp_enabled:
        mcp_server.start_in_thread(service, cfg.mcp_host, cfg.mcp_port, cfg.mcp_auth_token)

    try:
        while loop_thread.is_alive():
            loop_thread.join(timeout=5)
    except KeyboardInterrupt:
        logger.info("Stopping ...")
    finally:
        loop.stop()
        source.stop()
        time.sleep(0.5)
        os._exit(0)  # daemon threads (websocket, uvicorn) don't need a graceful shutdown


if __name__ == "__main__":
    main()
