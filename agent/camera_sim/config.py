"""Sim configuration: everything camera_v3 reads from the view-info API,
plus replay controls, in one YAML file. No API dependency.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(PACKAGE_DIR, "sim_config.yaml")


@dataclass
class SimConfig:
    # identity (stamped on every streamed message)
    device_id: str = "sim-camera-01"
    view_id: object = "sim-view"
    model_id: object = "sim-model"
    user_id: object = "sim-user"
    user_name: str = "sim"
    cat_names: dict[int, str] = field(default_factory=dict)

    # replay source
    image_dir: str = ""
    replay_start: str = ""
    replay_end: str = ""
    replay_speed: float = 1.0
    max_gap_sleep: float = 2.0
    loop: bool = False
    limit: int = 0
    exit_when_done: bool = True

    # geometry
    camera_img_size: tuple[int, int] = (800, 600)
    capture_size: tuple[int, int] | None = None

    # detection output
    ws_enabled: bool = True
    ws_url: str = "ws://127.0.0.1:8000"
    q_max: int = 100
    output_jsonl: str = ""
    loop_model: str = "yolov8s"

    # image saving: the sim never uploads or writes images
    image_save_place: str = "None"
    local_image_dir: str = ""
    history_image_bucket_name: str = ""

    # models (same keys ModelRegistry.from_config reads from camera_v3.Config)
    models_config: str | None = None
    model_s3_save_path: str = ""
    model_training_bucket_name: str = ""
    extra_models: dict = field(default_factory=dict)

    # look-back / tracks
    frame_buffer_size: int = 120
    frame_buffer_jpeg_quality: int = 85

    # MCP server
    mcp_enabled: bool = True
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8766
    mcp_auth_token: str | None = None

    log_level: str = "INFO"

    @property
    def effective_capture_size(self) -> tuple[int, int]:
        return self.capture_size or self.camera_img_size

    @property
    def model_img_size(self) -> tuple[int, int]:
        w, h = self.camera_img_size
        return (w // 32 * 32, h // 32 * 32)

    def log_summary(self) -> None:
        logger.info("===== camera_sim configuration =====")
        for f in (
            "device_id", "view_id", "model_id", "user_id", "user_name", "cat_names",
            "image_dir", "replay_start", "replay_end", "replay_speed", "max_gap_sleep",
            "loop", "limit", "exit_when_done", "camera_img_size", "capture_size",
            "ws_enabled", "ws_url", "q_max", "output_jsonl", "loop_model",
            "models_config", "model_s3_save_path", "model_training_bucket_name",
            "frame_buffer_size", "mcp_enabled", "mcp_host", "mcp_port",
        ):
            logger.info(f"  {f}: {getattr(self, f)}")
        logger.info(f"  mcp_auth_token: {'set (token required)' if self.mcp_auth_token else 'NONE (open)'}")


def _as_size(v) -> tuple[int, int] | None:
    if v is None:
        return None
    return (int(v[0]), int(v[1]))


def from_yaml(path: str = DEFAULT_CONFIG) -> SimConfig:
    import yaml

    with open(path) as f:
        d = yaml.safe_load(f) or {}
    base = os.path.dirname(os.path.abspath(path))
    models_config = d.get("models_config") or "models.json"
    if not os.path.isabs(models_config):
        models_config = os.path.join(base, models_config)
    token = d.get("mcp_auth_token") or os.environ.get("MCP_AUTH_TOKEN") or None
    return SimConfig(
        device_id=str(d.get("device_id", "sim-camera-01")),
        view_id=d.get("view_id", "sim-view"),
        model_id=d.get("model_id", "sim-model"),
        user_id=d.get("user_id", "sim-user"),
        user_name=str(d.get("user_name", "sim")),
        cat_names={int(k): str(v) for k, v in (d.get("cat_names") or {}).items()},
        image_dir=d.get("image_dir", "") or "",
        replay_start=str(d.get("replay_start") or ""),
        replay_end=str(d.get("replay_end") or ""),
        replay_speed=float(d.get("replay_speed", 1.0)),
        max_gap_sleep=float(d.get("max_gap_sleep", 2.0)),
        loop=bool(d.get("loop", False)),
        limit=int(d.get("limit", 0)),
        exit_when_done=bool(d.get("exit_when_done", True)),
        camera_img_size=_as_size(d.get("camera_img_size", [800, 600])),
        capture_size=_as_size(d.get("capture_size")),
        ws_enabled=bool(d.get("ws_enabled", True)),
        ws_url=d.get("ws_url", "ws://127.0.0.1:8000"),
        q_max=int(d.get("q_max", 100)),
        output_jsonl=d.get("output_jsonl") or "",
        loop_model=d.get("loop_model", "yolov8s"),
        models_config=models_config,
        model_s3_save_path=d.get("model_s3_save_path") or "",
        model_training_bucket_name=d.get("model_training_bucket_name") or "",
        extra_models=d.get("extra_models") or {},
        frame_buffer_size=int(d.get("frame_buffer_size", 120)),
        frame_buffer_jpeg_quality=int(d.get("frame_buffer_jpeg_quality", 85)),
        mcp_enabled=bool(d.get("mcp_enabled", True)),
        mcp_host=d.get("mcp_host", "127.0.0.1"),
        mcp_port=int(d.get("mcp_port", 8766)),
        mcp_auth_token=token,
        log_level=str(d.get("log_level", "INFO")).upper(),
    )
