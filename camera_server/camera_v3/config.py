"""Device identity and runtime configuration.

Everything the v2 script read into module-level constants now lives in one
`Config` object, loaded once at startup from the AuraAI view-info API.
Optional keys that v2 did not use have defaults, so the existing backend
response keeps working unchanged.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field, fields

import requests

logger = logging.getLogger(__name__)

VIEW_INFO_API = "https://auraaiinc.com/api/v1/ai-cat/camera-server/view-info/{device_id}"

# Environment variables read anywhere in camera_v3, logged at startup (secrets masked).
ENV_SETTINGS = (
    "DEVICE_ID_OVERRIDE", "MODELS_CONFIG", "MCP_ENABLED", "MCP_HOST", "MCP_PORT", "MCP_AUTH_TOKEN",
    "S3_AWS_DEFAULT_REGION", "S3_AWS_ACCESS_KEY", "S3_AWS_SECRET_ACCESS_KEY",
)
SECRET_ENV = {"MCP_AUTH_TOKEN", "S3_AWS_ACCESS_KEY", "S3_AWS_SECRET_ACCESS_KEY"}


@dataclass
class Config:
    device_id: str
    cat_names: dict[int, str]
    camera_img_size: tuple[int, int]  # (w, h) the backend sees; detection boxes use this space
    model_img_size: tuple[int, int]
    view_id: object
    model_id: object
    user_id: object
    user_name: str
    model_training_bucket_name: str
    history_image_bucket_name: str
    aura_server_ip: str
    image_save_place: str  # "Cloud" | "Local" | "Both" | anything else = none
    model_s3_save_path: str
    rtsp_stream_url: str
    log_level: str
    q_max: int

    # --- new in v3 (all optional in view-info) ---
    # Resolution the camera is opened at. Zoom crops come from this. Defaults to camera_img_size,
    # which reproduces v2 behaviour exactly; set it to the sensor's max to get real zoom detail.
    capture_size: tuple[int, int] | None = None
    frame_buffer_size: int = 120  # processed frames kept (as JPEG) for look-back tools
    frame_buffer_jpeg_quality: int = 85
    # Path of the models file (env MODELS_CONFIG); None = camera_v3/models.json.
    models_config: str | None = None
    # name -> same keys as a models file entry; added to / overrides the models file
    extra_models: dict = field(default_factory=dict)
    local_image_dir: str = "/home/muyang/Cat_monitor_data"
    ws_port: int = 8000

    # MCP server (from environment, not view-info, so the token never travels through the API)
    mcp_enabled: bool = True
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8765
    mcp_auth_token: str | None = None

    @property
    def effective_capture_size(self) -> tuple[int, int]:
        return self.capture_size or self.camera_img_size

    @classmethod
    def from_view_info(cls, device_id: str, info: dict) -> "Config":
        w, h = info["width"], info["height"]
        cap = info.get("capture_width"), info.get("capture_height")
        return cls(
            device_id=device_id,
            cat_names={int(k): v for k, v in info["cat_info"].items()},
            camera_img_size=(w, h),
            model_img_size=(w // 32 * 32, h // 32 * 32),
            view_id=info["view_id"],
            model_id=info["model_id"],
            user_id=info["user_id"],
            user_name=info["user_name"],
            model_training_bucket_name=info["model_training_bucket_name"],
            history_image_bucket_name=info["history_image_bucket_name"],
            aura_server_ip=info["aura_server_ip"],
            image_save_place=info["image_save_place"],
            model_s3_save_path=info["model_s3_save_path"],
            rtsp_stream_url=info["rtsp_stream_url"],
            log_level=info.get("log_level", "INFO").upper(),
            q_max=info["q_max"],
            capture_size=(int(cap[0]), int(cap[1])) if all(cap) else None,
            frame_buffer_size=int(info.get("frame_buffer_size", 120)),
            frame_buffer_jpeg_quality=int(info.get("frame_buffer_jpeg_quality", 85)),
            models_config=os.environ.get("MODELS_CONFIG") or None,
            extra_models=info.get("extra_models") or {},
            local_image_dir=info.get("local_image_dir", "/home/muyang/Cat_monitor_data"),
            ws_port=int(info.get("ws_port", 8000)),
            mcp_enabled=os.environ.get("MCP_ENABLED", "1") not in ("0", "false", "False"),
            mcp_host=os.environ.get("MCP_HOST", "0.0.0.0"),
            mcp_port=int(os.environ.get("MCP_PORT", "8765")),
            mcp_auth_token=os.environ.get("MCP_AUTH_TOKEN") or None,
        )

    def log_summary(self) -> None:
        """Log every setting at startup. Secrets are masked."""
        logger.info("===== camera_v3 configuration =====")
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "mcp_auth_token":
                value = "set (token required)" if value else "NONE (open)"
            logger.info(f"  {f.name}: {value}")
        logger.info(f"  effective_capture_size: {self.effective_capture_size}")
        logger.info("----- environment -----")
        for name in ENV_SETTINGS:
            value = os.environ.get(name)
            if value is None:
                value = "<unset>"
            elif name in SECRET_ENV:
                value = f"<set, {len(value)} chars>"
            logger.info(f"  {name}: {value}")


def wait_for_internet(url: str = "https://google.com") -> None:
    while True:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return
        except Exception:
            pass
        logger.info("connecting to internet ...")
        time.sleep(0.5)


def get_rpi_serial() -> str:
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("Serial"):
                    return line.strip().split(":")[1].strip()
    except Exception as e:
        return f"Error: {e}"
    return "Unknown"


def fetch_view_info(device_id: str, retry_s: float = 5.0) -> dict:
    """Fetch view info, retrying until it succeeds (v2 crashed on failure)."""
    url = VIEW_INFO_API.format(device_id=device_id)
    logger.info(f"View info API: {url}")
    while True:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.json()
            logger.error(f"Failed to get view info: {r.status_code} -> {r.text}")
        except Exception as e:
            logger.error(f"Error getting view info: {e}")
        time.sleep(retry_s)


def load_config() -> Config:
    wait_for_internet()
    device_id = os.environ.get("DEVICE_ID_OVERRIDE") or get_rpi_serial()
    logger.info(f"Device ID: {device_id}")
    return Config.from_view_info(device_id, fetch_view_info(device_id))
