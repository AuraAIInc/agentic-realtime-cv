"""Settings, from environment variables (detection_ingest_server.py loads .env via model.db)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 10002  # Flask API is on 10000, image MCP on 10001
    auth_token: str | None = None
    # "true" (default): always start, with auth only if a token is set; "auto": start only when a
    # token is set; "false": never
    enabled: str = "true"
    # Optional: the image SSD root (same layout as image-server), for include_images=true
    image_root: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            host=os.getenv("TRACKING_MCP_HOST", cls.host),
            port=int(os.getenv("TRACKING_MCP_PORT", str(cls.port))),
            auth_token=os.getenv("TRACKING_MCP_AUTH_TOKEN") or None,
            enabled=os.getenv("TRACKING_MCP_ENABLED", cls.enabled).lower(),
            image_root=os.getenv("TRACKING_IMAGE_ROOT") or None,
        )

    @property
    def should_start(self) -> bool:
        if self.enabled in ("1", "true", "yes", "on"):
            return True
        if self.enabled in ("0", "false", "no", "off"):
            return False
        return self.auth_token is not None
