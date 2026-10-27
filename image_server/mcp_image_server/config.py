"""Settings, from environment variables (.env is loaded by server.main)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    image_root: str = "/mnt/extreme/Cat_monitor_data"  # same folder the Flask API serves
    host: str = "0.0.0.0"
    port: int = 10001  # Flask API is on 10000
    auth_token: str | None = None
    max_batch: int = 1000
    max_list: int = 5000

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            image_root=os.getenv("IMAGE_ROOT", cls.image_root),
            host=os.getenv("IMAGE_MCP_HOST", cls.host),
            port=int(os.getenv("IMAGE_MCP_PORT", str(cls.port))),
            auth_token=os.getenv("IMAGE_MCP_AUTH_TOKEN") or None,
        )
