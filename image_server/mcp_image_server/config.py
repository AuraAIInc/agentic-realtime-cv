"""Settings, from environment variables (.env is loaded by server.main)."""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass

from mcp_image_server.s3_store import S3_DEFAULT_BUCKET, S3_DEFAULT_CUTOFF, S3_DEFAULT_PREFIX


@dataclass(frozen=True)
class Settings:
    image_root: str = "/mnt/extreme/Cat_monitor_data"  # same folder the Flask API serves
    host: str = "0.0.0.0"
    port: int = 10001  # Flask API is on 10000
    auth_token: str | None = None
    max_batch: int = 1000
    max_list: int = 5000
    # S3 backend for images newer than s3_cutoff (YYYY-MM-DD). Enabled only when
    # S3_AWS_ACCESS_KEY + S3_AWS_SECRET_ACCESS_KEY are set; otherwise local-only.
    s3_bucket: str = S3_DEFAULT_BUCKET
    s3_prefix: str = S3_DEFAULT_PREFIX
    s3_cutoff: str = "2026-10-20"
    s3_region: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            image_root=os.getenv("IMAGE_ROOT", cls.image_root),
            host=os.getenv("IMAGE_MCP_HOST", cls.host),
            port=int(os.getenv("IMAGE_MCP_PORT", str(cls.port))),
            auth_token=os.getenv("MCP_AUTH_TOKEN") or None,
            s3_bucket=os.getenv("IMAGE_S3_BUCKET") or os.getenv("HISTORY_IMAGE_BUCKET_NAME", S3_DEFAULT_BUCKET),
            s3_prefix=os.getenv("IMAGE_S3_PREFIX") or os.getenv("S3_IMAGE_PREFIX", S3_DEFAULT_PREFIX),
            s3_cutoff=os.getenv("IMAGE_S3_CUTOFF_DATE", "2026-10-20"),
            s3_region=os.getenv("S3_AWS_DEFAULT_REGION"),
            s3_access_key=os.getenv("S3_AWS_ACCESS_KEY"),
            s3_secret_key=os.getenv("S3_AWS_SECRET_ACCESS_KEY"),
        )

    @property
    def s3_enabled(self) -> bool:
        return bool(self.s3_access_key and self.s3_secret_key)

    @property
    def s3_cutoff_date(self) -> datetime.date:
        try:
            return datetime.datetime.strptime(self.s3_cutoff.strip(), "%Y-%m-%d").date()
        except ValueError:
            return S3_DEFAULT_CUTOFF
