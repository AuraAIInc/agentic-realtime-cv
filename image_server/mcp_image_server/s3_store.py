"""Read-only S3 backend for images newer than the SSD archive.

Layout (same keys the camera writes, see camera_v3/image_store.py):

    s3://<bucket>/<prefix>/YYYY-MM-DD/HH/MM/YYYY-MM-DD-HH-MM-SS-ffffff.jpg

Credentials come from the standard S3_AWS_* environment variables
(S3_AWS_ACCESS_KEY, S3_AWS_SECRET_ACCESS_KEY, S3_AWS_DEFAULT_REGION).
boto3 is imported lazily so local-only installs keep working without it.
"""

from __future__ import annotations

import datetime
import io
import logging
import os
import re

from mcp_image_server.jpeg_info import jpeg_dimensions
from mcp_image_server.timing import Timer

logger = logging.getLogger("mcp_image_server.s3")

S3_DEFAULT_BUCKET = "aura-history-image"
S3_DEFAULT_PREFIX = "14/c2d5f3ce14067817"
S3_DEFAULT_CUTOFF = datetime.date(2026, 10, 20)

_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(-\d{2}(-\d{2})?)?$")

# First bytes fetched for JPEG dimensions; the SOF marker is in the first KB.
_HEADER_RANGE_BYTES = 65536

_NOT_FOUND_CODES = {"404", "NoSuchKey", "NotFound", "NoSuchBucket"}


def parse_cutoff(value: str) -> datetime.date:
    return datetime.datetime.strptime(value.strip(), "%Y-%m-%d").date()


def is_s3_date(d: datetime.date, cutoff: datetime.date = S3_DEFAULT_CUTOFF) -> bool:
    """Images captured after the cutoff day live on S3; the cutoff day itself stays local."""
    return d > cutoff


def s3_key_for(prefix: str, t: datetime.datetime, image_name: str) -> str:
    return f"{prefix.strip('/')}/{t.strftime('%Y-%m-%d/%H/%M')}/{image_name}"


def s3_list_prefix_for(prefix: str, list_prefix: str) -> str:
    """Map a day/hour/minute list prefix onto its S3 key prefix."""
    list_prefix = list_prefix.strip()
    if not _PREFIX_RE.match(list_prefix):
        raise ValueError("prefix must be YYYY-MM-DD, YYYY-MM-DD-HH or YYYY-MM-DD-HH-MM")
    parts = list_prefix.split("-")
    day = "-".join(parts[:3])
    rest = "".join(f"/{p}" for p in parts[3:])
    return f"{prefix.strip('/')}/{day}{rest}/"


def _is_not_found(e: BaseException) -> bool:
    code = getattr(getattr(e, "response", None) or {}, "get", lambda *a: None)("Error", {})
    if isinstance(code, dict):
        code = code.get("Code", "")
    return str(code) in _NOT_FOUND_CODES or e.__class__.__name__ in ("NoSuchKey", "NotFound")


def _is_aws_error(e: BaseException) -> bool:
    module = e.__class__.__module__ or ""
    return module.split(".")[0] in ("boto3", "botocore") or hasattr(e, "response")


class S3Store:
    def __init__(
        self,
        bucket: str = S3_DEFAULT_BUCKET,
        prefix: str = S3_DEFAULT_PREFIX,
        client=None,
        region: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        cutoff: datetime.date = S3_DEFAULT_CUTOFF,
    ):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.cutoff = cutoff
        if client is not None:
            self._client = client
        else:
            self._client = self._build_client(region, access_key, secret_key)

    @staticmethod
    def _build_client(region, access_key, secret_key):
        try:
            import boto3
        except ImportError as e:
            raise OSError("boto3 is required for S3 images: pip install -r requirements-mcp.txt") from e
        access_key = access_key or os.environ.get("S3_AWS_ACCESS_KEY")
        secret_key = secret_key or os.environ.get("S3_AWS_SECRET_ACCESS_KEY")
        region = region or os.environ.get("S3_AWS_DEFAULT_REGION")
        if not access_key or not secret_key:
            raise OSError("S3 images need S3_AWS_ACCESS_KEY and S3_AWS_SECRET_ACCESS_KEY in the environment")
        return boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

    @classmethod
    def from_settings(cls, settings) -> "S3Store | None":
        """Build from Settings, or None when S3 credentials are absent (local-only mode)."""
        if not settings.s3_enabled:
            return None
        try:
            return cls(
                bucket=settings.s3_bucket,
                prefix=settings.s3_prefix,
                region=settings.s3_region,
                access_key=settings.s3_access_key,
                secret_key=settings.s3_secret_key,
                cutoff=settings.s3_cutoff_date,
            )
        except OSError as e:
            logger.warning(f"S3 disabled: {e}")
            return None

    # ---------------------------------------------------------------- lookup
    def key(self, image_name: str, t: datetime.datetime) -> str:
        return s3_key_for(self.prefix, t, image_name)

    def info(self, image_name: str, t: datetime.datetime, include_dimensions: bool, timer: Timer) -> dict:
        key = self.key(image_name, t)
        timer.mark("resolve")
        try:
            head = self._client.head_object(Bucket=self.bucket, Key=key)
        except Exception as e:
            timer.mark("s3_head")
            if _is_not_found(e):
                return {"image_name": image_name, "exists": False, "source": "s3"}
            if _is_aws_error(e):
                raise OSError(f"S3 head_object failed for {key}: {e}") from e
            raise
        timer.mark("s3_head")
        out = {
            "image_name": image_name,
            "exists": True,
            "source": "s3",
            "size_bytes": head.get("ContentLength"),
            "captured_at": t.isoformat(timespec="microseconds"),
            "modified_at": _isoformat(head.get("LastModified")),
            "relative_path": key,
            "mime_type": head.get("ContentType") or "image/jpeg",
            "s3_bucket": self.bucket,
            "s3_key": key,
        }
        if include_dimensions:
            try:
                resp = self._client.get_object(Bucket=self.bucket, Key=key, Range=f"bytes=0-{_HEADER_RANGE_BYTES - 1}")
                header = resp["Body"].read()
            except Exception as e:
                if _is_aws_error(e):
                    raise OSError(f"S3 range get failed for {key}: {e}") from e
                raise
            dims = jpeg_dimensions(io.BytesIO(header))
            out.update(dims or {"width": None, "height": None, "header_error": "could not read JPEG header"})
            timer.mark("read_header")
        return out

    def read(self, image_name: str, t: datetime.datetime, timer: Timer) -> tuple[dict, bytes | None]:
        key = self.key(image_name, t)
        timer.mark("resolve")
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=key)
            data = resp["Body"].read()
        except Exception as e:
            timer.mark("s3_get")
            if _is_not_found(e):
                return {"image_name": image_name, "exists": False, "source": "s3"}, None
            if _is_aws_error(e):
                raise OSError(f"S3 get_object failed for {key}: {e}") from e
            raise
        timer.mark("s3_get")
        info = {
            "image_name": image_name,
            "exists": True,
            "source": "s3",
            "size_bytes": len(data),
            "captured_at": t.isoformat(timespec="microseconds"),
            "modified_at": _isoformat(resp.get("LastModified")),
            "relative_path": key,
            "mime_type": resp.get("ContentType") or "image/jpeg",
            "s3_bucket": self.bucket,
            "s3_key": key,
        }
        return info, data

    def list(self, list_prefix: str, limit: int, timer: Timer) -> dict:
        s3_prefix = s3_list_prefix_for(self.prefix, list_prefix)
        timer.mark("resolve")
        names: list[str] = []
        truncated = False
        token = None
        try:
            while True:
                kwargs = {"Bucket": self.bucket, "Prefix": s3_prefix, "MaxKeys": min(max(limit - len(names), 1), 1000)}
                if token:
                    kwargs["ContinuationToken"] = token
                resp = self._client.list_objects_v2(**kwargs)
                for obj in resp.get("Contents", []):
                    fn = obj["Key"].rsplit("/", 1)[-1]
                    if not fn.endswith(".jpg"):
                        continue
                    if len(names) >= limit:
                        truncated = True
                        break
                    names.append(fn)
                if truncated or not resp.get("IsTruncated"):
                    break
                token = resp.get("NextContinuationToken")
        except Exception as e:
            timer.mark("s3_list")
            if _is_aws_error(e):
                raise OSError(f"S3 list failed for {s3_prefix}: {e}") from e
            raise
        timer.mark("s3_list")
        names.sort()
        return {"prefix": list_prefix.strip(), "count": len(names), "truncated": truncated, "images": names, "source": "s3"}


def _isoformat(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, datetime.datetime):
        return v.isoformat(timespec="milliseconds")
    return str(v)
