"""Model weight downloads into this project's models/ folder.

Two sources, same as camera_v3:
  * S3 (the real cat model best.pt): handled by ModelRegistry via LazyS3Store.
  * Ultralytics hub (yolov8*.pt): downloaded here so the file lives in the
    sim project instead of the global Ultralytics cache.
"""

from __future__ import annotations

import logging
import os
import shutil
import urllib.request

logger = logging.getLogger(__name__)

# GitHub release tags carrying the YOLOv8 .pt assets, newest first.
HUB_TAGS = ("v8.3.0",)
HUB_URL = "https://github.com/ultralytics/assets/releases/download/{tag}/{name}"


def ensure_hub_weights(weights_path: str, hub_name: str) -> str:
    """Make sure `weights_path` exists, downloading `hub_name` if needed."""
    if os.path.exists(weights_path):
        return weights_path
    os.makedirs(os.path.dirname(weights_path) or ".", exist_ok=True)
    last_err: Exception | None = None
    for tag in HUB_TAGS:
        url = HUB_URL.format(tag=tag, name=hub_name)
        try:
            logger.info(f"Downloading {hub_name} from {url} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "camera-sim"})
            with urllib.request.urlopen(req, timeout=120) as r, open(weights_path, "wb") as f:
                shutil.copyfileobj(r, f)
            logger.info(f"Saved {hub_name} to {weights_path} ({os.path.getsize(weights_path) / 1e6:.1f} MB)")
            return weights_path
        except Exception as e:
            last_err = e
            logger.warning(f"Download from {url} failed: {e}")
            if os.path.exists(weights_path):
                os.remove(weights_path)
    # Fallback: let Ultralytics resolve it (global cache), then copy it in.
    logger.info(f"Falling back to Ultralytics auto-download for {hub_name}")
    try:
        from ultralytics import YOLO

        cached = str(YOLO(hub_name).ckpt_path)
        shutil.copyfile(cached, weights_path)
        logger.info(f"Copied {cached} -> {weights_path}")
        return weights_path
    except Exception as e:
        raise RuntimeError(f"could not obtain {hub_name} (direct: {last_err}; ultralytics: {e})") from e


class LazyS3Store:
    """camera_v3.S3Store-compatible (put/download) but connects lazily.

    The sim never uploads images; download is only used for model weights.
    boto3 and the S3_AWS_* env vars are only needed when an s3_key is set.
    """

    def __init__(self):
        self._client = None

    def _connect(self):
        if self._client is None:
            import boto3

            try:
                self._client = boto3.client(
                    "s3",
                    aws_access_key_id=os.environ["S3_AWS_ACCESS_KEY"],
                    aws_secret_access_key=os.environ["S3_AWS_SECRET_ACCESS_KEY"],
                    region_name=os.environ["S3_AWS_DEFAULT_REGION"],
                )
            except KeyError:
                logger.error("Please set S3_AWS_ACCESS_KEY, S3_AWS_SECRET_ACCESS_KEY, S3_AWS_DEFAULT_REGION")
                raise
        return self._client

    def put(self, bucket: str, key: str, body: bytes, content_type: str = "image/jpeg") -> None:
        self._connect().put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)

    def download(self, bucket: str, key: str, path: str) -> None:
        self._connect().download_file(bucket, key, path)
