"""Image persistence: S3 history uploads (background thread), local disk, and
model downloads. Same keys and folders as v2.
"""

from __future__ import annotations

import datetime
import logging
import os
import threading
import time
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class S3Store:
    def __init__(self):
        self._client = None
        self._lock = threading.Lock()
        self.connect()

    def connect(self) -> None:
        import boto3

        try:
            client = boto3.client(
                "s3",
                aws_access_key_id=os.environ["S3_AWS_ACCESS_KEY"],
                aws_secret_access_key=os.environ["S3_AWS_SECRET_ACCESS_KEY"],
                region_name=os.environ["S3_AWS_DEFAULT_REGION"],
            )
        except KeyError:
            logger.error("Please set S3_AWS_ACCESS_KEY, S3_AWS_SECRET_ACCESS_KEY, S3_AWS_DEFAULT_REGION")
            raise
        with self._lock:
            self._client = client

    def put(self, bucket: str, key: str, body: bytes, content_type: str = "image/jpeg") -> None:
        self._client.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)

    def download(self, bucket: str, key: str, path: str) -> None:
        self._client.download_file(bucket, key, path)


class ImageStore:
    def __init__(self, cfg, s3: S3Store | None):
        self.cfg = cfg
        self.s3 = s3
        self.to_cloud = cfg.image_save_place in ("Cloud", "Both")
        self.to_local = cfg.image_save_place in ("Local", "Both")
        self._queue: deque[tuple[np.ndarray, datetime.datetime]] = deque()
        self._qlock = threading.Lock()

    def start(self) -> "ImageStore":
        if self.to_cloud:
            threading.Thread(target=self._upload_loop, name="upload_image_to_s3", daemon=True).start()
        return self

    def qsize(self) -> int:
        return len(self._queue)

    def save(self, frame: np.ndarray, t: datetime.datetime) -> None:
        """Called by the detection loop for every processed frame (display resolution)."""
        if self.to_cloud:
            with self._qlock:
                self._queue.append((frame, t))
                while len(self._queue) > self.cfg.q_max:
                    self._queue.popleft()
        if self.to_local:
            folder = os.path.join(self.cfg.local_image_dir, t.strftime("%Y-%m-%d/%H/%M"))
            os.makedirs(folder, exist_ok=True)
            cv2.imwrite(os.path.join(folder, t.strftime("%Y-%m-%d-%H-%M-%S-%f.jpg")), frame)

    def _upload_loop(self) -> None:
        while True:
            item = None
            with self._qlock:
                if self._queue:
                    item = self._queue.popleft()
            if item is None:
                time.sleep(0.1)
                continue
            image, t = item
            try:
                key = f"{self.cfg.user_id}/{self.cfg.device_id}/" + t.strftime("%Y-%m-%d/%H/%M/%Y-%m-%d-%H-%M-%S-%f.jpg")
                _, buf = cv2.imencode(".jpg", image)
                self.s3.put(self.cfg.history_image_bucket_name, key, buf.tobytes())
            except Exception as e:
                logger.error(f"Failed to upload image to S3: {e}")
                with self._qlock:
                    if len(self._queue) < self.cfg.q_max:
                        self._queue.appendleft(item)
                try:
                    self.s3.connect()
                except Exception:
                    pass
                time.sleep(1)
