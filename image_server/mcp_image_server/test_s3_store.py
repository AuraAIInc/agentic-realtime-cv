"""S3 backend for post-2026-10-20 images (no network: fake boto3 client).

Run from the image_server/ folder:
    python -m pytest mcp_image_server/test_s3_store.py -q
"""

from __future__ import annotations

import datetime
import io
import os

import pytest

from mcp_image_server.config import Settings
from mcp_image_server.image_store import ImageStore
from mcp_image_server.s3_store import (
    S3_DEFAULT_BUCKET,
    S3_DEFAULT_PREFIX,
    S3Store,
    is_s3_date,
    s3_key_for,
    s3_list_prefix_for,
)
from mcp_image_server.timing import Timer

NEW_NAME = "2026-10-25-01-23-20-879991.jpg"
NEW_KEY = "14/c2d5f3ce14067817/2026-10-25/01/23/2026-10-25-01-23-20-879991.jpg"
OLD_NAME = "2026-10-20-23-59-59-000001.jpg"


def _minimal_jpeg(width: int = 8, height: int = 6) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height)).save(buf, "JPEG")
    return buf.getvalue()


class FakeS3Client:
    """Minimal boto3 S3 client double: objects maps key -> bytes."""

    def __init__(self, objects: dict[str, bytes] | None = None):
        self.objects = objects or {}
        self.calls: list[tuple] = []

    def _not_found(self, op: str):
        from botocore.exceptions import ClientError

        return ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}},
            op,
        )

    def head_object(self, Bucket, Key):
        self.calls.append(("head", Bucket, Key))
        if Key not in self.objects:
            raise self._not_found("HeadObject")
        return {
            "ContentLength": len(self.objects[Key]),
            "LastModified": datetime.datetime(2026, 10, 25, 1, 23, 21),
            "ContentType": "image/jpeg",
        }

    def get_object(self, Bucket, Key, Range=None):
        self.calls.append(("get", Bucket, Key, Range))
        if Key not in self.objects:
            raise self._not_found("GetObject")
        body = self.objects[Key]
        if Range:
            # "bytes=start-end"
            start, end = Range.split("=")[1].split("-")
            body = body[int(start) : int(end) + 1]
        return {"Body": io.BytesIO(body), "ContentLength": len(body)}

    def list_objects_v2(self, Bucket, Prefix, MaxKeys=1000, ContinuationToken=None):
        self.calls.append(("list", Bucket, Prefix))
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        return {
            "Contents": [{"Key": k, "Size": len(self.objects[k])} for k in keys[:MaxKeys]],
            "IsTruncated": False,
        }


# ---------------------------------------------------------------- key format


def test_s3_key_matches_camera_upload_layout():
    t = datetime.datetime.strptime(NEW_NAME, "%Y-%m-%d-%H-%M-%S-%f.jpg")
    assert s3_key_for(S3_DEFAULT_PREFIX, t, NEW_NAME) == NEW_KEY
    assert S3_DEFAULT_BUCKET == "aura-history-image"
    assert S3_DEFAULT_PREFIX == "14/c2d5f3ce14067817"


def test_cutoff_new_images_go_to_s3_old_stay_local():
    assert is_s3_date(datetime.date(2026, 10, 19)) is False
    assert is_s3_date(datetime.date(2026, 10, 20)) is False
    assert is_s3_date(datetime.date(2026, 10, 21)) is True
    assert is_s3_date(datetime.date(2026, 10, 25)) is True


def test_list_prefix_mapping():
    assert s3_list_prefix_for(S3_DEFAULT_PREFIX, "2026-10-25") == f"{S3_DEFAULT_PREFIX}/2026-10-25/"
    assert s3_list_prefix_for(S3_DEFAULT_PREFIX, "2026-10-25-01") == f"{S3_DEFAULT_PREFIX}/2026-10-25/01/"
    assert s3_list_prefix_for(S3_DEFAULT_PREFIX, "2026-10-25-01-23") == f"{S3_DEFAULT_PREFIX}/2026-10-25/01/23/"


# ---------------------------------------------------------------- routing


def test_new_image_served_from_s3(tmp_path):
    data = _minimal_jpeg()
    s3 = S3Store(bucket="b", prefix="p", client=FakeS3Client({f"p/2026-10-25/01/23/{NEW_NAME}": data}))
    store = ImageStore(str(tmp_path), s3=s3)
    info = store.info(NEW_NAME, include_dimensions=True, timer=Timer())
    assert info["exists"] is True
    assert info["source"] == "s3"
    assert info["size_bytes"] == len(data)
    assert info["width"] == 8 and info["height"] == 6


def test_new_image_falls_back_to_local_when_s3_missing(tmp_path):
    data = _minimal_jpeg()
    folder = tmp_path / "2026-10-25" / "01" / "23"
    folder.mkdir(parents=True)
    (folder / NEW_NAME).write_bytes(data)
    s3 = S3Store(bucket="b", prefix="p", client=FakeS3Client({}))
    store = ImageStore(str(tmp_path), s3=s3)
    info = store.info(NEW_NAME, include_dimensions=False, timer=Timer())
    assert info["exists"] is True
    assert info["source"] == "local"


def test_old_image_never_hits_s3(tmp_path):
    data = _minimal_jpeg()
    folder = tmp_path / "2026-10-20" / "23" / "59"
    folder.mkdir(parents=True)
    (folder / OLD_NAME).write_bytes(data)
    client = FakeS3Client({})
    store = ImageStore(str(tmp_path), s3=S3Store(bucket="b", prefix="p", client=client))
    info = store.info(OLD_NAME, include_dimensions=False, timer=Timer())
    assert info["exists"] is True
    assert info["source"] == "local"
    assert client.calls == []


def test_read_returns_s3_bytes(tmp_path):
    data = _minimal_jpeg()
    s3 = S3Store(bucket="b", prefix="p", client=FakeS3Client({f"p/2026-10-25/01/23/{NEW_NAME}": data}))
    store = ImageStore(str(tmp_path), s3=s3)
    info, body = store.read(NEW_NAME, timer=Timer())
    assert info["exists"] is True and body == data


def test_list_new_hour_from_s3(tmp_path):
    key = f"p/2026-10-25/01/23/{NEW_NAME}"
    s3 = S3Store(bucket="b", prefix="p", client=FakeS3Client({key: b"x"}))
    store = ImageStore(str(tmp_path), s3=s3)
    out = store.list("2026-10-25-01", limit=500, timer=Timer())
    assert out["images"] == [NEW_NAME]
    assert out["source"] == "s3"


def test_list_old_day_from_local(tmp_path):
    folder = tmp_path / "2026-10-20" / "23" / "59"
    folder.mkdir(parents=True)
    (folder / OLD_NAME).write_bytes(b"fake-jpg")
    client = FakeS3Client({})
    store = ImageStore(str(tmp_path), s3=S3Store(bucket="b", prefix="p", client=client))
    out = store.list("2026-10-20", limit=500, timer=Timer())
    assert out["images"] == [OLD_NAME]
    assert client.calls == []


# ---------------------------------------------------------------- settings


def test_settings_s3_from_env():
    old = {k: os.environ.get(k) for k in ("IMAGE_S3_BUCKET", "HISTORY_IMAGE_BUCKET_NAME", "IMAGE_S3_PREFIX", "S3_AWS_ACCESS_KEY", "S3_AWS_SECRET_ACCESS_KEY", "S3_AWS_DEFAULT_REGION", "IMAGE_S3_CUTOFF_DATE")}
    try:
        os.environ["IMAGE_S3_BUCKET"] = "my-bucket"
        os.environ["IMAGE_S3_PREFIX"] = "u/d"
        os.environ["S3_AWS_ACCESS_KEY"] = "ak"
        os.environ["S3_AWS_SECRET_ACCESS_KEY"] = "sk"
        os.environ["S3_AWS_DEFAULT_REGION"] = "us-west-1"
        os.environ["IMAGE_S3_CUTOFF_DATE"] = "2026-10-20"
        s = Settings.from_env()
        assert s.s3_bucket == "my-bucket"
        assert s.s3_prefix == "u/d"
        assert s.s3_enabled is True
        assert s.s3_cutoff_date == datetime.date(2026, 10, 20)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_settings_s3_disabled_without_keys():
    old = {k: os.environ.get(k) for k in ("S3_AWS_ACCESS_KEY", "S3_AWS_SECRET_ACCESS_KEY")}
    try:
        os.environ.pop("S3_AWS_ACCESS_KEY", None)
        os.environ.pop("S3_AWS_SECRET_ACCESS_KEY", None)
        assert Settings.from_env().s3_enabled is False
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
