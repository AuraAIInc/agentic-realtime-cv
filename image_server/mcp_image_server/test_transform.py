"""Crop/resize gating and slim info payloads (no network, no S3).

Run from the image_server/ folder:
    python -m pytest mcp_image_server/test_transform.py -q
"""

from __future__ import annotations

import io
import os

import pytest

from mcp_image_server.image_store import (
    TransformUnavailable,
    check_transform,
    pil_available,
    transform,
)
from mcp_image_server.image_store import ImageStore
from mcp_image_server.timing import Timer

NAME = "2026-10-24-20-15-03-123456.jpg"


def _jpeg(width=64, height=48) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buf, "JPEG")
    return buf.getvalue()


def _seeded_store(tmp_path) -> ImageStore:
    folder = tmp_path / "2026-10-24" / "20" / "15"
    folder.mkdir(parents=True)
    (folder / NAME).write_bytes(_jpeg())
    return ImageStore(str(tmp_path))


def test_pil_available_in_test_env():
    assert pil_available() is True


def test_check_transform_fails_fast_without_pil(monkeypatch):
    monkeypatch.setattr("mcp_image_server.image_store.pil_available", lambda: False)
    with pytest.raises(TransformUnavailable, match="without box/max_side"):
        check_transform([0, 0, 10, 10], None)
    with pytest.raises(TransformUnavailable, match="make_contact_sheet"):
        check_transform(None, 384)
    check_transform(None, None)  # plain reads never need Pillow


def test_transform_without_pil_is_actionable(monkeypatch):
    monkeypatch.setattr("mcp_image_server.image_store.pil_available", lambda: False)
    with pytest.raises(TransformUnavailable, match="Do not retry"):
        transform(_jpeg(), 32, None, Timer())


def test_transform_crop_and_resize_with_pil():
    data, meta = transform(_jpeg(64, 48), 32, [0, 0, 64, 48], Timer())
    assert (meta["width"], meta["height"]) == (32, 24)
    assert meta["box"] == [0, 0, 64, 48]
    assert data[:2] == b"\xff\xd8"


def test_info_payload_is_slim(tmp_path):
    out = _seeded_store(tmp_path).info(NAME, True, Timer())
    assert out["exists"] is True and out["source"] == "local"
    assert {"image_name", "exists", "source", "size_bytes", "captured_at", "width", "height"} >= set(out) - {"channels", "progressive", "header_error"}
    assert not {"modified_at", "relative_path", "mime_type", "s3_bucket", "s3_key"} & set(out)


def test_read_payload_is_slim(tmp_path):
    info, data = _seeded_store(tmp_path).read(NAME, Timer())
    assert data and info["size_bytes"] == len(data)
    assert not {"modified_at", "relative_path", "mime_type", "s3_bucket", "s3_key"} & set(info)


def test_missing_image_payload(tmp_path):
    info, data = _seeded_store(tmp_path).read("2026-10-24-20-15-04-000000.jpg", Timer())
    assert (info["exists"], data) == (False, None)
    assert os.path.sep not in info["image_name"]
