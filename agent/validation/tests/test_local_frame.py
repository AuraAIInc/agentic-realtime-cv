"""Tests for local__get_frame (incident frames from local disk)."""
import asyncio

import pytest

from agent.validation.worker import ensure_supervisor_importable

ensure_supervisor_importable()

from supervisor_agent.local_tools.server import LocalTools, image_of, text_of  # noqa: E402


def _frame(root, stem, size=(800, 600), color=(10, 20, 30)):
    from PIL import Image

    p = root / f"{stem}.jpg"
    Image.new("RGB", size, color).save(p, "JPEG")
    return p


@pytest.fixture()
def roots(tmp_path):
    images = tmp_path / "2026-10-28-16" / "images"
    images.mkdir(parents=True)
    _frame(images, "5502_2026-10-28-16-33-00-304643")
    _frame(images, "5503_2026-10-28-16-33-00-664538")
    return [images]


def test_exact_iso_match_returns_image(roots):
    tools = LocalTools(frame_roots=roots)
    out = asyncio.run(tools.t_get_frame("2026-10-28T16:33:00.304643"))
    assert text_of(out)["delta_s"] == 0.0
    assert text_of(out)["frame"] == "5502_2026-10-28-16-33-00-304643"
    assert image_of(out) is not None


def test_camera_format_and_nearest_with_delta(roots):
    tools = LocalTools(frame_roots=roots)
    out = asyncio.run(tools.t_get_frame("2026-10-28-16-33-01-000000"))
    payload = text_of(out)
    assert payload["frame"] == "5503_2026-10-28-16-33-00-664538"
    assert 0.0 < payload["delta_s"] < 2.0
    assert image_of(out) is not None


def test_out_of_tolerance_is_text_error(roots):
    tools = LocalTools(frame_roots=roots)
    out = asyncio.run(tools.t_get_frame("2026-10-28T16:40:00.000000"))
    assert "no local frame" in out["error"]
    assert image_of(out) is None


def test_bad_timestamp_is_text_error(roots):
    tools = LocalTools(frame_roots=roots)
    out = asyncio.run(tools.t_get_frame("not-a-time"))
    assert "bad timestamp" in out["error"]


def test_box_crop_returns_smaller_image(roots):
    tools = LocalTools(frame_roots=roots)
    full = asyncio.run(tools.t_get_frame("2026-10-28T16:33:00.304643"))
    crop = asyncio.run(tools.t_get_frame("2026-10-28T16:33:00.304643", box=[100, 100, 200, 200]))
    assert text_of(crop)["size"][0] <= text_of(full)["size"][0]
    assert len(image_of(crop)) < len(image_of(full))


def test_call_dispatch_reports_unknown_tool(roots):
    tools = LocalTools(frame_roots=roots)
    out = asyncio.run(tools.call("nope", {}))
    assert "unknown local tool" in text_of(out)["error"]


def test_missing_roots_report_none(tmp_path):
    tools = LocalTools(frame_roots=[tmp_path / "absent"])
    out = asyncio.run(tools.t_get_frame("2026-10-28T16:33:00.304643"))
    assert "none" in out["error"]


def _track_run(tmp_path):
    import json

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        {"create_time": "2026-10-28T16:34:05.908104", "box_x1": 339.0, "box_y1": 172.0,
         "box_x2": 396.0, "box_y2": 244.0, "orig_cat_id": 26, "cat_id": 26, "period_id": 2},
        {"create_time": "2026-10-28T16:35:21.976102", "box_x1": 174.0, "box_y1": 457.0,
         "box_x2": 279.0, "box_y2": 585.0, "orig_cat_id": 25, "cat_id": 26, "period_id": 2},
        {"create_time": "2026-10-28T16:35:21.976102", "box_x1": 1.0, "box_y1": 1.0,
         "box_x2": 2.0, "box_y2": 2.0, "cat_id": 26, "period_id": 2, "deleted": True},
    ]
    (run_dir / "revised.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return run_dir


def test_get_track_box_exact_match(tmp_path):
    tools = LocalTools(track_run_dir=_track_run(tmp_path))
    out = asyncio.run(tools.t_get_track_box(2, "2026-10-28T16:34:05.908104"))
    assert out["box"] == [339.0, 172.0, 396.0, 244.0]
    assert out["delta_s"] == 0.0 and out["real"] is True


def test_get_track_box_nearest_with_delta_and_span_hint(tmp_path):
    tools = LocalTools(track_run_dir=_track_run(tmp_path))
    out = asyncio.run(tools.t_get_track_box(2, "2026-10-28T16:34:06.5"))
    assert out["frame_time"] == "2026-10-28T16:34:05.908104"
    assert 0.0 < out["delta_s"] < 2.0
    far = asyncio.run(tools.t_get_track_box(2, "2026-10-28T16:34:30"))
    assert "visit 2 spans 2026-10-28T16:34:05" in far["error"]
    assert "has no tracked rows" in asyncio.run(tools.t_get_track_box(9, "2026-10-28T16:34:05.908104"))["error"]


def test_get_track_box_skips_tombstones_and_bad_input(tmp_path, monkeypatch):
    tools = LocalTools(track_run_dir=_track_run(tmp_path))
    out = asyncio.run(tools.t_get_track_box(2, "2026-10-28T16:35:21.976102"))
    assert out["box"][0] == 174.0  # the live row, not the tombstone
    assert "bad timestamp" in asyncio.run(tools.t_get_track_box(2, "nope"))["error"]
    assert "bad period_id" in asyncio.run(tools.t_get_track_box("x", "2026-10-28T16:35:21.976102"))["error"]
    monkeypatch.delenv("TRACK_RUN_DIR", raising=False)
    bare = LocalTools()
    assert "TRACK_RUN_DIR" in asyncio.run(bare.t_get_track_box(2, "2026-10-28T16:35:21.976102"))["error"]
