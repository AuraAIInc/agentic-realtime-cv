"""Camera service helpers: raw captures, slim status, buffered-frame flags.

Run from the camera_server/ folder:
    python -m pytest camera_v3/tests -q
"""

from __future__ import annotations

import datetime
from types import SimpleNamespace

import numpy as np

from camera_v3.camera_service import CameraService
from camera_v3.frame_buffer import FrameBuffer
from camera_v3.track_history import TrackHistory

DET = {"cat_id": 26, "box": [10, 10, 30, 30], "confidence": 0.75, "track_id": 1, "label": "Qbo"}


def _frame(w=100, h=80):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_buffer_times_zero_is_empty():
    buf = FrameBuffer(10)
    buf.add(datetime.datetime(2026, 1, 1), _frame())
    assert buf.times(0) == []
    assert len(buf.times(5)) == 1


def _service(**kw):
    cfg = SimpleNamespace(
        device_id="d1", view_id=3, camera_img_size=(100, 80), effective_capture_size=(100, 80),
        cat_names={25: "Qb", 26: "Qbo"}, model_id=1,
    )
    t0 = datetime.datetime(2026, 1, 1, 12, 0, 0)
    source = SimpleNamespace(
        actual_size=(100, 80), read_fps=2.5, read_ms=7.0,
        get_latest=lambda **a: (_frame(), t0, 1),
        list_modes=lambda: [{"width": 100, "height": 80, "format": "MJPEG"}],
    )
    loop = SimpleNamespace(
        last_message={"detections": [DET], "create_time": "2026-01-01-12-00-00-000000"},
        snapshot=lambda: {"frames_processed": 1, "loop_fps": 2.5, "last_message": loop.last_message},
    )
    controller = SimpleNamespace(status=lambda: {
        "mode": "default", "settings": {}, "defaults": {}, "override_expires_in_s": None,
        "override_reason": None, "pending_jobs": 0})
    inference = SimpleNamespace(registry=SimpleNamespace(names=lambda: ["default"]))
    buffer = kw.pop("buffer", FrameBuffer(10))
    tracks = kw.pop("tracks", TrackHistory())
    return CameraService(cfg, source, controller, inference, buffer, tracks, loop,
                         SimpleNamespace(qsize=lambda: 0), lambda: {"connected": False, "queue": 3})


def test_status_slim_by_default_verbose_on_request():
    slim = _service().status()
    assert slim["sim"] is False
    assert "camera_read_ms" not in slim and "image_upload_queue" not in slim
    assert slim["websocket"] == {"connected": False}
    assert "pending_jobs" not in slim["control"]
    assert slim["capture_size"] == [100, 80] and slim["cat_names"] == {"25": "Qb", "26": "Qbo"}

    full = _service().status(verbose=True)
    assert full["camera_read_ms"] == 7.0 and full["image_upload_queue"] == 0
    assert full["websocket"]["queue"] == 3 and "defaults" in full["control"]


def test_track_history_flags_buffered_frames():
    svc = _service()
    t0 = datetime.datetime(2026, 1, 1, 12, 0, 0)
    svc.tracks.update(t0, [DET])
    svc.buffer.add(t0, _frame())
    (rec,) = svc.track_history()["tracks"]
    assert rec["first_frame_buffered"] is True and rec["last_frame_buffered"] is True
    svc.buffer.add(t0 + datetime.timedelta(minutes=5), _frame())
    svc.buffer._times.clear()  # buffer now starts after the track: both ends evicted
    svc.buffer._times.append(t0 + datetime.timedelta(minutes=5))
    (rec,) = svc.track_history()["tracks"]
    assert rec["first_frame_buffered"] is False


def test_capture_frame_returns_raw_pixels_only():
    svc = _service()
    meta, data = svc.capture_frame()
    assert data[:2] == b"\xff\xd8" and "detections" not in meta
    meta, _ = svc.capture_frame(box=[10, 10, 30, 30])
    assert meta["roi_display"] and "detections" not in meta


def test_buffered_times_default_is_span_only():
    svc = _service()
    out = svc.buffered_times(0)
    assert out["recent"] == [] and out["count"] == 0
