"""Camera service helpers: drawn detections, buffered detections, slim status.

Run from the camera_server/ folder:
    python -m pytest camera_v3/tests -q
"""

from __future__ import annotations

import datetime
from types import SimpleNamespace

import numpy as np

from camera_v3 import imaging
from camera_v3.camera_service import CameraService
from camera_v3.frame_buffer import FrameBuffer
from camera_v3.track_history import TrackHistory

DET = {"cat_id": 26, "box": [10, 10, 30, 30], "confidence": 0.75, "track_id": 1, "label": "Qbo"}


def _frame(w=100, h=80):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_draw_detection_marks_box_pixels():
    f = _frame()
    imaging.draw_detections(f, [DET], (100, 80))
    assert f.sum() > 0
    assert (f[10, 10] == [60, 220, 60]).all()  # cat green on the box corner


def test_draw_skips_boxes_outside_the_view():
    f = _frame()
    imaging.draw_detections(f, [{**DET, "box": [500, 500, 600, 600]}], (100, 80))
    assert f.sum() == 0


def test_draw_maps_crop_region():
    # Frame shows display region [10,10,50,50] at 80x80 px: box [10,10,30,30] lands at [0,0,40,40].
    f = np.zeros((80, 80, 3), dtype=np.uint8)
    imaging.draw_detections(f, [DET], (100, 80), view=(10, 10, 50, 50))
    assert (f[0, 0] == [60, 220, 60]).all()
    assert (f[39, 39] == [60, 220, 60]).all()


def test_buffer_keeps_detections_per_frame():
    buf = FrameBuffer(10)
    t0 = datetime.datetime(2026, 1, 1, 12, 0, 0)
    buf.add(t0, _frame(), [DET])
    buf.add(t0 + datetime.timedelta(seconds=1), _frame(), [])
    _, _, dets = buf.nearest(t0)
    assert dets == [DET]
    _, _, dets2 = buf.nearest(t0 + datetime.timedelta(seconds=1))
    assert dets2 == []


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
    svc.buffer.add(t0, _frame(), [DET])
    (rec,) = svc.track_history()["tracks"]
    assert rec["first_frame_buffered"] is True and rec["last_frame_buffered"] is True
    svc.buffer.add(t0 + datetime.timedelta(minutes=5), _frame(), [])
    svc.buffer._times.clear()  # buffer now starts after the track: both ends evicted
    svc.buffer._times.append(t0 + datetime.timedelta(minutes=5))
    (rec,) = svc.track_history()["tracks"]
    assert rec["first_frame_buffered"] is False


def test_capture_frame_draws_and_returns_same_frame_detections():
    svc = _service()
    meta, data = svc.capture_frame(draw_detections=True)
    assert meta["detections"] == [DET]
    assert meta["detections_frame"] == "2026-01-01-12-00-00-000000"
    assert data[:2] == b"\xff\xd8"
    plain, _ = svc.capture_frame()
    assert "detections" not in plain


def test_capture_frame_at_uses_buffered_detections():
    svc = _service()
    t0 = datetime.datetime(2026, 1, 1, 12, 0, 0)
    svc.buffer.add(t0, _frame(), [DET])
    meta, _ = svc.capture_frame(at=t0.isoformat(), draw_detections=True)
    assert meta["detections"] == [DET] and meta["detections_frame"].startswith("2026-01-01T12:00:00")


def test_buffered_times_default_is_span_only():
    svc = _service()
    out = svc.buffered_times(0)
    assert out["recent"] == [] and out["count"] == 0
