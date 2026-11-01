"""What the MCP tools actually do, independent of MCP.

Frame reads (latest frame, ring buffer) happen on the calling thread; anything
that runs a model is handed to the detection loop via Controller.submit so the
loop stays the only user of the YOLO models.

All boxes in and out are in display coordinates (camera_img_size), the same
space as the boxes in the WebSocket stream.
"""

from __future__ import annotations

import datetime
import logging
import time

from camera_v3 import imaging
from camera_v3.controller import Controller
from camera_v3.detection_loop import DetectionLoop
from camera_v3.frame_buffer import FrameBuffer
from camera_v3.frame_source import PIXEL_FORMAT, FrameSource
from camera_v3.image_store import ImageStore
from camera_v3.inference import Inference
from camera_v3.track_history import TrackHistory

logger = logging.getLogger(__name__)

JPEG_QUALITY = 92


def _iso(t: datetime.datetime) -> str:
    return t.isoformat(timespec="milliseconds")


def parse_time(s: str) -> datetime.datetime:
    """Accept ISO 8601 or the WebSocket create_time format (YYYY-mm-dd-HH-MM-SS-ffffff).

    Frame times are naive local time, so a time with a UTC offset is converted to local.
    """
    try:
        t = datetime.datetime.fromisoformat(s)
    except ValueError:
        return datetime.datetime.strptime(s, "%Y-%m-%d-%H-%M-%S-%f")
    return t.astimezone().replace(tzinfo=None) if t.tzinfo else t


class CameraService:
    def __init__(
        self,
        cfg,
        source: FrameSource,
        controller: Controller,
        inference: Inference,
        buffer: FrameBuffer,
        tracks: TrackHistory,
        loop: DetectionLoop,
        image_store: ImageStore,
        ws_status=None,
    ):
        self.cfg = cfg
        self.source = source
        self.controller = controller
        self.inference = inference
        self.buffer = buffer
        self.tracks = tracks
        self.loop = loop
        self.image_store = image_store
        self.ws_status = ws_status or (lambda: {})
        # (stream size, still size) -> Affine from stream pixels to still pixels. Each mode crops the
        # sensor differently, but the mapping is fixed per mode, so it's measured once and reused.
        self._view_maps: dict[tuple, imaging.Affine] = {}

    # ------------------------------------------------------------ helpers
    def _latest(self):
        got = self.source.get_latest(timeout=5.0)
        if got is None:
            raise RuntimeError("no frame from camera in the last 5 s")
        frame, t, _ = got
        return frame, t

    def _frame_at(self, at: str | None):
        """(frame, time, loop detections for it). Live frames (at=None) carry no detections;
        use the loop's last message for those."""
        if not at:
            frame, t = self._latest()
            return frame, t, []
        want = parse_time(at)
        found = self.buffer.nearest(want)
        if found is None:
            raise RuntimeError("frame buffer is empty")
        t, frame, dets = found
        if want < t and (t - want).total_seconds() > 1.0:
            span = self.buffer.span()
            raise RuntimeError(f"frame at {at} has already left the buffer (oldest kept: {span.get('oldest')})")
        return frame, t, dets

    def _render(self, frame, t, max_side) -> tuple[dict, bytes]:
        """JPEG for the client, shrunk so its longest side is <= max_side (None = as is)."""
        out = imaging.fit_max_side(frame, max_side)
        meta = {"captured_at": _iso(t), "source_size": list(imaging.frame_size(frame)), "returned_size": list(imaging.frame_size(out))}
        return meta, imaging.encode(out, JPEG_QUALITY)

    def _max_size(self) -> list[int] | None:
        sizes = self.supported_sizes()
        return [sizes[0]["width"], sizes[0]["height"]] if sizes else None

    # ------------------------------------------------------------ status
    def status(self, verbose: bool = False) -> dict:
        """Slim by default (what callers need to pick the next tool); verbose adds
        grab timings, controller defaults, queue depths and websocket internals."""
        control = self.controller.status()
        ws = self.ws_status()
        span = self.buffer.span()
        out = {
            "device_id": self.cfg.device_id,
            "view_id": self.cfg.view_id,
            "sim": False,
            "display_size": list(self.cfg.camera_img_size),
            "capture_size": list(self.source.actual_size or self.cfg.effective_capture_size),
            "camera_max_size": self._max_size(),
            "camera_read_fps": round(self.source.read_fps, 2),
            "control": {k: control[k] for k in ("mode", "settings", "override_expires_in_s", "override_reason") if k in control},
            "loop": self.loop.snapshot(),
            "models": self.inference.registry.names(),
            "cat_names": {str(k): v for k, v in self.cfg.cat_names.items()},
            "frame_buffer": {k: span[k] for k in ("count", "oldest", "newest") if k in span},
            "track_epoch": self.tracks.epoch,
            "websocket": {"connected": ws.get("connected")} if isinstance(ws, dict) else ws,
        }
        if verbose:
            out["camera_read_ms"] = getattr(self.source, "read_ms", None)  # mean frame-grab time, 5 s window
            out["control"] = control
            out["frame_buffer"] = span
            out["websocket"] = ws
            out["image_upload_queue"] = self.image_store.qsize()
        return out

    # ------------------------------------------------------------ captures
    def capture_frame(self, box=None, padding=0.15, max_side=1280, at=None, draw_detections=False):
        """Newest (or buffered) stream frame, optionally cropped to a padded display-coords box.

        With draw_detections, the loop's boxes for that same frame are drawn and
        returned in meta["detections"], so one call shows the frame and what the
        model saw in it. For live frames (at=None) the boxes come from the loop's
        last processed message, which may be a frame behind.
        """
        frame, t, dets = self._frame_at(at)
        if not at:
            last = self.loop.last_message or {}
            dets, dets_from = list(last.get("detections") or []), last.get("create_time")
        else:
            dets_from = _iso(t)
        if not box:
            if draw_detections and dets:
                frame = imaging.draw_detections(frame.copy(), dets, tuple(self.cfg.camera_img_size))
            meta, data = self._render(frame, t, max_side)
        else:
            roi = imaging.pad_box(box, padding, self.cfg.camera_img_size)
            crop, _ = imaging.crop_display_roi(frame, roi, self.cfg.camera_img_size)
            if draw_detections and dets:
                crop = crop.copy()  # the ROI is a view into the source frame; never draw on it
                imaging.draw_detections(crop, dets, tuple(self.cfg.camera_img_size), view=tuple(roi))
            meta, data = self._render(crop, t, max_side)
            meta["roi_display"] = roi
            meta["zoom_factor"] = round(imaging.frame_size(frame)[0] / self.cfg.camera_img_size[0], 3)
        if draw_detections:
            meta["detections"] = dets
            meta["detections_frame"] = dets_from
        return meta, data

    # ------------------------------------------------------------ high resolution
    def supported_sizes(self) -> list[dict]:
        """Sizes the camera lists in PIXEL_FORMAT, largest first (any format if it lists none)."""
        modes = self.source.list_modes()
        modes = [m for m in modes if m["format"] == PIXEL_FORMAT] or modes
        by_size = {(m["width"], m["height"]): m for m in modes}
        return sorted(by_size.values(), key=lambda m: -m["width"] * m["height"])

    def camera_modes(self) -> dict:
        sizes = self.supported_sizes()
        out = {
            "device": self.source.device_path,
            "capture_size": list(self.source.actual_size or self.cfg.effective_capture_size),
            "max_size": [sizes[0]["width"], sizes[0]["height"]] if sizes else None,
            "sizes": [f"{m['width']}x{m['height']}" for m in sizes],
        }
        if self.source.device_path is None:
            out["note"] = "RTSP / file source: resolution is fixed by the stream; capture_high_res returns its native size"
        elif not sizes:
            out["note"] = "modes unknown (install v4l-utils on the Pi); capture_high_res will request 3840x2160"
        return out

    def capture_high_res(self, width=None, height=None, box=None, padding=0.0, max_side=None, warmup=5):
        logger.info(f"High-res request: size={width}x{height} box={box} padding={padding} max_side={max_side}")
        t0 = time.monotonic()
        size, fourcc, how = self._pick_size(width, height)
        logger.info(f"High-res target: {size[0]}x{size[1]} {fourcc} ({how})")
        extra = {"requested_size": [width, height], "used_size": list(size), "size_choice": how}
        if box:
            # Reference view for the box: the always-on stream, grabbed before the camera switches mode.
            stream, _ = self._latest()
        frame, t = self.source.capture_still(size, fourcc, warmup)
        full_size = imaging.frame_size(frame)
        extra["full_size"] = list(full_size)
        logger.info(f"High-res frame: {full_size[0]}x{full_size[1]} (requested {size[0]}x{size[1]})")
        if box:
            frame = self._crop_hires(frame, stream, box, padding, extra)
        meta, data = self._render(frame, t, max_side)
        meta.update(extra)
        logger.info(
            f"High-res done in {time.monotonic() - t0:.2f}s: returned {meta['returned_size'][0]}x{meta['returned_size'][1]} "
            f"{len(data) / 1024:.0f} KiB"
        )
        return meta, data

    def _pick_size(self, width, height) -> tuple[tuple[int, int], str | None, str]:
        """Turn the request into a size the camera actually lists, so the first call succeeds.

        Omitted = largest listed size. An unlisted size, or only width / only height, snaps to
        the closest listed size (same aspect ratio preferred) instead of letting the driver
        silently pick one.
        """
        sizes = self.supported_sizes()
        if not sizes:  # modes unknown (no v4l2-ctl): ask for what was requested and let the driver decide
            if (width is None) != (height is None):
                raise ValueError("give both width and height, or neither for the camera's maximum (camera modes unknown)")
            size = (int(width), int(height)) if width else (3840, 2160)
            return size, PIXEL_FORMAT, "camera modes unknown, requested as is" if width else "camera modes unknown, trying 3840x2160"
        if width is None and height is None:
            m = sizes[0]
            return (m["width"], m["height"]), m["format"], "largest listed size"
        exact = [m for m in sizes if m["width"] == width and m["height"] == height]
        if exact:
            return (width, height), exact[0]["format"], "listed size"
        if width is None or height is None:
            same = [m for m in sizes if m["width"] == width or m["height"] == height]
            if same:
                m = same[0]  # largest with that width / height
                return (m["width"], m["height"]), m["format"], f"largest listed size with {'width' if width else 'height'} {width or height}"
            want_area = (width or height * 4 / 3) * (height or width * 3 / 4)
            aspect = None
        else:
            want_area, aspect = width * height, width / height
        same_aspect = [m for m in sizes if aspect and abs(m["width"] / m["height"] / aspect - 1) < 0.02]
        m = min(same_aspect or sizes, key=lambda m: abs(m["width"] * m["height"] - want_area))
        listed = ", ".join(f"{s['width']}x{s['height']}" for s in sizes)
        return (m["width"], m["height"]), m["format"], f"{width}x{height} not supported, snapped to nearest of: {listed}"

    def _crop_hires(self, frame, stream, box, padding, extra: dict):
        """Crop a display-coords box out of a high-res still.

        Each size can have a different field of view (sensor crop), so the box is mapped through
        the measured stream -> still alignment rather than by scaling. Falls back to plain scaling
        (same view assumed) when the frames can't be matched, e.g. in the dark.
        """
        display = self.cfg.camera_img_size
        stream_size, full_size = imaging.frame_size(stream), imaging.frame_size(frame)
        key = (stream_size, full_size)
        aff = self._view_maps.get(key)
        if aff is not None:
            how = "cached alignment"
        else:
            t0 = time.monotonic()
            found = imaging.match_view(stream, frame)
            if found:
                aff, n = found
                self._view_maps[key] = aff
                how = f"matched ({n} inliers)"
            else:
                aff = imaging.Affine(full_size[0] / stream_size[0], full_size[1] / stream_size[1])
                how = "not aligned, assuming same field of view"
                extra["warning"] = (
                    f"could not align the {full_size[0]}x{full_size[1]} view with the stream; the crop assumes the same "
                    "field of view and may be offset from the box"
                )
            logger.info(f"High-res view alignment {stream_size} -> {full_size}: {how} in {time.monotonic() - t0:.2f}s")
        roi = imaging.pad_box(box, padding, display)
        roi_full = aff.box(imaging.scale_box(roi, display, stream_size))
        clamped = imaging.clamp_box(roi_full, full_size)
        if clamped[2] - clamped[0] < 2 or clamped[3] - clamped[1] < 2:
            raise ValueError(f"box {roi} is outside the {full_size[0]}x{full_size[1]} field of view; use a larger size")
        if clamped != roi_full:
            extra["warning"] = f"box {roi} is partly outside the {full_size[0]}x{full_size[1]} field of view; crop was clipped"
        extra.update(
            roi_display=roi,
            roi_full=clamped,
            view_alignment=how,
            stream_view_in_full=aff.box([0, 0, *stream_size]),
            zoom_factor=round(aff.sx * stream_size[0] / display[0], 3),  # still pixels per display pixel
        )
        if "warning" in extra:
            logger.warning(f"High-res crop: {extra['warning']}")
        logger.info(
            f"High-res crop: display roi {roi} -> {clamped} in {full_size[0]}x{full_size[1]} "
            f"(zoom x{extra['zoom_factor']}, {how})"
        )
        x1, y1, x2, y2 = clamped
        return frame[y1:y2, x1:x2]

    # ------------------------------------------------------------ inference
    def models(self) -> dict:
        return {n: self.inference.registry.describe(n) for n in self.inference.registry.names()}

    def detect(self, model="default", at=None, roi=None, imgsz=None, conf=None, timeout=60.0) -> dict:
        frame, t, _ = self._frame_at(at)
        if roi:
            roi = imaging.clamp_box(roi, self.cfg.camera_img_size)
        frame = frame.copy()
        dets = self.controller.submit(lambda: self.inference.run(frame, model, roi, imgsz, conf, track=False), timeout=timeout)
        try:
            cat_classes = bool(self.inference.registry.describe(model).get("cat_classes"))
        except Exception:
            cat_classes = None  # unknown model; inference already ran, don't fail the reply over metadata
        return {"captured_at": _iso(t), "model": model, "cat_classes": cat_classes, "roi": roi, "imgsz": imgsz, "detections": dets}

    # ------------------------------------------------------------ look-back
    def track_history(self, limit=50, all_epochs=False, at=None, window_seconds=2.0) -> dict:
        when = parse_time(at) if at else None
        tracks = self.tracks.snapshot(limit, not all_epochs, when, window_seconds)
        span = self.buffer.span()
        oldest, newest = span.get("oldest"), span.get("newest")
        for r in tracks:
            # ISO timestamps sort lexicographically, so string comparison is enough.
            r["first_frame_buffered"] = bool(oldest and newest and oldest <= r["first_seen"] <= newest)
            r["last_frame_buffered"] = bool(oldest and newest and oldest <= r["last_seen"] <= newest)
        out = {"epoch": self.tracks.epoch, "tracks": tracks}
        if when:
            out.update(at=_iso(when), window_seconds=window_seconds)
        return out

    def track_frame(self, track_id: int, which="first", epoch=None, crop=True, padding=0.3, max_side=1024):
        rec = self.tracks.get(track_id, epoch)
        if rec is None:
            raise ValueError(f"track {track_id} not found (epoch {epoch if epoch is not None else self.tracks.epoch})")
        at = rec["first_seen"] if which == "first" else rec["last_seen"]
        box = rec["first_box"] if which == "first" else rec["last_box"]
        meta, data = self.capture_frame(box if crop else None, padding, max_side, at)
        meta.update({"track": rec, "which": which, "box": box})
        return meta, data

    def buffered_times(self, limit=50) -> dict:
        return {**self.buffer.span(), "recent": self.buffer.times(limit) if limit > 0 else []}

    # ------------------------------------------------------------ scheduling
    def set_schedule(self, duration_seconds, fps=None, roi=None, imgsz=None, model=None, reason="", merge=True) -> dict:
        return self.controller.set_override(duration_seconds, reason=reason, merge=merge, fps=fps, roi=roi, imgsz=imgsz, model=model)

    def reset_schedule(self) -> dict:
        return self.controller.clear_override()
