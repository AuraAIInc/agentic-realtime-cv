"""MCP server exposing camera control tools over streamable HTTP.

Endpoint: http://<pi>:<MCP_PORT>/mcp   (default port 8765)
If MCP_AUTH_TOKEN is set, clients must send  Authorization: Bearer <token>.

Runs on its own thread with its own event loop. Tool functions are sync; the SDK
runs them in a worker thread, so blocking on a detection-loop job is fine.
If the server crashes it is restarted; detection is never affected.
"""

from __future__ import annotations

import concurrent.futures
import hmac
import json
import logging
import threading
import time
from typing import Literal

from camera_v3.camera_service import CameraService

logger = logging.getLogger(__name__)

INSTRUCTIONS = """\
Controls the RCAT cat camera on a Raspberry Pi. The camera always runs a default
detection loop and streams results over WebSocket; these tools let you inspect
frames, run bigger models on hard frames, zoom into regions, and temporarily
change the loop's frame rate / crop / model.

Resolution: capture_frame reads the always-on stream (get_status.capture_size, often
well below the sensor); it is instant and can look back with `at`. For more detail
(e.g. zooming into a small box) use capture_high_res; {sizes_line}
Images are always returned as JPEG.

Coordinates: every box (input and output) is [x1, y1, x2, y2] in display pixels,
the same space as the WebSocket detections (see get_status.display_size).
Times: ISO 8601, or the WebSocket create_time string.
Schedule changes always expire (duration_seconds); the camera returns to defaults on its own.
"""

def build_mcp(service: CameraService):
    from mcp.server.mcpserver import Image, MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    sizes = [f"{m['width']}x{m['height']}" for m in service.supported_sizes()]
    sizes_line = (
        f"the camera supports {', '.join(sizes)} (largest first)."
        if sizes
        else "get_camera_modes lists what the camera supports."
    )
    logger.info(f"MCP capture_high_res sizes: {', '.join(sizes) or 'unknown'}")
    mcp = MCPServer(name="rcat-camera", instructions=INSTRUCTIONS.format(sizes_line=sizes_line))

    def guarded(fn):
        """Run fn; expected failures are returned to the client with their message."""
        try:
            return fn()
        except (ValueError, RuntimeError, TimeoutError, concurrent.futures.TimeoutError) as e:
            raise ToolError(f"{type(e).__name__}: {e}") from e

    def text(fn) -> str:
        return json.dumps(guarded(fn), default=str)

    def with_image(fn):
        meta, data = guarded(fn)
        return [json.dumps(meta, default=str), Image(data=data, format="jpeg")]

    @mcp.tool(structured_output=False)
    def get_status() -> str:
        """Current mode (default/override), effective settings and time left on any override,
        loop and camera FPS, last detection message, available models, cat names, frame
        buffer span, and WebSocket state."""
        return text(lambda: service.status())

    @mcp.tool(structured_output=False)
    def capture_frame(
        box: list[int] | None = None,
        padding: float = 0.15,
        max_side: int | None = 1280,
        at: str | None = None,
    ):
        """Grab the newest frame of the always-on stream (or the buffered frame nearest `at`).
        box crops it (display coords, padded by `padding` x box size); for a sharper zoom use
        capture_high_res with the same box. The image is shrunk so its longest side is
        <= max_side (null = no resize). Instant; does not interrupt the detection loop."""
        return with_image(lambda: service.capture_frame(box, padding, max_side, at))

    @mcp.tool(structured_output=False)
    def get_camera_modes() -> str:
        """Resolutions capture_high_res supports (sizes, largest first; max_size = largest)
        and the current always-on capture_size."""
        return text(lambda: service.camera_modes())

    supported = f"Supported sizes: {', '.join(sizes)}. " if sizes else ""

    @mcp.tool(
        structured_output=False,
        description=(
            "Take a still at a different resolution than the always-on stream. "
            + supported
            + "width/height omitted = the largest size. A size the camera "
            "doesn't list, or only width / only height, snaps to the nearest supported size "
            "(see used_size / size_choice in the reply). Every size shows the full camera view, "
            "like the always-on stream; only resolution (and aspect ratio) differ. A box (display "
            "coords, padded by `padding` x box size) is cropped from the still for zoom detail. "
            "Use this rather than capture_frame to zoom: the crop comes from the larger still, so it "
            "has more detail. max_side shrinks only the returned image (null = full resolution). Pauses the detection loop ~3 s while the camera switches resolution "
            "and back."
        ),
    )
    def capture_high_res(
        width: int | None = None,
        height: int | None = None,
        box: list[int] | None = None,
        padding: float = 0.0,
        max_side: int | None = None,
    ):
        return with_image(lambda: service.capture_high_res(width, height, box, padding, max_side))

    @mcp.tool(structured_output=False)
    def list_models() -> str:
        """Every model detect / set_schedule can use, from the camera's models file: description,
        cat_classes (true = classes are this user's cats, detections carry cat_id), whether it is
        loaded, its class names once loaded, and any other details the file gives."""
        return text(lambda: service.models())

    models = ", ".join(service.inference.registry.names())

    @mcp.tool(
        structured_output=False,
        description=(
            f"Escalate: run one detection pass with `model` (one of: {models}; list_models describes "
            "them) on the newest frame or the buffered frame nearest `at`, optionally only inside "
            "`roi` (full-res crop) and at input size `imgsz` [w, h]. Does not touch the tracker. "
            "Delays the default loop by one inference."
        ),
    )
    def detect(
        model: str = "default",
        at: str | None = None,
        roi: list[int] | None = None,
        imgsz: list[int] | None = None,
        conf: float | None = None,
    ) -> str:
        return text(lambda: service.detect(model, at, roi, imgsz, conf))

    @mcp.tool(structured_output=False)
    def get_track_history(
        limit: int = 50,
        all_epochs: bool = False,
        at: str | None = None,
        window_seconds: float = 2.0,
    ) -> str:
        """Tracks seen by the default loop: first/last seen time and box, hits, max confidence,
        and cat_id counts. identity_consistent=false means the track changed cat_id at least
        once (a likely ID switch worth validating). Track IDs restart (new epoch) whenever
        the loop's model/roi/imgsz changes. `at` returns only tracks seen within
        window_seconds of that time (from every epoch). Short-term only: kept in the camera's
        memory (last 500 tracks, lost on restart). For older or complete history use the
        backend, which records every track from the WebSocket stream."""
        return text(lambda: service.track_history(limit, all_epochs, at, window_seconds))

    @mcp.tool(structured_output=False)
    def get_track_frame(
        track_id: int,
        which: Literal["first", "last"] = "first",
        epoch: int | None = None,
        crop: bool = True,
        padding: float = 0.3,
        max_side: int | None = 1024,
    ):
        """Return the buffered frame where a track first (or last) appeared, cropped around its
        box by default, for re-checking how a track started. Recent tracks only: the frame
        must still be in the look-back buffer (see list_buffered_frames for the time span it
        covers, often under a minute); older frames fail with an error."""
        return with_image(lambda: service.track_frame(track_id, which, epoch, crop, padding, max_side))

    @mcp.tool(structured_output=False)
    def list_buffered_frames(limit: int = 50) -> str:
        """Timestamps of the most recent processed frames kept in the camera's look-back buffer
        (full capture resolution, a fixed number of frames, lost on restart). `at` in
        capture_frame / detect and get_track_frame can only reach frames between oldest and
        newest; at full frame rate that is often under a minute."""
        return text(lambda: service.buffered_times(limit))

    @mcp.tool(structured_output=False)
    def set_schedule(
        duration_seconds: float = 300,
        fps: float | None = None,
        roi: list[int] | None = None,
        imgsz: list[int] | None = None,
        model: str | None = None,
        merge: bool = True,
        reason: str = "",
    ) -> str:
        """Temporarily change the default loop for duration_seconds (max 24 h), then it reverts.
        fps: frames per second (e.g. 1, or 0.0333 for one per 30 s).
        roi: only run the loop inside this box (display coords), e.g. the sleeping spot.
        imgsz: model input [w, h] (multiples of 32). model: run the loop with another model.
        Calling again while an override is active replaces it and restarts the timer. Fields
        you omit keep their current override value with merge=true (e.g. set fps, then add a
        roi without losing the fps), or go back to defaults with merge=false."""
        return text(lambda: service.set_schedule(duration_seconds, fps, roi, imgsz, model, reason, merge))

    @mcp.tool(structured_output=False)
    def reset_schedule() -> str:
        """Cancel any override and return the loop to default settings now."""
        return text(lambda: service.reset_schedule())

    return mcp


class BearerAuth:
    """Tiny ASGI middleware: reject HTTP requests without the right bearer token."""

    def __init__(self, app, token: str):
        self.app = app
        self.expected = f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            auth = dict(scope.get("headers") or []).get(b"authorization", b"")
            if not hmac.compare_digest(auth, self.expected):
                await send({"type": "http.response.start", "status": 401, "headers": [(b"content-type", b"text/plain")]})
                await send({"type": "http.response.body", "body": b"unauthorized"})
                return
        await self.app(scope, receive, send)


def serve_forever(service: CameraService, host: str, port: int, token: str | None) -> None:
    import uvicorn

    while True:
        try:
            mcp = build_mcp(service)
            app = mcp.streamable_http_app(host=host, json_response=True)  # plain JSON, no SSE event size limit
            if token:
                app = BearerAuth(app, token)
            logger.info(f"MCP server listening on http://{host}:{port}/mcp (auth={'on' if token else 'off'})")
            uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning")).run()
            logger.warning("MCP server stopped; restarting in 5 s")
        except Exception:
            logger.exception("MCP server crashed; restarting in 5 s")
        time.sleep(5)


def start_in_thread(service: CameraService, host: str, port: int, token: str | None) -> threading.Thread:
    t = threading.Thread(target=serve_forever, args=(service, host, port, token), name="mcp_server", daemon=True)
    t.start()
    return t
