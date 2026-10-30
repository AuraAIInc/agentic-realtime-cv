"""MCP tool definitions and the HTTP / stdio entry point.

Every tool response carries a `timing` object:
    {"elapsed_ms": <total server-side time>, "steps_ms": {<step>: ms, ...}}
measured from the moment the tool starts until its response (including base64
encoding of any image) is built. Network transfer time is not included; the
client can measure that itself as round-trip minus elapsed_ms.
"""

from __future__ import annotations

import argparse
import base64
import hmac
import json
import logging
from dataclasses import replace

from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, ImageContent, TextContent

from mcp_image_server.config import Settings
from mcp_image_server.image_store import ImageStore, InvalidName, transform
from mcp_image_server.s3_store import S3Store
from mcp_image_server.timing import Timer

logger = logging.getLogger("mcp_image_server")

INSTRUCTIONS = """\
Serves cat-monitor images: the SSD archive, plus AWS S3 for captures newer
than the cutoff day (default 2026-10-20, see IMAGE_S3_CUTOFF_DATE).
Image names are capture timestamps: YYYY-MM-DD-HH-MM-SS-ffffff.jpg
(the same names the camera writes and the /v1/image-server/query API uses).
Use image_exists / images_exist first (no image data), then
get_image only for the images you actually need. Every response includes
timing.elapsed_ms (server-side time for that call) and source (s3 or local).
"""


def _result(payload: dict, timer: Timer, image: tuple[bytes, str] | None = None, error: bool = False) -> CallToolResult:
    content = []
    if image is not None:
        data, mime = image
        b64 = base64.b64encode(data).decode()
        timer.mark("base64")
        content.append(ImageContent(type="image", data=b64, mime_type=mime))
    payload = {**payload, "timing": timer.report()}
    content.insert(0, TextContent(type="text", text=json.dumps(payload)))
    return CallToolResult(content=content, structured_content=payload, is_error=error)


def _error(e: Exception, timer: Timer, **context) -> CallToolResult:
    return _result({**context, "error": f"{type(e).__name__}: {e}"}, timer, error=True)


def build_server(settings: Settings) -> MCPServer:
    s3 = S3Store.from_settings(settings)
    if s3 is not None:
        logger.info(f"S3 backend on: s3://{s3.bucket}/{s3.prefix}/ for captures after {s3.cutoff}")
    else:
        logger.info("S3 backend off (no S3_AWS_ACCESS_KEY / S3_AWS_SECRET_ACCESS_KEY): local SSD only")
    store = ImageStore(settings.image_root, s3=s3, s3_cutoff=settings.s3_cutoff_date)
    mcp = MCPServer(name="rcat-image-server", instructions=INSTRUCTIONS)

    @mcp.tool(structured_output=False)
    def image_exists(image_name: str, include_dimensions: bool = True) -> CallToolResult:
        """Fast check whether an image is on the SSD. image_name is the capture timestamp
        YYYY-MM-DD-HH-MM-SS-ffffff.jpg (24-hour clock, 6-digit microseconds), e.g.
        2026-10-24-20-15-03-123456.jpg. Returns exists and, if it does, size_bytes,
        captured_at, modified_at, relative_path, and (include_dimensions=true) width, height and
        channels read from the JPEG header without decoding. One stat() (+ a few hundred bytes
        read for dimensions); no image data is returned."""
        timer = Timer()
        try:
            return _result(store.info(image_name, include_dimensions, timer), timer)
        except (InvalidName, OSError) as e:
            return _error(e, timer, image_name=image_name)

    @mcp.tool(structured_output=False)
    def images_exist(image_names: list[str], include_dimensions: bool = False) -> CallToolResult:
        """Batch version of image_exists for up to 1000 names in one call, each named
        YYYY-MM-DD-HH-MM-SS-ffffff.jpg (e.g. 2026-10-24-20-15-03-123456.jpg). Invalid names are
        reported per item instead of failing the whole call."""
        timer = Timer()
        if len(image_names) > settings.max_batch:
            return _error(ValueError(f"at most {settings.max_batch} names per call"), timer)
        results, found = [], 0
        for name in image_names:
            try:
                r = store.info(name, include_dimensions, Timer())
            except (InvalidName, OSError) as e:
                r = {"image_name": name, "exists": False, "error": str(e)}
            found += bool(r.get("exists"))
            results.append(r)
        timer.mark("check_all")
        return _result({"requested": len(image_names), "found": found, "results": results}, timer)

    @mcp.tool(structured_output=False)
    def get_image(
        image_name: str,
        max_side: int | None = None,
        box: list[int] | None = None,
    ) -> CallToolResult:
        """Return a JPEG image named YYYY-MM-DD-HH-MM-SS-ffffff.jpg (e.g.
        2026-10-24-20-15-03-123456.jpg). By default the file bytes are sent as stored (no decode,
        fastest). Optional: box=[x1, y1, x2, y2] crops in source pixels, max_side shrinks so the
        longest side is <= max_side; either one re-encodes the result as JPEG.
        If the image does not exist, returns exists=false and no image."""
        timer = Timer()
        try:
            info, data = store.read(image_name, timer)
            if data is None:
                return _result(info, timer)
            mime = "image/jpeg"
            if box or max_side:
                data, meta = transform(data, max_side, box, timer)
                info.update(meta)
            info.update({"returned_bytes": len(data), "returned_mime_type": mime})
            return _result(info, timer, image=(data, mime))
        except (InvalidName, OSError, ImportError) as e:
            return _error(e, timer, image_name=image_name)

    @mcp.tool(structured_output=False)
    def list_images(prefix: str, limit: int = 500) -> CallToolResult:
        """List image names sorted by time. prefix is the leading part of an image name, cut at a
        day, hour or minute boundary (zero-padded, 24-hour): YYYY-MM-DD (e.g. 2026-10-24),
        YYYY-MM-DD-HH (e.g. 2026-10-24-20) or YYYY-MM-DD-HH-MM (e.g. 2026-10-24-20-15). Seconds
        or partial fields (2026-10, 2026-10-24-2) are rejected. Use a minute or hour prefix for
        speed; a whole day can be tens of thousands of files."""
        timer = Timer()
        try:
            return _result(store.list(prefix, min(int(limit), settings.max_list), timer), timer)
        except (InvalidName, OSError) as e:
            return _error(e, timer, prefix=prefix)

    return mcp


class BearerAuth:
    """ASGI middleware: reject HTTP requests without the right bearer token."""

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


def build_http_app(settings: Settings):
    # json_response: plain JSON replies, not SSE (no 1 MB SSE event limit on clients, lower latency)
    app = build_server(settings).streamable_http_app(host=settings.host, json_response=True)
    return BearerAuth(app, settings.auth_token) if settings.auth_token else app


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s")
    settings = Settings.from_env()

    p = argparse.ArgumentParser(description="MCP server for the cat-monitor image SSD")
    p.add_argument("--transport", choices=["http", "stdio"], default="http")
    p.add_argument("--host", default=settings.host)
    p.add_argument("--port", type=int, default=settings.port)
    p.add_argument("--root", default=settings.image_root)
    a = p.parse_args()
    settings = replace(settings, image_root=a.root, host=a.host, port=a.port)

    logger.info(f"Image root: {settings.image_root}")
    if a.transport == "stdio":
        build_server(settings).run("stdio")
        return

    import uvicorn

    logger.info(f"MCP image server on http://{settings.host}:{settings.port}/mcp (auth={'on' if settings.auth_token else 'OFF'})")
    uvicorn.run(build_http_app(settings), host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
