# mcp_image_server

MCP server for the cat-monitor images on the SSD (`/mnt/extreme/Cat_monitor_data`).
It runs next to the Flask API (`app.py`, port 10000) and reads the same folders.
It doesn't import `src/`, so it doesn't open MySQL or S3 connections.

```
python -m mcp_image_server                 # HTTP on 0.0.0.0:10001, endpoint /mcp
python -m mcp_image_server --transport stdio
python -m mcp_image_server --root /some/other/folder --port 10002
```

## Tools

Images are addressed by name, `YYYY-MM-DD-HH-MM-SS-ffffff.jpg`, the same names the
camera writes and `/v1/image-server/query/<name>` uses. The folder comes from the
name, so a lookup is one `stat()` call, with no directory scan.

| Tool | Returns |
|---|---|
| `image_exists(image_name, include_dimensions=true)` | `exists`; if true: `size_bytes`, `captured_at`, `modified_at`, `relative_path`, `mime_type`, and `width` / `height` / `channels` / `progressive` read from the JPEG header (no decode). No image data. |
| `images_exist(image_names, include_dimensions=false)` | Same, for up to 1000 names in one call; bad names reported per item. |
| `get_image(image_name, max_side?, box?)` | JPEG image content. By default the exact file bytes (no decode). `box` = crop `[x1,y1,x2,y2]`, `max_side` = downscale; either re-encodes as JPEG at quality 90 (needs Pillow). Missing image → `exists: false`, no image. |
| `list_images(prefix, limit=500)` | Image names under a day `YYYY-MM-DD`, hour `YYYY-MM-DD-HH` or minute `YYYY-MM-DD-HH-MM` (zero-padded, 24-hour, e.g. `2026-10-24-20-15`). |

## Timing

Every response (found, not found, or error) includes:

```json
"timing": {"elapsed_ms": 0.128, "steps_ms": {"resolve": 0.068, "stat": 0.03}}
```

`elapsed_ms` is server-side time from the tool starting until the response,
including base64 encoding of the image, is built. Network time isn't included;
the client can work it out as round trip minus `elapsed_ms`. The result is
in the text content (JSON) and also in `structured_content`.

Measured in testing (local disk, x86): `image_exists` ≈ 0.1 ms with dimensions,
`get_image` of a 1.1 MB original ≈ 4 ms (most of it base64), a crop + resize +
re-encode ≈ 110 ms (JPEG decode dominates). Expect several times slower on a Pi.

## Configuration (.env)

```
MCP_AUTH_TOKEN=<long random string>   # the single shared value: same line in every server's .env and the agent's
IMAGE_MCP_PORT=10001
IMAGE_MCP_HOST=0.0.0.0
IMAGE_ROOT=/mnt/extreme/Cat_monitor_data
```

Responses are plain JSON (not SSE streams). That keeps large images under the
MCP client's 1 MB per-event SSE limit.

## Install

Needs Python ≥ 3.10.

```
pip install -r requirements-mcp.txt
```
