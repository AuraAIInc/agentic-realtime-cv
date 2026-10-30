# mcp_image_server

MCP server for the cat-monitor images: the SSD archive (`/mnt/extreme/Cat_monitor_data`)
plus AWS S3 for captures after 2026-10-20.
It runs next to the Flask API (`app.py`, port 10000) and reads the same folders.
It doesn't import `src/`, so it doesn't open MySQL connections; S3 is only
contacted for recent images when S3 credentials are configured.

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

## Download an hour of images

[`../fetch_hour_images.py`](../fetch_hour_images.py) downloads every image of one hour
through this server's tools (`list_images`, then `get_image` per image). It runs on any
machine that can reach the server and needs only the Python standard library.

```
python image_server/fetch_hour_images.py 2025-05-05-19                  # YYYY-MM-DD-HH, 24-hour
python image_server/fetch_hour_images.py 2025-05-05-19 --out ~/Downloads/cats --workers 8
```

- Images land in `image_server/retrieved_images/<hour>/` (git-ignored) unless `--out` is given.
- Server and token come from `IMAGE_MCP_URL` and `MCP_AUTH_TOKEN` in the environment or the
  nearest `.env` (the agent's settings); `--url` overrides the server.
- Files already in the output folder are skipped, so rerun the same command to resume an
  interrupted download.
- Exits non-zero if the server is unreachable, the hour has no images, or any download fails.
- About 1 minute per 1,000 images (~150 MB) over Wi-Fi with 8 workers.

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

## Where images come from

Captures on or before `IMAGE_S3_CUTOFF_DATE` (default `2026-10-20`) are read
from the SSD (`IMAGE_ROOT`). Captures after that day are read from AWS S3 at
`s3://<bucket>/<prefix>/YYYY-MM-DD/HH/MM/<name>.jpg` (same keys the camera
writes), falling back to the SSD copy when S3 has no such key. Every response
carries `source: "s3"` or `"local"`. Without S3 credentials the server runs
local-only.

## Configuration (.env)

```
MCP_AUTH_TOKEN=<long random string>   # the single shared value: same line in every server's .env and the agent's
IMAGE_MCP_PORT=10001
IMAGE_MCP_HOST=0.0.0.0
IMAGE_ROOT=/mnt/extreme/Cat_monitor_data
IMAGE_S3_BUCKET=aura-history-image    # or HISTORY_IMAGE_BUCKET_NAME
IMAGE_S3_PREFIX=14/c2d5f3ce14067817
IMAGE_S3_CUTOFF_DATE=2026-10-20
S3_AWS_DEFAULT_REGION=<region, e.g. us-west-1>
S3_AWS_ACCESS_KEY=<access key id>         # same keys the camera uses
S3_AWS_SECRET_ACCESS_KEY=<secret access key>
```

Responses are plain JSON (not SSE streams). That keeps large images under the
MCP client's 1 MB per-event SSE limit.

## Install

Needs Python ≥ 3.10.

```
pip install -r requirements-mcp.txt
```

**Hackathon-temporary:** on the SSD host, `../run_mcp_server.sh` runs the code in
this repo folder (`~/agentic-realtime-cv/image_server`) but with the original
server's venv (`~/camera_server/.venv`, override with `ORIGINAL_VENV`), logging to
`mcp_server.log` there. By hand that is:

```
source ~/camera_server/.venv/bin/activate
cd ~/agentic-realtime-cv/image_server
nohup python -m mcp_image_server >> mcp_server.log 2>&1 &
tail -f mcp_server.log
```
