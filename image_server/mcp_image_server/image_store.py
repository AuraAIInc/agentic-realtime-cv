"""Image lookup: SSD archive plus S3 for recent captures.

Local layout (same as the Flask API):

    <root>/YYYY-MM-DD/HH/MM/YYYY-MM-DD-HH-MM-SS-ffffff.jpg

The folder is derived from the file name, so checking an image is a single
stat() call: no directory listing, no index, no database.

Images captured after the S3 cutoff day (default 2026-10-20) live on S3 at
s3://<bucket>/<prefix>/YYYY-MM-DD/HH/MM/<name>.jpg instead; see s3_store.py.
"""

from __future__ import annotations

import datetime
import io
import os
import re

from mcp_image_server.jpeg_info import jpeg_dimensions
from mcp_image_server.timing import Timer

NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{6}\.jpg$")
PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(-\d{2}(-\d{2})?)?$")


# Pillow's JPEG quality (1-95 useful range) for crops / downscales; 90 keeps fur and edge detail.
JPEG_QUALITY = 90


class InvalidName(ValueError):
    pass


class ImageStore:
    def __init__(self, root: str, s3=None, s3_cutoff: datetime.date | None = None):
        self.root = root
        self.s3 = s3
        self._cutoff = s3_cutoff or (s3.cutoff if s3 is not None else None)

    # ---------------------------------------------------------------- routing
    def _use_s3(self, t: datetime.datetime) -> bool:
        return self.s3 is not None and self._cutoff is not None and t.date() > self._cutoff

    def _use_s3_day(self, day: str) -> bool:
        # Day prefixes are zero-padded YYYY-MM-DD, so string order matches date order.
        return self.s3 is not None and self._cutoff is not None and day > self._cutoff.isoformat()

    # ---------------------------------------------------------------- paths
    def resolve(self, image_name: str) -> tuple[datetime.datetime, str]:
        """Validate the name and build its path. The strict pattern rules out path traversal."""
        image_name = image_name.strip()
        if not NAME_RE.match(image_name):
            raise InvalidName(f"image name must look like 2026-10-24-20-15-03-123456.jpg, got {image_name!r}")
        try:
            t = datetime.datetime.strptime(image_name, "%Y-%m-%d-%H-%M-%S-%f.jpg")
        except ValueError as e:
            raise InvalidName(f"not a valid timestamp: {image_name!r} ({e})") from None
        return t, os.path.join(self.root, t.strftime("%Y-%m-%d/%H/%M"), image_name)

    # ---------------------------------------------------------------- lookup
    def info(self, image_name: str, include_dimensions: bool, timer: Timer) -> dict:
        t, path = self.resolve(image_name)
        if self._use_s3(t):
            out = self.s3.info(image_name, t, include_dimensions, timer)
            if out.get("exists"):
                return out
            # fall through to the SSD copy when S3 has no such key
        else:
            timer.mark("resolve")
        try:
            st = os.stat(path)
        except FileNotFoundError:
            timer.mark("stat")
            return {"image_name": image_name.strip(), "exists": False, "source": "local"}
        timer.mark("stat")
        out = {
            "image_name": image_name.strip(),
            "exists": True,
            "source": "local",
            "size_bytes": st.st_size,
            "captured_at": t.isoformat(timespec="microseconds"),
            "modified_at": datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="milliseconds"),
            "relative_path": os.path.relpath(path, self.root),
            "mime_type": "image/jpeg",
        }
        if include_dimensions:
            with open(path, "rb") as f:
                dims = jpeg_dimensions(f)
            out.update(dims or {"width": None, "height": None, "header_error": "could not read JPEG header"})
            timer.mark("read_header")
        return out

    def read(self, image_name: str, timer: Timer) -> tuple[dict, bytes | None]:
        t, path = self.resolve(image_name)
        if self._use_s3(t):
            info, data = self.s3.read(image_name, t, timer)
            if data is not None:
                return info, data
            # fall through to the SSD copy when S3 has no such key
        else:
            timer.mark("resolve")
        try:
            st = os.stat(path)
        except FileNotFoundError:
            timer.mark("stat")
            return {"image_name": image_name.strip(), "exists": False, "source": "local"}, None
        timer.mark("stat")
        with open(path, "rb") as f:
            data = f.read()
        timer.mark("read_file")
        info = {
            "image_name": image_name.strip(),
            "exists": True,
            "source": "local",
            "size_bytes": st.st_size,
            "captured_at": t.isoformat(timespec="microseconds"),
            "modified_at": datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="milliseconds"),
            "relative_path": os.path.relpath(path, self.root),
            "mime_type": "image/jpeg",
        }
        return info, data

    def list(self, prefix: str, limit: int, timer: Timer) -> dict:
        """List image names under a day (YYYY-MM-DD), hour (YYYY-MM-DD-HH) or minute (YYYY-MM-DD-HH-MM)."""
        prefix = prefix.strip()
        if not PREFIX_RE.match(prefix):
            raise InvalidName("prefix must be YYYY-MM-DD, YYYY-MM-DD-HH or YYYY-MM-DD-HH-MM")
        parts = prefix.split("-")
        if self._use_s3_day("-".join(parts[:3])):
            return self.s3.list(prefix, limit, timer)
        folder = os.path.join(self.root, "-".join(parts[:3]), *parts[3:])
        timer.mark("resolve")
        names: list[str] = []
        truncated = False
        if os.path.isdir(folder):
            for dirpath, dirnames, filenames in os.walk(folder):
                dirnames.sort()
                for fn in sorted(filenames):
                    if fn.endswith(".jpg"):
                        if len(names) >= limit:
                            truncated = True
                            break
                        names.append(fn)
                if truncated:
                    break
        timer.mark("list")
        return {"prefix": prefix, "count": len(names), "truncated": truncated, "images": names, "source": "local"}


def transform(data: bytes, max_side: int | None, box: list[int] | None, timer: Timer) -> tuple[bytes, dict]:
    """Optional crop / downscale, re-encoded as JPEG. Needs Pillow; only used when asked for."""
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    img.load()
    timer.mark("decode")
    meta = {"source_width": img.width, "source_height": img.height}
    if box:
        if len(box) != 4:
            raise InvalidName("box must be [x1, y1, x2, y2] in source pixels")
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, x2 = sorted((max(0, min(img.width, x1)), max(0, min(img.width, x2))))
        y1, y2 = sorted((max(0, min(img.height, y1)), max(0, min(img.height, y2))))
        if x2 - x1 < 1 or y2 - y1 < 1:
            raise InvalidName("box is empty after clamping to the image")
        img = img.crop((x1, y1, x2, y2))
        meta["box"] = [x1, y1, x2, y2]
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    timer.mark("crop_resize")
    buf = io.BytesIO()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(buf, "JPEG", quality=JPEG_QUALITY)
    timer.mark("encode")
    meta.update({"width": img.width, "height": img.height})
    return buf.getvalue(), meta
