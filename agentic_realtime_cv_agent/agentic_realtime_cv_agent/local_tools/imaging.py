"""Image helpers: crops and numbered contact sheets (several crops sent as ONE image)."""

from __future__ import annotations

import io
import math

from PIL import Image, ImageDraw, ImageFont


def open_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def to_jpeg(im: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def crop(im: Image.Image, box, pad: float = 0.3, max_side: int | None = 384) -> Image.Image:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    c = im.crop((int(max(0, x1 - pad * w)), int(max(0, y1 - pad * h)), int(min(im.width, x2 + pad * w)), int(min(im.height, y2 + pad * h))))
    if max_side:
        c.thumbnail((max_side, max_side))
    return c


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def contact_sheet(tiles: list[tuple[Image.Image, str]], tile: int = 256, cols: int | None = None) -> Image.Image:
    """Grid of square tiles, each labelled with its number and caption."""
    n = len(tiles)
    cols = cols or min(n, 3 if n <= 6 else 4)
    rows = math.ceil(n / cols)
    label_h = max(18, tile // 12)
    sheet = Image.new("RGB", (cols * tile, rows * (tile + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font(max(12, tile // 18))
    for i, (im, caption) in enumerate(tiles):
        im = im.copy()
        im.thumbnail((tile, tile))
        x, y = (i % cols) * tile, (i // cols) * (tile + label_h)
        sheet.paste(im, (x + (tile - im.width) // 2, y + label_h + (tile - im.height) // 2))
        draw.text((x + 4, y + 2), f"{i + 1}: {caption}"[:48], fill="black", font=font)
        draw.rectangle([x, y, x + tile - 1, y + tile + label_h - 1], outline="#999999")
    return sheet
