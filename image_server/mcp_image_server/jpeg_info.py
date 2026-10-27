"""Read JPEG width/height from the file header without decoding the image.

Reads only up to the SOF marker (usually the first few hundred bytes), so it
costs about the same as a stat() on an SSD.
"""

from __future__ import annotations

from typing import BinaryIO

# Start-of-frame markers carry the dimensions. C4 (DHT), C8 (JPG), CC (DAC) are not SOF.
_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def jpeg_dimensions(f: BinaryIO) -> dict | None:
    """Return {"width", "height", "channels", "progressive"} or None if not a readable JPEG."""
    if f.read(2) != b"\xff\xd8":
        return None
    while True:
        b = f.read(1)
        while b and b != b"\xff":  # skip to next marker
            b = f.read(1)
        while b == b"\xff":  # skip fill bytes
            b = f.read(1)
        if not b:
            return None
        marker = b[0]
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:  # markers without a length field
            continue
        if marker == 0xD9:  # end of image
            return None
        raw_len = f.read(2)
        if len(raw_len) < 2:
            return None
        length = int.from_bytes(raw_len, "big")
        if marker in _SOF:
            d = f.read(6)
            if len(d) < 6:
                return None
            return {
                "width": int.from_bytes(d[3:5], "big"),
                "height": int.from_bytes(d[1:3], "big"),
                "channels": d[5],
                "progressive": marker in (0xC2, 0xC6, 0xCA, 0xCE),
            }
        f.seek(length - 2, 1)
