"""Pure image helpers: resizing, cropping, coordinate mapping, encoding."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass(frozen=True)
class Affine:
    """Maps pixel coords in one image to another: out = in * scale + offset."""

    sx: float = 1.0
    sy: float = 1.0
    ox: float = 0.0
    oy: float = 0.0

    def box(self, b) -> list[int]:
        x1, y1, x2, y2 = b
        return [
            int(round(x1 * self.sx + self.ox)),
            int(round(y1 * self.sy + self.oy)),
            int(round(x2 * self.sx + self.ox)),
            int(round(y2 * self.sy + self.oy)),
        ]


def scale_box(box, from_size: tuple[int, int], to_size: tuple[int, int]) -> list[int]:
    sx, sy = to_size[0] / from_size[0], to_size[1] / from_size[1]
    return Affine(sx, sy).box(box)


def clamp_box(box, size: tuple[int, int]) -> list[int]:
    w, h = size
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
    y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
    return [x1, y1, x2, y2]


def pad_box(box, pad_frac: float, size: tuple[int, int]) -> list[int]:
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad_frac, (y2 - y1) * pad_frac
    return clamp_box([x1 - px, y1 - py, x2 + px, y2 + py], size)


def frame_size(frame: np.ndarray) -> tuple[int, int]:
    return frame.shape[1], frame.shape[0]


def resize_to(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if frame_size(frame) == tuple(size):
        return frame
    return cv2.resize(frame, tuple(size), interpolation=cv2.INTER_AREA)


def fit_max_side(frame: np.ndarray, max_side: int | None) -> np.ndarray:
    if not max_side:
        return frame
    w, h = frame_size(frame)
    s = max_side / max(w, h)
    if s >= 1:
        return frame
    return cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def crop_display_roi(full: np.ndarray, roi_display, display_size: tuple[int, int]) -> tuple[np.ndarray, Affine]:
    """Crop `full` using a box given in display (camera_img_size) coords.

    Returns the crop and the Affine that maps crop pixel coords back to display coords.
    """
    full_size = frame_size(full)
    fx1, fy1, fx2, fy2 = clamp_box(scale_box(roi_display, display_size, full_size), full_size)
    if fx2 - fx1 < 2 or fy2 - fy1 < 2:
        raise ValueError(f"ROI {roi_display} is empty after clamping to the frame")
    crop = full[fy1:fy2, fx1:fx2]
    kx, ky = display_size[0] / full_size[0], display_size[1] / full_size[1]
    return crop, Affine(kx, ky, fx1 * kx, fy1 * ky)


def match_view(src: np.ndarray, dst: np.ndarray, max_side: int = 960, min_inliers: int = 15) -> tuple[Affine, int] | None:
    """Affine mapping `src` pixel coords to `dst` pixel coords, from ORB feature matches.

    The camera crops / bins its sensor differently per mode, so two frames of the same scene
    at different sizes are related by a scale + offset, not by a plain resize. Returns
    (affine, inlier count), or None when the views can't be matched (too dark, no texture).
    """

    def prep(img):
        k = min(1.0, max_side / max(img.shape[:2]))
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return (cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else g), k

    (gs, ks), (gd, kd) = prep(src), prep(dst)
    orb = cv2.ORB_create(4000)
    kp1, d1 = orb.detectAndCompute(gs, None)
    kp2, d2 = orb.detectAndCompute(gd, None)
    if d1 is None or d2 is None:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < min_inliers:
        return None
    a = np.float32([kp1[m.queryIdx].pt for m in good]) / ks
    b = np.float32([kp2[m.trainIdx].pt for m in good]) / kd
    m, mask = cv2.estimateAffine2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if m is None:
        return None
    inliers = int(mask.sum())
    (sx, r1, ox), (r2, sy, oy) = m
    if inliers < min_inliers or sx <= 0 or sy <= 0 or abs(r1) > 0.05 * sx or abs(r2) > 0.05 * sy:
        return None  # not a clean crop + scale: treat as unmatched
    return Affine(float(sx), float(sy), float(ox), float(oy)), inliers


def is_grey(frame: np.ndarray) -> bool:
    """True when all channels are equal (camera in IR / night mode)."""
    return bool(np.mean(np.std(frame, axis=2)) == 0)


def encode(frame: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("image encode failed")
    return buf.tobytes()


def decode(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
