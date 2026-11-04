"""Resolve tracker cat ids per camera view (Phase 3 of 5).

``cats.yaml`` stores each cat's per-camera tracker label under
``tracking_ids`` with keys like ``view_3_black_living_room``. Neither
``agent.yaml`` nor ``zones.yaml`` carries an explicit int -> key table
(``zones.yaml`` only keys views by bare int, e.g. ``"3"`` with the frame
size), so the view id is taken from the ``view_<id>_`` key prefix.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "agent_data" / "config"
CATS_FILE = "cats.yaml"
ZONES_FILE = "zones.yaml"
# Pre-view-size legacy default (640x480 camera assumption). Production runs
# resolve the real size via resolve_view_size; this only covers views missing
# from zones.yaml.
LEGACY_IMG_SIZE = (640, 480)

_VIEW_KEY_RE = re.compile(r"^view_(\d+)_")


def _config_dir(config_dir=None) -> Path:
    return Path(config_dir) if config_dir is not None else CONFIG_DIR


def load_cat_profiles(config_dir=None) -> list:
    """Load the cat profile list from ``cats.yaml``."""
    path = _config_dir(config_dir) / CATS_FILE
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["cats"]


def known_views(config_dir=None) -> dict:
    """Map known int view ids -> tracking_ids key (from key prefixes)."""
    views = {}
    for cat in load_cat_profiles(config_dir):
        for key in (cat.get("tracking_ids") or {}):
            m = _VIEW_KEY_RE.match(key)
            if m:
                views.setdefault(int(m.group(1)), key)
    return views


def resolve_cat_ids(view_id: int, config_dir=None) -> tuple:
    """Return ``(cat_ids, id_to_name)`` for camera view ``view_id``.

    Raises a ``ValueError`` listing the known views when ``view_id`` has no
    ``tracking_ids`` entry anywhere in ``cats.yaml``.
    """
    view_id = int(view_id)
    cat_ids, id_to_name = [], {}
    for cat in load_cat_profiles(config_dir):
        for key, tid in (cat.get("tracking_ids") or {}).items():
            m = _VIEW_KEY_RE.match(key)
            if m and int(m.group(1)) == view_id:
                cat_ids.append(int(tid))
                id_to_name[int(tid)] = cat.get("name", str(tid))
                break
    if not cat_ids:
        known = known_views(config_dir)
        known_list = sorted(known)
        detail = ", ".join(f"{vid} ({known[vid]})" for vid in known_list) or "none"
        raise ValueError(f"Unknown view_id {view_id}: no tracking_ids entry for it in {CATS_FILE}; known views: {detail}")
    return cat_ids, id_to_name


def resolve_view_size(view_id: int, config_dir=None) -> tuple:
    """Return ``(width, height)`` frame size for camera view ``view_id``.

    Source is ``zones.yaml`` (``views.<id>.size``), the same pixel space the
    view's detection boxes use. Views missing from zones.yaml (or an unreadable
    file) fall back to the legacy 640x480 size with a warning, preserving the
    pre-view-size behavior.
    """
    view_id = int(view_id)
    try:
        with open(_config_dir(config_dir) / ZONES_FILE, encoding="utf-8") as fh:
            views = (yaml.safe_load(fh) or {}).get("views") or {}
    except OSError:
        views = {}
    entry = views.get(str(view_id), views.get(view_id, {})) or {}
    size = entry.get("size")
    if (
        isinstance(size, (list, tuple))
        and len(size) == 2
        and all(isinstance(v, (int, float)) and v > 0 for v in size)
    ):
        return (int(size[0]), int(size[1]))
    logger.warning(
        "view %s has no size in %s; falling back to legacy %sx%s",
        view_id,
        ZONES_FILE,
        *LEGACY_IMG_SIZE,
    )
    return LEGACY_IMG_SIZE


def appearance_for_tracking_id(view_id: int, tracking_id: int, config_dir=None) -> str | None:
    """One-line appearance text for a tracker label, e.g. ``"Qbo: long-haired ..."``.

    Returns None when the view/label has no profile or no summary (callers
    must treat appearance as optional enrichment, never a requirement).
    """
    view_id, tracking_id = int(view_id), int(tracking_id)
    for cat in load_cat_profiles(config_dir):
        for key, tid in (cat.get("tracking_ids") or {}).items():
            m = _VIEW_KEY_RE.match(key)
            if m and int(m.group(1)) == view_id and int(tid) == tracking_id:
                summary = (cat.get("summary") or "").strip()
                if not summary:
                    return None
                return f"{cat.get('name', tracking_id)}: {summary}"
    return None


__all__ = [
    "resolve_cat_ids",
    "resolve_view_size",
    "known_views",
    "load_cat_profiles",
    "appearance_for_tracking_id",
    "CONFIG_DIR",
    "LEGACY_IMG_SIZE",
]
