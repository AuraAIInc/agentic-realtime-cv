"""Resolve tracker cat ids per camera view (Phase 3 of 5).

``cats.yaml`` stores each cat's per-camera tracker label under
``tracking_ids`` with keys like ``view_3_black_living_room``. Neither
``agent.yaml`` nor ``zones.yaml`` carries an explicit int -> key table
(``zones.yaml`` only keys views by bare int, e.g. ``"3"`` with the frame
size), so the view id is taken from the ``view_<id>_`` key prefix.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "agent_data" / "config"
CATS_FILE = "cats.yaml"

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


__all__ = ["resolve_cat_ids", "known_views", "load_cat_profiles", "CONFIG_DIR"]
