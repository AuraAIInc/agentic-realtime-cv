"""Known static objects the detector mistakes for a cat, per camera view.

``agent/agent_data/config/zones.yaml`` may list zones of ``kind: decoy``: the box of an object
that never moves and that the edge detector keeps labeling as a cat (view 3's exercise ball by
the back wall was tracked as "Qbo" for 26 minutes on 2026-11-01, 8 minutes on 10-25 and 6
minutes on 10-26). A detection whose box overlaps a decoy box at IoU >= ``iou`` is dropped at
ingest, before it can become a visit, steal a cat's identity from a real overlapping visit, or
reach the reviewer.

The match is deliberately strict: a cat lying *next to* the object produces a wider box
(IoU ~0.3-0.5 against the object alone) and is kept.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

ZONES_YAML = Path(__file__).resolve().parents[1] / "agent_data" / "config" / "zones.yaml"
DEFAULT_IOU = 0.6


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_decoys(view_id, path: Path | str | None = None) -> list[dict]:
    """Decoy zones for one view: [{'name', 'box', 'iou'}]. Missing file/view/yaml -> []."""
    p = Path(path) if path else ZONES_YAML
    try:
        import yaml

        data = yaml.safe_load(p.read_text()) or {}
    except Exception as exc:  # no file, no yaml, bad yaml: run without decoys
        logger.debug("no decoy zones (%s)", exc)
        return []
    view = ((data.get("views") or {}).get(str(view_id)) or {})
    out = []
    for z in view.get("zones") or []:
        if z.get("kind") != "decoy" or not isinstance(z.get("box"), (list, tuple)) or len(z["box"]) != 4:
            continue
        try:
            out.append({"name": str(z.get("name", "decoy")), "box": [float(v) for v in z["box"]],
                        "iou": float(z.get("min_iou", DEFAULT_IOU))})
        except (TypeError, ValueError):
            continue
    return out


def match_decoy(box, decoys: list[dict]) -> str | None:
    """Name of the decoy this box sits on, or None."""
    for d in decoys:
        if box_iou(box, d["box"]) >= d["iou"]:
            return d["name"]
    return None
