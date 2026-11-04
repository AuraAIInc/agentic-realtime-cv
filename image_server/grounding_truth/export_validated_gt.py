"""Export a tracking run's validated rows as YOLO ground truth for an hour folder.

The validated result (``validated.jsonl``) is taken as 100% accurate: every
run frame (``images.jsonl``) gets a ``labels/<same stem as the image>.txt``
YOLO file — one ``<class_id> xc yc w h`` line per validated box, or an empty
file when the frame validated with no boxes. Class ids follow the hour
folder's ``classes_map.json`` order; boxes are normalized against each
image's own JPEG size. Only run frames are touched — label files outside
the run's window are left exactly as they are.

Usage (Python >= 3.10, stdlib only):
    python image_server/grounding_truth/export_validated_gt.py \\
        output/tracking_runs/<run> image_server/retrieved_images/2026-10-28-16
    python image_server/grounding_truth/export_validated_gt.py <run> <hour> --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fetch_hour_images import jpeg_size, to_yolo  # noqa: E402  (stdlib-only module)

logger = logging.getLogger(__name__)


def _read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _frame_base(create_time: str) -> str:
    return datetime.datetime.fromisoformat(create_time).strftime("%Y-%m-%d-%H-%M-%S-%f")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export validated rows as YOLO ground truth.")
    parser.add_argument("run_dir", help="tracking run directory (reads images.jsonl + validated.jsonl)")
    parser.add_argument("hour_dir", help="retrieved hour folder (reads images/ + classes_map.json, writes labels/)")
    parser.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] %(message)s")

    images_dir = os.path.join(args.hour_dir, "images")
    labels_dir = os.path.join(args.hour_dir, "labels")
    with open(os.path.join(args.hour_dir, "classes_map.json"), encoding="utf-8") as fh:
        order = json.load(fh)["order"]
    cls_of = {int(cid): i for i, cid in enumerate(order)}

    # Map frame timestamps to image stems (<id>_<timestamp>).
    base_of = {}
    for fname in os.listdir(images_dir):
        if fname.endswith(".jpg") and len(fname) > 30:
            base_of[fname[-30:-4]] = fname[:-4]

    frames = _read_jsonl(os.path.join(args.run_dir, "images.jsonl"))
    boxes = {}
    degenerate = 0
    for row in _read_jsonl(os.path.join(args.run_dir, "validated.jsonl")):
        if row.get("deleted") or row.get("box_x1") is None or row.get("cat_id") is None:
            continue
        boxes.setdefault(row["create_time"], []).append(row)

    try:
        os.makedirs(labels_dir, exist_ok=True)
    except OSError:
        pass
    stats = {"frames": 0, "with_boxes": 0, "empty": 0, "written": 0, "identical": 0,
             "changed": 0, "emptied": 0, "orphans": [], "size_mismatch": []}
    ref_size = None
    for frame in frames:
        ts = frame["create_time"]
        base = _frame_base(ts)
        stem = base_of.get(base)
        if stem is None:
            stats["orphans"].append(ts)
            continue
        stats["frames"] += 1
        try:
            size = jpeg_size(os.path.join(images_dir, stem + ".jpg"))
        except OSError:
            stats["orphans"].append(ts)
            continue
        ref_size = ref_size or size
        if tuple(size) != tuple(ref_size):
            stats["size_mismatch"].append(stem)
        lines = []
        for row in sorted(boxes.get(ts, []), key=lambda r: cls_of.get(int(r["cat_id"]), 99)):
            yolo = to_yolo(row["box_x1"], row["box_y1"], row["box_x2"], row["box_y2"], *size)
            if yolo is None:
                degenerate += 1
                continue
            lines.append(f"{cls_of[int(row['cat_id'])]} {yolo[0]:.6f} {yolo[1]:.6f} {yolo[2]:.6f} {yolo[3]:.6f}\n")
        stats["with_boxes" if lines else "empty"] += 1
        path = os.path.join(labels_dir, stem + ".txt")
        try:
            with open(path, encoding="utf-8") as fh:
                old = fh.readlines()
        except OSError:
            old = None
        if old == lines:
            stats["identical"] += 1
            continue
        if old is None:
            stats["written"] += 1
        elif not lines:
            stats["emptied"] += 1
        else:
            stats["changed"] += 1
        if not args.dry_run:
            with open(path, "w", encoding="utf-8") as fh:
                fh.writelines(lines)

    print(f"run frames: {stats['frames']} ({stats['with_boxes']} with boxes, {stats['empty']} empty)")
    print(f"labels: {stats['written']} new, {stats['changed']} changed, "
          f"{stats['identical']} identical, {stats['emptied']} emptied (now box-free)")
    print(f"image size: {ref_size[0]}x{ref_size[1]}" + (f" ({len(stats['size_mismatch'])} mismatched)" if stats["size_mismatch"] else ""))
    if degenerate:
        print(f"WARNING: {degenerate} degenerate boxes skipped")
    if stats["orphans"]:
        print(f"WARNING: {len(stats['orphans'])} run frames have no image (first: {stats['orphans'][0]})")
    print("label files outside the run window: untouched")
    if args.dry_run:
        print("(dry run — nothing written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
