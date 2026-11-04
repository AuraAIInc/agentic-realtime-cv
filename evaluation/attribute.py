"""Attribute identity errors to the pipeline step that caused them.

For one run + hour, every revised and validated row is matched to the ground-truth
box in its frame (IoU >= 0.5, best match). Each row then has a GT class (or none =
false positive). The report tallies, per escalation/verdict that touched the row:

    fixed   revised class wrong -> validated class right
    broke   revised class right -> validated class wrong
    kept-ok / kept-bad   class unchanged, right / wrong
    dropped-tp / dropped-fp  row removed by validation (was a real cat / was not)

and, for the tracker step, raw detector class -> revised class on the same frames.

    .venv/bin/python evaluation/attribute.py --run output/tracking_runs/<run> \
        --hour image_server/retrieved_images/2026-10-28-16 [--gt-source labels]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import GT_SOURCES, jpeg_size, load_yolo_labels  # noqa: E402
from metrics import Box, iou  # noqa: E402


def rows(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def key(r: dict) -> tuple:
    return (r["create_time"], round(r["box_x1"]), round(r["box_y1"]), round(r["box_x2"]), round(r["box_y2"]))


def gt_class(r: dict, gt: dict, thr: float = 0.5):
    """GT cat id of the best-overlapping GT box, or None if no box overlaps enough."""
    best, cid = thr, None
    b = Box(r["box_x1"], r["box_y1"], r["box_x2"], r["box_y2"], -1)
    for g in gt.get(r["create_time"], []):
        v = iou(b, g)
        if v >= best:
            best, cid = v, g.cls
    return cid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--hour", required=True)
    ap.add_argument("--gt-source", default="labels", choices=list(GT_SOURCES))
    ap.add_argument("--json", help="write the tallies here")
    a = ap.parse_args()

    order = json.load(open(os.path.join(a.hour, "classes_map.json")))["order"]
    img = next(p for p in sorted(os.listdir(os.path.join(a.hour, "images"))) if p.endswith(".jpg"))
    w, h = jpeg_size(os.path.join(a.hour, "images", img))
    gt = load_yolo_labels(os.path.join(a.hour, GT_SOURCES[a.gt_source]), order, w, h)
    frames = {json.loads(l)["create_time"] for l in open(os.path.join(a.run, "images.jsonl"))}
    gt = {t: v for t, v in gt.items() if t in frames}

    raw = rows(os.path.join(a.run, "stream.jsonl"))
    rev = rows(os.path.join(a.run, "revised.jsonl"))
    val = rows(os.path.join(a.run, "validated.jsonl"))
    val_by_key = {key(r): r for r in val}

    # tracker step: real detections only (orig_cat_id set) -> did relabeling help?
    trk = collections.Counter()
    for r in rev:
        if r.get("orig_cat_id") is None:
            g = gt_class(r, gt)
            trk["interp-" + ("fp" if g is None else ("ok" if g == r["cat_id"] else "bad"))] += 1
            continue
        g = gt_class(r, gt)
        if g is None:
            trk["det-fp"] += 1
        elif r["orig_cat_id"] == r["cat_id"]:
            trk["kept-ok" if g == r["cat_id"] else "kept-bad"] += 1
        else:
            trk["relabel-fixed" if g == r["cat_id"] else ("relabel-broke" if g == r["orig_cat_id"] else "relabel-bad")] += 1

    by = collections.defaultdict(collections.Counter)
    for r in rev:
        g = gt_class(r, gt)
        v = val_by_key.get(key(r))
        if v is None:
            by["(removed)"]["dropped-" + ("fp" if g is None else "tp")] += 1
            continue
        tag = f"{v.get('validated_by') or '-'} {v.get('verdict') or '-'}"
        if g is None:
            by[tag]["fp-" + ("relabeled" if v["cat_id"] != r["cat_id"] else "kept")] += 1
        elif v["cat_id"] == r["cat_id"]:
            by[tag]["kept-ok" if g == r["cat_id"] else "kept-bad"] += 1
        else:
            by[tag]["fixed" if g == v["cat_id"] else "broke"] += 1
    added = [v for k, v in val_by_key.items() if k not in {key(r) for r in rev}]
    for v in added:
        g = gt_class(v, gt)
        tag = f"{v.get('validated_by') or '-'} {v.get('verdict') or '-'} (added)"
        by[tag]["added-" + ("fp" if g is None else ("ok" if g == v["cat_id"] else "bad"))] += 1

    print(f"# {os.path.basename(a.run.rstrip('/'))}")
    print("tracker:", dict(trk))
    gt_cls = collections.Counter(g.cls for v in gt.values() for g in v)
    print("gt boxes per class:", dict(gt_cls))
    net = lambda c: c["fixed"] - c["broke"]
    for tag, c in sorted(by.items(), key=lambda kv: net(kv[1])):
        print(f"{net(c):+6d}  {tag:42s} {dict(c)}")
    if a.json:
        json.dump({"tracker": trk, "by_verdict": by}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
