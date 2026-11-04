"""Score predicted YOLO label folders against labels/ ground truth.

Frame set: the tracking run's ingested frames (``images.jsonl``) that have
an image file. Ground truth comes from ``labels/<stem>.txt`` (missing or
empty means no boxes). Each ``--pred`` folder (e.g. ``muse_labels``,
``spark_labels``) is matched per frame with greedy same-class IoU matching
at ``--iou`` (default 0.5): a same-class pair at/above threshold is a true
positive; a leftover pair at/above threshold across classes counts as
misclassified (one FP + one FN); leftover predictions are FPs, leftover
ground truth is FNs. Missing prediction files count as empty predictions
(no boxes) and are reported as uncovered.

Usage (Python >= 3.10, stdlib only):
    python image_server/grounding_truth/evaluate_labels.py <run_dir> <hour_dir> \\
        --pred muse_labels --pred spark_labels [--iou 0.5]

Prints per-predictor tables and writes a Markdown + JSON report to
image_server/grounding_truth/reports/eval-<stamp>.md (and .json).
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)

GROUNDING_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(GROUNDING_DIR, "reports")


def _read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_yolo(path):
    """Return ([(class_id, x1, y1, x2, y2)] in normalized coords, bad_line_count)."""
    boxes, bad = [], 0
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return None, 0  # missing file: caller distinguishes from empty
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            bad += 1
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            bad += 1
            continue
        if w <= 0 or h <= 0:
            bad += 1
            continue
        boxes.append((cls, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes, bad


def _iou(a, b) -> float:
    ix1, iy1 = max(a[1], b[1]), max(a[2], b[2])
    ix2, iy2 = min(a[3], b[3]), min(a[4], b[4])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[3] - a[1]) * max(0.0, a[4] - a[2])
    bb = max(0.0, b[3] - b[1]) * max(0.0, b[4] - b[2])
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def _match(gt, pred, thresh):
    """Greedy match; returns (tp, fp, fn, misclassified, ious, per_class).

    Same-class pairs (>= thresh, best IoU first) match first; leftover
    cross-class pairs >= thresh are misclassified (scored as FP + FN).
    per_class maps class_id -> [tp, fp, fn] (a misclassified pair gives the
    GT class an FN and the predicted class an FP).
    """
    pairs = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            pairs.append((_iou(g, p), i, j))
    pairs.sort(reverse=True)
    used_g, used_p = set(), set()
    tp = fp = fn = mis = 0
    ious = []
    per_class = {}

    def bump(cls, k):
        per_class.setdefault(cls, [0, 0, 0])[k] += 1

    for ov, i, j in pairs:
        if ov < thresh or i in used_g or j in used_p:
            continue
        if gt[i][0] != pred[j][0]:
            continue
        used_g.add(i)
        used_p.add(j)
        tp += 1
        ious.append(ov)
        bump(gt[i][0], 0)
    for ov, i, j in pairs:
        if ov < thresh or i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        mis += 1
        fp += 1
        fn += 1
        bump(gt[i][0], 2)
        bump(pred[j][0], 1)
    for i, g in enumerate(gt):
        if i not in used_g:
            fn += 1
            bump(g[0], 2)
    for j, p in enumerate(pred):
        if j not in used_p:
            fp += 1
            bump(p[0], 1)
    return tp, fp, fn, mis, ious, per_class


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0 if not fn else 0.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(frames, gt_boxes, pred_dir, thresh):
    """Score one prediction folder; frames/stems aligned with gt_boxes."""
    agg = {"tp": 0, "fp": 0, "fn": 0, "mis": 0, "ious": [], "per_class": {},
           "covered": 0, "missing": 0, "bad_lines": 0, "exact_frames": 0,
           "empty_gt": 0, "empty_gt_with_fp": 0, "pred_boxes": 0, "gt_boxes": 0}
    for stem, gt in zip(frames, gt_boxes):
        pred, bad = _read_yolo(os.path.join(pred_dir, stem + ".txt"))
        agg["bad_lines"] += bad
        if pred is None:
            agg["missing"] += 1
            pred = []
        else:
            agg["covered"] += 1
        agg["pred_boxes"] += len(pred)
        agg["gt_boxes"] += len(gt)
        if not gt:
            agg["empty_gt"] += 1
            if pred:
                agg["empty_gt_with_fp"] += 1
        tp, fp, fn, mis, ious, pc = _match(gt, pred, thresh)
        agg["tp"] += tp
        agg["fp"] += fp
        agg["fn"] += fn
        agg["mis"] += mis
        agg["ious"] += ious
        for cls, (t, f, n) in pc.items():
            c = agg["per_class"].setdefault(cls, [0, 0, 0])
            c[0] += t
            c[1] += f
            c[2] += n
        if fp == 0 and fn == 0:
            agg["exact_frames"] += 1
    return agg


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Score YOLO label folders against labels/ ground truth.")
    parser.add_argument("run_dir", help="tracking run directory (frame set from images.jsonl)")
    parser.add_argument("hour_dir", help="retrieved hour folder (labels/ is ground truth)")
    parser.add_argument("--pred", action="append", required=True, help="prediction folder name (repeatable)")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU match threshold (default 0.5)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] %(message)s")
    started = datetime.datetime.now()

    with open(os.path.join(args.hour_dir, "classes.txt"), encoding="utf-8") as fh:
        names = [line.strip() for line in fh if line.strip()]

    base_of = {}
    images_dir = os.path.join(args.hour_dir, "images")
    for fname in os.listdir(images_dir):
        if fname.endswith(".jpg") and len(fname) > 30:
            base_of[fname[-30:-4]] = fname[:-4]
    frames, gt_boxes, gt_missing = [], [], 0
    for frame in _read_jsonl(os.path.join(args.run_dir, "images.jsonl")):
        base = datetime.datetime.fromisoformat(frame["create_time"]).strftime("%Y-%m-%d-%H-%M-%S-%f")
        stem = base_of.get(base)
        if stem is None:
            continue
        gt, _ = _read_yolo(os.path.join(args.hour_dir, "labels", stem + ".txt"))
        if gt is None:
            gt_missing += 1
            gt = []
        frames.append(stem)
        gt_boxes.append(gt)

    results = {}
    for pred in args.pred:
        agg = evaluate(frames, gt_boxes, os.path.join(args.hour_dir, pred), args.iou)
        p, r, f = _prf(agg["tp"], agg["fp"], agg["fn"])
        agg.update({"precision": p, "recall": r, "f1": f,
                    "mean_iou": sum(agg["ious"]) / len(agg["ious"]) if agg["ious"] else None})
        results[pred] = agg

    stamp = started.strftime("%Y%m%d-%H%M%S")
    report = {"stamp": stamp, "run_dir": args.run_dir, "hour_dir": args.hour_dir,
              "iou_threshold": args.iou, "frames": len(frames),
              "gt_boxes": sum(len(g) for g in gt_boxes),
              "gt_files_missing": gt_missing, "predictors": {}}
    for pred, agg in results.items():
        report["predictors"][pred] = {
            "covered_frames": agg["covered"], "missing_files": agg["missing"],
            "bad_lines": agg["bad_lines"], "pred_boxes": agg["pred_boxes"],
            "tp": agg["tp"], "fp": agg["fp"], "fn": agg["fn"],
            "misclassified": agg["mis"], "precision": round(agg["precision"], 4),
            "recall": round(agg["recall"], 4), "f1": round(agg["f1"], 4),
            "mean_iou": round(agg["mean_iou"], 4) if agg["mean_iou"] is not None else None,
            "exact_frames": agg["exact_frames"], "empty_gt_frames": agg["empty_gt"],
            "empty_gt_with_fp": agg["empty_gt_with_fp"],
            "per_class": {str(c): {"tp": t, "fp": f, "fn": n,
                                   "precision": round(*_prf(t, f, n)[:1], 4),
                                   "recall": round(_prf(t, f, n)[1], 4)}
                           for c, (t, f, n) in sorted(agg["per_class"].items())},
        }

    try:
        os.makedirs(REPORTS_DIR, exist_ok=True)
    except OSError:
        pass
    with open(os.path.join(REPORTS_DIR, f"eval-{stamp}.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    lines = [f"# Label evaluation — {stamp}", "",
             f"- Run: `{args.run_dir}` · hour: `{args.hour_dir}` · IoU ≥ {args.iou}",
             f"- Frames: {len(frames)} · GT boxes: {report['gt_boxes']} "
             f"({report['gt_boxes'] / len(frames):.2f}/frame) · GT files missing: {gt_missing}", ""]
    for pred, agg in results.items():
        r = report["predictors"][pred]
        lines += [f"## {pred}", "",
                  f"- Coverage: {r['covered_frames']}/{len(frames)} frames "
                  f"({r['missing_files']} files missing, {r['bad_lines']} bad lines)",
                  f"- Boxes: {r['pred_boxes']} predicted vs {report['gt_boxes']} GT",
                  f"- TP {r['tp']} · FP {r['fp']} · FN {r['fn']} · misclassified {r['misclassified']}",
                  f"- Precision {r['precision']:.4f} · Recall {r['recall']:.4f} · F1 {r['f1']:.4f} · "
                  f"mean IoU {r['mean_iou'] if r['mean_iou'] is not None else 'n/a'}",
                  f"- Exact-match frames: {r['exact_frames']}/{len(frames)} · "
                  f"empty GT frames with FP: {r['empty_gt_with_fp']}/{r['empty_gt_frames']}", "",
                  "| Class | TP | FP | FN | Precision | Recall |",
                  "|---|---|---|---|---|---|"]
        for c, m in r["per_class"].items():
            name = names[int(c)] if int(c) < len(names) else f"class {c}"
            lines.append(f"| {name} ({c}) | {m['tp']} | {m['fp']} | {m['fn']} | "
                         f"{m['precision']:.4f} | {m['recall']:.4f} |")
        lines.append("")
    md_path = os.path.join(REPORTS_DIR, f"eval-{stamp}.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print(f"(JSON: {os.path.join(REPORTS_DIR, f'eval-{stamp}.json')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
