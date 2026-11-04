"""Evaluate raw / revised / validated detections against YOLO ground truth.

Ground truth: <hour>/muse_labels/*.txt by default (SAM proposals + Muse
Spark classification; documented in image_server/grounding_truth/README.md
as the grounding-truth labels for evaluation). --gt-source selects the
label directory: muse | spark | labels.

Predictions: <run>/{stream,revised,validated}.jsonl from a tracking run
(stream = raw detector replay, revised = track/period corrector output,
validated = post-validation output). Frames are joined on create_time.

Example:
    python evaluation/evaluate.py \
        --run output/tracking_runs/view3_20261028-1633_1640_replay_db_20261102-231613-423659 \
        --hour image_server/retrieved_images/2026-10-28-16
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import Box, evaluate_frames  # noqa: E402

GT_SOURCES = {"muse": "muse_labels", "spark": "spark_labels", "labels": "labels"}
STAGES = (("raw", "stream.jsonl"), ("revised", "revised.jsonl"), ("validated", "validated.jsonl"))
STEM_RE = re.compile(r"^\d+_(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d+)$")


def stem_to_time(stem: str) -> str | None:
    """'5502_2026-10-28-16-33-00-304643' -> '2026-10-28T16:33:00.304643'."""
    m = STEM_RE.match(stem)
    if not m:
        return None
    y, mo, d, h, mi, s, us = m.groups()
    return f"{y}-{mo}-{d}T{h}:{mi}:{s}.{us}"


def jpeg_size(path: str) -> tuple | None:
    """Read (width, height) from a JPEG file without third-party libs."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    i = 2
    while i + 8 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3):
            h, w = struct.unpack(">HH", data[i + 5 : i + 9])
            return w, h
        if marker == 0xFF:
            i += 1
            continue
        i += 2 + struct.unpack(">H", data[i + 2 : i + 4])[0]
    return None


def load_yolo_labels(label_dir: str, order: list, img_w: int, img_h: int) -> dict:
    """Return {create_time: [Box, ...]} with YOLO ids mapped via order."""
    gt: dict = {}
    for path in sorted(glob.glob(os.path.join(label_dir, "*.txt"))):
        ts = stem_to_time(os.path.splitext(os.path.basename(path))[0])
        if ts is None:  # e.g. classes.txt
            continue
        boxes = []
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) != 5:
                    continue
                try:
                    cid, cx, cy, w, h = float(parts[0]), *map(float, parts[1:])
                except ValueError:
                    continue
                cls = order[int(cid)]
                x1 = (cx - w / 2) * img_w
                y1 = (cy - h / 2) * img_h
                x2 = (cx + w / 2) * img_w
                y2 = (cy + h / 2) * img_h
                boxes.append(Box(x1, y1, x2, y2, cls, 1.0))
        gt[ts] = boxes
    return gt


def load_jsonl(path: str) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_stage(run_dir: str, fname: str) -> dict:
    """Return {create_time: [Box, ...]} for one prediction stage."""
    path = os.path.join(run_dir, fname)
    preds: dict = {}
    if not os.path.exists(path):
        return preds
    for row in load_jsonl(path):
        preds.setdefault(row["create_time"], []).append(
            Box(
                row["box_x1"],
                row["box_y1"],
                row["box_x2"],
                row["box_y2"],
                row["cat_id"],
                row.get("conf") or 0.0,
            )
        )
    return preds


def evaluate_run(run_dir: str, hour_dir: str, gt_source: str, iou_thrs: list) -> dict:
    label_dir = os.path.join(hour_dir, GT_SOURCES[gt_source])
    with open(os.path.join(hour_dir, "classes_map.json")) as f:
        cmap = json.load(f)
    order = cmap["order"]
    names = {str(k): v for k, v in cmap["names"].items()}

    images = sorted(glob.glob(os.path.join(hour_dir, "images", "*.jpg")))
    size = jpeg_size(images[0]) if images else None
    img_w, img_h = size if size else (800, 600)

    gt = load_yolo_labels(label_dir, order, img_w, img_h)
    run_frames = [r["create_time"] for r in load_jsonl(os.path.join(run_dir, "images.jsonl"))]
    frame_set = [t for t in run_frames if t in gt]
    missing_gt = [t for t in run_frames if t not in gt]

    result = {
        "run": os.path.basename(os.path.normpath(run_dir)),
        "hour": os.path.basename(os.path.normpath(hour_dir)),
        "gt_source": gt_source,
        "gt_dir": label_dir,
        "class_names": names,
        "image_size": [img_w, img_h],
        "run_frames": len(run_frames),
        "eval_frames": len(frame_set),
        "frames_without_gt": len(missing_gt),
        "stages": {},
    }
    for stage, fname in STAGES:
        preds = load_stage(run_dir, fname)
        frames = {t: (preds.get(t, []), gt[t]) for t in frame_set}
        metrics = {f"iou_{t}": evaluate_frames(frames, t, classes=order) for t in iou_thrs}
        # Count predictions that fell on frames outside the GT coverage.
        outside = sum(len(v) for t, v in preds.items() if t not in gt)
        result["stages"][stage] = {
            "file": fname,
            "present": os.path.exists(os.path.join(run_dir, fname)),
            "preds_outside_gt_frames": outside,
            "metrics": metrics,
        }
    return result


def fmt(x: float) -> str:
    return f"{x:.3f}"


def stage_table(res: dict, iou_key: str, label: str, names: dict) -> str:
    lines = [
        f"| stage | P@{label} | R@{label} | F1@{label} | "
        "P(loc) | R(loc) | F1(loc) | mean IoU | cls-acc | TP | FP | FN |"
    ]
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for stage, _ in STAGES:
        s = res["stages"][stage]
        if not s["present"]:
            lines.append(f"| {stage} | — | — | — | — | — | — | — | — | — | — | — |")
            continue
        m = s["metrics"][iou_key]
        a = m["agnostic"]
        lines.append(
            f"| {stage} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} | "
            f"{fmt(a['precision'])} | {fmt(a['recall'])} | {fmt(a['f1'])} | "
            f"{fmt(m['mean_matched_iou'])} | {fmt(m['class_accuracy_on_matched'])} | "
            f"{m['tp']} | {m['fp']} | {m['fn']} |"
        )
    return "\n".join(lines)


def dfmt(x: float) -> str:
    return f"{x:+.3f}"


def delta_table(res: dict, iou_key: str) -> str:
    """Stage-to-stage deltas (revised − raw, validated − revised)."""
    lines = [
        "| step | ΔP | ΔR | ΔF1 | ΔP(loc) | ΔR(loc) | ΔF1(loc) | Δmean IoU | Δcls-acc |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for a_stage, b_stage in (("raw", "revised"), ("revised", "validated")):
        a = res["stages"][a_stage]
        b = res["stages"][b_stage]
        if not a["present"] or not b["present"]:
            lines.append(f"| {a_stage} → {b_stage} | — | — | — | — | — | — | — | — |")
            continue
        ma, mb = a["metrics"][iou_key], b["metrics"][iou_key]
        aa, ab = ma["agnostic"], mb["agnostic"]
        lines.append(
            f"| {a_stage} → {b_stage} "
            f"| {dfmt(mb['precision'] - ma['precision'])} "
            f"| {dfmt(mb['recall'] - ma['recall'])} "
            f"| {dfmt(mb['f1'] - ma['f1'])} "
            f"| {dfmt(ab['precision'] - aa['precision'])} "
            f"| {dfmt(ab['recall'] - aa['recall'])} "
            f"| {dfmt(ab['f1'] - aa['f1'])} "
            f"| {dfmt(mb['mean_matched_iou'] - ma['mean_matched_iou'])} "
            f"| {dfmt(mb['class_accuracy_on_matched'] - ma['class_accuracy_on_matched'])} |"
        )
    return "\n".join(lines)


def per_class_table(res: dict, iou_key: str, names: dict) -> str:
    lines = ["| stage | class | P | R | F1 | TP | FP | FN |"]
    lines.append("|---|---|---|---|---|---|---|---|")
    for stage, _ in STAGES:
        s = res["stages"][stage]
        if not s["present"]:
            continue
        for cid, mc in s["metrics"][iou_key]["per_class"].items():
            cname = names.get(str(cid), str(cid))
            lines.append(
                f"| {stage} | {cid} ({cname}) | {fmt(mc['precision'])} | "
                f"{fmt(mc['recall'])} | {fmt(mc['f1'])} | {mc['tp']} | "
                f"{mc['fp']} | {mc['fn']} |"
            )
    return "\n".join(lines)


def render_report(results: list, iou_thrs: list) -> str:
    parts = ["# Detection accuracy report", ""]
    for res in results:
        names = res["class_names"]
        parts += [
            f"## Run `{res['run']}`",
            "",
            f"- Hour: `{res['hour']}` — ground truth: `{res['gt_dir']}`",
            f"- Classes: {', '.join(f'{k}={v}' for k, v in names.items())}; "
            f"image size {res['image_size'][0]}x{res['image_size'][1]}",
            f"- Frames: {res['eval_frames']} evaluated "
            f"({res['run_frames']} in run, {res['frames_without_gt']} without GT)",
            "",
        ]
        for t in iou_thrs:
            key = f"iou_{t}"
            gt_n = res["stages"]["raw"]["metrics"][key]["n_gt"]
            parts += [
                f"### IoU >= {t} (GT boxes: {gt_n})",
                "",
                "P/R/F1 are class-aware (right class + box overlap). "
                "P/R/F1(loc) are class-agnostic (box overlap only). "
                "cls-acc = fraction of spatially matched pairs with the right class.",
                "",
                stage_table(res, key, str(t), names),
                "",
                delta_table(res, key),
                "",
                per_class_table(res, key, names),
                "",
            ]
    if len(results) > 1:
        key = f"iou_{iou_thrs[0]}"
        parts += [f"## Cross-run comparison (validated stage, IoU >= {iou_thrs[0]})", ""]
        parts.append("| run | P | R | F1 | P(loc) | R(loc) | mean IoU | cls-acc |")
        parts.append("|---|---|---|---|---|---|---|---|")
        for res in results:
            s = res["stages"]["validated"]
            if not s["present"]:
                parts.append(f"| `{res['run']}` | — | — | — | — | — | — | — |")
                continue
            m = s["metrics"][key]
            a = m["agnostic"]
            parts.append(
                f"| `{res['run']}` | {fmt(m['precision'])} | {fmt(m['recall'])} | "
                f"{fmt(m['f1'])} | {fmt(a['precision'])} | {fmt(a['recall'])} | "
                f"{fmt(m['mean_matched_iou'])} | {fmt(m['class_accuracy_on_matched'])} |"
            )
        parts.append("")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate detection stages vs YOLO GT.")
    ap.add_argument("--run", action="append", required=True, help="Tracking run dir (repeatable).")
    ap.add_argument("--hour", required=True, help="retrieved_images/<hour> dir with GT labels.")
    ap.add_argument("--gt-source", choices=sorted(GT_SOURCES), default="muse")
    ap.add_argument("--iou", type=float, action="append", default=[0.5, 0.75])
    ap.add_argument("--out", default=None, help="Report dir (default: evaluation/reports).")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(out_dir, exist_ok=True)

    results = [evaluate_run(r, args.hour, args.gt_source, args.iou) for r in args.run]
    for res in results:
        with open(os.path.join(out_dir, f"{res['run']}_metrics.json"), "w") as f:
            json.dump(res, f, indent=2)
    report = render_report(results, args.iou)
    report_path = os.path.join(out_dir, "ACCURACY_REPORT.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(report)
    print(f"\nwrote {report_path} + {len(results)} metrics.json file(s)")


if __name__ == "__main__":
    main()
