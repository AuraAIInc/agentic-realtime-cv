"""Replay YOLO label files as raw detections through the tracking correctors.

Takes a retrieved-images hour folder (``images/`` + a ``*_labels/`` folder of
per-image YOLO ``.txt`` files, e.g. the ``muse_labels/`` ground truth) and
runs the exact ``agent.main`` pipeline on it: TrackCorrector,
PeriodCorrector, live supervisor-agent validation, validated outputs,
charts, and the session report. Run from the repo root::

    .venv/bin/python -m agent.replay_labels --view-id 3 \\
        --hour-dir image_server/retrieved_images/2026-10-28-16 \\
        --labels muse_labels --validate

Label semantics: a present-but-empty ``.txt`` is an observed empty frame
(ingested, advances timeouts); a *missing* ``.txt`` is an unobserved frame
(skipped, never fed as empty). Boxes get ``--conf`` (default 1.0: ground
truth is certain) and must clear ``--min-conf`` like any raw detection.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import re
import sys
from pathlib import Path

logger = logging.getLogger("agent.replay_labels")

# 5502_2026-10-28-16-33-00-304643.jpg — numeric id, then a camera timestamp.
_STEM_RE = re.compile(r"^\d+_(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{1,6})$")


def parse_frame_time(stem: str) -> datetime.datetime | None:
    """Timestamp from an image/label stem, or None when the name does not match."""
    m = _STEM_RE.match(stem)
    if not m:
        return None
    year, mon, day, hh, mm, ss, frac = m.groups()
    micro = int(frac.ljust(6, "0")[:6])
    try:
        return datetime.datetime(int(year), int(mon), int(day), int(hh), int(mm), int(ss), micro)
    except ValueError:
        return None


def load_order(hour_dir: Path) -> list[int]:
    """YOLO class index -> tracking cat id from the hour's classes_map.json."""
    import json

    raw = json.loads((hour_dir / "classes_map.json").read_text())
    order = [int(v) for v in raw.get("order", [])]
    if not order:
        raise ValueError(f"{hour_dir}/classes_map.json has no 'order' list")
    return order


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def yolo_to_pixels(parts: list[str], width: int, height: int) -> tuple[int, float, float, float, float] | None:
    """Parse one YOLO line to (class_idx, x1, y1, x2, y2) pixels; None when unusable."""
    if len(parts) < 5:
        return None
    try:
        cid = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:5])
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    x1 = min(max((cx - w / 2) * width, 0.0), float(width))
    y1 = min(max((cy - h / 2) * height, 0.0), float(height))
    x2 = min(max((cx + w / 2) * width, 0.0), float(width))
    y2 = min(max((cy + h / 2) * height, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return cid, x1, y1, x2, y2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent.replay_labels",
        description="Replay YOLO label files as raw detections through the tracking pipeline.",
        epilog="window: --start/--end are minute-precision timestamps (YYYY-MM-DD-HH-MM); "
        "omitted, the window covers the full label span. Missing label files are "
        "skipped (unobserved); empty ones are ingested as empty frames.",
    )
    parser.add_argument("--view-id", type=int, required=True, help="camera view id (cat ids resolve from cats.yaml tracking_ids)")
    parser.add_argument("--hour-dir", required=True, help="retrieved-images hour folder (images/ + a labels dir + classes_map.json)")
    parser.add_argument("--labels", default="muse_labels", help="label folder inside --hour-dir (default muse_labels)")
    parser.add_argument("--start", default=None, help='window start, e.g. "2026-10-28-16-33" (default: first label)')
    parser.add_argument("--end", default=None, help="window end minute, same format; the whole minute is included (default: last label)")
    parser.add_argument("--min-conf", type=float, default=0.5, help="keep detections strictly above this confidence (default 0.5)")
    parser.add_argument("--conf", type=float, default=1.0, help="confidence stamped on every label box (default 1.0)")
    parser.add_argument("--limit", type=int, default=None, help="ingest at most N frames (oldest first; default: all)")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True, help="write original.png/revised.png charts (default --plot)")
    parser.add_argument("--out", default=None, help="run output base dir (default: the tracking runs dir)")
    parser.add_argument(
        "--memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="feed labeled period activity to long-term memory (default --no-memory)",
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="validate every escalation live with the supervisor agent and write validated_detection "
        "outputs (default --validate: this script exists to exercise the full pipeline)",
    )
    parser.add_argument(
        "--report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write a report.md session summary after finish (default --report)",
    )
    return parser


def main(argv=None) -> int:
    from agent.env import load_repo_env

    load_repo_env()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] %(filename)s:%(lineno)d: %(message)s")

    from agent.main import TrackingRun, resolve_cat_ids, resolve_window
    from agent.tracking import RunStore
    from agent.tracking.store import DEFAULT_RUNS_DIR

    hour_dir = Path(args.hour_dir)
    labels_dir = hour_dir / args.labels
    images_dir = hour_dir / "images"
    for path in (hour_dir, labels_dir, images_dir):
        if not path.is_dir():
            parser.error(f"not a directory: {path}")
    try:
        order = load_order(hour_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    try:
        cat_ids, id_to_name = resolve_cat_ids(args.view_id)
    except ValueError as exc:
        parser.error(str(exc))
    unknown = [tid for tid in order if tid not in set(cat_ids)]
    if unknown:
        parser.error(f"classes_map order {order} has tracking ids unknown to view {args.view_id} (cats {cat_ids}): {unknown}")

    frames: list[tuple[datetime.datetime, Path]] = []
    for name in sorted(os.listdir(images_dir)):
        if not name.lower().endswith((".jpg", ".jpeg")):
            continue
        when = parse_frame_time(os.path.splitext(name)[0])
        if when is None:
            logger.warning("skipping image with unparseable name: %s", name)
            continue
        frames.append((when, images_dir / name))
    frames.sort()
    if not frames:
        parser.error(f"no timestamped images in {images_dir}")
    labeled = [(when, path) for when, path in frames if (labels_dir / (path.stem + ".txt")).is_file()]
    if not labeled:
        parser.error(f"no label files matching {images_dir} in {labels_dir}")
    start_s = args.start or labeled[0][0].strftime("%Y-%m-%d-%H-%M")
    end_s = args.end or labeled[-1][0].strftime("%Y-%m-%d-%H-%M")
    try:
        start, end_exclusive = resolve_window(start_s, end_s)
    except ValueError as exc:
        parser.error(str(exc))
    frames = [(when, path) for when, path in frames if start <= when < end_exclusive]
    if args.limit is not None:
        frames = frames[: max(args.limit, 0)]
    if not frames:
        parser.error("window holds no frames (check --start/--end against the label span)")

    end_minute = end_exclusive - datetime.timedelta(seconds=60)
    now = datetime.datetime.now()
    run_name = f"view{args.view_id}_{start:%Y%m%d-%H%M}_{end_minute:%H%M}_labels_{now:%Y%m%d-%H%M%S-%f}"
    store = RunStore(run_name, base_dir=args.out or DEFAULT_RUNS_DIR)
    run = TrackingRun(
        view_id=args.view_id,
        start=start,
        end_exclusive=end_exclusive,
        cat_ids=cat_ids,
        id_to_name=id_to_name,
        store=store,
        min_conf=args.min_conf,
        plot=args.plot,
        mode="labels",
        args_dict=vars(args),
        record_raw=True,  # synthetic camera lines stay re-replayable via --replay
        memory=args.memory,
        validate=args.validate,
        report=args.report,
    )
    logger.info(f"run dir: {store.run_dir} (view {args.view_id}, cats {cat_ids}, {len(frames)} frames)")

    size_cache: dict[Path, tuple[int, int]] = {}
    skipped_missing = 0
    for i, (when, img_path) in enumerate(frames):
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.is_file():
            skipped_missing += 1  # unobserved: never fed as empty
            continue
        if img_path not in size_cache:
            try:
                size_cache[img_path] = image_size(img_path)
            except Exception:
                logger.warning("unreadable image %s; frame skipped", img_path.name)
                skipped_missing += 1
                continue
        width, height = size_cache[img_path]
        by_cat: dict[int, list] = {}
        try:
            lines = label_path.read_text().splitlines()
        except OSError:
            logger.warning("unreadable label %s; frame skipped", label_path.name)
            skipped_missing += 1
            continue
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            parsed = yolo_to_pixels(parts, width, height)
            if parsed is None:
                run.counts["labels_bad_line"] += 1
                continue
            cid, x1, y1, x2, y2 = parsed
            if cid < 0 or cid >= len(order):
                run.counts["labels_unknown_class"] += 1
                continue
            tid = order[cid]
            box = [x1, y1, x2, y2, tid, args.conf]
            prev = by_cat.get(tid)
            if prev is None or (x2 - x1) * (y2 - y1) > (prev[2] - prev[0]) * (prev[3] - prev[1]):
                if prev is not None:
                    run.counts["labels_duplicate_class"] += 1
                by_cat[tid] = box
            else:
                run.counts["labels_duplicate_class"] += 1
        msg = {
            "view_id": args.view_id,
            "create_time": when,
            "detections": [
                {"cat_id": tid, "confidence": box[5], "box": box[:4]} for tid, box in sorted(by_cat.items())
            ],
            "is_grey": 0,
            "model_id": None,
        }
        run.process_message(msg)
        if (i + 1) % 250 == 0:
            logger.info("ingested %d/%d frames (%d escs)", i + 1, len(frames), len(run._escs.all))
    logger.info("ingest done: %d frames fed, %d skipped (no label), %d escalations", len(frames) - skipped_missing, skipped_missing, len(run._escs.all))
    run_dir = run.finish()
    print(f"run written: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
