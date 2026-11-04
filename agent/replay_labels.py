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

After the run, validated rows are exported as YOLO files (real sightings
only — interpolated guesses never become ground truth) into
``<hour-dir>/validated_labels/`` — never into ``labels/``, which stays the
reference input. Use :func:`export_run` to backfill an already-finished run.
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


def collect_frames(images_dir: Path) -> list[tuple[datetime.datetime, Path]]:
    """All timestamped images, oldest first (unparseable names logged + skipped)."""
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
    return frames


def fed_frames(frames: list[tuple[datetime.datetime, Path]], labels_dir: Path) -> list[tuple[datetime.datetime, Path]]:
    """Frames having a label file (observable); missing files are never fed."""
    return [(when, path) for when, path in frames if (labels_dir / (path.stem + ".txt")).is_file()]


def export_validated_labels(
    run_dir: Path | str,
    fed: list[tuple[datetime.datetime, Path]],
    order: list[int],
    out_dir: Path | str,
    classes_src: Path | None = None,
) -> dict:
    """Write validated rows as YOLO ``<image-stem>.txt`` files; return stats.

    Ground truth keeps real sightings only: tombstones, interpolated rows
    (tracker guesses and merge gap-fills alike), unboxed rows, and unknown
    identities are all excluded. Every fed frame gets a file (empty when no
    box survived); unobserved frames get none.
    """
    import json

    from agent.validation.apply import VALIDATED_FILE

    run_dir, out_dir = Path(run_dir), Path(out_dir)
    stats = {"files": 0, "boxes": 0, "dropped_interp": 0, "unknown_cat": 0, "unmatched_time": 0, "bad_box": 0}
    if not fed:
        logger.warning("no fed frames; label export skipped")
        return stats
    rows = [json.loads(line) for line in (run_dir / VALIDATED_FILE).read_text().splitlines() if line.strip()]
    idx_of = {tid: i for i, tid in enumerate(order)}
    by_time: dict[datetime.datetime, Path] = {}
    for when, path in fed:
        by_time.setdefault(when, path)
    sizes: dict[Path, tuple[int, int]] = {}
    lines_by_stem: dict[str, list[str]] = {path.stem: [] for _, path in fed}
    for row in rows:
        if row.get("deleted") or row.get("orig_cat_id") is None:
            stats["dropped_interp"] += 1
            continue
        box = [row.get(k) for k in ("box_x1", "box_y1", "box_x2", "box_y2")]
        if not all(isinstance(v, (int, float)) for v in box):
            stats["bad_box"] += 1
            continue
        if row.get("cat_id") not in idx_of:
            stats["unknown_cat"] += 1
            continue
        try:
            ts = datetime.datetime.fromisoformat(row["create_time"])
        except (KeyError, ValueError):
            stats["unmatched_time"] += 1
            continue
        img_path = by_time.get(ts)
        if img_path is None:
            stats["unmatched_time"] += 1
            continue
        if img_path not in sizes:
            sizes[img_path] = image_size(img_path)
        width, height = sizes[img_path]
        x1, y1, x2, y2 = box
        cx = min(max((x1 + x2) / 2 / width, 0.0), 1.0)
        cy = min(max((y1 + y2) / 2 / height, 0.0), 1.0)
        w = min(max((x2 - x1) / width, 0.0), 1.0)
        h = min(max((y2 - y1) / height, 0.0), 1.0)
        if w <= 0 or h <= 0:
            stats["bad_box"] += 1
            continue
        lines_by_stem[img_path.stem].append(f"{idx_of[row['cat_id']]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, lines in lines_by_stem.items():
        (out_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    if classes_src is not None and Path(classes_src).is_file():
        (out_dir / "classes.txt").write_text(Path(classes_src).read_text())
    stats["files"] = len(lines_by_stem)
    stats["boxes"] = sum(len(v) for v in lines_by_stem.values())
    return stats


def export_run(run_dir: Path | str, hour_dir: Path | str, export_dir_name: str = "validated_labels") -> dict:
    """Backfill YOLO labels for an already-finished labels-mode run.

    Re-derives the run's fed frames from its ``run.json`` args (same
    window/limit/labels-dir the run used).
    """
    import json

    from agent.main import resolve_window

    run_dir, hour_dir = Path(run_dir), Path(hour_dir)
    meta = json.loads((run_dir / "run.json").read_text())
    args = meta.get("args") or {}
    labels_dir = hour_dir / (args.get("labels") or "muse_labels")
    images_dir = hour_dir / "images"
    order = load_order(hour_dir)
    frames = collect_frames(images_dir)
    labeled = fed_frames(frames, labels_dir)
    start_s = args.get("start") or labeled[0][0].strftime("%Y-%m-%d-%H-%M")
    end_s = args.get("end") or labeled[-1][0].strftime("%Y-%m-%d-%H-%M")
    start, end_exclusive = resolve_window(start_s, end_s)
    frames = [(when, path) for when, path in frames if start <= when < end_exclusive]
    if args.get("limit") is not None:
        frames = frames[: max(int(args["limit"]), 0)]
    fed = fed_frames(frames, labels_dir)
    return export_validated_labels(run_dir, fed, order, hour_dir / export_dir_name, hour_dir / "classes.txt")


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
    parser.add_argument(
        "--export-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="export validated rows as YOLO files (real sightings only) into "
        "<hour-dir>/<--export-dir> after finish (default --export-labels)",
    )
    parser.add_argument(
        "--export-dir",
        default="validated_labels",
        help="label export folder inside --hour-dir (default validated_labels)",
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

    frames = collect_frames(images_dir)
    if not frames:
        parser.error(f"no timestamped images in {images_dir}")
    labeled = fed_frames(frames, labels_dir)
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
    # Local track-box lookups resolve the revised track file through this.
    os.environ["TRACK_RUN_DIR"] = str(store.run_dir)
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
    fed: list[tuple[datetime.datetime, Path]] = []
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
        fed.append((when, img_path))
        if (i + 1) % 250 == 0:
            logger.info("ingested %d/%d frames (%d escs)", i + 1, len(frames), len(run._escs.all))
    logger.info("ingest done: %d frames fed, %d skipped (no label), %d escalations", len(frames) - skipped_missing, skipped_missing, len(run._escs.all))
    run_dir = run.finish()
    if args.export_labels:
        try:
            if not (run_dir / "validated.jsonl").is_file():
                logger.warning("no validated.jsonl (validation off?); label export skipped")
            else:
                stats = export_validated_labels(run_dir, fed, order, hour_dir / args.export_dir, hour_dir / "classes.txt")
                logger.info(
                    "exported validated labels to %s: %d files, %d boxes (%d interp excluded, %d unknown-cat, %d unmatched, %d bad-box)",
                    hour_dir / args.export_dir,
                    stats["files"],
                    stats["boxes"],
                    stats["dropped_interp"],
                    stats["unknown_cat"],
                    stats["unmatched_time"],
                    stats["bad_box"],
                )
        except Exception:
            logger.exception("label export failed; run artifacts are complete without it")
    print(f"run written: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
