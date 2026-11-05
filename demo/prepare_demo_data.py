"""Build the committed demo dataset (maintainers only; the demo itself never needs this).

Pulls everything ``run_demo.py`` needs for one camera window into
``demo/data/<hour>/`` so the end-to-end demo runs from this repo alone:

    raw_detections.jsonl   camera-format YOLO messages, one per frame (empty frames kept),
                           read-only SELECT from the shared MySQL (agent/tracking/db_stream.py)
    images/                the window's camera frames  (<idx>_<YYYY-MM-DD-HH-MM-SS-ffffff>.jpg)
    labels/                ground-truth YOLO labels for those frames (validated export)
    muse_labels/           MUSE SAM 3.1 offline label set   (scored as a reference stage)
    spark_labels/          MUSE SPARK 1.3 offline label set (scored as a reference stage)
    classes_map.json       YOLO class index -> cat id, classes.txt
    manifest.json          window, counts, sources

Sources: the MySQL ``images``/``detection`` tables (MYSQL_* in .env) and an hour folder
written by ``image_server/fetch_hour_images.py`` (default
``image_server/retrieved_images/<hour>``). Usage, from the repo root:

    python demo/prepare_demo_data.py                      # 2026-10-28 16:33-16:38, view 3
    python demo/prepare_demo_data.py --start 2026-10-28-16-33 --end 2026-10-28-16-38 \\
        --hour-dir image_server/retrieved_images/2026-10-28-16
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

STEM_RE = re.compile(r"^\d+_(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d+)$")
LABEL_DIRS = ("labels", "muse_labels", "spark_labels")
META_FILES = ("classes_map.json", "classes.txt")


def stem_time(stem: str) -> datetime.datetime | None:
    m = STEM_RE.match(stem)
    if not m:
        return None
    g = [int(v) for v in m.groups()[:6]]
    return datetime.datetime(*g, int(m.group(7).ljust(6, "0")[:6]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view-id", type=int, default=3)
    ap.add_argument("--start", default="2026-10-28-16-33", help="first minute, YYYY-MM-DD-HH-MM")
    ap.add_argument("--end", default="2026-10-28-16-38", help="last minute (included), YYYY-MM-DD-HH-MM")
    ap.add_argument("--hour-dir", default=None, help="retrieved hour folder (default image_server/retrieved_images/<hour>)")
    ap.add_argument("--out-root", default=str(REPO / "demo" / "data"))
    args = ap.parse_args(argv)

    from agent.env import load_repo_env
    from agent.tracking import db_stream

    load_repo_env()
    start = datetime.datetime.strptime(args.start, "%Y-%m-%d-%H-%M")
    end_excl = datetime.datetime.strptime(args.end, "%Y-%m-%d-%H-%M") + datetime.timedelta(minutes=1)
    hour = start.strftime("%Y-%m-%d-%H")
    if end_excl - datetime.timedelta(microseconds=1) > start.replace(minute=59, second=59):
        raise SystemExit("the window must stay inside one hour (the evaluator keys GT by hour)")
    src = Path(args.hour_dir) if args.hour_dir else REPO / "image_server" / "retrieved_images" / hour
    if not (src / "images").is_dir():
        raise SystemExit(f"no images in {src}; fetch the hour first: python image_server/fetch_hour_images.py {hour}")
    out = Path(args.out_root) / hour
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # 1. Raw detections, straight from MySQL (read-only).
    db = db_stream.connect()
    msgs = list(db_stream.fetch_messages(db, args.view_id, start, end_excl))
    with open(out / "raw_detections.jsonl", "w", encoding="utf-8") as fh:
        for m in msgs:
            fh.write(json.dumps(m) + "\n")
    n_det = sum(len(m["detections"]) for m in msgs)
    print(f"raw detections: {len(msgs)} frames, {n_det} boxes")

    # 2. Frames + label sets inside the window.
    kept: set[str] = set()
    for img in sorted((src / "images").glob("*.jpg")):
        t = stem_time(img.stem)
        if t is not None and start <= t < end_excl:
            (out / "images").mkdir(exist_ok=True)
            shutil.copy2(img, out / "images" / img.name)
            kept.add(img.stem)
    counts = {"images": len(kept)}
    for d in LABEL_DIRS:
        if not (src / d).is_dir():
            continue
        (out / d).mkdir()
        n = 0
        for stem in sorted(kept):
            f = src / d / f"{stem}.txt"
            if f.exists():
                shutil.copy2(f, out / d / f.name)
                n += 1
        counts[d] = n
    for name in META_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, out / name)
    print("copied:", counts)

    manifest = {
        "view_id": args.view_id,
        "window": {"start": start.isoformat(), "end_exclusive": end_excl.isoformat(),
                   "cli": {"--start": args.start, "--end": args.end}},
        "raw_detections": {"file": "raw_detections.jsonl", "frames": len(msgs), "boxes": n_det,
                           "source": "MySQL images/detection tables (read-only SELECT, agent/tracking/db_stream.py)"},
        "files": counts,
        "ground_truth": "labels/ (validated YOLO export; class order in classes_map.json)",
        "built": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
