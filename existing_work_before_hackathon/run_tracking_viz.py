"""Frame-by-frame visualization of TrackCorrector + PeriodCorrector on historical detections.

Read-only: pulls detections from MySQL, runs both correctors with save_db=False,
and writes one JPG per frame (cat index 0) plus an optional MP4.

Usage (from existing_work_before_hackathon/):
    python run_tracking_viz.py --view-id 0 \
        --start "2024-12-05 21:00" --end "2024-12-05 21:05"
"""
import argparse
import datetime
import glob
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("LOG_LEVEL", "ERROR")

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from tqdm.auto import tqdm

from model.db import MysqlDB
from model.period_corrector import PeriodCorrector
from model.track_corrector import TrackCorrector

MIN_CONFIDENCE = 0.5


def load_detections(db, view_id, start_time, end_time, table="detection"):
    # Rows before ~2025-05 predate views and have view_id NULL, so include them too.
    columns = ["create_time", "cat_id", "box_x1", "box_y1", "box_x2", "box_y2", "conf"]
    sql = f"""
    SELECT img.create_time, det.{", det.".join(columns[1:])}
    FROM images img
    LEFT JOIN {table} det
    ON img.create_time = det.create_time AND det.view_id <=> img.view_id
    WHERE (img.view_id = %s OR img.view_id IS NULL) AND img.create_time BETWEEN %s AND %s
    """
    detections = pd.DataFrame(db.execute_query(sql, (view_id, start_time, end_time)), columns=columns)
    detections["box"] = detections[["box_x1", "box_y1", "box_x2", "box_y2", "cat_id", "conf"]].values.tolist()
    return detections.sort_values("create_time")


def make_read_only(corrector):
    # The correctors write to MySQL even with save_db=False (new period ids, period updates),
    # so swap the write calls for no-ops and hand out fake period ids instead.
    counter = iter(range(1, 10**9))
    corrector.db.execute_insert = lambda *a, **k: next(counter)
    corrector.db.execute_update = lambda *a, **k: None
    corrector.db.execute_many_insert = lambda *a, **k: None
    corrector.db.execute_delete = lambda *a, **k: None
    return corrector


def plot_cat(track_corrector, period_corrector, cat_idx):
    # process_images(plot_progress=True) only plots cat index 0; hook select_target instead,
    # which is where process_images plots, so any cat can be visualized.
    select_target = track_corrector.select_target

    def select_target_and_plot(idx):
        select_target(idx)
        if idx == cat_idx:
            tc = track_corrector
            tc.plot_2mins(
                tc.buffer, tc.revised_detection, idx, tc.base, tc.target,
                frame_escalations=tc.escalations, period_escalations=period_corrector.escalations,
            )

    track_corrector.select_target = select_target_and_plot


def frames_to_video(frame_dir, video_path, fps):
    import cv2

    frames = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
    if not frames:
        print("No frames to stitch.")
        return
    height, width = cv2.imread(frames[0]).shape[:2]
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for f in frames:
        writer.write(cv2.resize(cv2.imread(f), (width, height)))
    writer.release()
    print(f"Video saved: {video_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view-id", type=int, required=True)
    parser.add_argument("--start", required=True, help='e.g. "2024-12-05 21:00" (seconds optional: "21:00:30.5")')
    parser.add_argument("--end", required=True, help='e.g. "2024-12-05 21:05" (seconds optional)')
    parser.add_argument("--cat-idx", type=int, default=0, help="index into the view's cat_ids to plot")
    parser.add_argument("--table", choices=["detection", "revised_detection"], default="detection",
                        help="detections to replay: raw model output or the stored revised ones")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    # absolute, because main() chdirs into it before writing the MP4
    args.out = os.path.abspath(args.out)

    start_time = datetime.datetime.fromisoformat(args.start)
    end_time = datetime.datetime.fromisoformat(args.end)

    detections = load_detections(MysqlDB(), args.view_id, start_time, end_time, args.table)
    print(f"Loaded {len(detections)} rows, {detections['create_time'].nunique()} frames")

    track_corrector = make_read_only(TrackCorrector(view_id=args.view_id))
    period_corrector = make_read_only(PeriodCorrector(view_id=args.view_id))
    # plot_2mins draws upcoming raw detections from this and can't handle empty images
    track_corrector.images = detections.dropna(subset=["cat_id"])
    plot_cat(track_corrector, period_corrector, args.cat_idx)
    counts = detections["cat_id"].dropna().astype(int).value_counts()
    print("Detections per cat_idx:", {track_corrector.id_to_idx.get(c, f"?{c}"): n for c, n in counts.items()})
    print(f"Plotting cat_idx {args.cat_idx} (cat_id {track_corrector.idx_to_id[args.cat_idx]})")

    # plot_2mins saves into ./tracking_process, so work inside the output dir
    os.makedirs(args.out, exist_ok=True)
    os.chdir(args.out)
    track_corrector.init_plot_progress()

    for create_time in tqdm(detections["create_time"].unique()):
        create_time = pd.Timestamp(create_time).to_pydatetime()
        det = detections[(detections["create_time"] == create_time) & (detections["conf"] > MIN_CONFIDENCE)]
        img_detected = dict(zip(det["cat_id"].values.astype(int), det["box"].values))
        for is_append, revised in track_corrector.process_images(
            create_time, img_detected, is_grey=0, model_id=0, view_id=args.view_id, device_id=0,
            save_db=False, plot_progress=False,
        ):
            if is_append:
                for period in period_corrector.process_detection(revised, view_id=args.view_id):
                    tqdm.write(f"Period: {list(period)}")
            else:
                period_corrector.unprocess_detection(revised)

    frame_dir = os.path.join(args.out, "tracking_process")
    print(f"Frames saved: {frame_dir}")
    if not args.no_video:
        frames_to_video(frame_dir, os.path.join(args.out, "tracking_process.mp4"), args.fps)


if __name__ == "__main__":
    main()
