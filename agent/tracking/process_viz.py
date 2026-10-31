"""Per-frame process visualization (ported from legacy plot_2mins / run_tracking_viz).

Renders one JPG per frame showing the buffer, committed revised track, base /
target markers, upcoming raw detections, and escalation overlays — then
stitches frames into an MP4. Explicit paths throughout; no chdir.
"""

from __future__ import annotations

import datetime
import glob
import os
import shutil

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

CAT_COLORS = ["#4A8CC3", "#E37939", "#8FBC89", "#F6A600", "#D5E4A8", "#9392BE", "#8891DB", "#E3A8A1", "#B5CDA3"]


def init_frame_dir(out_dir):
    """Create a fresh frame directory, removing any previous contents."""
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def render_process_frame(track_corrector, period_escalations, cat_idx, out_path, futures=None, width=640, height=480):
    """Render one process frame for ``cat_idx`` and save it to ``out_path``.

    ``futures`` is an optional DataFrame of upcoming raw detections (columns
    create_time, cat_id, box_x1..box_y2, conf); None skips the futures band.
    Returns ``out_path``, or None when the buffer is empty (nothing to draw).
    """
    buffer = track_corrector.buffer
    revised_detection = track_corrector.revised_detection
    base = track_corrector.base
    target = track_corrector.target
    frame_escalations = track_corrector.escalations
    time_range = 90

    if len(buffer[cat_idx]) == 0:
        return None
    plt.figure(figsize=(14, 5))
    create_time = buffer[cat_idx][-1][0]
    last_buffer = buffer[cat_idx][0][0]
    plt.fill_between([last_buffer, create_time], [0, 0], [2, 2], alpha=0.2)
    for b in buffer[cat_idx]:
        if b[1] is not None:
            plt.plot([b[0], b[0]], [b[1] / width, b[3] / width], color="#C7988C", lw=2)
            plt.plot([b[0], b[0]], [b[2] / height + 1, b[4] / height + 1], color="#C7988C", lw=2)

    for k in range(1, len(revised_detection[cat_idx]) + 1):
        b = revised_detection[cat_idx][-k]
        if b[1] is not None:
            plt.plot([b[0], b[0]], [b[1] / width, b[3] / width], color=CAT_COLORS[track_corrector.id_to_idx[b[7]]], lw=2)
            # plt.plot([b[0], b[0]], [b[2]/height+1, b[4]/height+1], color=CAT_COLORS[b[7]], lw=2)
        if b[0] < create_time - datetime.timedelta(seconds=time_range):
            break

    if futures is not None:
        upcoming = futures[(futures["create_time"] > create_time) & (futures["create_time"] < create_time + datetime.timedelta(seconds=time_range))]
        for i in range(len(upcoming)):
            img = upcoming.iloc[i]
            ct = img["create_time"]
            if ct < create_time:
                continue
            c_id = img["cat_id"]
            box_x1 = img["box_x1"]
            box_y1 = img["box_y1"]
            box_x2 = img["box_x2"]
            box_y2 = img["box_y2"]
            alpha = img["conf"]
            # print(create_time, c_id, cat_id, cat_id == c_id, box_x1, box_y1, box_x2, box_y2, [ct, ct], [box_x1/width, box_x2/width])
            plt.plot(
                [ct, ct],
                [box_x1 / width, box_x2 / width],
                color="#8891DB" if c_id == track_corrector.idx_to_id[cat_idx] else "grey",
                lw=3,
                alpha=alpha if c_id == track_corrector.idx_to_id[cat_idx] else 0.1,
            )
            plt.plot(
                [ct, ct],
                [box_y1 / height + 1, box_y2 / height + 1],
                color="#8891DB" if c_id == track_corrector.idx_to_id[cat_idx] else "grey",
                lw=3,
                alpha=alpha if c_id == track_corrector.idx_to_id[cat_idx] else 0.1,
            )

    if base[cat_idx] is not None and base[cat_idx][1] is not None:
        plt.plot([base[cat_idx][0], base[cat_idx][0]], [base[cat_idx][1] / width, base[cat_idx][4] / height + 1], color="#D5E4A8", lw=10, alpha=0.5)
    if target[cat_idx] is not None and target[cat_idx][1] is not None:
        plt.plot(
            [target[cat_idx][0], target[cat_idx][0]], [target[cat_idx][1] / width, target[cat_idx][4] / height + 1], color="#9392BE", lw=10, alpha=0.5
        )

    window_left = create_time - datetime.timedelta(seconds=time_range)
    window_right = create_time + datetime.timedelta(seconds=time_range)
    for esc in frame_escalations or []:
        if window_left <= esc.time <= window_right:
            plt.axvline(esc.time, color="red", lw=1.5, alpha=0.85)
    for esc in period_escalations or []:
        span_start, span_end = esc.time, esc.end_time or esc.time
        if span_end >= window_left and span_start <= window_right:
            plt.axvspan(max(span_start, window_left), min(span_end, window_right), color="red", alpha=0.12)

    # plt.text(buffer[cat_id][-1][0], 2.05, buffer[cat_id][-1][0], ha='center')
    plt.title(
        f"Create Time: {create_time}, \nBase: {list(base[cat_idx]) if base[cat_idx] is not None else None}\n,"
        + f"Target: {list(target[cat_idx]) if target[cat_idx] is not None else None}"
    )
    plt.tight_layout()
    plt.ylim(0, 1)
    plt.xlim(create_time - datetime.timedelta(seconds=time_range), create_time + datetime.timedelta(seconds=time_range))
    plt.savefig(out_path)
    plt.close()
    return out_path


def stitch_mp4(frame_dir, video_path, fps):
    """Stitch ``*.jpg`` frames in ``frame_dir`` into an MP4 at ``video_path``."""
    import cv2

    frames = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
    if not frames:
        print("No frames to stitch.")
        return None
    height, width = cv2.imread(frames[0]).shape[:2]
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for f in frames:
        writer.write(cv2.resize(cv2.imread(f), (width, height)))
    writer.release()
    print(f"Video saved: {video_path}")
    return video_path


__all__ = ["render_process_frame", "init_frame_dir", "stitch_mp4", "CAT_COLORS"]
