import pandas as pd
import datetime
import matplotlib.pyplot as plt
from model.db import MysqlDB
import seaborn as sns
import numpy as np
import logging

CAT_COLORS = ["#4A8CC3", "#E37939", "#8FBC89", "#F6A600", "#D5E4A8", "#9392BE", "#8891DB", "#E3A8A1", "#B5CDA3"]
CAT_NAMES = ["Qb", "Qbo", "Yuki", "Momo", "Luna", "Neko", "Tora", "Hana", "Kumo"]
ROUND_SECONDS = datetime.timedelta(seconds=10)
sns.set_theme()


def _get_data_from_sql_for_plot(start_time, end_time, revised):
    detection_table = "revised_detection" if revised else "detection"
    db = MysqlDB()

    columns = ["image_name", "create_time", "light", "detected", "is_grey"]
    sql = f"""
    SELECT {", ".join(columns)} 
    FROM aura_cat.images 
    WHERE create_time > '{start_time}' and create_time < '{end_time}'
    """
    logging.info(sql)
    images = pd.DataFrame(data=db.execute_query(sql), columns=columns)
    logging.info(f"Retrieved image data: {len(images)}")
    images = images.sort_values("create_time")

    # detection tables key frames by create_time; image_name was dropped from them
    columns = ["create_time", "cat_id", "box_x1", "box_y1", "box_x2", "box_y2", "conf"]
    sql = f"""
    SELECT {", ".join(columns)} 
    FROM aura_cat.{detection_table} 
    WHERE create_time > '{start_time}' and create_time < '{end_time}'
    """
    logging.info(sql)
    detections = pd.DataFrame(data=db.execute_query(sql), columns=columns)
    logging.info(f"Retrieved detection data: {len(detections)}")
    
    columns = ["cat_id", "time_from", "time_to", "detected_frames", "total_frames", "total_seconds", "average_conf"]
    sql = f"""
    SELECT {", ".join(columns)} from aura_cat.detected_period
    WHERE time_from < '{end_time}' and time_to > '{start_time}'
    """
    logging.info(sql)
    detected_period = pd.DataFrame(data=db.execute_query(sql), columns=columns)
    logging.info(f"Retrieved detected_period data: {len(detected_period)}")

    columns = ["cat_id", "time_from", "time_to", "detected_frames", "total_frames", "total_seconds",
               "average_conf", "is_active", "speed_mean", "speed_std", "speed_min", "speed_max", "mean_box_x1",
               "mean_box_y1", "mean_box_x2", "mean_box_y2"]
    sql = f"""
    SELECT {", ".join(columns)} 
    FROM aura_cat.cat_states 
    WHERE time_to > '{start_time}' and time_from < '{end_time}'
    """
    logging.info(sql)
    cat_states = pd.DataFrame(data=db.execute_query(sql), columns=columns)
    logging.info(f"Retrieved cat_states data: {len(cat_states)}")

    return images, detections, detected_period, cat_states


def _get_data_from_csv_for_plot(start_time, end_time, file_path):
    images = pd.read_csv(f"{file_path}/images.csv", index_col=False)
    num_columns = len(images.columns)
    images = images.set_axis(["id", "db_indx", "image_name", "create_time", "light", "detected", "is_grey"][-num_columns:], axis=1)
    if "id" in images.columns:
        images = images.drop(columns=["id", "db_indx"])
    images["create_time"] = pd.to_datetime(images["create_time"])
    dict_image_time = dict(zip(images['image_name'], images['create_time']))
    images = images[(images["create_time"] > start_time) & (images["create_time"] < end_time)]
    images = images.sort_values("create_time")
    
    detections = pd.read_csv(f"{file_path}/detection.csv", index_col=False)
    num_columns = len(detections.columns)
    detections = detections.set_axis(["id", "db_indx", "image_name", "cat_id", "box_x1", "box_y1",
                                      "box_x2", "box_y2", "conf", "light", "model_id"][-num_columns:], axis=1)
    if "id" in detections.columns:
        detections = detections.drop(columns=["id", "db_indx"])
    detections["create_time"] = detections["image_name"].map(dict_image_time)
    detections = detections[(detections["create_time"] > start_time) & (detections["create_time"] < end_time)]

    detected_period = pd.read_csv(f"{file_path}/detected_period.csv", index_col=False)
    if len(detected_period) == 0:
        detected_period = pd.DataFrame(columns=range(9))
    num_columns = len(detected_period.columns)
    detected_period = detected_period.set_axis(["id", "db_indx", "cat_id", "time_from", "time_to", "detected_frames",
                                                "total_frames", "total_seconds", "average_conf"][-num_columns:], axis=1)
    detected_period["time_from"] = pd.to_datetime(detected_period["time_from"])
    detected_period["time_to"] = pd.to_datetime(detected_period["time_to"])
    if "id" in detected_period.columns:
        detected_period = detected_period.drop(columns=["id", "db_indx"])
    detected_period = detected_period[(detected_period["time_from"] < end_time) & (detected_period["time_to"] > start_time)]

    cat_states = pd.read_csv(f"{file_path}/cat_states.csv", index_col=False)
    if len(cat_states) == 0:
        cat_states = pd.DataFrame(columns=range(14))
    num_columns = len(cat_states.columns)
    cat_states = cat_states.set_axis(["id", "db_indx", "cat_id", "time_from", "time_to", "detected_frames",
                                      "total_frames", "total_seconds", "average_conf", "is_active", "speed_mean",
                                      "speed_std", "speed_min", "speed_max", "mean_box_x1", "mean_box_y1",
                                      "mean_box_x2", "mean_box_y2"][-num_columns:], axis=1)
    cat_states["time_from"] = pd.to_datetime(cat_states["time_from"])
    cat_states["time_to"] = pd.to_datetime(cat_states["time_to"])
    if "id" in cat_states.columns:
        cat_states = cat_states.drop(columns=["id", "db_indx"])
    cat_states = cat_states[(cat_states["time_from"] < end_time) & (cat_states["time_to"] > start_time)]

    return images, detections, detected_period, cat_states


def _lanes(*frames):
    """Sorted unique cat ids across the frames; old ids 0..2 keep lanes 0..2."""
    ids = set()
    for f in frames:
        if f is not None and "cat_id" in f.columns:
            ids.update(int(c) for c in f["cat_id"].dropna().unique())
    return sorted(ids) or [0, 1, 2]


def _lane_of(cat_id, lane_of):
    """Chart lane for a cat id, or None when the row has no usable id."""
    try:
        return lane_of[int(cat_id)]
    except (TypeError, ValueError, KeyError):
        return None


def _color(lane):
    return CAT_COLORS[lane % len(CAT_COLORS)]


def get_period_speed(period, detections, cat_id):
    detections = detections.sort_values("create_time")
    start_time = period["time_from"]
    end_time = period["time_to"]
    detections = detections[(detections["create_time"] >= start_time) & (detections["create_time"] <= end_time)]
    detections = detections[detections["cat_id"] == cat_id]
    speed_curves = []
    for i in range(1, len(detections)):
        this_frame = detections.iloc[i]
        last_frame = detections.iloc[i-1]
        d = np.sqrt(((this_frame["box_x1"]+this_frame["box_x2"])/2 - (last_frame["box_x1"]+last_frame["box_x2"])/2)**2
                    + ((this_frame["box_y1"]+last_frame["box_y2"])/2-(last_frame["box_y1"]+last_frame["box_y2"])/2)**2)
        t = (this_frame["create_time"] - last_frame["create_time"]).total_seconds()
        speed_curves.append([this_frame["create_time"], d/t])
    return np.array(speed_curves).T


def plot_detection_details(start_time, end_time, source="sql", revised=False, plot_conf=False, rolling_window=50, width=640, height=480,
                           save_path=None, show=True):
    if source == "sql":
        images, detections, detected_period, cat_states = _get_data_from_sql_for_plot(start_time, end_time, revised)
    else:
        images, detections, detected_period, cat_states = _get_data_from_csv_for_plot(start_time, end_time, source, revised)
    
    if len(images) == 0:
        return False

    images["create_time"] = pd.to_datetime(images["create_time"])
    detections["create_time"] = pd.to_datetime(detections["create_time"])
    lanes = _lanes(detections, detected_period, cat_states)
    lane_of = {c: i for i, c in enumerate(lanes)}
    n = len(lanes)

    plt.figure(figsize=(14, max(4, n + 1)))
    for i, cid in enumerate(lanes):
        name = CAT_NAMES[i] if i < len(CAT_NAMES) else f"cat {cid}"
        if cid != i:
            name += f" ({cid})"
        plt.plot([start_time, end_time], [i, i], c="white", lw=0.8, alpha=0.8)
        plt.text(start_time, i+0.12, name, c=_color(i), ha="left", va="center")
    plt.plot([start_time, end_time], [n, n], c="white", lw=0.8, alpha=0.8)
    plt.fill_between([start_time, end_time], [-0.5, -0.5], [0,0], color=["white"], alpha=0.4)
    plt.fill_between([start_time, end_time], [n, n], [n+0.5, n+0.5], color=["white"], alpha=0.4)

    for i in range(len(detected_period)):
        row = detected_period.iloc[i]
        lane = _lane_of(row["cat_id"], lane_of)
        if lane is None:
            continue
        plt.fill_between([row["time_from"]+ROUND_SECONDS, row["time_to"]-ROUND_SECONDS],
                         [lane, lane], [lane+1, lane+1], color=_color(lane), alpha=0.1)
        speed_curve = get_period_speed(row, detections, row["cat_id"])
        if len(speed_curve) != 0:
            plt.plot(speed_curve[0], np.clip(speed_curve[1]/100, 0, 1) + lane, c=_color(lane), lw=0.5)

    for i in range(len(detections)):
        det = detections.iloc[i]
        lane = _lane_of(det["cat_id"], lane_of)
        if lane is None:
            continue
        if plot_conf:
            plt.scatter(det["create_time"], det["conf"]+ lane, c=_color(lane), s=1)
        else:
            plt.plot([det["create_time"], det["create_time"]], [det["box_x1"]/width + lane, det["box_x2"]/width + lane],
                    color=_color(lane), lw=1)

    for lane, cid in enumerate(lanes):
        detections_cat = detections[detections["cat_id"] == cid]
        img_cat = images.merge(detections_cat, on="create_time", how="inner")
        img_cat["cumulative_conf"] = img_cat["conf"].rolling(window=rolling_window, min_periods=1).mean()
        plt.plot(img_cat["create_time"], img_cat["cumulative_conf"] / 1 + lane, c=_color(lane), lw=0.5)

    images['create_time_minutes'] = images["create_time"].dt.floor('Min')
    min_aggregation = images['create_time_minutes'].value_counts()
    is_grey = images.groupby("create_time_minutes")["is_grey"].max() == 1

    frequency_width = max(20 / (((end_time - start_time).total_seconds()/60) / 30),10)
    for dt in min_aggregation.index:
        dt_is_grey = is_grey.loc[dt]
        plt.plot([dt+datetime.timedelta(seconds=30), dt+datetime.timedelta(seconds=30)],
                 [-0.5, min_aggregation.loc[dt]/300-0.5], lw=frequency_width,
                 color='#cccccc' if dt_is_grey else '#ffe699', solid_capstyle="round")

    # new images rows have NULL light (inserts no longer write it); plot only rows that have one
    light_rows = images.dropna(subset=["light"]) if "light" in images.columns else images.iloc[0:0]
    light_periods = []
    lp = []
    for i in range(len(light_rows)):
        row = light_rows.iloc[i]
        if not lp or (row["create_time"] - lp[-1][0]).total_seconds() < 30:
            lp.append([row["create_time"], row["light"]])
        else:
            light_periods.append(lp.copy())
            lp = [[row["create_time"], row["light"]]]
    if lp:
        light_periods.append(lp.copy())
    for lp in light_periods:
        lp = np.array(lp).T
        plt.plot(lp[0], lp[1] / 200 / 2 + n, color="red", lw=0.5)
        plt.fill_between(list(lp[0]), [n]*len(lp[0]), list(lp[1] / 200 / 2 + n),color="red", alpha=0.04)

    for i in range(len(cat_states)):
        row = cat_states.iloc[i]
        lane = _lane_of(row["cat_id"], lane_of)
        if lane is None:
            continue
        plt.plot([row["time_from"] + ROUND_SECONDS, row["time_to"] + ROUND_SECONDS], [lane, lane],
                 lw=5, solid_capstyle="round", color=_color(lane), alpha=1 if row["is_active"] else 0.4)

    plt.xlim(start_time, end_time)
    plt.ylim(-0.5, n + 0.5)
    plt.yticks([], [])
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return True
