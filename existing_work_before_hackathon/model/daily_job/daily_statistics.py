import time
import datetime
import pandas as pd

from model.my_logging import logger
from model.db import MysqlDB


time_zone_offset = 8  # UTC+8


def query_data(db, date, cat_id, view_id):
    start_time = datetime.datetime.strptime(date, "%Y-%m-%d") + datetime.timedelta(hours=time_zone_offset)
    end_time = start_time + datetime.timedelta(days=1)

    sql = f"""
    select box_x1, box_y1, box_x2, box_y2, create_time, period_id, conf
    from aura_cat.revised_detection
    where period_id in (
        select period_id
        from aura_cat.period
        where start_time > '{start_time}' and start_time < '{end_time}'
        and coordinated_cat_id = {cat_id} and view_id = {view_id}
    )
    """
    time_0 = time.time()
    data = db.execute_query(sql)

    time_1 = time.time()
    logger.info(f"Queried {len(data)} of {date} active detections in {time_1 - time_0:.2f} secs")
    data = pd.DataFrame(
        data=data,
        columns=["box_x1", "box_y1", "box_x2", "box_y2", "create_time", "period_id", "conf"],
    )
    data["box_x"] = (data["box_x1"] + data["box_x2"]) / 2
    data["box_y"] = (data["box_y1"] + data["box_y2"]) / 2
    data["last_time"] = data["create_time"].shift(1)
    data["last_x"] = data["box_x"].shift(1)
    data["last_y"] = data["box_y"].shift(1)
    data["distance"] = ((data["box_x"] - data["last_x"]) ** 2 + (data["box_y"] - data["last_y"]) ** 2) ** 0.5
    data["time_diff"] = (data["create_time"] - data["last_time"]).apply(lambda x: x.total_seconds())
    data = data[data["time_diff"] > 0]
    data["speed"] = data["distance"] / data["time_diff"]
    data["last_period_id"] = data["period_id"].shift(1)
    data = data[data["period_id"] == data["last_period_id"]]

    return data


def prepare_data(db: MysqlDB, date, cat_id, view_id, data, speed_threshold=3):
    start_time = datetime.datetime.strptime(date, "%Y-%m-%d") + datetime.timedelta(hours=time_zone_offset)
    end_time = start_time + datetime.timedelta(days=1)
    
    if len(data) == 0:
        return [], []
    row = [start_time, cat_id, view_id]
    columns = ["date", "cat_id", "view_id"]
    row.append(float(data["time_diff"].sum()))
    columns.append("total_duration")

    row.append(float(data["time_diff"][data["speed"] < speed_threshold].sum()))
    columns.append("rest_duration")

    row.append(float(data["time_diff"][data["speed"] >= speed_threshold].sum()))
    columns.append("active_duration")

    row.append(float(data["distance"].sum()))
    columns.append("total_distance")

    row.append(float(data["speed"].mean()))
    columns.append("speed_mean")

    row.append(len(data["period_id"].unique()))
    columns.append("total_period")

    period_duration = data.groupby("period_id")["time_diff"].sum()
    period_duration = period_duration.sort_values(ascending=False)

    sql = f"""
    select count(*) 
    from aura_cat.images 
    where create_time > '{start_time}' and create_time < '{end_time}'
    """
    total_frames = db.execute_query(sql)[0][0]
    row.append(total_frames)
    columns.append("total_frames")

    row.append(datetime.datetime.now())
    columns.append("create_time")

    row.append(float(data["conf"].mean()))
    columns.append("conf_mean")
    
    detected_percentage = row[3] / 86400 if 86400 > 0 else 0
    active_percentage = row[5] / row[3] if row[3] > 0 else 0
    average_conf = data["conf"].mean() if len(data) > 0 else 0
    score = detected_percentage * 0.4 + active_percentage * 0.4 + average_conf * 0.2
    row.append(int(score * 100))
    columns.append("score")

    print(row)
    return row, columns


def prepare_hourly_data(db: MysqlDB, date, cat_id, view_id, data, hour_interval=1):
    if len(data) == 0:
        return [], []
    rows = []
    columns = ["stat_time", "cat_id", "view_id", "duration", "distance", "average_speed"]
    start_time = datetime.datetime.strptime(date, "%Y-%m-%d") + datetime.timedelta(hours=time_zone_offset)
    end_time = start_time + datetime.timedelta(days=1)
    current_time = start_time
    while current_time < end_time:
        next_time = current_time + datetime.timedelta(hours=hour_interval)
        hour_data = data[(data["create_time"] >= current_time) & (data["create_time"] < next_time)]
        if len(hour_data) == 0:
            row = [current_time.strftime("%Y-%m-%d %H:%M:%S"), cat_id, view_id, 0.0, 0.0, 0.0]
            rows.append(row)
            current_time = next_time
            continue
        duration = float(hour_data["time_diff"].sum())
        distance = float(hour_data["distance"].sum())
        average_speed = distance / duration if duration > 0 else 0.0
        row = [current_time.strftime("%Y-%m-%d %H:%M:%S"), cat_id, view_id, duration, distance, average_speed]
        rows.append(row)
        current_time = next_time
    return rows, columns


def insert_hourly_data(db: MysqlDB, rows, columns):
    if len(rows) == 0 or len(columns) == 0:
        return None
    values_list = []
    for row in rows:
        if len(row) != len(columns):
            logger.error("Row and columns length mismatch")
            return None
        values_list.append(tuple(row))
    sql = f"insert into aura_cat.hourly_statistics ({', '.join(columns)}) values ({', '.join(['%s'] * len(columns))})"
    try:
        stat_id = db.execute_many_insert(sql, values_list)
    except Exception as e:
        logger.error(f"Failed to insert hourly data: {e}")
        return None
    return stat_id


def insert_data(db: MysqlDB, row, columns, table_name="aura_cat.daily_statistics"):
    if len(row) != len(columns):
        logger.error("Row and columns length mismatch")
        return None
    if len(row) == 0:
        return None
    sql = f"""insert into {table_name} ({', '.join(columns)}) values ('{"', '".join([str(r) for r in row])}')"""
    try:
        stat_id = db.execute_insert(sql)
    except Exception as e:
        logger.error(f"Failed to insert data: {e}")
        return None
    return stat_id
