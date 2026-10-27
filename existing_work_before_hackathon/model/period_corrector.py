import heapq
import numpy as np
import pandas as pd
import time
from collections import defaultdict
from model.db import MysqlDB
from model.my_logging import logger

# detection = [create_time, box_x1, box_y1, box_x2, box_y2, original_cat_id, conf, corrected_cat_id, period_id]


class PeriodCorrector:
    def __init__(self, view_id):
        self.view_id = view_id
        self.db = MysqlDB()
        self.cat_ids = self._get_view_cat_ids(view_id)
        # self.detections = pd.DataFrame(columns=["create_time", "box_x1", "box_y1", "box_x2", "box_y2", "original_cat_id",
        # "conf", "corrected_cat_id", "period_id"])
        self.detections = []
        self.started_period_starts = []
        self.connected_periods = {}
        self.number_of_cats = len(self.cat_ids)
        self.id_to_idx = {cat_id: idx for idx, cat_id in enumerate(self.cat_ids)}
        self.idx_to_id = {idx: cat_id for idx, cat_id in enumerate(self.cat_ids)}
        self.min_score = 1
        self.cat_period = {cat_id: None for cat_id in self.cat_ids}
        self.last_period_time = {}
        self.now_detecting_date = None
    
    def _get_view_cat_ids(self, view_id):
        sql = f"""
        SELECT cat_ids
        FROM view
        WHERE view_id = %s
        """
        result = self.db.execute_query(sql, (view_id,))
        if not result:
            raise ValueError(f"View {view_id} not found in database.")
        cat_ids = [int(cat_id) for cat_id in result[0][0].split(",")]
        return cat_ids

    def coordinate_period_groups(self):
        heap = []
        coordinated_cat_id_dict = {}
        period_stats = {}
        period_score = {}
        coord_idx_period = {i: [] for i in range(self.number_of_cats)}
        for period_id in self.connected_periods.keys():
            # period = self.detections[self.detections["period_id"] == period_id]
            period = [d for d in self.detections if d[8] == period_id]
            stats = self.get_period_stats(period)
            score = get_stats_score(stats)
            period_stats[period_id] = stats
            period_score[period_id] = score
            heapq.heappush(heap, (-score.max(), score.argmax(), period_id, score, (period[0][0], period[-1][0])))
        while heap:
            max_score, scored_idx, period_id, score, period = heapq.heappop(heap)
            max_score = -max_score
            if max_score < self.min_score:
                coordinated_cat_id_dict[period_id] = None
                continue
            if is_conflict(coord_idx_period[scored_idx], period):
                score[scored_idx] = 0
                heapq.heappush(heap, (-score.max(), score.argmax(), period_id, score, period))
            else:
                coord_idx_period[scored_idx].append(period)
                coordinated_cat_id_dict[period_id] = self.idx_to_id[int(scored_idx)]
        assert len(coordinated_cat_id_dict) == len(period_stats), {k: v for k, v in period_stats.items() if k not in coordinated_cat_id_dict}
        return coordinated_cat_id_dict, period_stats, period_score

    def save_period_to_db(self, coordinated_cat_id_dict, period_stats, period_score, view_id):
        sql = f"""
        UPDATE period SET start_time = %s, end_time = %s, coordinated_cat_id = %s, detect_length = %s, 
        detect_conf = %s, detect_rate = %s, detect_area_account = %s, score = %s, 
        all_detect_lengths = %s, all_detect_confs = %s, all_detect_rates = %s, 
        all_detect_area_accounts = %s, all_scores = %s, view_id = %s 
        WHERE period_id = %s
        """
        # print("Saving period to db", coordinate_id, period_stats, period_score)
        logger.debug(f"Saving period to db: {coordinated_cat_id_dict}, {period_stats}, {period_score}")
        for period_id in period_stats.keys():
            stats = period_stats[period_id]
            score = period_score[period_id]
            period = self.connected_periods[period_id]
            if len(period) == 1:
                start_time = period[0]
                end_time = self.last_period_time[period_id]
            else:
                start_time, end_time = period

            row = [
                start_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                end_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                int(coordinated_cat_id_dict[period_id]) if coordinated_cat_id_dict[period_id] is not None else None,
            ]

            coordinated_cat_id = coordinated_cat_id_dict[period_id]
            # print("Cat_id", cat_id)
            # print("Stats", stats)
            if coordinated_cat_id is not None:
                row += [
                    stats["detect_length"].tolist()[self.id_to_idx[coordinated_cat_id]],
                    stats["detect_conf"].tolist()[self.id_to_idx[coordinated_cat_id]],
                    stats["detect_rate"].tolist()[self.id_to_idx[coordinated_cat_id]],
                    stats["period_area_count"].tolist()[self.id_to_idx[coordinated_cat_id]],
                    score[self.id_to_idx[coordinated_cat_id]].tolist(),
                ]
            else:
                row += [None, None, None, None, None]
            row += [
                str(stats["detect_length"].tolist()),
                str(stats["detect_conf"].tolist()),
                str(stats["detect_rate"].tolist()),
                str(stats["period_area_count"].tolist()),
                str(score.tolist()),
            ]
            row.append(view_id)
            row.append(period_id)
            yield row
            # print("Data", row)
            logger.debug(f"Saving period to db SQL: {sql}")
            logger.debug(f"Saving period to db: {row}")
            time0 = time.time()
            self.db.execute_update(sql, row)
            logger.debug(f"Saving period to db time: {time.time() - time0}")

    def process_detection(self, detection, view_id):
        time0 = time.time()
        if self.now_detecting_date is not None and self.now_detecting_date != detection[0].date():
            for _ in self.finish_up(view_id):
                ...
        self.now_detecting_date = detection[0].date()
        logger.debug(f"Process detection condition::: detection: {detection}, cat_period: {self.cat_period}")
        if len(detection) == 8 and detection[7] in self.cat_period and self.cat_period[detection[7]] is not None:
            detection_time = detection[0]
            close_cat_id = detection[7]
            close_period_id = self.cat_period[close_cat_id]
            close_period_time = self.last_period_time[close_period_id]
            if close_period_id in self.started_period_starts:
                self.started_period_starts.remove(close_period_id)
                self.connected_periods[close_period_id].append(close_period_time)
                del self.last_period_time[close_period_id]
                self.cat_period[close_cat_id] = None
            else:
                logger.warning(f"Len 8: Period {close_period_id} is not in started_period_starts: {self.started_period_starts}")
                logger.warning(f"Connected periods: {self.connected_periods}")
                logger.warning(f"Last period time: {self.last_period_time}")
                logger.warning(f"Cat period: {self.cat_period}")

            logger.debug(f"Decided to save period to DB?: {detection_time}, {self.started_period_starts}")
            if len(self.started_period_starts) == 0:
                coordinated_cat_id_dict, period_stats, period_score = self.coordinate_period_groups()
                for one_row in self.save_period_to_db(coordinated_cat_id_dict, period_stats, period_score, view_id):
                    # print(f"Corrected period (gap): {one_row}")
                    yield one_row
                self.connected_periods = {}
        elif len(detection) == 9:
            # assert len(detection) == 9, f"Detection should have 9 elements, but got {len(detection)}: {list(detection)}"
            detection_time = detection[0]
            detection_correct_cat_id = detection[7]
            detection_period_id = detection[8]

            self.last_period_time[detection_period_id] = detection_time
            logger.debug(f"Detected period_id: {detection_period_id} last period_id: {self.cat_period[detection_correct_cat_id]}")
            if detection_period_id != self.cat_period[detection_correct_cat_id]:
                last_period = [d for d in self.detections if d[7] == detection_correct_cat_id]
                if len(last_period) > 0 and last_period[-1][8] in self.started_period_starts:
                    close_period_id = last_period[-1][8]
                    close_period_time = last_period[-1][0]
                    if close_period_id in self.started_period_starts:
                        logger.debug(f"Popping period {close_period_id} from started_period_starts: {self.started_period_starts}")
                        self.started_period_starts.remove(close_period_id)
                        self.connected_periods[close_period_id].append(close_period_time)
                        del self.last_period_time[close_period_id]

                    else:
                        logger.warning(f"Len 9: Period {close_period_id} is not in started_period_starts: {self.started_period_starts}")
                        logger.warning(f"Connected periods: {self.connected_periods}")
                        logger.warning(f"Last period time: {self.last_period_time}")

            # print(detection_time, self.started_period_starts)
            logger.debug(f"Decided to save period to DB?: {detection_time}, {self.started_period_starts}")
            if len(self.started_period_starts) == 0:
                coordinated_cat_id_dict, period_stats, period_score = self.coordinate_period_groups()
                for one_row in self.save_period_to_db(coordinated_cat_id_dict, period_stats, period_score, view_id):
                    # print(f"Corrected period (change): {one_row}")
                    yield one_row
                self.connected_periods = {}

            if detection_period_id != self.cat_period[detection_correct_cat_id]:
                self.started_period_starts.append(detection_period_id)
                self.connected_periods[detection_period_id] = [detection_time]
            self.detections.append(detection)
            self.cat_period[detection_correct_cat_id] = detection_period_id
        logger.debug(f"PeriodCorrector process detection time: {time.time() - time0}")

    def unprocess_detection(self, detection):
        for k, d in enumerate(self.detections):
            if np.array_equal(d, detection):
                self.detections.pop(k)
                return

    def finish_up(self, view_id):
        logger.info("Finishing up Periods ... ")
        coordinate_id, period_stats, period_score = self.coordinate_period_groups()
        for one_row in self.save_period_to_db(coordinate_id, period_stats, period_score, view_id):
            # print(f"Corrected period (change): {one_row}")
            yield one_row
        self.detections = []
        self.started_period_starts = []
        self.connected_periods = {}
        self.cat_period = {cat_id: None for cat_id in self.cat_ids}
        self.last_period_time = {}

        yield None

    def get_period_stats(self, period, vote_unit=50):
        # detection = [create_time, box_x1, box_y1, box_x2, box_y2, original_cat_id, conf, corrected_cat_id, period_id]
        stats = {}
        # original_detect = period[period["original_cat_id"].isna() == False]
        original_detect = [p for p in period if pd.isna(p[5]) == False]
        cats_original_detect = [[d for d in original_detect if d[5] == i] for i in self.cat_ids]
        stats["detect_rate"] = np.array([len(cats_original_detect[i]) / len(period) for i in range(self.number_of_cats)]).round(4)
        stats["detect_conf"] = np.array(
            [(np.mean([d[6] for d in cats_original_detect[i]]) if len(cats_original_detect[i]) > 0 else 0) for i in range(self.number_of_cats)]
        ).round(4)
        stats["detect_length"] = np.array([len(cats_original_detect[i]) for i in range(self.number_of_cats)])
        stats["length"] = len(period)
        period_conf = {i: defaultdict(list) for i in self.cat_ids}
        for p in range(len(original_detect)):
            row = original_detect[p]
            x_center = (row[1] + row[3]) / 2
            y_center = (row[2] + row[4]) / 2
            period_conf[int(row[5])][(x_center // vote_unit, y_center // vote_unit)].append(row[6])
        stats["period_area_count"] = np.array([len([k1 for k1, v1 in v.items() if np.mean(v1) > 0.5]) for k, v in period_conf.items()])
        stats["period_conf"] = period_conf
        return stats


def is_conflict(periods_group, period):
    for p in periods_group:
        if p[0] < period[1] and period[0] < p[1]:
            return True
    return False


def get_stats_score(stats):
    score = 0
    score += (stats["detect_conf"] == stats["detect_conf"].max()) * 3
    score += (stats["detect_length"] == stats["detect_length"].max()) * 2
    score += stats["detect_length"] > 20
    score += stats["detect_rate"] > 0.5
    score += stats["detect_conf"] > 0.5
    score += stats["detect_rate"] > 0.3
    score = score.astype(float) + (stats["period_area_count"] == stats["period_area_count"].max()) * 4.2
    score = score.astype(float) * ((stats["detect_length"] >= 10).astype(float) + (stats["detect_length"] < 10).astype(float) * 0.2)
    return score
