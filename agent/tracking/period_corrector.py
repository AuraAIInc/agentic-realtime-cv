"""Period-level correction (migrated from legacy model/period_corrector.py).

MySQL is decoupled: ``cat_ids`` are passed in and period rows go to an
injected ``Sink``. Close/gap/coordinate logic and P-code emits are unchanged.
"""

from __future__ import annotations

import datetime
import heapq
import itertools
import logging
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from agent.escalation import Escalation, EscalationConfig, box_distance, box_iou, even_pick, max_displacement, top_two_margin, window_is_static
from agent.escalation import emit as emit_escalation
from agent.tracking.sinks import InMemorySink

logger = logging.getLogger(__name__)

# detection = [create_time, box_x1, box_y1, box_x2, box_y2, original_cat_id, conf, corrected_cat_id, period_id]


class PeriodCorrector:
    def __init__(self, cat_ids: list[int], escalation_config=None, escalation_seq=None):
        self.cat_ids = list(cat_ids)
        self.escalation_config = escalation_config or EscalationConfig.from_env()
        # Shared run-wide counter (itertools.count) so frame and period escalations
        # number continuously; a private one when the caller passes nothing.
        self._esc_seq = escalation_seq if escalation_seq is not None else itertools.count(1)
        self.escalations = []
        self.prev_period_end = {}  # cat_id -> (end_time, last_box, period_id, trail boxes)
        self._open_prev = {}  # period_id -> prev_period_end snapshot at open (or None)
        self._p7_checked = set()  # period_ids already assessed for P7
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

    def _escalate(self, code, reason, start, end, cat_id=None, period_id=None, metrics=None):
        emit_escalation(
            self.escalations,
            Escalation(time=start, end_time=end, scope="period", code=code, reason=reason, cat_id=cat_id, period_id=period_id, metrics=metrics or {}, seq=next(self._esc_seq)),
        )

    def coordinate_period_groups(self):
        heap = []
        coordinated_cat_id_dict = {}
        period_stats = {}
        period_score = {}
        period_bounds = {}
        period_length = {}
        original_top = {}
        reassigned = set()
        coord_idx_period = {i: [] for i in range(self.number_of_cats)}
        for period_id in self.connected_periods.keys():
            # period = self.detections[self.detections["period_id"] == period_id]
            period = [d for d in self.detections if d[8] == period_id]
            stats = self.get_period_stats(period)
            score = get_stats_score(stats)
            period_stats[period_id] = stats
            period_score[period_id] = score
            period_bounds[period_id] = (period[0][0], period[-1][0])
            period_length[period_id] = len(period)
            top_idx, margin = top_two_margin(score)
            original_top[period_id] = top_idx
            heapq.heappush(heap, (-score.max(), score.argmax(), period_id, score, (period[0][0], period[-1][0])))
            self._check_vote_escalations(period_id, stats, score, top_idx, margin, period_bounds[period_id])
        while heap:
            max_score, scored_idx, period_id, score, period = heapq.heappop(heap)
            max_score = -max_score
            if max_score < self.min_score:
                coordinated_cat_id_dict[period_id] = None
                self._check_unknown_escalation(period_id, period_bounds[period_id], period_length[period_id])
                continue
            if is_conflict(coord_idx_period[scored_idx], period):
                reassigned.add(period_id)
                score[scored_idx] = 0
                heapq.heappush(heap, (-score.max(), score.argmax(), period_id, score, period))
            else:
                coord_idx_period[scored_idx].append(period)
                coordinated_cat_id_dict[period_id] = self.idx_to_id[int(scored_idx)]
                if period_id in reassigned and int(scored_idx) != original_top[period_id]:
                    start, end = period_bounds[period_id]
                    wanted = self.idx_to_id[original_top[period_id]]
                    taker, taker_span = None, None
                    for pid2, cat2 in coordinated_cat_id_dict.items():
                        if cat2 != wanted or pid2 not in period_bounds:
                            continue
                        s2, e2 = period_bounds[pid2]
                        if s2 < end and start < e2:
                            taker, taker_span = pid2, [s2.isoformat(timespec="microseconds"), e2.isoformat(timespec="microseconds")]
                            break
                    self._escalate(
                        "P3",
                        f"period {period_id} wanted cat {wanted} but it was taken by an overlapping period; assigned cat {self.idx_to_id[int(scored_idx)]} instead",
                        start,
                        end,
                        cat_id=self.idx_to_id[int(scored_idx)],
                        period_id=period_id,
                        metrics={
                            "wanted_cat_id": wanted,
                            "taken_by_period_id": taker,
                            "taken_by_span": taker_span,
                            "key_frames": self._sample_period_frames(period_id),
                        },
                    )
        assert len(coordinated_cat_id_dict) == len(period_stats), {k: v for k, v in period_stats.items() if k not in coordinated_cat_id_dict}
        for period_id, voted in coordinated_cat_id_dict.items():
            self._check_tracker_vote_split(period_id, voted, period_stats[period_id], period_score[period_id], period_bounds[period_id])
        return coordinated_cat_id_dict, period_stats, period_score

    def _check_tracker_vote_split(self, period_id, voted, stats, score, bounds):
        """P8: the tracker's frame labels disagree with the coordinated vote.

        The tracker files boxes by continuity (one early label can drag a
        whole visit along); the vote counts the camera's original labels.
        When they name different cats, at least one is wrong — this is the
        period-scale signature of continuity hijacking, invisible to the
        frame-level relabel codes.
        """
        if voted is None:
            return
        tracked = Counter(
            int(d[7]) for d in self._period_rows(period_id) if len(d) > 7 and not pd.isna(d[7])
        )
        if not tracked:
            return
        top = tracked.most_common(2)
        if len(top) > 1 and top[0][1] == top[1][1]:
            return  # no dominant tracked label; nothing coherent to dispute
        tracked_cat = top[0][0]  # d[7] is already a cat id (see row format above)
        if tracked_cat == voted:
            return
        relabeled = sum(
            1
            for d in self._period_rows(period_id)
            if len(d) > 7 and not pd.isna(d[5]) and not pd.isna(d[7]) and int(d[5]) != int(d[7])
        )
        start, end = bounds
        self._escalate(
            "P8",
            f"period {period_id} tracked as cat {tracked_cat} ({top[0][1]} frames) but voted cat {voted}; "
            f"continuity may have hijacked the visit",
            start,
            end,
            cat_id=voted,
            period_id=period_id,
            metrics={
                "tracked_cat_id": int(tracked_cat),
                "voted_cat_id": int(voted),
                "tracked_frames": int(top[0][1]),
                "relabeled_frames": int(relabeled),
                "votes": {
                    str(self.idx_to_id[i]): {"score": round(float(score[i]), 1), "frames": int(stats["detect_length"][i])}
                    for i in range(len(score))
                },
                "key_frames": self._sample_period_frames(period_id),
            },
        )

    def _period_rows(self, period_id) -> list:
        """Committed rows of one period, in time order."""
        return [d for d in self.detections if len(d) > 8 and d[8] == period_id]

    def _median_winner_box(self, period_id, cat_id) -> list | None:
        """Median box of the winner's real sightings (None when there are none)."""
        boxes = [[float(v) for v in d[1:5]] for d in self._period_rows(period_id) if d[1] is not None and d[5] == cat_id]
        if not boxes:
            return None
        cols = [sorted(c) for c in zip(*boxes)]
        return [cols[i][len(cols[i]) // 2] for i in range(4)]

    def _sample_period_frames(self, period_id, cap: int = 6) -> list:
        """Sampled frames across a period: time, box (or None), real-sighting flag."""
        rows = [d for d in self._period_rows(period_id) if d[0] is not None]
        out = []
        for d in (rows[i] for i in even_pick(len(rows), cap)):
            box = [float(v) for v in d[1:5]] if d[1] is not None else None
            out.append({"t": d[0].isoformat(timespec="microseconds"), "box": box, "real": bool(box) and not pd.isna(d[5])})
        return out

    def _check_vote_escalations(self, period_id, stats, score, top_idx, margin, bounds):
        cfg = self.escalation_config
        start, end = bounds
        top_cat = self.idx_to_id[top_idx]
        if margin < cfg.vote_margin:
            order = sorted(range(len(score)), key=lambda i: -score[i])
            runner_cat = self.idx_to_id[order[1]]
            self._escalate(
                "P1",
                f"period {period_id} identity is a close vote: cat {top_cat} {float(score[top_idx]):.1f} vs cat {runner_cat} {float(score[order[1]]):.1f} (margin {margin:.1f})",
                start,
                end,
                cat_id=top_cat,
                period_id=period_id,
                metrics={
                    "scores": list(score),
                    "margin": margin,
                    "runner_up_cat_id": runner_cat,
                    "votes": {
                        str(self.idx_to_id[i]): {"score": round(float(score[i]), 1), "frames": int(stats["detect_length"][i])}
                        for i in range(len(score))
                    },
                    "key_frames": self._sample_period_frames(period_id),
                },
            )
        weak_winner = float(score[top_idx]) < cfg.weak_winner_score or stats["detect_length"][top_idx] < cfg.weak_winner_min_length
        if weak_winner:
            self._escalate(
                "P2",
                f"period {period_id} winner cat {top_cat} is weak: score {float(score[top_idx]):.1f}, {int(stats['detect_length'][top_idx])} original frames",
                start,
                end,
                cat_id=top_cat,
                period_id=period_id,
                metrics={
                    "score": float(score[top_idx]),
                    "detect_length": int(stats["detect_length"][top_idx]),
                    "key_frames": self._sample_period_frames(period_id),
                },
            )
        winner_conf = float(stats["detect_conf"][top_idx])
        winner_rate = float(stats["detect_rate"][top_idx])
        # A thin visit fires P2 already with the same key frames and the same
        # keep/drop question (plus a merge option) — a P5 on top is pure cost.
        if (winner_conf < cfg.low_mean_conf or winner_rate < cfg.low_detect_rate) and not weak_winner:
            self._escalate(
                "P5",
                f"period {period_id} cat {top_cat} is low quality: mean conf {winner_conf:.2f}, original-frame rate {winner_rate:.2f}",
                start,
                end,
                cat_id=top_cat,
                period_id=period_id,
                metrics={
                    "detect_conf": winner_conf,
                    "detect_rate": winner_rate,
                    "total_frames": int(stats["length"]),
                    "detect_counts": {str(self.idx_to_id[i]): int(n) for i, n in enumerate(stats["detect_length"]) if int(n) > 0},
                    "winner_box": self._median_winner_box(period_id, top_cat),
                    "key_frames": self._sample_period_frames(period_id),
                },
            )

    def _check_unknown_escalation(self, period_id, bounds, n_rows):
        cfg = self.escalation_config
        start, end = bounds
        seconds = (end - start).total_seconds()
        if seconds > cfg.unknown_min_seconds or n_rows > cfg.unknown_min_frames:
            counts: dict = {}
            for d in self._period_rows(period_id):
                if len(d) > 5 and not pd.isna(d[5]):
                    key = str(int(d[5]))
                    counts[key] = counts.get(key, 0) + 1
            self._escalate(
                "P4",
                f"period {period_id} got no cat but ran {seconds:.0f} s / {n_rows} frames; a long track nobody could label",
                start,
                end,
                period_id=period_id,
                metrics={
                    "seconds": seconds,
                    "rows": n_rows,
                    "detect_counts": counts,
                    "key_frames": self._sample_period_frames(period_id),
                },
            )

    def save_period_to_db(self, coordinated_cat_id_dict, period_stats, period_score, view_id, sink):
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
            logger.debug(f"Saving period to sink: {row}")
            time0 = time.time()
            sink.save_period(row)
            logger.debug(f"Saving period to sink time: {time.time() - time0}")

    def process_detection(self, detection, view_id=None, sink=None):
        time0 = time.time()
        sink = sink if sink is not None else InMemorySink()
        if self.now_detecting_date is not None and self.now_detecting_date != detection[0].date():
            for _ in self.finish_up(view_id, sink):
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
                self._check_static_link(close_period_id)
                self._record_prev_end(close_cat_id, close_period_id, close_period_time)
            else:
                logger.warning(f"Len 8: Period {close_period_id} is not in started_period_starts: {self.started_period_starts}")
                logger.warning(f"Connected periods: {self.connected_periods}")
                logger.warning(f"Last period time: {self.last_period_time}")
                logger.warning(f"Cat period: {self.cat_period}")

            logger.debug(f"Decided to save period to DB?: {detection_time}, {self.started_period_starts}")
            if len(self.started_period_starts) == 0:
                coordinated_cat_id_dict, period_stats, period_score = self.coordinate_period_groups()
                for one_row in self.save_period_to_db(coordinated_cat_id_dict, period_stats, period_score, view_id, sink):
                    # print(f"Corrected period (gap): {one_row}")
                    yield one_row
                self._drop_link_state(self.connected_periods)
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
                        self._check_static_link(close_period_id)
                        self._record_prev_end(detection_correct_cat_id, close_period_id, close_period_time)

                    else:
                        logger.warning(f"Len 9: Period {close_period_id} is not in started_period_starts: {self.started_period_starts}")
                        logger.warning(f"Connected periods: {self.connected_periods}")
                        logger.warning(f"Last period time: {self.last_period_time}")

            # print(detection_time, self.started_period_starts)
            logger.debug(f"Decided to save period to DB?: {detection_time}, {self.started_period_starts}")
            if len(self.started_period_starts) == 0:
                coordinated_cat_id_dict, period_stats, period_score = self.coordinate_period_groups()
                for one_row in self.save_period_to_db(coordinated_cat_id_dict, period_stats, period_score, view_id, sink):
                    # print(f"Corrected period (change): {one_row}")
                    yield one_row
                self._drop_link_state(self.connected_periods)
                self.connected_periods = {}

            if detection_period_id != self.cat_period[detection_correct_cat_id]:
                self.started_period_starts.append(detection_period_id)
                self.connected_periods[detection_period_id] = [detection_time]
                self._open_prev[detection_period_id] = self.prev_period_end.get(detection_correct_cat_id)
                self._check_merge_candidate(detection_correct_cat_id, detection_period_id, detection)
            self.detections.append(detection)
            self.cat_period[detection_correct_cat_id] = detection_period_id
        logger.debug(f"PeriodCorrector process detection time: {time.time() - time0}")

    def _record_prev_end(self, cat_id, period_id, end_time):
        """Remember a closed period's end for the P6/P7 link checks (cleared if fully retracted)."""
        window = datetime.timedelta(seconds=self.escalation_config.static_window_seconds)
        trail = []
        for d in self.detections:
            if d[8] == period_id and d[1] is not None and end_time - window <= d[0] <= end_time:
                trail.append([d[1], d[2], d[3], d[4]])
        if trail:
            self.prev_period_end[cat_id] = (end_time, trail[-1], period_id, trail)
        else:
            self.prev_period_end.pop(cat_id, None)

    def _check_merge_candidate(self, cat_id, period_id, detection):
        prev = self.prev_period_end.get(cat_id)
        if prev is None:
            return
        cfg = self.escalation_config
        prev_end, prev_box, prev_period_id, _trail = prev
        gap = (detection[0] - prev_end).total_seconds()
        if gap < 0 or gap > cfg.merge_gap_seconds:
            return
        distance = box_distance(prev_box, detection[1:5])
        if distance < cfg.merge_max_distance:
            prev_span = self.connected_periods.get(prev_period_id) or []
            self._escalate(
                "P6",
                f"period {period_id} may connect to previous period {prev_period_id} of cat {cat_id}: gap {gap:.1f} s, box distance {distance:.0f} px",
                prev_end,
                detection[0],
                cat_id=cat_id,
                period_id=period_id,
                metrics={
                    "prev_period_id": prev_period_id,
                    "gap_seconds": gap,
                    "distance_px": distance,
                    "prev_start": prev_span[0] if prev_span else None,
                    "prev_end_box": [float(v) for v in prev_box],
                    "new_start_box": [float(v) for v in detection[1:5]],
                },
            )

    def _check_static_link(self, period_id):
        """P7: the closing period starts static where the previous one ended static.

        Runs at close (the head window is only complete then), plus a sweep in
        finish_up for periods still open at day end. Shades the gap like P6.
        """
        if period_id in self._p7_checked:
            return
        self._p7_checked.add(period_id)
        snap = self._open_prev.get(period_id)
        if snap is None:
            return
        cfg = self.escalation_config
        prev_end, prev_box, prev_period_id, prev_trail = snap
        rows = [(d[0], [d[1], d[2], d[3], d[4]]) for d in self.detections if d[8] == period_id and d[1] is not None]
        if len(rows) < 2:
            return
        cat_id = next(d[7] for d in self.detections if d[8] == period_id)
        start = rows[0][0]
        head = [box for t, box in rows if t <= start + datetime.timedelta(seconds=cfg.static_window_seconds)]
        if len(head) < 2:
            return
        gap = (start - prev_end).total_seconds()
        if gap < 0 or gap > cfg.static_link_max_gap:
            return
        new_disp = max_displacement(head)
        prev_disp = max_displacement(prev_trail)
        overlap = box_iou(head[0], prev_box)
        if window_is_static(head, cfg.static_max_px) and window_is_static(prev_trail, cfg.static_max_px) and overlap > cfg.static_overlap_iou:
            prev_span = self.connected_periods.get(prev_period_id) or []
            self._escalate(
                "P7",
                f"period {period_id} starts static where period {prev_period_id} ended static (overlap {overlap:.2f}, gap {gap:.0f} s): possibly one sleeping cat split by weak detections",
                prev_end,
                start,
                cat_id=cat_id,
                period_id=period_id,
                metrics={
                    "prev_period_id": prev_period_id,
                    "gap_seconds": gap,
                    "overlap_iou": overlap,
                    "new_displacement_px": new_disp,
                    "prev_displacement_px": prev_disp,
                    "new_head_frames": len(head),
                    "prev_trail_frames": len(prev_trail),
                    "prev_start": prev_span[0] if prev_span else None,
                    "prev_end_box": [float(v) for v in prev_box],
                    "new_start_box": [float(v) for v in head[0]],
                },
            )

    def _drop_link_state(self, period_ids):
        for pid in period_ids:
            self._open_prev.pop(pid, None)
            self._p7_checked.discard(pid)

    def unprocess_detection(self, detection):
        for k, d in enumerate(self.detections):
            if np.array_equal(d, detection):
                self.detections.pop(k)
                return

    def finish_up(self, view_id=None, sink=None):
        logger.info("Finishing up Periods ... ")
        sink = sink if sink is not None else InMemorySink()
        for pid in list(self.connected_periods):
            self._check_static_link(pid)
        coordinate_id, period_stats, period_score = self.coordinate_period_groups()
        for one_row in self.save_period_to_db(coordinate_id, period_stats, period_score, view_id, sink):
            # print(f"Corrected period (change): {one_row}")
            yield one_row
        self.detections = []
        self.started_period_starts = []
        self.connected_periods = {}
        self.cat_period = {cat_id: None for cat_id in self.cat_ids}
        self.last_period_time = {}
        self.escalations = []
        self.prev_period_end = {}
        self._open_prev = {}
        self._p7_checked = set()

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
