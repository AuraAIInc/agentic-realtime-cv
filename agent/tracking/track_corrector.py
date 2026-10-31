"""Frame-level track correction (migrated from legacy model/track_corrector.py).

MySQL is decoupled: ``cat_ids`` are passed in, period ids come from
``period_id_factory``, and every persisted row goes to an injected ``Sink``.
All corrector math, thresholds, escalation emit points, and the
``(is_append, revised)`` yield protocol are unchanged from legacy.
"""

from __future__ import annotations

import datetime
import itertools
import logging
from collections import Counter, deque

import numpy as np

from agent.escalation import Escalation, EscalationConfig
from agent.escalation import emit as emit_escalation
from agent.tracking.sinks import InMemorySink

logger = logging.getLogger(__name__)


class TrackCorrector:
    def __init__(self, cat_ids: list[int], escalation_config=None, period_id_factory=None):
        self.cat_ids = list(cat_ids)
        self.escalation_config = escalation_config or EscalationConfig.from_env()
        self._period_ids = period_id_factory if period_id_factory is not None else itertools.count(1)
        self.escalations = []
        self.min_conf = 0.4
        self.buffer_range = 15  # sec
        self.max_undetected_gap = 60  # sec
        self.img_width = 640
        self.img_height = 480
        self.number_of_cats = len(self.cat_ids)
        self.idx_to_id = {i: cat_id for i, cat_id in enumerate(self.cat_ids)}
        self.id_to_idx = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        self.revised_detection = [[] for _ in range(self.number_of_cats)]
        self.original_detection = [[] for _ in range(self.number_of_cats)]
        # Create Time, Box_x1, Box_y1, Box_x2, Box_y2, Original_Cat_ID, Conf, Cat_ID
        self.buffer = [deque([]) for _ in range(self.number_of_cats)]
        self.buffer_origin = [deque([]) for _ in range(self.number_of_cats)]
        self.base = [None for _ in range(self.number_of_cats)]
        self.target = [None for _ in range(self.number_of_cats)]
        self.last_detection_index = [None for _ in range(self.number_of_cats)]
        self.last_detection = [None for _ in range(self.number_of_cats)]
        self.max_buffer_conf = [0 for _ in range(self.number_of_cats)]
        self.period_id = [0 for _ in range(self.number_of_cats)]
        self.max_period_conf = [0 for _ in range(self.number_of_cats)]
        self.detecting = [False for _ in range(self.number_of_cats)]
        self.last_revised_detection = [None for _ in range(self.number_of_cats)]
        self._synth_run = [[None, False] for _ in range(self.number_of_cats)]
        logger.info("TrackCorrector initialized")

        self.now_detecting_date = None

    def get_new_period_id(self):
        if callable(self._period_ids):
            return self._period_ids()
        return next(self._period_ids)

    def remove_misdetection(self, new_det, cat_idx):
        new_box = np.array(new_det)[1:5]
        is_last_detection = True
        for h in range(1, len(self.buffer_origin[cat_idx]) + 1):
            if self.buffer_origin[cat_idx][-h][1] is None:
                continue
            if get_distance(new_box, self.buffer_origin[cat_idx][-h][1:5]) < self.img_width / 4:
                if not is_last_detection:
                    for k in range(1, h):
                        self.buffer[cat_idx][-k][1:6] = [None] * 5
                        self.buffer_origin[cat_idx][-k][1:6] = [None] * 5
                return
            is_last_detection = False

    def is_resonable_detection(self, box, new_detected_cat_idx):
        for previous_box in self.buffer[new_detected_cat_idx]:
            if previous_box[1] is None:
                continue
            if get_iou(box, previous_box[1:5]) > 0.5:
                return True
        return False

    def match_buffer_history_rule(self, new_cat_id, img_detected, cat_idx):
        """Which history rule the detection matches ("iou_last", "distance", "iou_buffer"), else None."""
        if self.last_detection[cat_idx] is None and self.base[cat_idx] is None:
            return None

        det_reference = self.last_detection[cat_idx] if self.last_detection[cat_idx] is not None else self.base[cat_idx]
        new_box = img_detected[new_cat_id][:4]
        iou = get_iou(new_box, det_reference[1:5])
        logger.debug(f"{new_cat_id}, {cat_idx}, IOU: {iou}. new_box: {new_box}, ref_box:{det_reference[1:5]}")

        if iou > 0.2:
            return "iou_last"

        if cat_idx not in img_detected and get_distance(new_box, det_reference[1:5]) < self.img_width / 4:
            return "distance"

        for b in self.buffer[cat_idx]:
            if b[1] is None:
                continue
            iou = get_iou(new_box, b[1:5])
            if iou > 0.8:
                return "iou_buffer"
        return None

    def match_buffer_history(self, new_cat_id, img_detected, cat_idx):
        return self.match_buffer_history_rule(new_cat_id, img_detected, cat_idx) is not None

    def _escalate(self, code, reason, time, cat_id=None, period_id=None, metrics=None):
        emit_escalation(
            self.escalations,
            Escalation(time=time, scope="frame", code=code, reason=reason, cat_id=cat_id, period_id=period_id, metrics=metrics or {}),
        )

    def smooth_buffer(self, cat_idx):
        original_detections = []
        for i in range(len(self.buffer[cat_idx])):
            if self.buffer[cat_idx][i][5] is not None:
                original_detections.append([i, self.buffer[cat_idx][i]])
            while len(original_detections) > 3:
                original_detections.pop(0)
            if len(original_detections) == 3:
                box_previsous = original_detections[0][1][1:5]
                box_current = original_detections[1][1][1:5]
                box_next = original_detections[2][1][1:5]
                if get_iou(box_previsous, box_next) > 0.5 and get_iou(box_previsous, box_current) < 0.2:
                    for k in range(original_detections[0][0], original_detections[2][0] + 1):
                        smoothed_buffer = np.array([original_detections[1][1][0]] + list((box_previsous + box_next) / 2) + [None, 0, None])
                        self.buffer[cat_idx][k] = smoothed_buffer
                        self.buffer_origin[cat_idx][k] = smoothed_buffer
                        logger.debug(f"Smoothed: {box_previsous} -> {box_current} -> {box_next} -> {(box_previsous + box_next) / 2}")

    def add_new_detection_to_buffer(self, img_detected):
        if len(img_detected) == 0:
            for cat_idx, cat_id in enumerate(self.cat_ids):
                self.buffer[cat_idx].append(np.array([self.right_time, None, None, None, None, None, 0, None]))
                self.buffer_origin[cat_idx].append(np.array([self.right_time, None, None, None, None, None, 0, None]))
                if self.last_detection_index[cat_idx] is not None:
                    self.last_detection_index[cat_idx] -= 1
            return

        multi_label_kills = []
        img_detected = remove_multi_labeling(img_detected, multi_label_kills)
        for killed_id, kept_id, dup_iou, conf_margin in multi_label_kills:
            if conf_margin < self.escalation_config.duplicate_tie_margin:
                self._escalate(
                    "T3",
                    f"duplicate boxes (IoU {dup_iou:.2f}) kept cat {kept_id} over cat {killed_id} on a {conf_margin:.2f} confidence margin",
                    self.right_time,
                    cat_id=kept_id,
                    metrics={"killed_cat_id": killed_id, "iou": dup_iou, "conf_margin": conf_margin},
                )

        correct_from_to = {}
        logger.debug(f"Last detection: {self.last_detection}")

        reasonable_detection = set([])
        for new_detected_cat_id in img_detected:
            if self.is_resonable_detection(img_detected[new_detected_cat_id][:4], self.id_to_idx[new_detected_cat_id]):
                logger.debug(f"Reasonable detection: {new_detected_cat_id}")
                reasonable_detection.add(new_detected_cat_id)
                continue
        for cat_idx, cat_id in enumerate(self.cat_ids):
            # if cat_id not in img_detected:
            if cat_id in reasonable_detection:
                continue
            for new_detected_cat_id in img_detected:
                if new_detected_cat_id in reasonable_detection or cat_id == new_detected_cat_id:
                    continue
                if self.match_buffer_history(new_detected_cat_id, img_detected, cat_idx):
                    # img_detected[cat_id] = img_detected.pop(new_detected_cat_id)
                    correct_from_to[new_detected_cat_id] = cat_id
                    logger.debug(f"Corrected cat {new_detected_cat_id} to {cat_id}: {img_detected}")
                    break
        # T1/T2: relabel ambiguity, reassessed read-only (same img_detected the loop used).
        for relabeled_id, assigned_id in correct_from_to.items():
            matched = {}
            for c_idx, c_id in enumerate(self.cat_ids):
                if c_id == relabeled_id:
                    continue
                rule = self.match_buffer_history_rule(relabeled_id, img_detected, c_idx)
                if rule is not None:
                    matched[c_id] = rule
            if len(matched) >= 2:
                self._escalate(
                    "T1",
                    f"detection labeled cat {relabeled_id} fits {len(matched)} cats {sorted(matched)}; kept first match cat {assigned_id}",
                    self.right_time,
                    cat_id=assigned_id,
                    metrics={"original_cat_id": relabeled_id, "candidates": matched, "chosen_cat_id": assigned_id},
                )
            elif matched.get(assigned_id) == "distance":
                self._escalate(
                    "T2",
                    f"detection labeled cat {relabeled_id} relabeled to cat {assigned_id} by center distance alone, no IoU support",
                    self.right_time,
                    cat_id=assigned_id,
                    metrics={"original_cat_id": relabeled_id, "rule": "distance"},
                )
        corrected_img_detected = {}
        for cat_id in img_detected:
            corrected_img_detected[correct_from_to.get(cat_id, cat_id)] = img_detected[cat_id]
        img_detected = corrected_img_detected
        logger.debug(f"Corrected ID: {correct_from_to}: new_img_detected: {img_detected}")

        for cat_idx, cat_id in enumerate(self.cat_ids):
            if cat_id in img_detected:
                # detected
                x1, y1, x2, y2, original_cat_id, conf = img_detected[cat_id]
                new_det = [self.right_time, x1, y1, x2, y2, original_cat_id, conf] + [correct_from_to.get(cat_id, cat_id)]

                jump_iou = (
                    get_iou(self.base[cat_idx][1:5], [x1, y1, x2, y2])
                    if self.last_detection[cat_idx] is None and self.base[cat_idx] is not None
                    else None
                )
                if jump_iou is not None and jump_iou < 0.2:
                    self._escalate(
                        "T4",
                        f"cat {cat_id} new box missed the base (IoU {jump_iou:.2f}); wiped {len(self.buffer[cat_idx])} buffered rows and closed the period",
                        self.right_time,
                        cat_id=cat_id,
                        period_id=self.period_id[cat_idx] or None,
                        metrics={"base_iou": jump_iou, "wiped_rows": len(self.buffer[cat_idx])},
                    )
                    for k in range(len(self.buffer[cat_idx])):
                        self.buffer[cat_idx][k][1:7] = [None] * 5 + [0]
                        self.buffer_origin[cat_idx][k][1:7] = [None] * 5 + [0]
                    self.base[cat_idx] = None
                    self.detecting[cat_idx] = False
                    self.last_detection_index[cat_idx] = None
                else:
                    # self.remove_misdetection(new_det, cat_id)
                    self.last_detection_index[cat_idx] = -1
                    self.last_detection[cat_idx] = new_det
            else:
                # create time, x1, y1, x2, y2, original_cat_id, conf, cat_id
                new_det = [self.right_time, None, None, None, None, None, 0, None]

                if self.last_detection_index[cat_idx] is not None:
                    self.last_detection_index[cat_idx] -= 1

            # collect detection
            logger.debug(f"ADD new detection to Buffer: {new_det}")
            self.buffer[cat_idx].append(np.array(new_det))
            self.buffer_origin[cat_idx].append(np.array(new_det))
            # self.smooth_buffer(cat_id)

    def check_buffer_label(self):
        for cat_idx, cat_id in enumerate(self.cat_ids):
            if len(self.buffer[cat_idx]) == 0:
                continue
            original_id_counter = Counter([b[6] for b in self.buffer[cat_idx]])
            id_common, id_count = original_id_counter.most_common(1)[0]
            if id_common is not None and id_common != cat_id and id_count >= len(self.buffer[cat_idx]) - 2:
                for k in range(1, 1 + len(self.buffer[cat_idx])):
                    if k < len(self.buffer[id_common]):
                        self.buffer[self.id_to_idx[id_common]][k] = self.buffer[cat_idx].pop()
                        self.buffer_origin[self.id_to_idx[id_common]][k] = self.buffer_origin[cat_idx].pop()
                    else:
                        self.buffer[self.id_to_idx[id_common]].appendleft(self.buffer[cat_idx].pop())
                        self.buffer_origin[self.id_to_idx[id_common]].appendleft(self.buffer_origin[cat_idx].pop())
                self.base[cat_idx] = None
                self.detecting[cat_idx] = False
                self.last_detection_index[cat_idx] = None

    def select_target(self, cat_idx):
        if len(self.buffer[cat_idx]) == 0:
            self.target[cat_idx] = None
            self.max_buffer_conf[cat_idx] = 0
            return

        if len(self.buffer[cat_idx]) == 1:
            self.target[cat_idx] = self.buffer[cat_idx][0]
            self.max_buffer_conf[cat_idx] = self.buffer[cat_idx][0][6]
            logger.debug(f"Cat: {cat_idx} --- Target (first in buffer): {list(self.buffer[cat_idx][0])}")
            return

        target_det, max_conf = None, 0

        last_det = self.base[cat_idx]
        # If high overlay rate with last detection, select it
        for k, b in enumerate(self.buffer[cat_idx]):
            if b[5] is None:
                continue
            if last_det is not None and last_det[1] is not None and get_iou(b[1:5], last_det[1:5]) > 0.7:
                self.max_buffer_conf[cat_idx] = max(self.max_buffer_conf[cat_idx], b[6])
                self.target[cat_idx] = b
                logger.debug(f"Cat: {cat_idx} --- Target (most overlay): {list(b)}")
                return

        # If high confidence, select it
        for k, b in enumerate(self.buffer[cat_idx]):
            if b[5] is None:
                continue
            if b[6] > 0.5 or (last_det is not None and last_det[1] is not None and get_iou(b[1:5], last_det[1:5]) > 0):
                self.target[cat_idx] = b
                self.max_buffer_conf[cat_idx] = b[6]
                logger.debug(f"Cat: {cat_idx} --- Target (high conf): {list(b)}")
                return
            if b[6] > max_conf:
                max_conf = b[6]
                target_det = b.copy()
            last_det = b
        self.target[cat_idx] = target_det
        logger.debug(f"Cat: {cat_idx} --- Target (normal): {list(target_det) if target_det is not None else None}")
        self.max_buffer_conf[cat_idx] = max_conf
        if target_det is not None:
            self._escalate(
                "T5",
                f"cat {self.idx_to_id[cat_idx]} target is a fallback guess: nothing overlaps the base and nothing above 0.5 conf (best {max_conf:.2f})",
                self.right_time,
                cat_id=self.idx_to_id[cat_idx],
                metrics={"target_conf": max_conf, "target_time": target_det[0]},
            )
        elif self.detecting[cat_idx] and any(b[1] is not None for b in self.buffer[cat_idx]):
            self._escalate(
                "T5",
                f"cat {self.idx_to_id[cat_idx]} has an open period but no target: buffer holds only empty or interpolated rows",
                self.right_time,
                cat_id=self.idx_to_id[cat_idx],
                period_id=self.period_id[cat_idx] or None,
                metrics={"buffer_rows": len(self.buffer[cat_idx])},
            )

    def collect_buffer(self, cat_idx, sink, view_id):
        # self.check_buffer_label()

        base_k, k = -1, 0
        max_poop_time = self.right_time - datetime.timedelta(seconds=self.buffer_range)
        if self.target[cat_idx] is not None:
            max_poop_time = min(max_poop_time, self.target[cat_idx][0])

        while self.buffer[cat_idx][k][0] <= max_poop_time:
            revised = self.buffer[cat_idx][k].copy()
            revised[7] = self.idx_to_id[cat_idx]
            original = self.buffer_origin[cat_idx][k].copy()
            if revised[1] is not None:
                base_k = k
                # print(f"start revise: {cat_id} - {k}: {list(revised)}")
                self.base[cat_idx] = revised.copy()
                self.max_period_conf[cat_idx] = max(self.max_period_conf[cat_idx], revised[6])
                if not self.detecting[cat_idx]:
                    self.period_id[cat_idx] = self.get_new_period_id()
                self.detecting[cat_idx] = True

                if self.last_revised_detection[cat_idx] is not None and revised[5] is not None:
                    last_revised = self.last_revised_detection[cat_idx]
                    if last_revised[1] is not None:
                        distance = get_distance(last_revised[1:5], revised[1:5])
                        # speed = get_speed(last_revised, revised)
                        # logger.debug(f"Continuous distance: {distance:.2f} (Time:{revised[0]} - Cat:{revised[-2]})")
                        if distance > 300:
                            old_period_id = self.period_id[cat_idx]
                            self.period_id[cat_idx] = self.get_new_period_id()
                            retracted = 0
                            while len(self.revised_detection[cat_idx]) > 0 and self.revised_detection[cat_idx][-1][5] is None:
                                remove_detection = self.revised_detection[cat_idx].pop()
                                self.original_detection[cat_idx].pop()
                                self.remove_revised_detection_from_db(remove_detection, sink)
                                # print("Remove detection:", list(remove_detection))
                                retracted += 1
                                yield False, remove_detection
                            self._escalate(
                                "T6",
                                f"cat {self.idx_to_id[cat_idx]} box jumped {distance:.0f} px; split period {old_period_id} -> {self.period_id[cat_idx]} and retracted {retracted} interpolated rows",
                                revised[0],
                                cat_id=self.idx_to_id[cat_idx],
                                period_id=self.period_id[cat_idx],
                                metrics={"distance_px": distance, "box_conf": revised[6], "old_period_id": old_period_id, "retracted_rows": retracted},
                            )
                # create_time, box_x1, box_y1, box_x2, box_y2, conf, cat_id, period_id = revised_detection

                # Create Time, Box_x1, Box_y1, Box_x2, Box_y2, Original_Cat_ID, Conf, Cat_ID, Period_ID
                if len(revised) == 8:
                    new_revise_detection = np.concatenate((revised, np.array([self.period_id[cat_idx]])))
                else:
                    revised[8] = self.period_id[cat_idx]
                    new_revise_detection = revised

                self.buffer[cat_idx][k] = new_revise_detection
                # print(f"revised cat: {cat_idx} - {k}: {list(new_revise_detection)}")

                if new_revise_detection[5] is not None and new_revise_detection[6] > self.min_conf:
                    self.last_revised_detection[cat_idx] = new_revise_detection

                self.save_revised_detection_to_db(new_revise_detection, view_id, sink)
                self.revised_detection[cat_idx].append(new_revise_detection)
                self.original_detection[cat_idx].append(np.concatenate((original, np.array([self.period_id[cat_idx]]))))
                # yield True, new_revise_detection
            # else:
            # yield True, revised
            k += 1

        if base_k == -1 and self.buffer[cat_idx][0][0] < self.right_time - datetime.timedelta(seconds=self.max_undetected_gap):
            self.base[cat_idx] = None
            self.detecting[cat_idx] = False
            self.last_revised_detection[cat_idx] = None
        poop = [0, None]
        kk = 0
        logger.debug(
            f"cat id:{cat_idx}, kk: {kk}, base_k: {base_k}, buffer: {len(self.buffer[cat_idx])},  poop time: {self.buffer[cat_idx][0][0]} gap_threshold: {self.right_time - datetime.timedelta(seconds=self.max_undetected_gap)}"
        )
        while kk <= base_k or self.buffer[cat_idx][0][0] < self.right_time - datetime.timedelta(seconds=self.max_undetected_gap):
            self.buffer_origin[cat_idx].popleft()
            poop = self.buffer[cat_idx].popleft()
            poop[7] = self.idx_to_id[cat_idx]
            # print(f"pooped cat: {cat_idx} - {kk}: {list(poop)}")
            yield True, poop
            if poop[1] is not None and poop[5] is None:
                run_start, run_flagged = self._synth_run[cat_idx]
                if run_start is None:
                    self._synth_run[cat_idx] = [poop[0], False]
                elif not run_flagged and (poop[0] - run_start).total_seconds() > self.escalation_config.synth_gap_seconds:
                    self._escalate(
                        "T7",
                        f"cat {self.idx_to_id[cat_idx]} committed {(poop[0] - run_start).total_seconds():.1f} s of interpolated-only frames with no real detection",
                        poop[0],
                        cat_id=self.idx_to_id[cat_idx],
                        period_id=self.period_id[cat_idx] or None,
                        metrics={"gap_start": run_start, "gap_seconds": (poop[0] - run_start).total_seconds()},
                    )
                    self._synth_run[cat_idx][1] = True
            else:
                self._synth_run[cat_idx] = [None, False]
            if poop[1] is None:
                self.detecting[cat_idx] = False
                self.last_revised_detection[cat_idx] = None
            if self.last_detection_index[cat_idx] is not None and -self.last_detection_index[cat_idx] > len(self.buffer[cat_idx]):
                self.last_detection_index[cat_idx] = None
                self.last_detection[cat_idx] = None
            kk += 1
        if poop[1] is not None:
            self.detecting[cat_idx] = True

    def interplate_buffer(self, cat_idx):
        if self.base[cat_idx] is None or self.target[cat_idx] is None or self.target[cat_idx][6] == 0:
            return

        # TODO check this
        if self.base[cat_idx][0] == self.target[cat_idx][0]:
            return
        base_target_iou = get_iou(self.base[cat_idx][1:5], self.target[cat_idx][1:5])

        for k, b in enumerate(self.buffer[cat_idx]):
            if b[0] < self.target[cat_idx][0]:
                update_weight = (
                    (max(0, b[6] - self.min_conf) / (self.max_buffer_conf[cat_idx] - self.min_conf))
                    if base_target_iou < 0.7 and b[1] is not None and self.max_buffer_conf[cat_idx] != self.min_conf
                    else 0
                )

                interplate_box = self.base[cat_idx][1:5] + (self.target[cat_idx][1:5] - self.base[cat_idx][1:5]) * (
                    (b[0] - self.base[cat_idx][0]).total_seconds() / (self.target[cat_idx][0] - self.base[cat_idx][0]).total_seconds()
                )
                update_box = b[1:5] * update_weight + interplate_box * (1 - update_weight) if update_weight != 0 else interplate_box
                self.buffer[cat_idx][k][1:5] = update_box
            else:
                return

    def process_images(self, create_time, img_detected, is_grey, model_id, view_id, device_id, sink=None):
        time_0 = datetime.datetime.now()
        sink = sink if sink is not None else InMemorySink()
        logger.debug("----------------------------- Track Process Started ------------------------")
        # img_detected: {cat_id: [box_x1, box_y1, box_x2, box_y2, original_cat_id, conf]}
        if self.now_detecting_date is not None and self.now_detecting_date != create_time.date():
            self.finish_up(sink)
        self.now_detecting_date = create_time.date()

        logger.debug(f"Process image: {create_time}")
        logger.debug(f"Detected boxes: {img_detected}")
        # print("Take time:", (time_1 - time_0).total_seconds(), [len(b) for b in self.buffer])
        logger.debug("Before Process ----")
        logger.debug(f"Base:   {list(self.base[1]) if self.base[1] is not None else None}")
        logger.debug(f"Target: {list(self.target[1]) if self.target[1] is not None else None}")

        self.save_image_to_db(create_time, len(img_detected), is_grey, view_id, device_id, sink)
        self.save_original_detection_to_db(create_time, img_detected, model_id, view_id, sink)

        self.right_time = create_time

        self.add_new_detection_to_buffer(img_detected)

        for cat_idx in range(self.number_of_cats):
            logger.debug(f"Buffer {cat_idx} length: {len(self.buffer[cat_idx])}")
            for b in self.buffer[cat_idx]:
                if b[2] is not None:
                    logger.debug(f"Buffer {cat_idx}: {list(b)}")
            logger.debug("-------------------------------")

        for cat_idx in range(self.number_of_cats):
            self.select_target(cat_idx)
            if self.max_buffer_conf[cat_idx] < self.min_conf and self.max_period_conf[cat_idx] < self.min_conf:
                for k, b in enumerate(self.buffer[cat_idx]):
                    self.buffer[cat_idx][k][1:7] = [None] * 5 + [0]
                self.last_detection_index[cat_idx] = -(len(self.buffer[cat_idx]) + 1)
                self.target[cat_idx] = None
            self.interplate_buffer(cat_idx)
            for new_revise_detection in self.collect_buffer(cat_idx, sink=sink, view_id=view_id):
                yield new_revise_detection

        time_1 = datetime.datetime.now()
        # print("Take time:", (time_1 - time_0).total_seconds(), [len(b) for b in self.buffer])
        logger.debug("After Process ----")
        logger.debug(f"Base:   {list(self.base[1]) if self.base[1] is not None else None}")
        logger.debug(f"Target: {list(self.target[1]) if self.target[1] is not None else None}")
        logger.debug("-------------------- Track Process Finished ------------------------")
        logger.debug(f"TrackCorrector process image time: {(time_1 - time_0).total_seconds():.2f} seconds")

    def save_revised_detection_to_db(self, revised_detection, view_id, sink, finish_up=False):
        if revised_detection is not None and len(revised_detection) > 0:
            sink.save_revised(list(revised_detection) + [view_id])
        if finish_up:
            sink.flush()

    def remove_revised_detection_from_db(self, revised_detection, sink):
        sink.delete_revised(list(revised_detection))

    def finish_up(self, sink=None):
        logger.info("Finishing up Tracker ... ")
        if sink is not None:
            sink.flush()

        self.revised_detection = [[] for _ in range(self.number_of_cats)]
        self.original_detection = [[] for _ in range(self.number_of_cats)]
        self.buffer = [deque([]) for _ in range(self.number_of_cats)]
        self.buffer_origin = [deque([]) for _ in range(self.number_of_cats)]
        self.base = [None for _ in range(self.number_of_cats)]
        self.target = [None for _ in range(self.number_of_cats)]
        self.last_detection_index = [None for _ in range(self.number_of_cats)]
        self.last_detection = [None for _ in range(self.number_of_cats)]
        self.max_buffer_conf = [0 for _ in range(self.number_of_cats)]
        self.period_id = [0 for _ in range(self.number_of_cats)]
        self.max_period_conf = [0 for _ in range(self.number_of_cats)]
        self.detecting = [False for _ in range(self.number_of_cats)]
        self.last_revised_detection = [None for _ in range(self.number_of_cats)]
        self.escalations = []
        self._synth_run = [[None, False] for _ in range(self.number_of_cats)]

    def save_image_to_db(self, create_time, detected, is_grey, view_id, device_id, sink, finish_up=False):
        if create_time is not None:
            sink.save_image((create_time, detected, is_grey, view_id, device_id))
        if finish_up:
            sink.flush()

    def save_original_detection_to_db(self, create_time, img_detection, model_id, view_id, sink, finish_up=False):
        if create_time is not None:
            rows = []
            for cat_id in img_detection:
                x1, y1, x2, y2, original_cat_id, conf = img_detection[cat_id]
                rows.append((create_time, cat_id, x1, y1, x2, y2, conf, model_id, view_id))
            sink.save_original(rows)
        if finish_up:
            sink.flush()


def remove_multi_labeling(img_detected, kills=None):
    keys = list(img_detected.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if keys[i] not in img_detected or keys[j] not in img_detected:
                continue
            iou = get_iou(img_detected[keys[i]][:4], img_detected[keys[j]][:4])
            if iou > 0.7:
                kill_detection = keys[i] if img_detected[keys[i]][5] < img_detected[keys[j]][5] else keys[j]
                kept_detection = keys[j] if kill_detection == keys[i] else keys[i]
                if kills is not None:
                    kills.append((kill_detection, kept_detection, iou, abs(img_detected[keys[i]][5] - img_detected[keys[j]][5])))
                img_detected.pop(kill_detection)
                logger.debug(f"Remove multi labeling of cat {kill_detection}: {img_detected}")
    return img_detected


def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def get_distance(boxA, boxB):
    centerAx = (boxA[0] + boxA[2]) / 2
    centerAy = (boxA[1] + boxA[3]) / 2
    centerBx = (boxB[0] + boxB[2]) / 2
    centerBy = (boxB[1] + boxB[3]) / 2
    distance = np.sqrt((centerAx - centerBx) ** 2 + (centerAy - centerBy) ** 2)
    return distance


def get_speed(detectA, detectB):
    distance = get_distance(detectA[1:5], detectB[1:5])
    second_diff = (detectB[0] - detectA[0]).total_seconds()
    speed = distance / second_diff
    return speed
