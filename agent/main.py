"""Runnable tracking entry point: live camera ingest or stream replay (Phase 4 of 5).

Detections-only: consumes detector JSON pushed by the camera over a websocket
(--live) or recorded in a JSONL file (--replay), runs the
TrackCorrector -> PeriodCorrector ingest loop, and writes a run directory with
JSONL logs, timeline charts, optional per-frame process frames + MP4, and
run.json metadata. No pixels are fetched; no database is touched.

Window: --start/--end are REQUIRED minute-precision timestamps
("YYYY-MM-DD-HH-MM"). A message at time t is processed iff
start <= t < end + 60s, i.e. the --end minute is fully included. --live stops
on the first message at/after end+60s (or Ctrl-C); --replay filters its input
to the same window.

Run directory (<out>/<run_name>/):
  stream.jsonl      canonical per-detection rows (FileSink) plus, in --live
                    runs, one raw camera message line per received message
  revised.jsonl     committed revised rows (+ deleted:true tombstones)
  periods.jsonl     period verdicts; images.jsonl: per-frame capture log
  escalations.jsonl one Escalation JSON object per line
  validations.jsonl (only with --validate) one validation op per escalation
                    (each op carries its agent_run stats + trace.md path)
  validated.jsonl + validated.db (only with --validate) verdict-corrected rows
  original.png / revised.png  timeline strip charts (unless --no-plot)
  validated.png (only with --validate, unless --no-plot) validated strip chart
  process_frames/*.jpg + process.mp4  (only with --process-viz)
  run.json          args, view, cat mapping, started/finished/elapsed wall time,
                    counts incl. escalations by code
  report.md         session summary (unless --no-report): metrics, charts,
                    escalations + verdicts, LLM trace links, memory stores,
                    images/tokens/cost; regenerate with
                    `python -m agent.tracking.report <run_dir>`

Importing this module has no side effects; tests import the helpers directly.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import itertools
import json
import logging
import math
import time
from collections import Counter
from pathlib import Path

from agent.escalation import Escalation, EscalationConfig, emit as emit_escalation
from agent.tracking import FileSink, PeriodCorrector, RunStore, TrackCorrector, plot_detection_charts, plot_validated_chart, resolve_cat_ids
from agent.tracking.process_viz import init_frame_dir, render_process_frame, stitch_mp4
from agent.tracking.report import REPORT_FILE, write_session_report
from agent.tracking.store import DEFAULT_RUNS_DIR

logger = logging.getLogger(__name__)

CAMERA_TIME_FORMAT = "%Y-%m-%d-%H-%M-%S-%f"
ARG_TIME_FORMAT = "%Y-%m-%d-%H-%M"
WINDOW_RULE = "start <= t < end + 60s"
LIVE_HOST = "0.0.0.0"


# ---------------------------------------------------------------- CLI / window


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent.main",
        description="Run the tracking correctors live from camera pushes or from a recorded stream.",
        epilog=(
            "window: --start/--end are minute-precision timestamps (YYYY-MM-DD-HH-MM); a message at time t is "
            "processed iff start <= t < end + 60s, i.e. the --end minute is fully included. --live stops on the "
            "first message at/after end+60s (or Ctrl-C). --replay filters its input file to the same window."
        ),
    )
    parser.add_argument("--view-id", type=int, required=True, help="camera view id (cat ids resolve from cats.yaml tracking_ids)")
    parser.add_argument("--start", required=True, help='window start, e.g. "2026-10-27-14-00"')
    parser.add_argument("--end", required=True, help="window end minute, same format; the whole minute is included (window is start <= t < end+60s)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="serve a websocket ingest endpoint; the camera pushes until --end or Ctrl-C")
    mode.add_argument("--replay", metavar="PATH", help="replay a camera-format stream.jsonl file filtered to the window")
    mode.add_argument("--replay-db", action="store_true", help="replay raw detections straight from MySQL (read-only, no intermediate file)")
    parser.add_argument("--port", type=int, default=8000, help="listen port for --live (default 8000)")
    parser.add_argument("--min-conf", type=float, default=0.5, help="keep detections strictly above this confidence (default 0.5)")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True, help="write original.png/revised.png charts (default --plot)")
    parser.add_argument("--process-viz", action="store_true", help="render per-frame process frames for --cat-idx (default off)")
    parser.add_argument("--cat-idx", type=int, default=0, help="index into the view's cat ids to visualize (default 0)")
    parser.add_argument("--fps", type=int, default=5, help="MP4 frame rate for --process-viz (default 5)")
    parser.add_argument("--no-video", action="store_true", help="render process frames but skip stitching process.mp4")
    parser.add_argument("--out", default=str(DEFAULT_RUNS_DIR), help=f"run output base dir (default {DEFAULT_RUNS_DIR})")
    parser.add_argument(
        "--memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="feed labeled period activity to long-term memory and "
        "every period + episode to short-term day memory (default --no-memory)",
    )
    parser.add_argument(
        "--bootstrap-frame",
        metavar="PATH",
        default=None,
        help="image file used once to bootstrap memory vision descriptions (optional)",
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="validate every escalation live with the supervisor agent and write validated_detection "
        "outputs (default --no-validate: escalations are only logged)",
    )
    parser.add_argument(
        "--report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write a report.md session summary after finish (charts, escalations + verdicts, "
        "LLM trace links, memory stores, images/tokens/cost; default --report)",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=None,
        help="quiet-window spot-check interval in stream seconds (default from ESCALATION_HEARTBEAT_SECONDS, "
        "300; 0 disables the periodic H1)",
    )
    parser.add_argument(
        "--heartbeat-at-end",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="emit one H1 spot-check when a run ends with zero escalations "
        "(default from ESCALATION_HEARTBEAT_AT_END, on)",
    )
    return parser


def parse_minute(value: str, name: str) -> datetime.datetime:
    """Parse a YYYY-MM-DD-HH-MM argument; raises ValueError with the format hint."""
    try:
        return datetime.datetime.strptime(value, ARG_TIME_FORMAT)
    except (TypeError, ValueError):
        raise ValueError(f"--{name} {value!r}: want YYYY-MM-DD-HH-MM, e.g. 2026-10-27-14-30")


def resolve_window(start_s: str, end_s: str) -> tuple:
    """Return (start, end_exclusive) with end_exclusive = --end + 60s.

    The window is start <= t < end + 60s, so the whole --end minute counts.
    Raises ValueError when the format is wrong or --end precedes --start.
    """
    start = parse_minute(start_s, "start")
    end = parse_minute(end_s, "end")
    if end < start:
        raise ValueError(f"--end {end_s} is before --start {start_s}")
    return start, end + datetime.timedelta(seconds=60)


# ---------------------------------------------------------------- messages


def is_camera_message(obj) -> bool:
    """True for camera-format lines (dicts carrying a "detections" key).

    Canonical FileSink rows share stream.jsonl in live runs; they have no
    "detections" key, so replay skips them via this check.
    """
    return isinstance(obj, dict) and "detections" in obj


def parse_message_time(value) -> datetime.datetime:
    """Parse a camera create_time ("%Y-%m-%d-%H-%M-%S-%f"); ISO-8601 accepted as fallback."""
    if isinstance(value, datetime.datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.datetime.strptime(value, CAMERA_TIME_FORMAT)
        except ValueError:
            pass
        try:
            return datetime.datetime.fromisoformat(value)
        except ValueError:
            pass
    raise ValueError(f"bad create_time {value!r}: want {CAMERA_TIME_FORMAT}")


def view_matches(msg_view, view_id: int) -> bool:
    """True when the message belongs to this run's view (missing counts as a match)."""
    if msg_view is None:
        return True
    try:
        return int(msg_view) == int(view_id)
    except (TypeError, ValueError):
        return str(msg_view) == str(view_id)


def build_img_detected(detections, *, min_conf: float, known_ids) -> tuple:
    """Filter one message's detections into the {cat_id: [x1,y1,x2,y2,orig,conf]} ingest map.

    Keeps detections strictly above min_conf (legacy semantics) whose cat id is
    known for this view. Returns (img_detected, stats Counter with raw, kept,
    low_conf, unknown_cat, bad_box, bad_list counts).
    """
    known = set(known_ids)
    stats = Counter(raw=0, kept=0, low_conf=0, unknown_cat=0, bad_box=0, bad_list=0)
    img_detected = {}
    if not isinstance(detections, list):
        stats["bad_list"] += 1
        return img_detected, stats
    for det in detections:
        stats["raw"] += 1
        try:
            cat_id = int(det["cat_id"])
            conf = float(det["confidence"])
            x1, y1, x2, y2 = (float(v) for v in det["box"][:4])
        except (KeyError, TypeError, ValueError, IndexError):
            stats["bad_box"] += 1
            continue
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2, conf)):
            stats["bad_box"] += 1
            continue
        if conf <= min_conf:
            stats["low_conf"] += 1
            continue
        if cat_id not in known:
            stats["unknown_cat"] += 1
            continue
        img_detected[cat_id] = [x1, y1, x2, y2, cat_id, conf]
    stats["kept"] = len(img_detected)
    return img_detected, stats


# ---------------------------------------------------------------- run state


class _EscCollector:
    """Collect every escalation, surviving finish_up's list rebinds.

    emit() appends to the corrector's current list, but finish_up() rebinds
    `corrector.escalations` to a fresh list (also on day rollover mid-run), so
    holding one reference would strand items. drain() harvests new items and
    follows rebinds; call it after every frame and after finish_up.
    """

    def __init__(self):
        self.all: list = []
        self._seen: dict = {}

    def drain(self, corrector) -> list:
        key = id(corrector)
        cur = corrector.escalations
        entry = self._seen.get(key)
        if entry is None:
            entry = [cur, 0]
            self._seen[key] = entry
        obj, idx = entry
        new = []
        if obj is not cur:
            new.extend(obj[idx:])
            obj, idx = cur, 0
        new.extend(cur[idx:])
        entry[0], entry[1] = cur, len(cur)
        self.all.extend(new)
        return new


# ---------------------------------------------------------------- memory builders
# agent.memory is imported lazily (inside these helpers) so tracking runs even
# when memory is disabled or the memory package cannot load.


PERIOD_ROW_TS_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def _parse_period_ts(value) -> datetime.datetime:
    if isinstance(value, datetime.datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.datetime.strptime(value, PERIOD_ROW_TS_FORMAT)
        except ValueError:
            pass
        return datetime.datetime.fromisoformat(value)
    raise ValueError(f"bad period timestamp {value!r}")


def _period_text(period_id, start, end, coord_cat, name, det_len, det_conf, score) -> str:
    span = f"{start:%H:%M}-{end:%H:%M}"
    who = f"{name} (cat {coord_cat})" if coord_cat is not None else "no cat assigned"
    bits = [f"Period {period_id} ({span}): {who}."]
    if det_len is not None:
        bits.append(f"{int(det_len)} detections")
    if det_conf is not None:
        bits.append(f"mean conf {float(det_conf):.2f}")
    if score is not None:
        bits.append(f"score {float(score):.1f}")
    if len(bits) > 1:
        return bits[0] + " " + ", ".join(bits[1:]) + "."
    return bits[0]


def _image_name_for(when: datetime.datetime) -> str:
    """Original image file name for a frame time (capture-time naming)."""
    return when.strftime(CAMERA_TIME_FORMAT) + ".jpg"


def _keyframe_names(start: datetime.datetime, end: datetime.datetime, n: int = 3) -> list[str]:
    """Evenly spread image names across a span (endpoints included)."""
    if n <= 1 or end <= start:
        return [_image_name_for(start)]
    span = (end - start).total_seconds()
    return [_image_name_for(start + datetime.timedelta(seconds=span * i / (n - 1))) for i in range(n)]


def _appearance_for(view_id: int, cat_id) -> str:
    """Cat-profile appearance line for a tracker label ("" when unavailable)."""
    if cat_id is None:
        return ""
    try:
        from agent.tracking.cat_config import appearance_for_tracking_id

        return appearance_for_tracking_id(view_id, cat_id) or ""
    except Exception:
        return ""


def period_row_to_memory_event(row: list, *, view_id: int, id_to_name=None, box=None):
    """Build a MemoryEvent from a PeriodCorrector list row (live hook)."""
    from agent.memory import MemoryEvent

    start = _parse_period_ts(row[0])
    end = _parse_period_ts(row[1])
    coord = row[2]
    pid = row[14] if len(row) > 14 else None
    name = (id_to_name or {}).get(coord) if coord is not None else None
    det_len, det_conf = row[3], row[4]
    text = _period_text(pid, start, end, coord, name, det_len, det_conf, row[7])
    cid = int(coord) if coord is not None else None
    return MemoryEvent(
        time=start,
        end_time=end,
        view_id=int(view_id),
        box=box,
        cat_id=cid,
        period_id=pid,
        kind="period",
        text=text,
        confidence=float(det_conf) if det_conf is not None else None,
        image_names=_keyframe_names(start, end),
        detection={"box": list(box) if box else None, "cat_id": cid, "conf": det_conf, "period_id": pid},
        appearance=_appearance_for(view_id, cid),
    )


def period_dict_to_memory_event(row: dict, *, view_id: int, id_to_name=None, box=None):
    """Build a MemoryEvent from a periods.jsonl dict (replay hook)."""
    from agent.memory import MemoryEvent

    start = _parse_period_ts(row["start"])
    end = _parse_period_ts(row["end"])
    coord = row.get("coordinated_cat_id")
    pid = row.get("period_id")
    winner = row.get("winner") or {}
    name = (id_to_name or {}).get(coord) if coord is not None else None
    det_len, det_conf = winner.get("detect_length"), winner.get("detect_conf")
    text = _period_text(pid, start, end, coord, name, det_len, det_conf, winner.get("score"))
    cid = int(coord) if coord is not None else None
    return MemoryEvent(
        time=start,
        end_time=end,
        view_id=int(view_id),
        box=box,
        cat_id=cid,
        period_id=pid,
        kind="period",
        text=text,
        confidence=float(det_conf) if det_conf is not None else None,
        image_names=_keyframe_names(start, end),
        detection={"box": list(box) if box else None, "cat_id": cid, "conf": det_conf, "period_id": pid},
        appearance=_appearance_for(view_id, cid),
    )


def escalation_to_memory_event(esc, *, view_id: int, box=None):
    """Build a MemoryEvent from an Escalation object (short-term hook).

    Escalation occurrences are tracker uncertainty, so they carry no
    confidence and only ever reach short-term day context — never long-term
    memory (see LongTermMemory._store_qualifies).
    """
    from agent.memory import MemoryEvent

    end = esc.end_time or esc.time
    return MemoryEvent(
        time=esc.time,
        end_time=esc.end_time,
        view_id=int(view_id),
        box=box,
        cat_id=esc.cat_id,
        period_id=esc.period_id,
        kind="escalation",
        text=f"[{esc.name or esc.code}] {esc.reason}",
        code=esc.code,
        image_names=_keyframe_names(esc.time, end, 2),
        detection={"cat_id": esc.cat_id, "period_id": esc.period_id},
    )


def escalation_dict_to_memory_event(row: dict, *, view_id: int, box=None):
    """Build a MemoryEvent from an escalations.jsonl dict (short-term hook)."""
    from agent.memory import MemoryEvent

    time = parse_message_time(row["time"])
    end_raw = row.get("end_time")
    end = parse_message_time(end_raw) if end_raw else time
    return MemoryEvent(
        time=time,
        end_time=parse_message_time(end_raw) if end_raw else None,
        view_id=int(view_id),
        box=box,
        cat_id=row.get("cat_id"),
        period_id=row.get("period_id"),
        kind="escalation",
        text=f"[{row.get('name') or row.get('code')}] {row.get('reason')}",
        code=row.get("code"),
        image_names=_keyframe_names(time, end, 2),
        detection={"cat_id": row.get("cat_id"), "period_id": row.get("period_id")},
    )


class TrackingRun:
    """One bounded tracking run: ingest loop + artifacts + run.json."""

    def __init__(
        self,
        *,
        view_id: int,
        start: datetime.datetime,
        end_exclusive: datetime.datetime,
        cat_ids,
        id_to_name: dict,
        store: RunStore,
        min_conf: float = 0.5,
        process_viz: bool = False,
        cat_idx: int = 0,
        fps: int = 5,
        no_video: bool = False,
        plot: bool = True,
        mode: str = "replay",
        args_dict: dict | None = None,
        record_raw: bool = False,
        memory: bool = False,  # disconnected for now; pass True / --memory to re-enable
        memory_store_path=None,
        memory_llm_client=None,
        bootstrap_frame: str | None = None,
        validate: bool = False,
        validator=None,  # inject a ValidationWorker (tests); otherwise built when validate=True
        report: bool = True,
    ):
        self.started_at = datetime.datetime.now()
        self._t0 = time.perf_counter()
        self.report = report
        self.view_id = int(view_id)
        self.start = start
        self.end_exclusive = end_exclusive
        self.cat_ids = list(cat_ids)
        self.id_to_name = dict(id_to_name)
        self._known = set(self.cat_ids)
        self.store = store
        self.sink = FileSink(store)
        self._esc_seq = itertools.count(1)  # run-wide escalation numbering, shared by both correctors
        self.track = TrackCorrector(cat_ids=self.cat_ids, escalation_seq=self._esc_seq)
        self.period = PeriodCorrector(cat_ids=self.cat_ids, escalation_seq=self._esc_seq)
        self.min_conf = min_conf
        self.process_viz = process_viz
        self.cat_idx = cat_idx
        self.fps = fps
        self.no_video = no_video
        self.plot = plot
        self.mode = mode
        self.args_dict = dict(args_dict or {})
        if process_viz and not 0 <= cat_idx < len(self.cat_ids):
            raise ValueError(f"--cat-idx {cat_idx} out of range for {len(self.cat_ids)} cats (0..{len(self.cat_ids) - 1})")
        self.frame_dir = None
        if process_viz:
            self.frame_dir = Path(init_frame_dir(str(store.run_dir / "process_frames")))
        self._viz_n = 0
        self.counts: Counter = Counter()
        self._escs = _EscCollector()
        self._raw_fh = None
        if record_raw:
            self._raw_fh = open(store.run_dir / RunStore.STREAM_FILE, "a", encoding="utf-8", buffering=1)
        self.memory = None
        if memory:
            try:
                self.memory = self._init_memory(memory_store_path, memory_llm_client)
            except Exception:
                logger.exception("memory init failed; continuing without memory")
                self.memory = None
        if self.memory is not None and bootstrap_frame:
            try:
                self.memory.bootstrap(Path(bootstrap_frame).read_bytes())
            except Exception:
                logger.exception("memory bootstrap failed; continuing in text-only mode")
        self.short_term = None
        if self.memory is not None:
            try:
                from agent.memory import DailyShortTermMemory

                self.short_term = DailyShortTermMemory(self.memory.config, self.view_id, self.memory)
                self.short_term.start_session(start, store.run_dir.name)
            except Exception:
                logger.exception("short-term memory init failed; continuing without it")
                self.short_term = None
        # Period activity path: closed periods combine across small gaps, each
        # member is labeled from its key frames, and completed records land in
        # both memories. Without it, periods store directly (legacy path).
        self._activity_agg = None
        self._labeler = None
        self._resolved: dict = {}  # period_id -> real keyframe names from the labeler
        if self.memory is not None:
            try:
                from agent.memory import PeriodActivityAggregator

                self._activity_agg = PeriodActivityAggregator(
                    gap_minutes=self.memory.config.activity_combine_gap_minutes
                )
            except Exception:
                logger.exception("activity aggregator init failed; periods store unlabeled")
                self._activity_agg = None
        if self._activity_agg is not None:
            try:
                from agent.validation import PeriodLabeler

                self._labeler = PeriodLabeler(sink=self._on_period_label, llm_client=self.memory.llm)
                self._labeler.start()
            except Exception:
                logger.exception("period labeler init failed; activity stays unknown")
                self._labeler = None
        self.validator = validator
        if self.validator is None and validate:
            from agent.validation import ValidationWorker

            self.validator = ValidationWorker(
                journal=store.append_validation,
                memory=self.memory,
                box_lookup=self._esc_box,
                short_term=self.short_term,
            )
        if self.validator is not None:
            self.validator.start()

    def _esc_box(self, esc):
        """Median box of the escalation's period for memory context (None when unavailable)."""
        try:
            return self._period_box(esc.period_id)
        except Exception:
            return None

    # -- memory (all hooks are best-effort; they never break the run) --

    def _init_memory(self, store_path, llm_client):
        from agent.memory import LongTermMemory, MemoryConfig

        cfg = MemoryConfig.from_yaml()
        cfg.view_id = self.view_id
        if llm_client is None:
            try:
                from agent.memory import MetaModelClient

                llm_client = MetaModelClient(role=cfg.llm_model_role, trace_dir=cfg.trace_dir)
            except Exception:
                logger.warning("memory LLM unavailable; using offline stub", exc_info=True)
                from agent.memory import StubLLMClient

                llm_client = StubLLMClient()
        return LongTermMemory(config=cfg, store_path=store_path, llm_client=llm_client)

    def _memory_tick(self, now) -> None:
        if self.memory is None:
            return
        try:
            self.memory.tick(now)
        except Exception:
            logger.exception("memory tick failed")
        if self.short_term is not None:
            try:
                self.short_term.rollover(now)
            except Exception:
                logger.exception("short-term rollover failed")

    def _period_box(self, period_id) -> tuple | None:
        """Median box of a closed period's detections (None when unavailable)."""
        try:
            boxes = [[d[1], d[2], d[3], d[4]] for d in self.period.detections if len(d) >= 9 and d[8] == period_id and d[1] is not None]
        except Exception:
            return None
        if not boxes:
            return None
        try:
            cols = sorted(boxes, key=lambda b: (b[0], b[1], b[2], b[3]))
            mid = cols[len(cols) // 2]
            vals = [float(v) for v in mid]
            if not all(math.isfinite(v) for v in vals):
                return None
            return (vals[0], vals[1], vals[2], vals[3])
        except (TypeError, ValueError):
            return None

    def _memory_store_period_row(self, row) -> None:
        if self.memory is None or row is None:
            return
        try:
            pid = row[14] if isinstance(row, list) and len(row) > 14 else None
            box = self._period_box(pid)
            event = period_row_to_memory_event(
                row,
                view_id=self.view_id,
                id_to_name=self.id_to_name,
                box=box,
            )
            if self._activity_agg is None:
                self._legacy_memory_store(event, row, pid, box)
                return
            from agent.memory import PeriodMember

            member = PeriodMember(
                period_id=pid,
                cat_id=event.cat_id,
                cat_name=self.id_to_name.get(event.cat_id) if event.cat_id is not None else None,
                start=event.time,
                end=event.end_time or event.time,
                detections=int(row[3] or 0),
                mean_conf=event.confidence,
                box=box,
                keyframes=list(event.image_names),
            )
            emitted = self._activity_agg.add(member)
            if member.label is None and self._labeler is not None and member.keyframes:
                self._labeler.label_period(
                    pid,
                    cat_id=member.cat_id,
                    cat_name=member.cat_name,
                    span=f"{member.start.strftime('%H:%M')} to {member.end.strftime('%H:%M')}",
                    keyframes=list(member.keyframes),
                    boxes=[box for _ in member.keyframes],
                )
            for record in emitted:
                self._store_activity_record(record)
        except Exception:
            logger.exception("memory period store failed; dropping event")

    def _legacy_memory_store(self, event, row, pid, box) -> None:
        """Direct period store when the activity path is unavailable."""
        self.memory.store(event)  # kept only when high-confidence (trust gate)
        if self.short_term is not None:
            self.short_term.record_period(
                period_id=pid,
                cat_id=event.cat_id,
                cat_name=self.id_to_name.get(event.cat_id) if event.cat_id is not None else None,
                start=event.time,
                end=event.end_time or event.time,
                detections=int(row[3] or 0),
                mean_conf=event.confidence,
                box=box,
                keyframes=event.image_names,
            )

    def _on_period_label(self, period_id, label, resolved_keyframes) -> None:
        """Labeler sink (labeler thread): attach the label, store completions."""
        try:
            if resolved_keyframes:
                self._resolved[period_id] = list(resolved_keyframes)
            if self._activity_agg is None:
                return
            for record in self._activity_agg.set_label(period_id, label):
                self._store_activity_record(record)
        except Exception:
            logger.exception("period label store failed; dropping label")

    def _store_activity_record(self, record) -> None:
        """One completed activity record lands in both memories (best-effort)."""
        from agent.memory import MemoryEvent

        members = list(record.members)
        labels = [m.label for m in members]
        # Real keyframe names where the labeler resolved them, else synthesized.
        names: list[str] = []
        for m in members:
            for k in self._resolved.get(m.period_id, m.keyframes):
                if k not in names:
                    names.append(str(k))
        acts = [l.activity for l in labels if l is not None and l.activity]
        distinct = list(dict.fromkeys(acts))
        if len(distinct) == 1:
            activity: str | None = distinct[0]
            confs = [l.confidence for l in labels if l is not None and l.confidence is not None]
            activity_conf = min(confs) if confs else None
        elif distinct:
            activity = ", then ".join(
                f"{m.label.activity or 'unknown'} "
                f"{m.start.strftime('%H:%M')}–{m.end.strftime('%H:%M')}"
                for m in members
                if m.label is not None
            ) or None
            activity_conf = None
        else:
            activity, activity_conf = None, None
        looks = "; ".join(dict.fromkeys(l.appearance for l in labels if l is not None and l.appearance))
        scene = "; ".join(dict.fromkeys(l.background for l in labels if l is not None and l.background))
        mean_conf = record.mean_conf
        # Long-term memory only keeps high-confidence records: the detector
        # and every member label must all agree at strength, else the trust
        # gate drops it (short-term still keeps the full-fidelity entry).
        label_confs = [l.confidence for l in labels if l is not None]
        trusted = (
            bool(members)
            and all(l is not None and not l.degraded and l.activity for l in labels)
            and all(isinstance(c, (int, float)) for c in label_confs)
            and isinstance(mean_conf, (int, float))
        )
        confidence = min(label_confs + [mean_conf]) if trusted else 0.0
        first_pid = members[0].period_id if members else None
        box = record.box
        event = MemoryEvent(
            time=record.start,
            end_time=record.end,
            view_id=self.view_id,
            box=box,
            cat_id=record.cat_id,
            period_id=first_pid,
            kind="period",
            text=record.narrative(),
            confidence=float(confidence),
            image_names=names,
            detection={
                "box": list(box) if box else None,
                "cat_id": record.cat_id,
                "conf": mean_conf,
                "period_id": first_pid,
                "period_ids": record.period_ids,
                "activity": activity,
            },
            appearance=looks or _appearance_for(self.view_id, record.cat_id),
            background=scene,
        )
        self.memory.store(event)
        if self.short_term is not None:
            self.short_term.record_period(
                period_id=first_pid,
                cat_id=record.cat_id,
                cat_name=record.cat_name,
                start=record.start,
                end=record.end,
                detections=record.detections,
                mean_conf=mean_conf,
                box=box,
                keyframes=names,
                period_ids=record.period_ids,
                activity=activity,
                activity_confidence=activity_conf,
                appearance=looks,
                background=scene,
            )

    def _finish_activity_labels(self) -> None:
        """Drain the labeler, then flush combined records (finish() only)."""
        if self._labeler is not None:
            try:
                self._labeler.stop()  # queued jobs land via set_label first
            except Exception:
                logger.exception("period labeler stop failed; leftovers degrade to unknown")
            finally:
                self._labeler = None
        if self._activity_agg is None:
            return
        try:
            for record in self._activity_agg.flush():
                self._store_activity_record(record)
        except Exception:
            logger.exception("activity flush failed; dropping leftover periods")

    def _memory_store_escalation(self, esc) -> None:
        # Escalation occurrences are tracker uncertainty: short-term day
        # context only, never long-term knowledge.
        if self.short_term is None:
            return
        try:
            event = escalation_to_memory_event(esc, view_id=self.view_id)
            self.short_term.record_episode(
                name=esc.name or esc.code,
                code=esc.code,
                time=event.time,
                reason=esc.reason,
                cat_id=esc.cat_id,
                period_id=esc.period_id,
                image_names=event.image_names,
            )
        except Exception:
            logger.exception("short-term episode store failed; dropping event")

    def _shutdown_memory(self) -> None:
        if self.memory is None:
            return
        try:
            if self.short_term is not None:
                self.short_term.merge_into(self.memory)
                self.memory.promote()
        except Exception:
            logger.exception("short-term merge failed; day stays unmerged")
        try:
            self.memory.shutdown()
        except Exception:
            logger.exception("memory shutdown failed")
        finally:
            self.memory = None
            self.short_term = None

    # -- ingest --

    def record_message(self, msg: dict) -> None:
        """Record one raw camera message line (live runs only; replay input is the record)."""
        if self._raw_fh is not None:
            self._raw_fh.write(json.dumps(msg) + "\n")

    def _skip(self, reason: str) -> str:
        self.counts[f"messages_{reason}"] += 1
        logger.debug(f"skip ({reason})")
        return reason

    def process_message(self, msg: dict) -> str:
        """Run one message through the window filter + ingest loop.

        Returns the outcome: "processed", or a skip reason ("not_a_message",
        "other_view", "bad_time", "before_window", "past_end"). Empty frames
        are ingested too: they advance buffer timeouts and close periods.
        """
        self.counts["messages_seen"] += 1
        if not is_camera_message(msg):
            return self._skip("not_a_message")
        if not view_matches(msg.get("view_id"), self.view_id):
            return self._skip("other_view")
        try:
            create_time = parse_message_time(msg.get("create_time"))
        except ValueError:
            return self._skip("bad_time")
        if create_time < self.start:
            return self._skip("before_window")
        if create_time >= self.end_exclusive:
            return self._skip("past_end")
        img_detected, stats = build_img_detected(msg.get("detections"), min_conf=self.min_conf, known_ids=self._known)
        for key, value in stats.items():
            self.counts[f"detections_{key}"] += value
        self._ingest(create_time, img_detected, msg)
        self._memory_tick(create_time)
        self._persist_new_escalations()
        if self.process_viz:
            self._render_frame()
        self.counts["messages_processed"] += 1
        # logger.info(
        #     "frame #%d %s kept=%d escs=%d",
        #     self.counts["messages_processed"],
        #     create_time.strftime("%m-%d %H:%M:%S"),
        #     stats["kept"],
        #     len(self._escs.all),
        # )
        return "processed"

    def _ingest(self, create_time, img_detected: dict, msg: dict) -> None:
        for is_append, revised in self.track.process_images(
            create_time,
            img_detected,
            is_grey=msg.get("is_grey", 0),
            model_id=msg.get("model_id"),
            view_id=self.view_id,
            device_id=msg.get("device_id"),
            sink=self.sink,
        ):
            if is_append:
                for row in self.period.process_detection(revised, view_id=self.view_id, sink=self.sink):
                    self._log_period_row(row)
                    self._memory_store_period_row(row)
            else:
                self.period.unprocess_detection(revised)

    def _log_period_row(self, row) -> None:
        """One summary line per closed period (same fields as the memory event)."""
        if row is None:  # finish_up yields padding Nones between real rows
            return
        coord = row[2]
        logger.info(
            "period closed: %s",
            _period_text(
                row[14] if len(row) > 14 else None,
                _parse_period_ts(row[0]),
                _parse_period_ts(row[1]),
                coord,
                self.id_to_name.get(coord) if coord is not None else None,
                row[3],
                row[4],
                row[7],
            ),
        )

    def _persist_new_escalations(self) -> None:
        for esc in self._escs.drain(self.track):
            self.store.append_escalation(esc)
            self._memory_store_escalation(esc)
            if self.validator is not None:
                self.validator.enqueue(esc)
        for esc in self._escs.drain(self.period):
            self.store.append_escalation(esc)
            self._memory_store_escalation(esc)
            if self.validator is not None:
                self.validator.enqueue(esc)

    def _render_frame(self) -> None:
        path = self.frame_dir / f"frame_{self._viz_n:06d}.jpg"
        rendered = render_process_frame(self.track, self.period.escalations, self.cat_idx, str(path), futures=None)
        if rendered is not None:
            self._viz_n += 1
            self.counts["viz_frames"] += 1

    # -- finalize --

    def finish(self) -> Path:
        """finish_up both correctors, then write charts, video, and run.json."""
        logger.info("finalizing run ...")
        self.track.finish_up(self.sink)
        self._persist_new_escalations()
        for row in self.period.finish_up(self.view_id, self.sink):
            self._log_period_row(row)
            self._memory_store_period_row(row)
        self._persist_new_escalations()
        self._finish_activity_labels()
        if self._raw_fh is not None:
            self._raw_fh.close()
            self._raw_fh = None
        self.store.flush()

        run_dir = self.store.run_dir
        artifacts = ["stream.jsonl", "revised.jsonl", "periods.jsonl", "images.jsonl", "escalations.jsonl"]
        ops = []
        validated_jsonl = None
        if self.validator is not None:
            ops = self.validator.stop()
            try:
                from agent.validation import materialize

                revised_rows = read_revised_rows(run_dir / RunStore.REVISED_FILE)
                jsonl_path, db_path, _ = materialize(run_dir, revised_rows, ops)
                validated_jsonl = jsonl_path
                artifacts += [jsonl_path.name, db_path.name, RunStore.VALIDATIONS_FILE]
            except Exception:
                logger.exception("validated outputs failed; run keeps revised rows only")
        if self.plot:
            try:
                original_rows = read_canonical_stream_rows(run_dir / RunStore.STREAM_FILE)
                revised_rows = read_revised_rows(run_dir / RunStore.REVISED_FILE)
                original_png, revised_png = plot_detection_charts(
                    original_rows,
                    revised_rows,
                    run_dir,
                    self.cat_ids,
                    self.id_to_name,
                    self._escs.all,
                    width=self.track.img_width,
                )
                artifacts += [original_png.name, revised_png.name]
            except Exception:
                logger.exception("charts failed; continuing without them")
            if validated_jsonl is not None:
                try:
                    validated_rows = read_revised_rows(validated_jsonl)
                    validated_png = plot_validated_chart(
                        validated_rows,
                        run_dir,
                        self.cat_ids,
                        self.id_to_name,
                        self._escs.all,
                        width=self.track.img_width,
                    )
                    artifacts.append(validated_png.name)
                except Exception:
                    logger.exception("validated chart failed; continuing without it")
        if self.process_viz and not self.no_video:
            try:
                video = stitch_mp4(str(self.frame_dir), str(run_dir / "process.mp4"), self.fps)
                if video is not None:
                    artifacts.append("process.mp4")
            except ImportError:
                logger.warning("cv2 not installed; skipping process.mp4 (frames are kept)")
            except Exception:
                logger.exception("video stitch failed; frames are kept")
        if self.process_viz:
            artifacts.append("process_frames/")
        artifacts.append("run.json")
        if self.report:
            artifacts.append(REPORT_FILE)

        by_code: Counter = Counter(esc.code for esc in self._escs.all)
        counts = dict(self.store.counters)
        counts.update(self.counts)
        counts["escalations_total"] = len(self._escs.all)
        counts["escalations_by_code"] = dict(sorted(by_code.items()))
        if self.validator is not None:
            by_verdict: Counter = Counter((op.get("review") or {}).get("verdict", "?") for op in ops)
            counts["validations_total"] = len(ops)
            counts["validations_by_verdict"] = dict(sorted(by_verdict.items()))
        if self._resolved:
            counts["labeler_keyframes_total"] = sum(len(v) for v in self._resolved.values())
        finished_at = datetime.datetime.now()
        end_minute = self.end_exclusive - datetime.timedelta(seconds=60)
        meta = {
            "run_name": run_dir.name,
            "mode": self.mode,
            "args": self.args_dict,
            "view_id": self.view_id,
            "cat_ids": self.cat_ids,
            "cat_names": self.id_to_name,
            "window": {
                "start": self.start.isoformat(timespec="minutes"),
                "end": end_minute.isoformat(timespec="minutes"),
                "end_exclusive": self.end_exclusive.isoformat(timespec="seconds"),
                "rule": WINDOW_RULE,
            },
            "counts": counts,
            "artifacts": artifacts,
            "started_at": self.started_at.isoformat(timespec="microseconds"),
            "finished_at": finished_at.isoformat(timespec="microseconds"),
            "elapsed_s": round(time.perf_counter() - self._t0, 1),
            "created_at": finished_at.isoformat(timespec="microseconds"),
            "run_dir": str(run_dir),
        }
        self.store.write_run_json(meta)
        self.store.close()
        self._shutdown_memory()
        # Memory shutdown (merge + promote) trails run.json by up to minutes;
        # re-stamp the finish so the report's trace window covers it.
        finished_at = datetime.datetime.now()
        meta["finished_at"] = finished_at.isoformat(timespec="microseconds")
        meta["elapsed_s"] = round(time.perf_counter() - self._t0, 1)
        self.store.write_run_json(meta)
        if self.report:
            try:
                write_session_report(run_dir)
            except Exception:
                logger.exception("session report failed; run artifacts are complete without it")
        logger.info(f"run written: {run_dir}")
        return run_dir


# ---------------------------------------------------------------- finalize helpers


def _read_jsonl(path: Path) -> list:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError:
                logger.warning(f"skipping non-JSON line in {path}")
    return rows


def read_canonical_stream_rows(path: Path) -> list:
    """Canonical FileSink rows from stream.jsonl (skips raw live message lines)."""
    return [row for row in _read_jsonl(path) if isinstance(row, dict) and "box_x1" in row and "detections" not in row]


def read_revised_rows(path: Path) -> list:
    """Revised rows from revised.jsonl (tombstones filtered later by the charts)."""
    return [row for row in _read_jsonl(path) if isinstance(row, dict)]


# ---------------------------------------------------------------- live / replay drivers


async def _live_serve(run: TrackingRun, port: int) -> None:
    import websockets

    stop = asyncio.Event()

    async def handler(websocket, path=None):  # websockets >= 14 passes only the connection
        peer = getattr(websocket, "remote_address", "?")
        logger.info(f"camera connected: {peer}")
        try:
            async for raw in websocket:
                if isinstance(raw, bytes):
                    try:
                        raw = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        run.counts["messages_seen"] += 1
                        run.counts["messages_undecodable"] += 1
                        continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    run.counts["messages_seen"] += 1
                    run.counts["messages_bad_json"] += 1
                    logger.warning("skipping non-JSON message")
                    continue
                run.record_message(msg)  # every parsed message is recorded
                try:
                    outcome = run.process_message(msg)
                except Exception:
                    run.counts["messages_error"] += 1
                    logger.exception("dropping message after ingest error")
                    continue
                if outcome == "past_end":
                    logger.info("--end reached; stopping")
                    stop.set()
                    try:
                        await websocket.close()
                    except Exception:
                        pass
                    return
        except websockets.ConnectionClosed:
            logger.info(f"camera disconnected: {peer}")

    logger.info(f"serving live ingest on {LIVE_HOST}:{port} (stop: t >= {run.end_exclusive} or Ctrl-C)")
    async with websockets.serve(handler, LIVE_HOST, port):
        await stop.wait()


def run_live(run: TrackingRun, port: int) -> int:
    try:
        import websockets  # noqa: F401
    except ImportError:
        logger.error("the 'websockets' package is required for --live (pip install websockets)")
        run.finish()
        return 1
    try:
        asyncio.run(_live_serve(run, port))
    except KeyboardInterrupt:
        logger.warning("interrupted (Ctrl-C); finalizing partial run")
    except OSError as e:
        logger.error(f"live server error: {e}")
        run.finish()
        return 1
    run.finish()
    return 0


def run_replay(run: TrackingRun, path: str) -> Path:
    src = Path(path)
    try:
        with open(src, encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    run.counts["messages_seen"] += 1
                    run.counts["messages_bad_json"] += 1
                    logger.warning(f"skipping non-JSON line in {src}")
                    continue
                try:
                    run.process_message(msg)
                except Exception:
                    run.counts["messages_error"] += 1
                    logger.exception("dropping message after ingest error")
    except KeyboardInterrupt:
        logger.warning("interrupted (Ctrl-C); finalizing partial run")
    logger.info(f"replayed {src}: {run.counts['messages_processed']} processed, {run.counts['messages_seen']} seen")
    return run.finish()


def run_replay_db(run: TrackingRun) -> Path:
    from agent.tracking.db_stream import connect, fetch_messages

    try:
        db = connect()
    except Exception as e:
        raise SystemExit(f"cannot connect to MySQL for --replay-db: {e}")
    try:
        for msg in fetch_messages(db, run.view_id, run.start, run.end_exclusive):
            try:
                run.process_message(msg)
            except Exception:
                run.counts["messages_error"] += 1
                logger.exception("dropping message after ingest error")
    except KeyboardInterrupt:
        logger.warning("interrupted (Ctrl-C); finalizing partial run")
    logger.info(f"replayed db: {run.counts['messages_processed']} processed, {run.counts['messages_seen']} seen")
    return run.finish()


# ---------------------------------------------------------------- main


def main(argv=None) -> int:
    from agent.env import load_repo_env

    load_repo_env()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] %(filename)s:%(lineno)d: %(message)s")
    try:
        start, end_exclusive = resolve_window(args.start, args.end)
    except ValueError as e:
        parser.error(str(e))
    if args.replay is not None and not Path(args.replay).is_file():
        parser.error(f"replay file not found: {args.replay}")
    try:
        cat_ids, id_to_name = resolve_cat_ids(args.view_id)
    except ValueError as e:
        parser.error(str(e))
    if args.process_viz and not 0 <= args.cat_idx < len(cat_ids):
        parser.error(f"--cat-idx {args.cat_idx} out of range for {len(cat_ids)} cats (0..{len(cat_ids) - 1})")

    mode = "live" if args.live else ("replay_db" if args.replay_db else "replay")
    now = datetime.datetime.now()
    end_minute = end_exclusive - datetime.timedelta(seconds=60)
    run_name = f"view{args.view_id}_{start:%Y%m%d-%H%M}_{end_minute:%H%M}_{mode}_{now:%Y%m%d-%H%M%S-%f}"
    store = RunStore(run_name, base_dir=args.out)
    run = TrackingRun(
        view_id=args.view_id,
        start=start,
        end_exclusive=end_exclusive,
        cat_ids=cat_ids,
        id_to_name=id_to_name,
        store=store,
        min_conf=args.min_conf,
        process_viz=args.process_viz,
        cat_idx=args.cat_idx,
        fps=args.fps,
        no_video=args.no_video,
        plot=args.plot,
        mode=mode,
        args_dict=vars(args),
        record_raw=args.live,
        memory=args.memory,
        bootstrap_frame=args.bootstrap_frame,
        validate=args.validate,
        report=args.report,
    )
    logger.info(f"run dir: {store.run_dir} (view {args.view_id}, cats {cat_ids})")
    if args.live:
        return run_live(run, args.port)
    if args.replay_db:
        return run_replay_db(run)
    run_replay(run, args.replay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
