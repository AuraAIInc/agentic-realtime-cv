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
  original.png / revised.png  timeline strip charts (unless --no-plot)
  process_frames/*.jpg + process.mp4  (only with --process-viz)
  run.json          args, view, cat mapping, counts incl. escalations by code

Importing this module has no side effects; tests import the helpers directly.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import math
from collections import Counter
from pathlib import Path

from agent.tracking import FileSink, PeriodCorrector, RunStore, TrackCorrector, plot_detection_charts, resolve_cat_ids
from agent.tracking.process_viz import init_frame_dir, render_process_frame, stitch_mp4
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
    parser.add_argument("--end", required=True, help='window end minute, same format; the whole minute is included (window is start <= t < end+60s)')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="serve a websocket ingest endpoint; the camera pushes until --end or Ctrl-C")
    mode.add_argument("--replay", metavar="PATH", help="replay a camera-format stream.jsonl file filtered to the window")
    parser.add_argument("--port", type=int, default=8000, help="listen port for --live (default 8000)")
    parser.add_argument("--min-conf", type=float, default=0.5, help="keep detections strictly above this confidence (default 0.5)")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True, help="write original.png/revised.png charts (default --plot)")
    parser.add_argument("--process-viz", action="store_true", help="render per-frame process frames for --cat-idx (default off)")
    parser.add_argument("--cat-idx", type=int, default=0, help="index into the view's cat ids to visualize (default 0)")
    parser.add_argument("--fps", type=int, default=5, help="MP4 frame rate for --process-viz (default 5)")
    parser.add_argument("--no-video", action="store_true", help="render process frames but skip stitching process.mp4")
    parser.add_argument("--out", default=str(DEFAULT_RUNS_DIR), help=f"run output base dir (default {DEFAULT_RUNS_DIR})")
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
    ):
        self.view_id = int(view_id)
        self.start = start
        self.end_exclusive = end_exclusive
        self.cat_ids = list(cat_ids)
        self.id_to_name = dict(id_to_name)
        self._known = set(self.cat_ids)
        self.store = store
        self.sink = FileSink(store)
        self.track = TrackCorrector(cat_ids=self.cat_ids)
        self.period = PeriodCorrector(cat_ids=self.cat_ids)
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
        self._persist_new_escalations()
        if self.process_viz:
            self._render_frame()
        self.counts["messages_processed"] += 1
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
                for _ in self.period.process_detection(revised, view_id=self.view_id, sink=self.sink):
                    pass
            else:
                self.period.unprocess_detection(revised)

    def _persist_new_escalations(self) -> None:
        for esc in self._escs.drain(self.track):
            self.store.append_escalation(esc)
        for esc in self._escs.drain(self.period):
            self.store.append_escalation(esc)

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
        for _ in self.period.finish_up(self.view_id, self.sink):
            pass
        self._persist_new_escalations()
        if self._raw_fh is not None:
            self._raw_fh.close()
            self._raw_fh = None
        self.store.flush()

        run_dir = self.store.run_dir
        artifacts = ["stream.jsonl", "revised.jsonl", "periods.jsonl", "images.jsonl", "escalations.jsonl"]
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

        by_code: Counter = Counter(esc.code for esc in self._escs.all)
        counts = dict(self.store.counters)
        counts.update(self.counts)
        counts["escalations_total"] = len(self._escs.all)
        counts["escalations_by_code"] = dict(sorted(by_code.items()))
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
            "created_at": datetime.datetime.now().isoformat(timespec="microseconds"),
            "run_dir": str(run_dir),
        }
        self.store.write_run_json(meta)
        self.store.close()
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


# ---------------------------------------------------------------- main


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
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

    mode = "live" if args.live else "replay"
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
    )
    logger.info(f"run dir: {store.run_dir} (view {args.view_id}, cats {cat_ids})")
    if args.live:
        return run_live(run, args.port)
    run_replay(run, args.replay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

