"""MCP tools over the cat tracking state, served from inside detection_ingest_server.py.

Two data sources, and every reply says which one a value came from:
* memory: the live TrackCorrector buffer (last ~15 s, can still be rewritten) and the
  PeriodCorrector's open / not-yet-coordinated periods. Frames marked final=false, periods
  status ongoing / ended_pending.
* MySQL: committed revised_detection rows and coordinated period rows (status final).
Replies use plain terms (visit, detection, interpolated) so a client needs no tracker knowledge.

All tools are async. Memory is read before the first await, on the event loop that runs
process_image(), so a snapshot never sees a half-processed frame. DB queries run in a thread.
Every response carries timing = {"elapsed_ms", "steps_ms"} like the image server.
"""

from __future__ import annotations

import asyncio
import base64
import datetime
import hmac
import io
import json
import logging
import os
from collections import Counter

from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, ImageContent, TextContent

from tracking_mcp import motion
from tracking_mcp.config import Settings
from tracking_mcp.history import History
from tracking_mcp.live_state import LiveState
from tracking_mcp.serialize import (
    box_of,
    has_box,
    image_name,
    is_missing,
    parse_time,
    py,
    row_to_dict,
    seconds_between,
    to_int,
)
from tracking_mcp.timing import Timer

logger = logging.getLogger("tracking_mcp")

RELINK_PX = 160  # max center jump when chaining raw pre-roll boxes (TrackCorrector's relabel distance)
MAX_SEGMENTS = 30

INSTRUCTIONS = """\
Cat tracking results for the home cameras, used to check detections and find reference frames.

Terms:
- detection: one box the detector model found on one frame, with conf (0-1) and the cat it
  thinks it is (detector_cat_id).
- track: the tracker links detections frame to frame into one moving box per cat. Short gaps
  are filled in; those boxes have interpolated=true and no detection behind them.
- period: one continuous visit of one cat on screen, from when it appears until it leaves.
  Its cat_id is decided from all detections in the visit, so it is more reliable than the
  label on any single frame. evidence_by_cat shows, per cat, how many detections named it,
  their average conf, and a score; the cat with the highest score is chosen.
- period status: ongoing (cat still on screen), ended_pending (visit ended, cat_id may still
  change), final (saved, will not change).
- final=false on a frame: the box is from the last ~15 s and can still be corrected.

Boxes are [x1, y1, x2, y2] in the camera view's pixels (640x480 unless list_views says
otherwise). Every frame has image_name: pass it to images__get_image (box=crop_box to zoom on
the cat). edge is the image border nearest the box (left, right, top, bottom or none).
Call list_views first to map cat names to cat_id.
"""


def _result(payload: dict, timer: Timer, images: list[tuple[bytes, str]] | None = None, error: bool = False) -> CallToolResult:
    payload = py({**payload, "timing": timer.report()})
    content = [TextContent(type="text", text=json.dumps(payload))]
    for data, mime in images or []:
        content.append(ImageContent(type="image", data=base64.b64encode(data).decode(), mime_type=mime))
    return CallToolResult(content=content, structured_content=payload, is_error=error)


def _error(e: Exception, timer: Timer, **context) -> CallToolResult:
    return _result({**context, "error": f"{type(e).__name__}: {e}"}, timer, error=True)


def build_server(live: LiveState, history: History, settings: Settings) -> MCPServer:
    mcp = MCPServer(name="rcat-tracking", instructions=INSTRUCTIONS)

    async def db(fn, *args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def names() -> dict:
        try:
            return {str(k): v for k, v in (await db(history.cat_names)).items()}
        except Exception as e:  # names are a nicety; don't fail the call over them
            logger.warning(f"cat names unavailable: {e}")
            return {}

    def views_for(view_id, cat_id=None) -> list:
        if view_id is not None:
            return [view_id]
        return live.views_with_cat(cat_id) if cat_id is not None else live.view_ids()

    # ------------------------------------------------------------------ period bundle
    async def load_period(period_id: int, timer: Timer) -> dict:
        """Merge everything known about one period: memory first (sync), then DB."""
        now = datetime.datetime.now()
        view = live.find_period_view(period_id)
        mem_rows = live.committed_rows(view, period_id) if view is not None else []
        pending = live.pending_rows(view, period_id) if view is not None else []
        state = next((p for p in live.period_states(view, now) if p["period_id"] == period_id), None) if view is not None else None
        finalized = next((p for p in live.finalized(view) if to_int(p.get("period_id")) == period_id), None) if view is not None else None
        timer.mark("memory")
        db_period = await db(history.period, period_id)
        db_rows = await db(history.period_rows, period_id)
        timer.mark("db")

        merged = {}
        for rows, final in ((db_rows, True), (mem_rows, True), (pending, False)):
            for r in rows:
                if has_box(r):
                    merged[(r[0], to_int(r[7]))] = (r, final)
        rows = sorted(merged.values(), key=lambda rf: rf[0][0])
        if view is None and db_period is not None:
            view = db_period["view_id"]
        vinfo = await db(history.view, view) if view is not None else {"width": 640, "height": 480}

        if db_period is not None and db_period["status"] == "final":
            status, summary = "final", db_period
        elif state is not None:
            status, summary = state["status"], state
        elif finalized is not None:
            status, summary = "final", finalized
        elif rows:
            status, summary = "ended_pending", db_period
        else:
            raise LookupError(f"period {period_id} not found in memory or the database")
        return {
            "period_id": period_id,
            "view_id": to_int(view) if to_int(view) is not None else view,
            "status": status,
            "summary": summary,
            "rows": rows,
            "pending_rows": len(pending),
            "width": vinfo.get("width") or 640,
            "height": vinfo.get("height") or 480,
        }

    def frames_summary(b: dict) -> dict:
        rows = [r for r, _ in b["rows"]]
        real = [r for r in rows if not is_missing(r[5])]
        cat = (b["summary"] or {}).get("cat_id")
        labels = Counter(to_int(r[5]) for r in real)
        return {
            "frames": len(rows),
            "detected": len(real),
            "interpolated": len(rows) - len(real),
            "not_final": b["pending_rows"],
            "detector_cat_counts": {str(k): v for k, v in labels.items()},
            "detector_agrees_fraction": round(labels.get(cat, 0) / len(real), 3) if real and cat is not None else None,
        }

    def frame_out(row, final, b, role=None, source="period") -> dict:
        d = row_to_dict(row, final=final)
        box = box_of(row)
        d.update({"edge": motion.edge_of(box, b["width"], b["height"]), "crop_box": motion.pad_box(box, b["width"], b["height"]), "source": source})
        if role:
            d = {"role": role, **d}
        return d

    def motion_of(b, speed_threshold=3.0, min_moving_s=2.0, merge_gap_s=3.0, window_s=2.0, min_distance_px=20.0) -> dict:
        points = [(r[0], [float(v) for v in r[1:5]]) for r, _ in b["rows"]]
        return motion.compute_motion(points, speed_threshold, min_moving_s, merge_gap_s, window_s, min_distance_px)

    async def latest_period_id(cat_id: int, view_id=None) -> int | None:
        best = None  # (time, period_id) of the newest committed or pending row
        for v in views_for(view_id, cat_id):
            snap = live.tracker_cat(v, cat_id)
            if snap and snap["current_period_id"] is not None:
                return snap["current_period_id"]
            rows = live.committed_rows(v, cat_id=cat_id)
            if rows and (best is None or rows[-1][0] > best[0]):
                best = (rows[-1][0], to_int(rows[-1][8]))
        if best:
            return best[1]
        periods = await db(history.periods, view_id, cat_id, None, None, 1)
        if periods:
            return periods[0]["period_id"]
        last = await db(history.last_revised, cat_id, view_id)
        return to_int(last[0][8]) if last else None

    # ------------------------------------------------------------------ tools
    @mcp.tool(structured_output=False)
    async def list_views() -> CallToolResult:
        """List the camera views: view_id, name, frame size and the cats in each view
        (cat_id -> name). live_views are the views this server is receiving frames for now.
        Call this first to turn a cat's name into its cat_id."""
        timer = Timer()
        live_views = live.view_ids()
        try:
            views = await db(history.views)
            timer.mark("db")
            return _result({"views": views, "live_views": live_views}, timer)
        except Exception as e:
            return _error(e, timer, live_views=live_views)

    @mcp.tool(structured_output=False)
    async def get_live_status(view_id: int | None = None) -> CallToolResult:
        """What the camera sees right now. Per view: the last frame received (time,
        seconds_ago, image_name) and every detection on it, including ones below
        min_confidence that the tracker ignores. Per cat: name, on_screen,
        current_period_id (the ongoing visit, null if not on screen), last_period_id, and
        latest_detection (newest frame where the detector found it; final=false).
        view_id limits the reply to one view."""
        timer = Timer()
        try:
            now = datetime.datetime.now()
            out = []
            for v in views_for(view_id):
                tc = live.tracker(v)
                cats = [live.tracker_cat(v, c) for c in (tc.cat_ids if tc else [])]
                out.append({"view_id": v, "last_frame": live.frame_snapshot(v, now), "cats": cats})
            timer.mark("memory")
            cat_names = await names()
            for v in out:
                for c in v["cats"]:
                    c["name"] = cat_names.get(str(c["cat_id"]))
            return _result({"now": now, "views": out, "server_started_at": live.started_at}, timer)
        except Exception as e:
            logger.exception("get_live_status")
            return _error(e, timer, view_id=view_id)

    @mcp.tool(structured_output=False)
    async def get_last_seen(cat_id: int | None = None, view_id: int | None = None, lookback_days: float = 7) -> CallToolResult:
        """When and where each cat (or only cat_id) was last seen: time, seconds_ago, box,
        conf, cat_id (tracked) and detector_cat_id, period_id and image_name. source is
        live (last few minutes, may still change if final=false), saved (from the database;
        period_cat_id / period_final tell the visit's cat and whether it is final) or
        detection_only (the detector saw it but it never became a track). Searches back
        lookback_days; last_seen is null if the cat was not seen in that time."""
        timer = Timer()
        try:
            now = datetime.datetime.now()
            found, cats = {}, set()
            for v in views_for(view_id, cat_id):
                tc = live.tracker(v)
                for c in tc.cat_ids if tc else []:
                    if cat_id is not None and c != cat_id:
                        continue
                    cats.add(c)
                    snap = live.tracker_cat(v, c)
                    hit = None
                    if snap and snap["latest_detection"]:
                        hit = {**snap["latest_detection"], "source": "live", "period_id": snap["current_period_id"]}
                    else:
                        rows = [r for r in live.committed_rows(v, cat_id=c) if has_box(r)]
                        if rows:
                            hit = {**row_to_dict(rows[-1], final=True), "source": "live"}
                    if hit and (c not in found or hit["time"] > found[c]["time"]):
                        found[c] = {**hit, "view_id": v}
            timer.mark("memory")
            if cat_id is not None:
                cats.add(cat_id)
            elif not cats:
                for vi in await db(history.views):
                    if view_id is None or vi["view_id"] == view_id:
                        cats.update(vi["cat_ids"])
            for c in sorted(cats):
                if c in found:
                    continue
                since = now - datetime.timedelta(days=lookback_days)
                last = await db(history.last_revised, c, view_id, since)
                if last:
                    row, meta = last
                    found[c] = {**row_to_dict(row, final=True), "source": "saved", **meta}
                    continue
                raw = await db(history.last_raw, c, view_id, since)
                if raw:
                    found[c] = {**row_to_dict(raw, final=True), "source": "detection_only"}
            timer.mark("db")
            cat_names = await names()
            result = []
            for c in sorted(cats):
                hit = found.get(c)
                if hit:
                    hit["seconds_ago"] = seconds_between(datetime.datetime.fromisoformat(hit["time"]), now)
                result.append({"cat_id": c, "name": cat_names.get(str(c)), "last_seen": hit})
            return _result({"now": now, "lookback_days": lookback_days, "cats": result}, timer)
        except Exception as e:
            logger.exception("get_last_seen")
            return _error(e, timer, cat_id=cat_id, view_id=view_id)

    @mcp.tool(structured_output=False)
    async def get_current_periods(view_id: int | None = None, final_limit: int = 10) -> CallToolResult:
        """Visits happening now or that just ended, before they are saved. Use it for
        "what is on camera now / in the last few minutes"; use get_periods for anything
        older. not_final: visits with status ongoing or ended_pending, each with start, end,
        duration_s, the cat_id it would get if decided now (can still change),
        evidence_by_cat, frame count and first / last frame. recently_final: the newest
        visits (up to final_limit) that became final since this server started."""
        timer = Timer()
        try:
            now = datetime.datetime.now()
            out = [
                {"view_id": v, "not_final": live.period_states(v, now), "recently_final": live.finalized(v)[: max(0, final_limit)]}
                for v in views_for(view_id)
            ]
            timer.mark("memory")
            return _result({"now": now, "views": out}, timer)
        except Exception as e:
            logger.exception("get_current_periods")
            return _error(e, timer, view_id=view_id)

    @mcp.tool(structured_output=False)
    async def get_periods(
        cat_id: int | None = None,
        view_id: int | None = None,
        time_from: str | None = None,
        time_to: str | None = None,
        limit: int = 20,
        min_duration_s: float = 0,
    ) -> CallToolResult:
        """Search saved, final visits, newest first. Use it for history ("when was Mochi
        here yesterday"); visits still ongoing or just ended are not here yet, use
        get_current_periods for those. Filters: cat_id, view_id, time_from / time_to
        ('YYYY-MM-DD HH:MM[:SS]', any overlap counts), min_duration_s. Each visit: period_id,
        start, end, duration_s, cat_id and evidence_by_cat."""
        timer = Timer()
        try:
            periods = await db(history.periods, view_id, cat_id, parse_time(time_from), parse_time(time_to), min(int(limit), 500), min_duration_s)
            timer.mark("db")
            return _result({"count": len(periods), "periods": periods}, timer)
        except Exception as e:
            logger.exception("get_periods")
            return _error(e, timer, cat_id=cat_id, view_id=view_id)

    @mcp.tool(structured_output=False)
    async def get_period_detail(
        period_id: int,
        max_points: int = 60,
        speed_threshold: float = 3.0,
        min_moving_s: float = 2.0,
        merge_gap_s: float = 3.0,
        window_s: float = 2.0,
        min_distance_px: float = 20.0,
    ) -> CallToolResult:
        """Look inside one visit by period_id (from get_current_periods, get_periods or
        get_last_seen). Works for ongoing, ended_pending and final visits. Returns:
        - period: status, start, end, cat_id, evidence_by_cat.
        - frames: how many frames have a box, how many were detected vs interpolated, how
          many are not final, and how often the detector's label matched the visit's cat_id
          (a low detector_agrees_fraction means the cat_id is less certain).
        - first_frame, last_frame, best_frame (highest conf), each with image_name and
          crop_box to view it, and came_from / last_edge: the image border where the cat
          entered and where it was last.
        - path: up to max_points box centers over time.
        - motion: moving and resting segments. The cat counts as moving when its box center
          travels faster than speed_threshold px/s averaged over window_s; pauses up to
          merge_gap_s are joined; moves shorter than min_moving_s or covering less than
          min_distance_px are ignored."""
        timer = Timer()
        try:
            b = await load_period(period_id, timer)
            rows = b["rows"]
            real = [(r, f) for r, f in rows if not is_missing(r[5])]
            mo = motion_of(b, speed_threshold, min_moving_s, merge_gap_s, window_s, min_distance_px)
            timer.mark("analyze")
            best = max(real, key=lambda rf: float(rf[0][6])) if real else None
            first = real[0] if real else (rows[0] if rows else None)
            last = rows[-1] if rows else None
            payload = {
                "period_id": period_id,
                "view_id": b["view_id"],
                "status": b["status"],
                "period": b["summary"],
                "frames": frames_summary(b),
                "first_frame": frame_out(*first, b, "first") if first else None,
                "last_frame": frame_out(*last, b, "last") if last else None,
                "best_frame": frame_out(*best, b, "best_conf") if best else None,
                "came_from": motion.edge_of(box_of(first[0]), b["width"], b["height"]) if first else None,
                "last_edge": motion.edge_of(box_of(last[0]), b["width"], b["height"]) if last else None,
                "path": motion.downsample([(r[0], box_of(r)) for r, _ in rows], max(2, int(max_points))),
                "motion": {**mo, "segments_total": len(mo["segments"]), "segments": mo["segments"][-MAX_SEGMENTS:]},
            }
            return _result(payload, timer)
        except Exception as e:
            logger.exception("get_period_detail")
            return _error(e, timer, period_id=period_id)

    @mcp.tool(structured_output=False)
    async def get_arrival_path(
        period_id: int | None = None,
        cat_id: int | None = None,
        view_id: int | None = None,
        n_frames: int = 5,
        pre_roll_s: float = 10,
        settle_s: float = 10,
        include_images: bool = False,
        max_side: int = 384,
    ) -> CallToolResult:
        """Key frames showing where a cat came from and how it got to where it is, so you
        can check the track by eye. Give period_id, or cat_id to use its newest visit.
        key_frames are in time order, each with a role:
        - empty_before: the last frame with no cat, to show the empty scene.
        - entry: the earliest box, also searching detections up to pre_roll_s before the
          visit started.
        - first_tracked: the first frame of the visit itself.
        - path: frames spread evenly by distance travelled (n_frames in total).
        - settled: where the cat first rested for settle_s or more; latest if it never did.
        Each has image_name, box, crop_box, conf, cat_id, detector_cat_id and edge. Fetch
        them with images__get_image in one turn and check the same cat moves along the path.
        include_images=true attaches the frames with the box drawn (max_side px), if this
        server can read the image folder."""
        timer = Timer()
        try:
            n_frames = max(2, min(int(n_frames), 10))
            if period_id is None:
                if cat_id is None:
                    raise ValueError("give period_id or cat_id")
                period_id = await latest_period_id(cat_id, view_id)
                if period_id is None:
                    raise LookupError(f"no period found for cat {cat_id}")
            b = await load_period(period_id, timer)
            real = [(r, f) for r, f in b["rows"] if not is_missing(r[5])]
            if not real:
                raise LookupError(f"period {period_id} has no real detections")
            t_first = real[0][0][0]

            # Raw YOLO boxes just before the period that chain back from its first box.
            raw = await db(history.raw_detections, t_first - datetime.timedelta(seconds=pre_roll_s), t_first - datetime.timedelta(microseconds=1), None, b["view_id"], 500)
            chain, head = [], box_of(real[0][0])
            for d in reversed(raw):
                if motion.center_distance(d["box"], head) <= RELINK_PX:
                    chain.append(d)
                    head = d["box"]
            chain.reverse()
            entry_time = parse_time(chain[0]["time"]) if chain else t_first
            empty = await db(history.last_empty_frame, b["view_id"], entry_time, max(60.0, pre_roll_s * 6))
            timer.mark("pre_roll")

            mo = motion_of(b)
            settle = next((s for s in mo["segments"] if s["state"] == "resting" and s["duration_s"] >= settle_s), None)
            settle_time = datetime.datetime.fromisoformat(settle["start"]) if settle else None

            # Candidate path: raw pre-roll chain, then the period's real detections up to settling.
            path = [([parse_time(d["time"]), *d["box"], d["cat_id"], d["conf"], d["cat_id"], None], True, "detection_only") for d in chain]
            path += [(r, f, "period") for r, f in real if settle_time is None or r[0] <= settle_time]
            picks = set(motion.picks_by_distance([(p[0][0], box_of(p[0])) for p in path], n_frames))
            keep = {0, len(path) - 1}
            if chain and len(chain) < len(path):
                keep.add(len(chain))  # first committed frame of the period
            picks = sorted(picks | keep)
            while len(picks) > n_frames + 1 and any(k not in keep for k in picks):
                middle = [k for k in picks if k not in keep]
                picks.remove(middle[len(middle) // 2])
            frames = []
            if empty is not None:
                frames.append({"role": "empty_before", "time": empty, "image_name": image_name(empty), "box": None})
            for k in picks:
                r, f, source = path[k]
                if k == 0:
                    role = "entry"
                elif source == "period" and (k == len(chain)):
                    role = "first_tracked"
                elif k == len(path) - 1:
                    role = "settled" if settle else "latest"
                else:
                    role = "path"
                frames.append(frame_out(r, f, b, role, source))
            timer.mark("pick")

            images = []
            if include_images:
                if not settings.image_root:
                    raise RuntimeError("include_images needs TRACKING_IMAGE_ROOT on the tracking server; use images__get_image instead")
                images = await db(_render_frames, settings.image_root, frames, max_side)
                timer.mark("images")
            payload = {
                "period_id": period_id,
                "view_id": b["view_id"],
                "status": b["status"],
                "cat_id": (b["summary"] or {}).get("cat_id"),
                "came_from": frames[1 if empty is not None else 0].get("edge"),
                "frames_before_visit": len(chain),
                "settled_at": settle["start"] if settle else None,
                "key_frames": frames,
                "path": motion.downsample([(p[0][0], box_of(p[0])) for p in path], 40),
                "images_attached": [fr["image_name"] for fr in frames if fr.get("image_attached")],
                "validate_hint": "Fetch each key_frames image_name with images__get_image (max_side 384, or box=crop_box "
                "to check identity) and confirm the same cat appears in each frame and moves along the path.",
            }
            return _result(payload, timer, images=images)
        except Exception as e:
            logger.exception("get_arrival_path")
            return _error(e, timer, period_id=period_id, cat_id=cat_id)

    @mcp.tool(structured_output=False)
    async def get_last_movement(
        cat_id: int,
        view_id: int | None = None,
        speed_threshold: float = 3.0,
        min_moving_s: float = 2.0,
        merge_gap_s: float = 3.0,
        window_s: float = 2.0,
        min_distance_px: float = 20.0,
        lookback_hours: float = 24,
        max_periods: int = 20,
    ) -> CallToolResult:
        """When cat_id last moved and how long it has stayed still since. Checks the cat's
        visits newest first (up to max_periods, within lookback_hours) and returns
        last_seen, last_movement (start, end, duration, distance, and image names of its
        first and last frame) and resting_since_movement (from the end of that move to the
        last seen frame). The cat counts as moving when its box center travels faster than
        speed_threshold px/s averaged over window_s (this hides box jitter); pauses up to
        merge_gap_s are joined; moves shorter than min_moving_s or ending less than
        min_distance_px from where they started are ignored."""
        timer = Timer()
        try:
            now = datetime.datetime.now()
            candidates = []  # (last time, period_id), newest first
            for v in views_for(view_id, cat_id):
                snap = live.tracker_cat(v, cat_id)
                if snap and snap["current_period_id"] is not None:
                    candidates.append((now, snap["current_period_id"]))
                last_by_period = {}
                for r in live.committed_rows(v, cat_id=cat_id):
                    last_by_period[to_int(r[8])] = r[0]
                candidates += [(t, p) for p, t in last_by_period.items()]
            timer.mark("memory")
            for p in await db(history.periods, view_id, cat_id, now - datetime.timedelta(hours=lookback_hours), None, int(max_periods)):
                candidates.append((datetime.datetime.fromisoformat(p["end"]), p["period_id"]))
            timer.mark("db_periods")
            order, seen = [], set()
            for _, p in sorted(candidates, key=lambda c: c[0], reverse=True):
                if p not in seen:
                    seen.add(p)
                    order.append(p)
            if not order:
                raise LookupError(f"no periods for cat {cat_id} in the last {lookback_hours} h")

            last_seen, movement, searched = None, None, []
            for p in order[: int(max_periods)]:
                b = await load_period(p, timer)
                if not b["rows"]:
                    continue
                if last_seen is None:
                    last_seen = {**frame_out(*b["rows"][-1], b, "last_seen"), "period_id": p, "period_status": b["status"]}
                mo = motion_of(b, speed_threshold, min_moving_s, merge_gap_s, window_s, min_distance_px)
                moving = [s for s in mo["segments"] if s["state"] == "moving"]
                searched.append({"period_id": p, "status": b["status"], "moving_segments": len(moving), "moving_s": mo["totals"]["moving_s"]})
                if moving:
                    movement = {**moving[-1], "period_id": p, "currently_moving": mo["segments"][-1] is moving[-1] and b["status"] == "open"}
                    break
            timer.mark("analyze")
            rest = None
            if movement and last_seen:
                end = datetime.datetime.fromisoformat(movement["end"])
                seen_t = datetime.datetime.fromisoformat(last_seen["time"])
                rest = {
                    "from": movement["end"],
                    "to": last_seen["time"],
                    "duration_s": seconds_between(end, seen_t),
                    "same_period": movement["period_id"] == last_seen["period_id"],
                    "periods_since": len(searched) - 1,
                    "box": last_seen["box"],
                    "image_name": last_seen["image_name"],
                }
            payload = {
                "cat_id": cat_id,
                "now": now,
                "last_seen": last_seen,
                "seconds_since_seen": seconds_between(datetime.datetime.fromisoformat(last_seen["time"]), now) if last_seen else None,
                "last_movement": movement,
                "resting_since_movement": rest,
                "params": {"speed_threshold_px_s": speed_threshold, "min_moving_s": min_moving_s, "merge_gap_s": merge_gap_s, "window_s": window_s, "min_distance_px": min_distance_px},
                "searched_periods": searched,
                "note": None if movement else f"no movement found in the {len(searched)} newest periods",
            }
            return _result(payload, timer)
        except Exception as e:
            logger.exception("get_last_movement")
            return _error(e, timer, cat_id=cat_id)

    @mcp.tool(structured_output=False)
    async def get_raw_detections(
        time_from: str,
        time_to: str,
        cat_id: int | None = None,
        view_id: int | None = None,
        limit: int = 500,
    ) -> CallToolResult:
        """Detector output before tracking (conf above 0.5) between time_from and time_to
        ('YYYY-MM-DD HH:MM[:SS]'), oldest first: time, cat_id, box, conf. Use it to compare
        what the detector saw on each frame with the tracked result."""
        timer = Timer()
        try:
            rows = await db(history.raw_detections, parse_time(time_from), parse_time(time_to), cat_id, view_id, min(int(limit), 5000))
            timer.mark("db")
            return _result({"count": len(rows), "detections": rows}, timer)
        except Exception as e:
            logger.exception("get_raw_detections")
            return _error(e, timer, time_from=time_from, time_to=time_to)

    return mcp


def _render_frames(image_root: str, frames: list[dict], max_side: int) -> list[tuple[bytes, str]]:
    """Key frames from the image SSD (layout <root>/YYYY-MM-DD/HH/MM/<name>), box drawn, downscaled."""
    from PIL import Image, ImageDraw

    out = []
    for fr in frames:
        t = datetime.datetime.strptime(fr["image_name"], "%Y-%m-%d-%H-%M-%S-%f.jpg")
        path = os.path.join(image_root, t.strftime("%Y-%m-%d/%H/%M"), fr["image_name"])
        if not os.path.exists(path):
            fr["image_attached"] = False
            continue
        img = Image.open(path).convert("RGB")
        if fr.get("box"):
            ImageDraw.Draw(img).rectangle(fr["box"], outline=(255, 0, 0), width=3)
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=80)
        out.append((buf.getvalue(), "image/jpeg"))
        fr["image_attached"] = True
    return out


class BearerAuth:
    """ASGI middleware: reject HTTP requests without the right bearer token."""

    def __init__(self, app, token: str):
        self.app = app
        self.expected = f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            auth = dict(scope.get("headers") or []).get(b"authorization", b"")
            if not hmac.compare_digest(auth, self.expected):
                await send({"type": "http.response.start", "status": 401, "headers": [(b"content-type", b"text/plain")]})
                await send({"type": "http.response.body", "body": b"unauthorized"})
                return
        await self.app(scope, receive, send)


def build_http_app(live: LiveState, history: History, settings: Settings):
    # json_response: plain JSON replies, not SSE (same as the image server)
    app = build_server(live, history, settings).streamable_http_app(host=settings.host, json_response=True)
    return BearerAuth(app, settings.auth_token) if settings.auth_token else app


def start_in_loop(live: LiveState, history: History, settings: Settings | None = None) -> asyncio.Task | None:
    """Serve the MCP app on the running event loop (call from inside detection_ingest_server's main())."""
    settings = settings or Settings.from_env()
    if not settings.should_start:
        logger.info("Tracking MCP server disabled (TRACKING_MCP_ENABLED is false, or auto without MCP_AUTH_TOKEN)")
        return None
    import uvicorn

    # ws="none": MCP is plain HTTP, and uvicorn's websocket support needs a newer `websockets`
    # than detection_ingest_server.py runs with (its "auto" import fails before the port is bound)
    config = uvicorn.Config(build_http_app(live, history, settings), host=settings.host, port=settings.port, ws="none", log_level="warning")
    server = uvicorn.Server(config)
    url = f"http://{settings.host}:{settings.port}/mcp"

    # Nothing awaits this task, so log its fate here; never let it take down the websocket server
    async def serve():
        try:
            await server.serve()
        except asyncio.CancelledError:
            raise
        except BaseException:  # uvicorn calls sys.exit(1) on bind failure
            logger.exception(f"Tracking MCP server on {url} crashed")
            return
        if not server.started:
            logger.error(f"Tracking MCP server exited without binding {url} (see uvicorn errors above)")

    async def report_started():
        for _ in range(30):
            if server.started:
                logger.info(f"Tracking MCP server listening on {url} (auth={'on' if settings.auth_token else 'OFF'})")
                return
            await asyncio.sleep(1)
        logger.error(f"Tracking MCP server still not listening on {url} after 30s")

    if not settings.auth_token:
        logger.warning("MCP_AUTH_TOKEN is not set: the tracking MCP server accepts requests without auth")
    logger.info(f"Starting tracking MCP server on {url}")
    task = asyncio.create_task(serve())
    asyncio.create_task(report_started())
    return task
