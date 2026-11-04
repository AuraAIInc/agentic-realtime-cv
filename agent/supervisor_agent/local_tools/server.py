"""The `local__*` tools: an in-process tool server registered on the MCPHub next to the
camera / images / tracking servers. Bookkeeping and math run here in code so the model
never spends tokens on them, and every image-returning tool keeps images small.
"""

from __future__ import annotations

import base64
import bisect
import datetime
import json
import logging
import os
import re
from pathlib import Path

from mcp.types import CallToolResult, ImageContent, TextContent

from supervisor_agent.local_tools import imaging, trajectory
from supervisor_agent.local_tools.store import Store

logger = logging.getLogger(__name__)

BOX = {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4, "description": "[x1, y1, x2, y2] in the view's pixels"}

#: Camera frame stems: 5502_2026-10-28-16-33-00-304643.jpg (same shape as replay_labels.parse_frame_time).
_STEM_RE = re.compile(r"^\d+_(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{1,6})$")

#: Nearest-frame tolerance for get_frame: beyond this the pixels may show another visit.
FRAME_MATCH_TOLERANCE_S = 2.0

TOOLS: list[dict] = [
    {
        "name": "get_cat_profiles",
        "description": "Written profiles of the household cats: name, tracker ids per view, summary, low-res and camera cues, "
        "how to tell them apart. detail='full' adds the long description and high-res cues (eyes, nose). "
        "cat_ids may be profile ids or tracker ids (e.g. 25/26 on view 3). With view_id, cat_id is that "
        "view's tracker id (answer with it); without view_id it is the profile id.",
        "input_schema": {"type": "object", "properties": {
            "cat_ids": {"type": "array", "items": {"type": "integer"}},
            "view_id": {"type": "integer"},
            "detail": {"type": "string", "enum": ["low_res", "full"], "default": "low_res"}}},
    },
    {
        "name": "find_references",
        "description": "Rank this cat's reference images WITHOUT returning any image: owner photos (source=profile) and "
        "confirmed camera crops (source=scene). Scored by same view, nearby box position and size, same pose and lighting, "
        "and photo quality. Fetch only the top one or two with get_reference_image.",
        "input_schema": {"type": "object", "required": ["cat_id"], "properties": {
            "cat_id": {"type": "integer"}, "view_id": {"type": "integer"}, "box": BOX,
            "pose": {"type": "string", "description": "sitting, lying, walking, crouching, sleeping, standing, playing"},
            "lighting": {"type": "string", "enum": ["day", "low_light", "night_ir"]},
            "source": {"type": "string", "enum": ["profile", "scene"]},
            "k": {"type": "integer", "default": 2}}},
    },
    {
        "name": "get_reference_image",
        "description": "Return one reference image (already cropped to the cat) by the `file` find_references gave. Counts against the image budget.",
        "input_schema": {"type": "object", "required": ["file"], "properties": {
            "file": {"type": "string"}, "max_side": {"type": "integer", "default": 384}}},
        "returns_image": True,
    },
    {
        "name": "get_frame",
        "description": "Return a camera frame from LOCAL disk by timestamp — the same pixels the images server would "
        "serve. Use this when images__get_image errors or the images server is down. Matches the exact frame when the "
        "timestamp hits one, else the nearest frame within 2 s (the actual frame time and delta are reported — judge "
        "from the returned pixels, noting any delta). Optional box crops around the cat. Counts against the image budget.",
        "input_schema": {"type": "object", "required": ["timestamp"], "properties": {
            "timestamp": {"type": "string", "description": "ISO-8601 (2026-10-28T16:34:42.769881) or camera (2026-10-28-16-34-42-769881) frame time"},
            "box": BOX, "pad": {"type": "number", "default": 0.3}, "max_side": {"type": "integer", "default": 768}}},
        "returns_image": True,
    },
    {
        "name": "get_track_box",
        "description": "Exact TRACKED box of one visit (period) at a timestamp, from this run's revised track file — "
        "for bisection times with no listed key frame. Fetch this first, then crop that moment with get_frame(box=...) "
        "using the returned box. Never invent boxes. No image; text only.",
        "input_schema": {"type": "object", "required": ["period_id", "timestamp"], "properties": {
            "period_id": {"type": "integer"},
            "timestamp": {"type": "string", "description": "ISO-8601 or camera frame time"},
            "tolerance_s": {"type": "number", "default": 2.0}}},
    },
    {
        "name": "make_contact_sheet",
        "description": "Build ONE numbered grid image from several crops, so several frames cost one image call. Each item is "
        "a stored frame {image_name, box} (cropped around the box with padding) or a reference {reference_file}. "
        "Tiles are numbered 1..n in the order given.",
        "input_schema": {"type": "object", "required": ["items"], "properties": {
            "items": {"type": "array", "maxItems": 9, "items": {"type": "object", "properties": {
                "image_name": {"type": "string"}, "box": BOX, "reference_file": {"type": "string"}, "caption": {"type": "string"}}}},
            "tile": {"type": "integer", "default": 256}, "pad": {"type": "number", "default": 0.3}}},
        "returns_image": True,
    },
    {
        "name": "analyze_trajectory",
        "description": "Physics check of one visit with no images: rebuilds the visit's own boxes from raw detections and flags "
        "teleports, mid-frame entry/exit, long gaps, size jumps, detector identity flips, another cat close by, and a box "
        "that never moves. Also returns numbered keyframes that test each doubt.",
        "input_schema": {"type": "object", "required": ["period_id"], "properties": {
            "period_id": {"type": "integer"}, "pre_roll_s": {"type": "number", "default": 5}}},
    },
    {
        "name": "get_label_set",
        "description": "The label set: activity labels (pick exactly one) and context labels (any number), with descriptions, "
        "what evidence each needs, and code rules. Includes the version number.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "add_label",
        "description": "Add a label to the label set. With confirm=false (default) it only validates and returns the draft, "
        "problems and similar existing labels: show that to the user. Call again with confirm=true only after the user says yes.",
        "input_schema": {"type": "object", "required": ["name", "kind", "description"], "properties": {
            "name": {"type": "string", "description": "lower_snake_case, e.g. in_tent"},
            "kind": {"type": "string", "enum": ["activity", "context"]},
            "description": {"type": "string"},
            "rule": {"type": "object", "description": "context labels only, e.g. {zone: sofa, min_overlap: 0.5}"},
            "needs": {"type": "array", "items": {}, "description": "e.g. [image], [image, motion], [{zone: tent}]"},
            "confirm": {"type": "boolean", "default": False}}},
    },
    {
        "name": "get_zones",
        "description": "Named zones of a view (places, occluders, entries) with boxes or polygons, and the view size.",
        "input_schema": {"type": "object", "required": ["view_id"], "properties": {"view_id": {"type": "integer"}}},
    },
    {
        "name": "zones_for_box",
        "description": "Which zones a box overlaps, and by what fraction of the box's area. No image.",
        "input_schema": {"type": "object", "required": ["view_id", "box"], "properties": {"view_id": {"type": "integer"}, "box": BOX}},
    },
    {
        "name": "save_zone",
        "description": "Define or replace a named zone on a view. confirm=false returns the draft for the user to check; "
        "confirm=true writes it (only after the user agrees).",
        "input_schema": {"type": "object", "required": ["view_id", "name", "kind"], "properties": {
            "view_id": {"type": "integer"}, "name": {"type": "string"},
            "kind": {"type": "string", "enum": ["place", "occluder", "entry"]},
            "box": BOX, "polygon": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "confirm": {"type": "boolean", "default": False}}},
    },
]
IMAGE_TOOLS = {t["name"] for t in TOOLS if t.get("returns_image")}


def text_of(result) -> dict:
    """First JSON text block of an MCP CallToolResult."""
    for block in getattr(result, "content", []) or []:
        if getattr(block, "type", None) == "text":
            try:
                return json.loads(block.text)
            except (TypeError, ValueError):
                return {"text": block.text}
    return {}


def image_of(result) -> bytes | None:
    for block in getattr(result, "content", []) or []:
        if getattr(block, "type", None) == "image":
            return base64.b64decode(block.data)
    return None


class LocalTools:
    server_name = "local"

    def __init__(self, store: Store | None = None, hub=None, frame_roots=None, track_run_dir=None):
        self.store = store or Store()
        self.hub = hub  # set by MCPHub.add_local; used to read tracking data and stored frames
        self._sizes: dict = {}
        self._frame_roots = [Path(p) for p in frame_roots] if frame_roots else None
        self._frame_index: dict | None = None  # datetime -> Path, built lazily
        self._track_run_dir = Path(track_run_dir) if track_run_dir else None
        self._track_index: dict | None = None  # (period_id, datetime) -> (box, real)
        self._track_mtime: float | None = None

    def specs(self) -> list[dict]:
        return TOOLS

    # ------------------------------------------------------------------ plumbing
    async def call(self, name: str, args: dict) -> CallToolResult:
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            return _json({"error": f"unknown local tool {name}"}, error=True)
        try:
            out = await fn(**args)
        except Exception as e:  # report to the model, don't crash the run
            logger.exception(f"local__{name}")
            return _json({"error": f"{type(e).__name__}: {e}"}, error=True)
        return out if isinstance(out, CallToolResult) else _json(out)

    async def remote(self, qualified: str, args: dict) -> dict:
        if self.hub is None or qualified not in self.hub.tools:
            raise RuntimeError(f"{qualified} is not available (server down?)")
        outcome = await self.hub.call(qualified, args)
        if outcome.error:
            raise RuntimeError(f"{qualified}: {outcome.error}")
        data = text_of(outcome.result)
        if getattr(outcome.result, "is_error", False) or (isinstance(data, dict) and data.get("error")):
            raise RuntimeError(f"{qualified}: {data.get('error') if isinstance(data, dict) else data}")
        return data

    async def frame_crop(self, image_name: str, box, pad: float = 0.3, max_side: int = 384) -> bytes:
        """A stored frame cropped around box (padding clamped to the frame). The frame is
        fetched whole (camera frames are ~100 KB on the LAN) and cropped here, so padding
        never asks the image server for pixels outside the frame. Only the crop reaches the model."""
        outcome = await self.hub.call("images__get_image", {"image_name": image_name})
        if outcome.error:
            raise RuntimeError(outcome.error)
        data = image_of(outcome.result)
        if data is None:
            raise FileNotFoundError(f"{image_name} not on the image server")
        im = imaging.open_bytes(data)
        out = imaging.crop(im, box, pad, max_side) if box else im
        if not box and max_side:
            out.thumbnail((max_side, max_side))
        return imaging.to_jpeg(out)

    def _resolve_frame_roots(self) -> list[Path]:
        """Images dirs get_frame searches: explicit roots, LOCAL_FRAME_DIRS, else retrieved_images."""
        if self._frame_roots is not None:
            return [p for p in self._frame_roots if p.is_dir()]
        env = os.getenv("LOCAL_FRAME_DIRS")
        if env:
            return [Path(p) for p in env.split(os.pathsep) if p and Path(p).is_dir()]
        retrieved = Path(__file__).resolve().parents[3] / "image_server" / "retrieved_images"
        if not retrieved.is_dir():
            return []
        return sorted(p for p in retrieved.glob("*/images") if p.is_dir())

    def _frame_index_build(self) -> dict:
        index: dict = {}
        for root in self._resolve_frame_roots():
            try:
                names = sorted(os.listdir(root))
            except OSError:
                continue
            for name in names:
                if not name.lower().endswith((".jpg", ".jpeg")):
                    continue
                m = _STEM_RE.match(os.path.splitext(name)[0])
                if not m:
                    continue
                try:
                    when = datetime.datetime(*[int(v) for v in m.groups()[:6]], int(m.group(7).ljust(6, "0")[:6]))
                except ValueError:
                    continue
                index.setdefault(when, root / name)
        return index

    def find_frame(self, when: datetime.datetime) -> tuple | None:
        """(path, hit_time, delta_s) for the exact or nearest-in-tolerance frame, else None."""
        if self._frame_index is None:
            self._frame_index = self._frame_index_build()
        if when in self._frame_index:
            return self._frame_index[when], when, 0.0
        keys = sorted(self._frame_index)
        if not keys:
            return None
        i = bisect.bisect_left(keys, when)
        cands = ([keys[i - 1]] if i > 0 else []) + ([keys[i]] if i < len(keys) else [])
        hit = min(cands, key=lambda k: abs((k - when).total_seconds()))
        delta = abs((hit - when).total_seconds())
        if delta > FRAME_MATCH_TOLERANCE_S:
            return None
        return self._frame_index[hit], hit, delta

    async def view_size(self, view_id) -> tuple[int, int]:
        if view_id is None:
            return (640, 480)
        size = self.store.view_size(view_id) or self._sizes.get(view_id)
        if size:
            return size
        try:
            views = (await self.remote("tracking__list_views", {})).get("views", [])
            for v in views if isinstance(views, list) else views.values():
                vid = v.get("view_id", v.get("id"))
                if vid is not None and int(vid) == int(view_id) and v.get("width"):
                    self._sizes[view_id] = (int(v["width"]), int(v["height"]))
                    return self._sizes[view_id]
        except Exception as e:
            logger.warning(f"view size for {view_id}: {e}")
        return (640, 480)

    # ------------------------------------------------------------------ tools
    async def t_get_cat_profiles(self, cat_ids=None, view_id=None, detail="low_res"):
        return {"cats": self.store.profiles(cat_ids, view_id, detail)}

    async def t_find_references(self, cat_id, view_id=None, box=None, pose=None, lighting=None, source=None, k=None):
        size = await self.view_size(view_id) if (view_id is not None and box) else None
        refs = self.store.find_references(cat_id, view_id, box, size, pose, lighting, source, k)
        return {"references": refs, "hint": "fetch the top one or two with local__get_reference_image"}

    async def t_get_reference_image(self, file, max_side=384):
        p = self.store.reference_path(file)
        im = imaging.Image.open(p).convert("RGB")
        im.thumbnail((max_side, max_side))
        return _json({"file": file, "size": list(im.size)}, image=imaging.to_jpeg(im))

    def _track_file(self) -> Path | None:
        run_dir = self._track_run_dir or (Path(os.getenv("TRACK_RUN_DIR")) if os.getenv("TRACK_RUN_DIR") else None)
        if run_dir is None:
            return None
        candidate = run_dir / "revised.jsonl"
        return candidate if candidate.is_file() else None

    def _track_index_build(self) -> dict:
        index: dict = {}
        path = self._track_file()
        if path is None:
            return index
        try:
            self._track_mtime = path.stat().st_mtime
        except OSError:
            return index
        try:
            lines = path.read_text().splitlines()
        except OSError:
            return index
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("deleted"):
                continue
            try:
                pid = int(row.get("period_id"))
                when = datetime.datetime.fromisoformat(row.get("create_time"))
            except (TypeError, ValueError):
                continue
            box = [row.get(k) for k in ("box_x1", "box_y1", "box_x2", "box_y2")]
            if not all(isinstance(v, (int, float)) for v in box):
                continue
            index[(pid, when)] = ([float(v) for v in box], row.get("orig_cat_id") is not None)
        return index

    def find_track_box(self, period_id: int, when: datetime.datetime, tolerance_s: float = 2.0):
        """(box, hit_time, delta_s, real) for a visit at a time, else None."""
        path = self._track_file()
        if path is None:
            return None
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        if self._track_index is None or mtime != self._track_mtime:
            self._track_index = self._track_index_build()
        key = (int(period_id), when)
        if key in self._track_index:
            box, real = self._track_index[key]
            return box, when, 0.0, real
        times = sorted(t for p, t in self._track_index if p == int(period_id))
        if not times:
            return None
        i = bisect.bisect_left(times, when)
        cands = ([times[i - 1]] if i > 0 else []) + ([times[i]] if i < len(times) else [])
        hit = min(cands, key=lambda t: abs((t - when).total_seconds()))
        delta = abs((hit - when).total_seconds())
        if delta > tolerance_s:
            return None
        box, real = self._track_index[(int(period_id), hit)]
        return box, hit, delta, real

    def track_span(self, period_id: int):
        """(first, last) datetimes of a visit's tracked rows, or None."""
        if self._track_file() is None:
            return None
        if self._track_index is None:
            self._track_index = self._track_index_build()
        times = sorted(t for p, t in self._track_index if p == int(period_id))
        return (times[0], times[-1]) if times else None

    @staticmethod
    def _parse_frame_time(value):
        try:
            return datetime.datetime.fromisoformat(value)
        except (TypeError, ValueError):
            pass
        try:
            return datetime.datetime.strptime(value, "%Y-%m-%d-%H-%M-%S-%f")
        except (TypeError, ValueError):
            return None

    async def t_get_track_box(self, period_id, timestamp, tolerance_s=2.0):
        when = self._parse_frame_time(timestamp)
        if when is None:
            return {"error": f"bad timestamp {timestamp!r}: want ISO-8601 or YYYY-MM-DD-HH-MM-SS-ffffff"}
        if self._track_file() is None:
            return {"error": "no revised track file (TRACK_RUN_DIR not set or no revised.jsonl yet)"}
        try:
            pid = int(period_id)
        except (TypeError, ValueError):
            return {"error": f"bad period_id {period_id!r}: want an integer visit id"}
        found = self.find_track_box(pid, when, float(tolerance_s or 0))
        if found is None:
            span = self.track_span(pid)
            hint = f" (visit {pid} spans {span[0].isoformat()}–{span[1].isoformat()})" if span else f" (visit {pid} has no tracked rows)"
            return {"error": f"no tracked box for visit {pid} within {tolerance_s:g} s of {when.isoformat()}{hint}"}
        box, hit, delta, real = found
        return {"period_id": pid, "frame_time": hit.isoformat(), "delta_s": round(delta, 3),
                "box": [round(v, 1) for v in box], "real": real}

    async def t_get_frame(self, timestamp, box=None, pad=0.3, max_side=768):
        when = self._parse_frame_time(timestamp)
        if when is None:
            return {"error": f"bad timestamp {timestamp!r}: want ISO-8601 or YYYY-MM-DD-HH-MM-SS-ffffff"}
        found = self.find_frame(when)
        if found is None:
            roots = [str(p) for p in self._resolve_frame_roots()]
            return {"error": f"no local frame within {FRAME_MATCH_TOLERANCE_S:g} s of {when.isoformat()} (roots: {roots or 'none'})"}
        path, hit, delta = found
        im = imaging.Image.open(path).convert("RGB")
        out = imaging.crop(im, box, pad, max_side) if box else im
        if not box and max_side:
            out.thumbnail((max_side, max_side))
        return _json(
            {"frame": path.stem, "frame_time": hit.isoformat(), "delta_s": round(delta, 3), "size": list(out.size)},
            image=imaging.to_jpeg(out),
        )

    async def t_make_contact_sheet(self, items, tile=256, pad=0.3):
        tiles, used = [], []
        for i, it in enumerate(items[:9], 1):
            caption = it.get("caption") or ""
            if it.get("reference_file"):
                im = imaging.Image.open(self.store.reference_path(it["reference_file"])).convert("RGB")
                caption = caption or "reference"
            elif it.get("image_name"):
                im = imaging.open_bytes(await self.frame_crop(it["image_name"], it.get("box"), pad, max_side=tile * 2))
                caption = caption or it["image_name"][11:19]
            else:
                continue
            tiles.append((im, caption))
            used.append({"tile": len(tiles), **{k: v for k, v in it.items() if k != "caption"}, "caption": caption})
        if not tiles:
            return {"error": "no usable items"}
        sheet = imaging.contact_sheet(tiles, tile)
        return _json({"tiles": used, "size": list(sheet.size)}, image=imaging.to_jpeg(sheet, 82))

    async def t_analyze_trajectory(self, period_id, pre_roll_s=5):
        cfg = self.store.agent_config().get("trajectory", {})
        detail = await self.remote("tracking__get_period_detail", {"period_id": int(period_id), "max_points": 240})
        period = detail.get("period") or {}
        start = trajectory.parse_t(period.get("start") or detail["first_frame"]["time"])
        end = trajectory.parse_t(period.get("end") or detail["last_frame"]["time"])
        fmt = "%Y-%m-%d %H:%M:%S.%f"
        raw = await self.remote("tracking__get_raw_detections", {
            "time_from": (start - datetime.timedelta(seconds=pre_roll_s)).strftime(fmt),
            "time_to": (end + datetime.timedelta(seconds=1)).strftime(fmt),
            "view_id": detail.get("view_id"), "limit": 5000})
        size = await self.view_size(detail.get("view_id"))
        return trajectory.analyze(detail, raw.get("detections", []), size, self.store.view_zones(detail.get("view_id")), cfg)

    async def t_get_label_set(self):
        return self.store.labels()

    async def t_add_label(self, name, kind, description, rule=None, needs=None, confirm=False):
        return self.store.add_label(name, kind, description, rule, needs, confirm)

    async def t_get_zones(self, view_id):
        return {"view_id": view_id, "size": list(await self.view_size(view_id)), "zones": self.store.view_zones(view_id)}

    async def t_zones_for_box(self, view_id, box):
        hits = []
        for z in self.store.view_zones(view_id):
            f = trajectory.box_zone_overlap(box, z)
            if f > 0:
                hits.append({"name": z["name"], "kind": z.get("kind"), "overlap": f})
        return {"view_id": view_id, "box": box, "zones": sorted(hits, key=lambda x: -x["overlap"])}

    async def t_save_zone(self, view_id, name, kind, box=None, polygon=None, confirm=False):
        return self.store.save_zone(view_id, name, kind, box, polygon, confirm)


def _json(payload: dict, image: bytes | None = None, error: bool = False) -> CallToolResult:
    content = [TextContent(type="text", text=json.dumps(payload, default=str))]
    if image:
        content.append(ImageContent(type="image", data=base64.b64encode(image).decode(), mime_type="image/jpeg"))
    return CallToolResult(content=content, is_error=error)
