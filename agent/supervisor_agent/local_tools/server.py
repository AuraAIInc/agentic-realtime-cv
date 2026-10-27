"""The `local__*` tools: an in-process tool server registered on the MCPHub next to the
camera / images / tracking servers. Bookkeeping and math run here in code so the model
never spends tokens on them, and every image-returning tool keeps images small.
"""

from __future__ import annotations

import base64
import datetime
import json
import logging

from mcp.types import CallToolResult, ImageContent, TextContent

from supervisor_agent.local_tools import imaging, trajectory
from supervisor_agent.local_tools.store import Store

logger = logging.getLogger(__name__)

BOX = {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4, "description": "[x1, y1, x2, y2] in the view's pixels"}

TOOLS: list[dict] = [
    {
        "name": "get_cat_profiles",
        "description": "Written profiles of the household cats: name, tracker ids per view, summary, low-res and camera cues, "
        "how to tell them apart. detail='full' adds the long description and high-res cues (eyes, nose). "
        "cat_ids may be profile ids or tracker ids (e.g. 25/26 on view 3).",
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

    def __init__(self, store: Store | None = None, hub=None):
        self.store = store or Store()
        self.hub = hub  # set by MCPHub.add_local; used to read tracking data and stored frames
        self._sizes: dict = {}

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
