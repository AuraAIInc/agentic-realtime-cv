"""Offline tests for the task agent: fake tracking / image servers and a scripted model.

Run from the agentic_realtime_cv_agent/ folder:
    python -m pytest agentic_realtime_cv_agent/tests -q
"""

from __future__ import annotations

import asyncio
import base64
import datetime as dt
import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp.types import CallToolResult, ImageContent, TextContent
from PIL import Image

from agentic_realtime_cv_agent.agent import MCPAgent
from agentic_realtime_cv_agent.config import AgentConfig
from agentic_realtime_cv_agent.local_tools import LocalTools, Store, trajectory
from agentic_realtime_cv_agent.mcp_hub import MCPHub
from agentic_realtime_cv_agent.tasks import VALIDATE_SCHEMA, TaskRunner, _strict, label_schema

DATA = Path(__file__).resolve().parents[2] / "agent_data"
T0 = dt.datetime(2026, 10, 26, 2, 0, 0)


# ---------------------------------------------------------------------- fixtures
def ts(s: float) -> str:
    return (T0 + dt.timedelta(seconds=s)).isoformat(sep=" ")


def walk(n=30, step=8.0, x0=400.0, y0=300.0, size=(90, 70), cat_id=26, t_step=0.5, start_s=0.0):
    out = []
    for i in range(n):
        x = x0 + i * step
        out.append({"time": ts(start_s + i * t_step), "cat_id": cat_id, "conf": 0.8, "box": [x - size[0] / 2, y0 - size[1] / 2, x + size[0] / 2, y0 + size[1] / 2]})
    return out


def detail_from(raw, status="final", cat_id=26, view_id=3, period_id=1, segments=None, frames=None):
    path = [{"time": r["time"], "x": (r["box"][0] + r["box"][2]) / 2, "y": (r["box"][1] + r["box"][3]) / 2} for r in raw]
    best = max(raw, key=lambda r: r["conf"])
    return {
        "period_id": period_id, "view_id": view_id, "status": status,
        "period": {"cat_id": cat_id, "start": raw[0]["time"], "end": raw[-1]["time"], "duration_s": 15, "status": status},
        "frames": frames or {"frames": len(raw), "detected": len(raw), "detector_agrees_fraction": 0.9},
        "first_frame": {"time": raw[0]["time"], "box": raw[0]["box"]}, "last_frame": {"time": raw[-1]["time"], "box": raw[-1]["box"]},
        "best_frame": {"time": best["time"], "box": best["box"], "image_name": trajectory.image_name(trajectory.parse_t(best["time"]))},
        "path": path, "motion": {"totals": {"duration_s": 15, "moving_s": 15}, "segments": segments or []},
    }


def jpeg(w=800, h=600, color=(200, 200, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "JPEG")
    return buf.getvalue()


class FakeServer:
    """Plays tracking__* and images__* through MCPHub.add_local."""

    def __init__(self, name, handlers):
        self.server_name = name
        self.handlers = handlers
        self.calls = []

    def specs(self):
        return [{"name": n, "description": n, "input_schema": {"type": "object", "properties": {}}} for n in self.handlers]

    async def call(self, name, args):
        self.calls.append((name, args))
        out = self.handlers[name](**args)
        if isinstance(out, bytes):
            return CallToolResult(content=[TextContent(type="text", text="{}"), ImageContent(type="image", data=base64.b64encode(out).decode(), mime_type="image/jpeg")])
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(out))])


class ScriptedModel:
    """Fake Responses API client: returns the scripted responses in order and records requests."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.responses = SimpleNamespace(create=self._create)

    async def _create(self, **kw):
        self.requests.append(kw)
        step = self.script.pop(0)
        if isinstance(step, list):  # tool calls
            items = [SimpleNamespace(type="function_call", name=n, arguments=json.dumps(a), call_id=f"c{i}") for i, (n, a) in enumerate(step)]
            return SimpleNamespace(id=f"r{len(self.requests)}", output=items, output_text="", usage=None)
        return SimpleNamespace(id=f"r{len(self.requests)}", output=[], output_text=json.dumps(step) if isinstance(step, dict) else step, usage=None)


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "agent_data"
    shutil.copytree(DATA / "config", root / "config")
    shutil.copytree(DATA / "references", root / "references", ignore=shutil.ignore_patterns("*.MOV"))
    return Store(root)


def make_runner(store, raw, detail, model_script):
    hub = MCPHub([])
    tracking = FakeServer("tracking", {
        "get_period_detail": lambda **a: detail,
        "get_raw_detections": lambda **a: {"count": len(raw), "detections": raw},
        "list_views": lambda **a: {"views": [{"view_id": 3, "width": 800, "height": 600}]},
    })
    images = FakeServer("images", {"get_image": lambda **a: jpeg()})
    hub.add_local(tracking)
    hub.add_local(images)
    local = LocalTools(store)
    hub.add_local(local)
    cfg = AgentConfig(model="fake", save_dir=None)
    model = ScriptedModel(model_script)
    agent = MCPAgent(cfg, hub, client=model)
    return TaskRunner(agent, local, store), model, images


# ---------------------------------------------------------------------- trajectory physics
def test_continuous_walk_has_no_flags():
    raw = walk(n=30, x0=60, step=25)  # enters at the left edge, leaves at the right
    a = trajectory.analyze(detail_from(raw), raw, (800, 600))
    assert a["flags"] == [] and a["continuous"] is True
    assert 1 <= len(a["keyframes"]) <= 6


def test_teleport_and_identity_flip_are_flagged():
    raw = walk(n=10, x0=60) + walk(n=10, x0=650, start_s=5.0, cat_id=25)
    a = trajectory.analyze(detail_from(raw), raw, (800, 600))
    kinds = {f["type"] for f in a["flags"]}
    assert "teleport" in kinds and "identity_flip" in kinds
    roles = {k["role"] for k in a["keyframes"]}
    assert "before_teleport" in roles and "after_teleport" in roles


def test_gap_behind_occluder_is_not_a_teleport():
    raw = walk(n=10, x0=60) + walk(n=10, x0=650, start_s=5.0)
    zones = [{"name": "behind_sofa", "kind": "occluder", "box": [600, 200, 800, 400]}]
    a = trajectory.analyze(detail_from(raw), raw, (800, 600), zones)
    assert "teleport" not in {f["type"] for f in a["flags"]}


def test_mid_frame_entry_is_flagged():
    raw = walk(n=10, x0=380, step=2)
    a = trajectory.analyze(detail_from(raw), raw, (800, 600))
    assert "impossible_entry" in {f["type"] for f in a["flags"]}


def test_other_cat_close_is_flagged_and_kept_out_of_own_track():
    raw = walk(n=10, x0=60)
    other = [{**r, "cat_id": 25, "box": [r["box"][0] + 30, r["box"][1], r["box"][2] + 30, r["box"][3]]} for r in raw[4:7]]
    a = trajectory.analyze(detail_from(raw), raw + other, (800, 600))
    assert a["own_detections"] == 10
    assert "other_cat_close" in {f["type"] for f in a["flags"]}


# ---------------------------------------------------------------------- store
def test_tracker_ids_map_to_profiles(store):
    assert store.cat(25, 3)["name"] == "Qb"
    assert store.cat(26)["name"] == "Qbo"
    assert store.cat(0)["name"] == "Qb"


def test_scene_reference_near_the_box_ranks_first(store):
    refs = store.find_references(26, view_id=3, box=[223, 346, 310, 526], view_size=(800, 600), lighting="low_light", k=3)
    assert refs[0]["source"] == "scene" and "same view" in refs[0]["why"]
    assert all(r["file"].endswith("_crop.jpg") for r in refs)
    assert store.reference_path(refs[0]["file"]).exists()


def test_painted_portrait_is_never_a_reference(store):
    files = [r["file"] for r in store.find_references(0, k=50)]
    assert not any("painted" in f for f in files)


def test_add_label_needs_confirmation_and_bumps_version(store):
    v = store.labels()["version"]
    draft = store.add_label("running", "activity", "Fast gallop", needs=["motion"])
    assert draft["saved"] is False and not draft["problems"]
    assert "running" not in store.label_names("activity")
    saved = store.add_label("running", "activity", "Fast gallop", needs=["motion"], confirm=True)
    assert saved["saved"] and store.labels()["version"] == v + 1
    assert store.add_label("running", "activity", "dup")["problems"]


def test_place_label_requires_its_zone(store):
    assert store.add_label("in_box", "context", "In the box", rule={"zone": "box"})["problems"]
    store.save_zone(3, "box", "place", box=[10, 10, 100, 100], confirm=True)
    assert not store.add_label("in_box", "context", "In the box", rule={"zone": "box"})["problems"]


def test_suggestions_reuse_a_similar_folder(store):
    a = store.suggest_label({"name": "zoomies", "image_name": "a.jpg"}, jpeg(50, 50))
    b = store.suggest_label({"name": "zoomie", "image_name": "b.jpg"}, jpeg(50, 50))
    assert a["folder"] == b["folder"] == "zoomies"
    assert (store.path("label_suggestions", "zoomies", "b_crop.jpg")).exists()


# ---------------------------------------------------------------------- schemas
def test_schemas_are_strict():
    def check(s):
        if isinstance(s, dict):
            if s.get("type") == "object":
                assert set(s["required"]) == set(s["properties"]) and s["additionalProperties"] is False
            for v in s.values():
                check(v)
        elif isinstance(s, list):
            for v in s:
                check(v)

    check(_strict(VALIDATE_SCHEMA))
    check(_strict(label_schema({"activity": [{"name": "walking"}], "context": []})))


# ---------------------------------------------------------------------- task runner
def test_clean_trajectory_needs_no_model(store):
    raw = walk(n=30, x0=60, step=25)
    runner, model, _ = make_runner(store, raw, detail_from(raw), [])
    out = asyncio.run(runner.run("check_trajectory", {"period_id": 1}))
    assert out["model_called"] is False and out["result"]["action"] == "accept"
    assert model.requests == []
    assert store.path("logs", "check_trajectory.jsonl").exists()


def test_flagged_trajectory_asks_the_model_once_with_a_contact_sheet(store):
    raw = walk(n=10, x0=60) + walk(n=10, x0=650, start_s=5.0, cat_id=25)
    answer = {"continuous": False, "segments": [], "suggest_split_at": [ts(5.0)], "not_cat_tiles": [], "action": "split", "reason": "two cats"}
    detail = detail_from(raw)
    runner, model, images = make_runner(store, raw, detail, [
        [("local__make_contact_sheet", {"items": [{"image_name": "2026-10-26-02-00-00-000000.jpg", "box": raw[0]["box"]}]})],
        answer,
    ])
    out = asyncio.run(runner.run("check_trajectory", {"period_id": 1}))
    assert out["result"]["action"] == "split" and out["images_used"] == 1
    req = model.requests[0]
    assert req["text"]["format"]["strict"] is True
    assert {t["name"] for t in req["tools"]} <= {"images__get_image", "local__make_contact_sheet", "local__get_cat_profiles",
                                                 "local__find_references", "local__get_reference_image"}
    assert "Check a visit's continuity" in req["instructions"]


def test_image_budget_refuses_extra_looks(store):
    raw = walk(n=10, x0=380, step=2)  # mid-frame entry -> model is asked
    detail = detail_from(raw, frames={"frames": 10, "detected": 10, "detector_agrees_fraction": 0.5})
    look = ("images__get_image", {"image_name": "2026-10-26-02-00-00-000000.jpg"})
    answer = {"is_cat": True, "cat_id": 26, "cat_name": "Qbo", "confidence": 0.7, "decided_at_step": 2, "evidence": ["dark plume tail"],
              "ruled_out": [], "action": "accept", "reason": "tail"}
    runner, model, images = make_runner(store, raw, detail, [[look, look, look, look], answer])
    payload = {"image_name": "2026-10-26-02-00-00-000000.jpg", "view_id": 3, "box": raw[0]["box"], "period_id": 1,
               "detector": {"is_cat_conf": 0.41, "cat_id": 26, "candidates": [{"cat_id": 26, "conf": 0.41}, {"cat_id": 25, "conf": 0.33}]}}
    out = asyncio.run(runner.run("validate_detection", payload))
    # budget 3, one crop preloaded -> 2 of the 4 looks allowed, 2 refused
    assert out["images_used"] == 3
    assert len(images.calls) == 3  # preloaded crop + 2 allowed looks
    refusals = [o for o in model.requests[1]["input"] if "image budget reached" in json.dumps(o["output"])]
    assert len(refusals) == 2
    first_input = model.requests[0]["input"][0]["content"]
    assert any(c["type"] == "input_image" for c in first_input)
    cats = json.loads(first_input[0]["text"])["cats"]
    assert {c["name"] for c in cats} == {"Qb", "Qbo"}


def test_solid_visit_is_accepted_without_the_model(store):
    raw = walk(n=30, x0=60)
    detail = detail_from(raw, frames={"frames": 30, "detected": 30, "detector_agrees_fraction": 0.95})
    runner, model, _ = make_runner(store, raw, detail, [])
    payload = {"image_name": "2026-10-26-02-00-05-000000.jpg", "view_id": 3, "box": raw[10]["box"], "period_id": 1,
               "detector": {"cat_id": 25, "candidates": [{"cat_id": 25, "conf": 0.45}]}}
    out = asyncio.run(runner.run("validate_detection", payload))
    assert out["model_called"] is False
    assert out["result"]["cat_id"] == 26 and out["result"]["action"] == "relabel"


def test_unknown_activity_files_a_suggestion(store):
    raw = walk(n=10, x0=60)
    detail = detail_from(raw)
    answer = {"activity": "unknown", "activity_confidence": 0.6, "context_labels": [], "evidence": ["gallop"],
              "suggestion": {"name": "running", "description": "Fast gallop", "nearest_existing": "walking"}, "action": "suggested_new"}
    runner, model, _ = make_runner(store, raw, detail, [answer])
    out = asyncio.run(runner.run("assign_label", {"image_name": "2026-10-26-02-00-02-000000.jpg", "view_id": 3, "box": raw[4]["box"]}))
    assert out["result"]["action"] == "suggested_new"
    assert store.path("label_suggestions", "running", "suggestions.jsonl").exists()
    enum = model.requests[0]["text"]["format"]["schema"]["properties"]["activity"]["enum"]
    assert "walking" in enum and "unknown" in enum


def test_moving_cat_rules_out_resting_labels(store):
    raw = walk(n=10, x0=60)
    seg = {"state": "moving", "start": raw[0]["time"], "end": raw[-1]["time"], "start_image": "a.jpg", "end_image": "b.jpg",
           "start_box": raw[0]["box"], "end_box": raw[-1]["box"], "duration_s": 5}
    detail = detail_from(raw, segments=[seg])
    runner, _, _ = make_runner(store, raw, detail, [])
    rules = asyncio.run(runner._label_rules({"image_name": "2026-10-26-02-00-02-000000.jpg", "view_id": 3, "box": raw[4]["box"], "period_id": 1}))
    assert rules["motion_state"] == "moving"
    assert "sleeping" not in rules["candidate_activities"] and "walking" in rules["candidate_activities"]
    assert len(rules["extra_frames"]) == 2


def test_contact_sheet_is_one_image(store):
    raw = walk(n=3)
    runner, _, images = make_runner(store, raw, detail_from(raw), [])
    local = runner.local
    res = asyncio.run(local.call("make_contact_sheet", {"items": [
        {"image_name": "2026-10-26-02-00-00-000000.jpg", "box": raw[0]["box"]},
        {"reference_file": "qbo/profile/chat_crouching_face_crop.jpg"}]}))
    assert sum(1 for c in res.content if c.type == "image") == 1
    assert json.loads(res.content[0].text)["tiles"][1]["caption"] == "reference"
