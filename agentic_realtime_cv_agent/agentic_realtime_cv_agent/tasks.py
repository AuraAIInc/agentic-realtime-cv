"""Task runner: the CV pipeline (or the CLI) hands over a task envelope

    {"task": "validate_detection" | "check_trajectory" | "assign_label" | "label_visit",
     "payload": {...}}

and code routes it. For each task the runner

1. does the cheap, deterministic part itself (tracking metadata, trajectory physics, zones,
   motion state), and returns without calling the model when that already decides it;
2. otherwise loads ONE skill (skills/<name>/SKILL.md) as the system instructions, exposes only
   that task's tools, attaches one small crop, and asks for a strict JSON answer;
3. validates the answer, retries once on a bad answer, then falls back to human review;
4. appends the result to agent_data/logs/<task>.jsonl (and review_queue.jsonl when needed).

Payloads (boxes are [x1, y1, x2, y2] in the view's pixels; cat ids are the tracker's):

validate_detection: {image_name, view_id, box, detector: {is_cat_conf, cat_id, candidates:
                     [{cat_id, conf}]}, period_id?, reason?}
check_trajectory:   {period_id}
assign_label:       {image_name, view_id, box, cat_id?, period_id?, extra_frames?: [{image_name, box}]}
label_visit:        {period_id}
"""

from __future__ import annotations

import base64
import copy
import datetime
import json
import logging
import re
from pathlib import Path

from agentic_realtime_cv_agent.agent import MCPAgent, RunOptions, RunResult
from agentic_realtime_cv_agent.local_tools import LocalTools, Store, trajectory

logger = logging.getLogger(__name__)
SKILLS_DIR = Path(__file__).resolve().parent / "skills"

TASK_BASE = """\
You are the supervisor agent for RCAT, a home cat-monitoring system: an overhead camera, a
small YOLO detector, a tracker that groups detections into visits (periods), and stored frames.
You are given ONE task with the evidence code has already gathered. Use the cheapest evidence
that settles it: metadata before pixels, small crops before full frames, written cat profiles
before reference photos, one contact sheet instead of several images. Image calls are budgeted.
Say only what you can see or what the data shows. When you cannot decide, say so through the
answer's human-review action instead of guessing. Answer with the JSON object only.
"""


def load_skill(name: str) -> str:
    text = (SKILLS_DIR / name / "SKILL.md").read_text()
    return re.sub(r"^---\n.*?\n---\n", "", text, count=1, flags=re.S).strip()


def _strict(obj: dict) -> dict:
    """Mark every property required and forbid extras (Responses API strict json_schema)."""
    obj = copy.deepcopy(obj)

    def walk(s):
        if isinstance(s, dict):
            if s.get("type") == "object" and "properties" in s:
                s["required"] = list(s["properties"])
                s["additionalProperties"] = False
            for v in s.values():
                walk(v)
        elif isinstance(s, list):
            for v in s:
                walk(v)

    walk(obj)
    return obj


def _fmt(name: str, schema: dict) -> dict:
    return {"type": "json_schema", "name": name, "schema": _strict(schema), "strict": True}


NULL_INT = {"type": ["integer", "null"]}
ACTIONS_VALIDATE = ["accept", "relabel", "reject_not_cat", "send_to_human"]
ACTIONS_TRAJ = ["accept", "split", "relabel", "reject_not_cat", "send_to_human"]
ACTIONS_LABEL = ["labelled", "suggested_new", "send_to_human"]

VALIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_cat": {"type": "boolean"},
        "cat_id": NULL_INT,
        "cat_name": {"type": ["string", "null"]},
        "confidence": {"type": "number"},
        "decided_at_step": {"type": "integer"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "ruled_out": {"type": "array", "items": {"type": "object", "properties": {"cat_id": {"type": "integer"}, "why": {"type": "string"}}}},
        "action": {"type": "string", "enum": ACTIONS_VALIDATE},
        "reason": {"type": "string"},
    },
}
TRAJ_SCHEMA = {
    "type": "object",
    "properties": {
        "continuous": {"type": "boolean"},
        "segments": {"type": "array", "items": {"type": "object", "properties": {
            "from": {"type": "string"}, "to": {"type": "string"}, "same_animal_as_first": {"type": "boolean"},
            "is_cat": {"type": "boolean"}, "cat_id": NULL_INT, "needs_identity": {"type": "boolean"}}}},
        "suggest_split_at": {"type": "array", "items": {"type": "string"}},
        "not_cat_tiles": {"type": "array", "items": {"type": "integer"}},
        "action": {"type": "string", "enum": ACTIONS_TRAJ},
        "reason": {"type": "string"},
    },
}


def label_schema(labels: dict) -> dict:
    acts = [a["name"] for a in labels["activity"]] or ["unknown"]
    if "unknown" not in acts:
        acts.append("unknown")
    ctx = [c["name"] for c in labels["context"]]
    ctx_item = {"type": "object", "properties": {"name": {"type": "string", **({"enum": ctx} if ctx else {})}, "confidence": {"type": "number"}}}
    return {
        "type": "object",
        "properties": {
            "activity": {"type": "string", "enum": acts},
            "activity_confidence": {"type": "number"},
            "context_labels": {"type": "array", "items": ctx_item},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "suggestion": {"anyOf": [{"type": "object", "properties": {
                "name": {"type": "string"}, "description": {"type": "string"}, "nearest_existing": {"type": "string"}}}, {"type": "null"}]},
            "action": {"type": "string", "enum": ACTIONS_LABEL},
        },
    }


def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


def _user(text_obj: dict, images: list[bytes] = ()) -> list[dict]:
    content = [{"type": "input_text", "text": json.dumps(text_obj, default=str)}]
    content += [{"type": "input_image", "image_url": _data_url(j), "detail": "auto"} for j in images]
    return [{"role": "user", "content": content}]


class TaskRunner:
    def __init__(self, agent: MCPAgent, local: LocalTools, store: Store | None = None):
        self.agent = agent
        self.local = local
        self.store = store or local.store
        self.cfg = self.store.agent_config()

    # ------------------------------------------------------------------ entry point
    async def run(self, task: str, payload: dict) -> dict:
        handler = getattr(self, f"_task_{task}", None)
        if handler is None:
            raise ValueError(f"unknown task {task!r}; use validate_detection, check_trajectory, assign_label or label_visit")
        started = datetime.datetime.now()
        out = await handler(payload)
        out.setdefault("task", task)
        out.setdefault("label_set_version", self.store.labels()["version"])
        out["input"] = payload
        out["elapsed_s"] = round((datetime.datetime.now() - started).total_seconds(), 2)
        if task != "label_visit":
            out["log_id"] = self.store.log(task, out)
            if (out.get("result") or {}).get("action") == "send_to_human":
                self.store.log("review_queue", {"task": task, "input": payload, "result": out.get("result"), "source_log_id": out["log_id"]})
        return out

    def _task_cfg(self, task: str) -> dict:
        return (self.cfg.get("tasks") or {}).get(task, {})

    async def _ask(self, task: str, user_input: list, schema_fmt: dict, preloaded_images: int = 0) -> tuple[dict | None, RunResult]:
        tc = self._task_cfg(task)
        budget = tc.get("image_budget")
        opts = RunOptions(
            instructions=TASK_BASE + "\n" + load_skill(tc.get("skill")),
            tool_names=tc.get("tools"),
            text_format=schema_fmt,
            image_budget=None if budget is None else max(0, budget - preloaded_images),
            high_res_budget=tc.get("high_res_budget"),
            max_steps=tc.get("max_steps"),
            reasoning_effort=tc.get("reasoning_effort"),
            tag=task,
        )
        run = await self.agent.run(user_input, options=opts)
        parsed = _parse(run.answer)
        if parsed is None:  # one retry, continuing the same conversation
            retry = await self.agent.run("Your last message was not the required JSON object. Reply with the JSON object only.",
                                         previous_response_id=run.last_response_id, options=RunOptions(**{**opts.__dict__, "image_budget": 0, "max_steps": 2}))
            parsed = _parse(retry.answer)
            _merge_runs(run, retry)
        run.images_used += preloaded_images
        return parsed, run

    @staticmethod
    def _meta(run: RunResult | None, cfg_model: str | None = None) -> dict:
        if run is None:
            return {"model_called": False, "images_used": 0, "tool_calls": [], "usage": {}}
        return {
            "model_called": True, "model": run.model, "turns": run.turns, "images_used": run.images_used,
            "tool_calls": [s.tool for s in run.tool_steps], "usage": run.usage, "llm_s": round(run.llm_ms / 1000, 2),
            "stopped_reason": run.stopped_reason, "run_dir": run.run_dir,
        }

    # ------------------------------------------------------------------ evidence helpers
    async def _safe(self, coro, default=None):
        try:
            return await coro
        except Exception as e:
            logger.warning(f"evidence step failed: {e}")
            return default

    async def _concurrent_cats(self, image_name: str, view_id, box, window_s: float = 2.0) -> list[dict]:
        """Detections of any cat, any view, within window_s of the frame, other than this box."""
        t = trajectory.parse_t(image_name)
        fmt = "%Y-%m-%d %H:%M:%S.%f"
        raw = await self.local.remote("tracking__get_raw_detections", {
            "time_from": (t - datetime.timedelta(seconds=window_s)).strftime(fmt),
            "time_to": (t + datetime.timedelta(seconds=window_s)).strftime(fmt), "limit": 500})
        out = []
        for d in raw.get("detections", []):
            same_box = d.get("view_id", view_id) == view_id and box and trajectory.dist(d["box"], box) < 0.5 * trajectory.diag(box)
            if not same_box:
                out.append({k: d.get(k) for k in ("time", "view_id", "cat_id", "conf", "box")})
        return out

    async def _period_summary(self, period_id) -> dict | None:
        d = await self.local.remote("tracking__get_period_detail", {"period_id": int(period_id), "max_points": 2})
        return {
            "period_id": period_id, "status": d.get("status"), "visit_cat_id": (d.get("period") or {}).get("cat_id"),
            "duration_s": (d.get("period") or {}).get("duration_s"), "evidence_by_cat": (d.get("period") or {}).get("evidence_by_cat"),
            "frames": d.get("frames"), "best_frame": d.get("best_frame"), "came_from": d.get("came_from"),
        }

    # ------------------------------------------------------------------ Task 1
    async def _task_validate_detection(self, p: dict) -> dict:
        det = p.get("detector") or {}
        cands = [c.get("cat_id") for c in det.get("candidates", [])] or ([det["cat_id"]] if det.get("cat_id") is not None else [])
        period = await self._safe(self._period_summary(p["period_id"])) if p.get("period_id") is not None else None
        others = await self._safe(self._concurrent_cats(p["image_name"], p.get("view_id"), p.get("box")), [])
        context = {"visit": period, "other_cats_same_moment": others}

        # Rung 1 in code: visit consensus.
        vc = (self.cfg.get("validate") or {}).get("visit_consensus", {"min_frames": 20, "min_agree": 0.8})
        fr = (period or {}).get("frames") or {}
        if period and period.get("status") == "final" and (fr.get("detected") or 0) >= vc["min_frames"] and (fr.get("detector_agrees_fraction") or 0) >= vc["min_agree"]:
            cat = self.store.cat(period["visit_cat_id"], p.get("view_id"))
            vcat = period["visit_cat_id"]
            result = {
                "is_cat": True, "cat_id": vcat, "cat_name": cat.get("name") if cat else None, "confidence": round(fr["detector_agrees_fraction"], 3),
                "decided_at_step": 1, "evidence": [f"final visit with {fr['detected']} detections, {fr['detector_agrees_fraction']:.0%} agree on cat {vcat}"],
                "ruled_out": [], "action": "accept" if det.get("cat_id") == vcat else "relabel", "reason": "visit consensus",
            }
            return {"result": result, "context": context, **self._meta(None)}

        crop = await self.local.frame_crop(p["image_name"], p.get("box"), pad=0.3, max_side=384)
        profiles = self.store.profiles(cands or None, p.get("view_id"))
        parsed, run = await self._ask("validate_detection", _user({"task": "validate_detection", "detection": p, "context": context, "cats": profiles,
                                                                   "crop": "attached image (the box, padded 30%)"}, [crop]),
                                      _fmt("validation", VALIDATE_SCHEMA), preloaded_images=1)
        return {"result": parsed or _fallback("validate"), "context": context, **self._meta(run)}

    # ------------------------------------------------------------------ Task 2
    async def _task_check_trajectory(self, p: dict) -> dict:
        analysis = await self.local.t_analyze_trajectory(p["period_id"], p.get("pre_roll_s", 5))
        tcfg = self.cfg.get("trajectory") or {}
        if not analysis["flags"] and tcfg.get("accept_without_model_if_no_flags", True):
            cat_id = analysis.get("tracked_cat_id")
            result = {"continuous": True, "segments": [], "suggest_split_at": [], "not_cat_tiles": [], "action": "accept",
                      "reason": f"no physics flags over {analysis['own_detections']} detections; kept cat {cat_id}"}
            return {"result": result, "analysis": _slim(analysis), **self._meta(None)}
        profiles = self.store.profiles(None, analysis.get("view_id"))
        parsed, run = await self._ask("check_trajectory", _user({"task": "check_trajectory", "analysis": _slim(analysis), "cats": profiles}),
                                      _fmt("trajectory_check", TRAJ_SCHEMA))
        return {"result": parsed or _fallback("trajectory"), "analysis": _slim(analysis), **self._meta(run)}

    # ------------------------------------------------------------------ Task 3
    async def _label_rules(self, p: dict) -> dict:
        labels = self.store.labels()
        view_id, box = p.get("view_id"), p.get("box")
        zones = {z["name"]: z for z in self.store.view_zones(view_id)}
        overlaps = {n: trajectory.box_zone_overlap(box, z) for n, z in zones.items()} if box else {}
        state, extra = None, list(p.get("extra_frames") or [])
        if p.get("period_id") is not None:
            detail = await self._safe(self.local.remote("tracking__get_period_detail", {"period_id": int(p["period_id"])}))
            seg = trajectory.motion_state_at(detail, trajectory.parse_t(p["image_name"])) if detail else None
            if seg:
                state = seg["state"]
                if state == "moving" and not extra:
                    extra = [{"image_name": seg["start_image"], "box": seg["start_box"]}, {"image_name": seg["end_image"], "box": seg["end_box"]}]
        others = await self._safe(self._concurrent_cats(p["image_name"], view_id, box, window_s=1.0), [])
        bl = trajectory.diag(box) if box else 100
        cat_near = any(o.get("view_id", view_id) == view_id and trajectory.dist(o["box"], box) <= 1.5 * bl for o in others) if box else False

        def possible(item) -> bool:
            for need in item.get("needs", []) or []:
                if isinstance(need, dict) and "zone" in need:
                    if need["zone"] in zones and overlaps.get(need["zone"], 0) < 0.2:
                        return False
                    if need["zone"] not in zones:
                        continue  # zone not mapped yet: can't rule it out
                if need == "other_cat_nearby" and not cat_near:
                    return False
            n = item["name"]
            if state == "moving" and n in ("sleeping", "resting", "sitting", "eating", "drinking", "grooming", "standing"):
                return False
            if state == "resting" and n == "walking":
                return False
            return True

        cands = [a["name"] for a in labels["activity"] if possible(a)]
        ctx_decided, ctx_open = [], []
        for c in labels["context"]:
            rule = c.get("rule") or {}
            needs_image = "image" in (c.get("needs") or [])
            if rule.get("zone"):
                if rule["zone"] not in zones:
                    continue
                inside = overlaps.get(rule["zone"], 0) >= rule.get("min_overlap", 0.5)
                if inside and not needs_image:
                    ctx_decided.append(c["name"])
                elif inside:
                    ctx_open.append(c["name"])
            else:
                ctx_open.append(c["name"])
        return {"labels": labels, "motion_state": state, "zone_overlaps": overlaps, "other_cat_nearby": cat_near,
                "candidate_activities": cands, "context_decided": ctx_decided, "context_open": ctx_open,
                "extra_frames": extra if state == "moving" else []}

    async def _task_assign_label(self, p: dict) -> dict:
        rules = await self._label_rules(p)
        labels = rules.pop("labels")
        real = [c for c in rules["candidate_activities"] if c != "unknown"]
        lcfg = self.cfg.get("labels") or {}
        if lcfg.get("rules_only_when_decisive", True) and len(real) == 1 and not rules["context_open"]:
            act = real[0]
            spec = next(a for a in labels["activity"] if a["name"] == act)
            if "image" not in (spec.get("needs") or []):
                result = {"activity": act, "activity_confidence": 0.7, "context_labels": [{"name": n, "confidence": 0.9} for n in rules["context_decided"]],
                          "evidence": [f"rules: motion={rules['motion_state']}, zones={rules['zone_overlaps']}"], "suggestion": None, "action": "labelled"}
                return {"result": result, "rules": rules, "label_set_version": labels["version"], **self._meta(None)}

        crop = await self.local.frame_crop(p["image_name"], p.get("box"), pad=0.3, max_side=384)
        brief = {"version": labels["version"], "activity": labels["activity"], "context": labels["context"]}
        parsed, run = await self._ask("assign_label", _user({"task": "assign_label", "detection": p, "labels": brief, "rules": rules,
                                                             "crop": "attached image (the cat, padded 30%)"}, [crop]),
                                      _fmt("activity_label", label_schema(labels)), preloaded_images=1)
        result = parsed or _fallback("label")
        if result.get("suggestion") and result.get("activity") == "unknown":
            sug = {**result["suggestion"], "image_name": p["image_name"], "box": p.get("box"), "view_id": p.get("view_id"), "cat_id": p.get("cat_id"),
                   "confidence": result.get("activity_confidence")}
            result["suggestion_filed"] = self.store.suggest_label(sug, crop)
            result["action"] = "suggested_new"
        return {"result": result, "rules": rules, "label_set_version": labels["version"], **self._meta(run)}

    async def _task_label_visit(self, p: dict) -> dict:
        """One label per motion segment (not per frame): a 40-minute nap costs one image."""
        detail = await self.local.remote("tracking__get_period_detail", {"period_id": int(p["period_id"])})
        segs = ((detail.get("motion") or {}).get("segments") or [])[: (self.cfg.get("labels") or {}).get("visit_max_segments", 8)]
        if not segs:
            bf = detail.get("best_frame") or {}
            segs = [{"state": None, "start": bf.get("time"), "end": bf.get("time"), "end_image": bf.get("image_name"), "end_box": bf.get("box")}]
        out = []
        for s in segs:
            payload = {"image_name": s["end_image"], "box": s["end_box"], "view_id": detail.get("view_id"), "period_id": p["period_id"],
                       "cat_id": (detail.get("period") or {}).get("cat_id")}
            if s.get("state") == "moving":
                payload["extra_frames"] = [{"image_name": s["start_image"], "box": s["start_box"]}, {"image_name": s["end_image"], "box": s["end_box"]}]
            r = await self.run("assign_label", payload)
            out.append({"segment": {k: s.get(k) for k in ("state", "start", "end", "duration_s")}, "activity": r["result"].get("activity"),
                        "context_labels": r["result"].get("context_labels"), "action": r["result"].get("action"), "log_id": r.get("log_id"),
                        "images_used": r.get("images_used", 0)})
        return {"result": {"period_id": p["period_id"], "segments": out}, "images_used": sum(o["images_used"] for o in out), "model_called": any(o["images_used"] for o in out)}


def _parse(text: str | None) -> dict | None:
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(json)?|```$", "", t, flags=re.M).strip()
    try:
        v = json.loads(t)
        return v if isinstance(v, dict) else None
    except ValueError:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except ValueError:
                return None
    return None


def _fallback(kind: str) -> dict:
    reason = "model did not return a valid answer"
    if kind == "validate":
        return {"is_cat": False, "cat_id": None, "cat_name": None, "confidence": 0.0, "decided_at_step": 6, "evidence": [], "ruled_out": [], "action": "send_to_human", "reason": reason}
    if kind == "trajectory":
        return {"continuous": False, "segments": [], "suggest_split_at": [], "not_cat_tiles": [], "action": "send_to_human", "reason": reason}
    return {"activity": "unknown", "activity_confidence": 0.0, "context_labels": [], "evidence": [], "suggestion": None, "action": "send_to_human"}


def _slim(analysis: dict) -> dict:
    a = dict(analysis)
    runs = a.get("identity_runs") or []
    if len(runs) > 12:
        a["identity_runs"] = runs[:6] + [{"note": f"{len(runs) - 12} runs omitted"}] + runs[-6:]
    return a


def _merge_runs(a: RunResult, b: RunResult) -> None:
    a.turns += b.turns
    a.tool_steps += b.tool_steps
    a.llm_ms += b.llm_ms
    for k, v in b.usage.items():
        a.usage[k] = a.usage.get(k, 0) + v
    a.answer = b.answer or a.answer
    a.last_response_id = b.last_response_id


def chat_options(store: Store) -> RunOptions:
    """Options for the interactive label-command chat (manage-labels skill)."""
    tc = (store.agent_config().get("tasks") or {}).get("manage_labels", {})
    return RunOptions(instructions=TASK_BASE.replace("Answer with the JSON object only.", "Talk to the user in plain sentences.") + "\n" + load_skill(tc.get("skill", "manage-labels")),
                      tool_names=tc.get("tools"), image_budget=tc.get("image_budget"), max_steps=tc.get("max_steps"), tag="manage_labels")
