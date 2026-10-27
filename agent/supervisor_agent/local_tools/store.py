"""File-backed data the task agent reads and writes: cat profiles, reference images, the
label set, zones, and result logs. Everything lives under agent_data/ (override with
$AGENT_DATA_DIR) so it can be edited by hand and mirrored to S3 with the same keys.

    agent_data/config/{cats,labels,zones,agent}.yaml
    agent_data/references/<cat>/{profile,scene,extra}/  + index.json
    agent_data/logs/<task>.jsonl, review_queue.jsonl
    agent_data/label_suggestions/<name>/suggestions.jsonl + crops
"""

from __future__ import annotations

import datetime
import json
import math
import os
import re
import threading
from pathlib import Path

import yaml

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "agent_data"
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class Store:
    def __init__(self, root: str | os.PathLike | None = None):
        self.root = Path(root or os.getenv("AGENT_DATA_DIR") or DEFAULT_DIR)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ yaml helpers
    def path(self, *parts) -> Path:
        return self.root.joinpath(*parts)

    def _yaml(self, name: str) -> dict:
        p = self.path("config", name)
        if not p.exists():
            return {}
        with open(p) as f:
            return yaml.safe_load(f) or {}

    def _write_yaml(self, name: str, data: dict, header: str = "") -> None:
        p = self.path("config", name)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w") as f:
            if header:
                f.write(header)
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, width=100)
        tmp.replace(p)

    def agent_config(self) -> dict:
        return self._yaml("agent.yaml")

    # ------------------------------------------------------------------ cats
    def cats(self) -> list[dict]:
        return self._yaml("cats.yaml").get("cats", [])

    def cat(self, any_id: int | None, view_id: int | None = None) -> dict | None:
        """Find a cat by its profile cat_id or any tracker id in tracking_ids."""
        if any_id is None:
            return None
        any_id = int(any_id)
        for c in self.cats():
            if c.get("cat_id") == any_id:
                return c
        for c in self.cats():
            ids = (c.get("tracking_ids") or {})
            if view_id is not None:
                for k, v in ids.items():
                    if v == any_id and k.startswith(f"view_{view_id}_"):
                        return c
            if any_id in ids.values():
                return c
        return None

    def cat_ids_for(self, cat: dict) -> set[int]:
        return {cat["cat_id"], *[v for v in (cat.get("tracking_ids") or {}).values() if isinstance(v, int)]}

    def profiles(self, ids: list[int] | None = None, view_id: int | None = None, detail: str = "low_res") -> list[dict]:
        """Trimmed profiles for the model. detail: low_res (default), camera, or full."""
        cats = self.cats() if not ids else [c for c in (self.cat(i, view_id) for i in ids) if c]
        seen, out = set(), []
        for c in cats:
            if c["cat_id"] in seen:
                continue
            seen.add(c["cat_id"])
            p = {
                "cat_id": c["cat_id"],
                "name": c.get("name"),
                "tracking_ids": c.get("tracking_ids", {}),
                "summary": c.get("summary"),
                "low_res_cues": c.get("low_res_cues", []),
                "camera_cues": c.get("camera_cues", []),
                "confused_with": c.get("confused_with", []),
            }
            if detail in ("full", "high_res"):
                p["description"] = (c.get("description") or "").strip()
                p["high_res_cues"] = c.get("high_res_cues", [])
            out.append(p)
        return out

    # ------------------------------------------------------------------ references
    def reference_index(self, cat: dict) -> list[dict]:
        folder = self.path(cat.get("folder") or f"references/{cat['name'].lower()}")
        p = folder / "index.json"
        if not p.exists():
            return []
        with open(p) as f:
            data = json.load(f)
        for r in data.get("images", []):
            r["_dir"] = str(folder)
        return data.get("images", [])

    def find_references(
        self,
        cat_id: int,
        view_id: int | None = None,
        box: list[float] | None = None,
        view_size: tuple[int, int] | None = None,
        pose: str | None = None,
        lighting: str | None = None,
        source: str | None = None,
        k: int | None = None,
    ) -> list[dict]:
        cat = self.cat(cat_id, view_id)
        if cat is None:
            raise LookupError(f"no cat profile for id {cat_id}")
        cfg = self.agent_config().get("references", {})
        w = {"view": 1.0, "position": 1.0, "size": 1.0, "pose": 0.5, "lighting": 1.0, "quality": 0.5, **cfg.get("weights", {})}
        k = k or cfg.get("max_per_cat", 2)
        out = []
        for r in self.reference_index(cat):
            if not r.get("use_for_matching", True):
                continue
            if source and r.get("source") != source:
                continue
            s, why = score_reference(r, w, view_id, box, view_size, pose, lighting)
            out.append({
                "cat_id": cat["cat_id"],
                "name": cat.get("name"),
                "file": f"{Path(r['_dir']).name}/{r.get('crop_file') or r['file']}",
                "source": r.get("source"),
                "score": round(s, 3),
                "why": why,
                "pose": r.get("pose"),
                "lighting": r.get("lighting"),
                "view_id": r.get("view_id"),
                "quality": r.get("quality"),
                "note": r.get("note"),
            })
        out.sort(key=lambda x: -x["score"])
        return out[:k]

    def reference_path(self, file: str) -> Path:
        """file as returned by find_references: '<cat folder>/<relative path>'."""
        p = (self.path("references") / file).resolve()
        if not str(p).startswith(str(self.path("references").resolve())):
            raise ValueError("reference path escapes the references folder")
        if not p.exists():
            raise FileNotFoundError(file)
        return p

    # ------------------------------------------------------------------ labels
    def labels(self) -> dict:
        d = self._yaml("labels.yaml")
        d.setdefault("version", 1)
        d.setdefault("activity", [])
        d.setdefault("context", [])
        return d

    def label_names(self, kind: str) -> list[str]:
        return [x["name"] for x in self.labels().get(kind, [])]

    def add_label(self, name: str, kind: str, description: str, rule: dict | None = None, needs: list | None = None, confirm: bool = False) -> dict:
        """Validate a new label; write it only when confirm=True. Returns the draft + problems."""
        labels = self.labels()
        problems = []
        if kind not in ("activity", "context"):
            problems.append("kind must be 'activity' or 'context'")
        if not NAME_RE.match(name or ""):
            problems.append("name must be lower_snake_case, 2-41 chars, starting with a letter")
        existing = {x["name"] for x in labels["activity"] + labels["context"]}
        if name in existing:
            problems.append(f"label '{name}' already exists")
        similar = [n for n in existing if _similar(n, name)]
        if rule and rule.get("zone"):
            view_zones = {z["name"] for v in self.zones().get("views", {}).values() for z in v.get("zones", [])}
            if rule["zone"] not in view_zones:
                problems.append(f"zone '{rule['zone']}' is not defined in zones.yaml yet; define it first (save_zone)")
        entry = {"name": name, "description": description}
        if needs:
            entry["needs"] = needs
        if rule:
            entry["rule"] = rule
        result = {"draft": entry, "kind": kind, "problems": problems, "similar_existing": similar, "saved": False, "version": labels["version"]}
        if confirm and not problems:
            with self._lock:
                labels = self.labels()
                labels[kind].append(entry)
                labels["version"] = int(labels.get("version", 1)) + 1
                self._write_yaml("labels.yaml", labels, _header("labels.yaml", self))
            result.update(saved=True, version=labels["version"])
            self.log("label_changes", {"action": "add", "kind": kind, "label": entry, "version": labels["version"]})
        return result

    # ------------------------------------------------------------------ zones
    def zones(self) -> dict:
        d = self._yaml("zones.yaml")
        d.setdefault("views", {})
        return d

    def view_zones(self, view_id) -> list[dict]:
        return (self.zones()["views"].get(str(view_id)) or {}).get("zones") or []

    def view_size(self, view_id) -> tuple[int, int] | None:
        s = (self.zones()["views"].get(str(view_id)) or {}).get("size")
        return tuple(s) if s else None

    def save_zone(self, view_id: int, name: str, kind: str, box: list[float] | None = None, polygon: list | None = None, confirm: bool = False) -> dict:
        problems = []
        if not NAME_RE.match(name or ""):
            problems.append("name must be lower_snake_case")
        if kind not in ("place", "occluder", "entry"):
            problems.append("kind must be place, occluder or entry")
        if (box is None) == (polygon is None):
            problems.append("give exactly one of box or polygon")
        entry = {"name": name, "kind": kind, **({"box": [round(float(v), 1) for v in box]} if box else {}), **({"polygon": polygon} if polygon else {})}
        result = {"draft": entry, "view_id": view_id, "problems": problems, "saved": False}
        if confirm and not problems:
            with self._lock:
                z = self.zones()
                v = z["views"].setdefault(str(view_id), {"zones": []})
                v.setdefault("zones", [])
                v["zones"] = [x for x in v["zones"] if x.get("name") != name] + [entry]
                z["version"] = int(z.get("version", 1)) + 1
                self._write_yaml("zones.yaml", z, _header("zones.yaml", self))
            result["saved"] = True
            self.log("label_changes", {"action": "save_zone", "view_id": view_id, "zone": entry})
        return result

    # ------------------------------------------------------------------ logs
    def log(self, task: str, record: dict) -> str:
        p = self.path("logs", f"{task}.jsonl")
        p.parent.mkdir(parents=True, exist_ok=True)
        rid = f"{task}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        with self._lock, open(p, "a") as f:
            f.write(json.dumps({"id": rid, "logged_at": _now(), **record}, default=str) + "\n")
        return rid

    def suggest_label(self, suggestion: dict, crop_jpeg: bytes | None) -> dict:
        """File an unassignable crop under label_suggestions/<name>/, reusing an existing
        folder whose name is the same or very close."""
        name = re.sub(r"[^a-z0-9_]+", "_", (suggestion.get("name") or "unnamed").lower()).strip("_") or "unnamed"
        root = self.path("label_suggestions")
        root.mkdir(parents=True, exist_ok=True)
        existing = [p.name for p in root.iterdir() if p.is_dir()]
        match = next((e for e in existing if e == name or _similar(e, name)), None)
        folder = root / (match or name)
        folder.mkdir(exist_ok=True)
        rec = {k: v for k, v in suggestion.items()}
        rec["folder"] = folder.name
        if crop_jpeg:
            stem = (suggestion.get("image_name") or _now()).removesuffix(".jpg")
            img = folder / f"{stem}_crop.jpg"
            img.write_bytes(crop_jpeg)
            rec["crop_file"] = img.name
        with self._lock, open(folder / "suggestions.jsonl", "a") as f:
            f.write(json.dumps({"logged_at": _now(), **rec}, default=str) + "\n")
        return rec


def score_reference(r: dict, w: dict, view_id, box, view_size, pose, lighting) -> tuple[float, list[str]]:
    """Higher = more useful as a reference for this query (see design doc, Task 1)."""
    s, why = 0.0, []
    same_view = view_id is not None and r.get("view_id") == view_id
    if same_view:
        s += w["view"]
        why.append("same view")
    if same_view and box and r.get("box") and view_size:
        vw, vh = view_size
        (ax, ay), (bx, by) = _center(box), _center(r["box"])
        dist = math.hypot((ax - bx) / vw, (ay - by) / vh) / math.sqrt(2)
        s += w["position"] * (1 - dist)
        ratio = max(_area(box), 1) / max(_area(r["box"]), 1)
        s += w["size"] * max(0.0, 1 - abs(math.log(ratio)) / 2)
        if dist < 0.15:
            why.append("nearby spot")
    if pose and r.get("pose") == pose:
        s += w["pose"]
        why.append("same pose")
    if lighting and r.get("lighting") == lighting:
        s += w["lighting"]
        why.append("same lighting")
    s += w["quality"] * (r.get("quality") or 3) / 5
    return s, why


def _center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _similar(a: str, b: str) -> bool:
    """Cheap near-duplicate check for label names (zoomies vs zoomie, on_sofa vs on_the_sofa)."""
    if a == b:
        return False
    ta, tb = set(a.replace("the_", "").split("_")), set(b.replace("the_", "").split("_"))
    if ta == tb:
        return True
    sa, sb = a.rstrip("s"), b.rstrip("s")
    return sa == sb


def _header(name: str, store: Store) -> str:
    p = store.path("config", name)
    if not p.exists():
        return ""
    lines = []
    with open(p) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                lines.append(line)
            else:
                break
    return "".join(lines)
