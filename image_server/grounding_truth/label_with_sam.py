"""Label retrieved hour images with Muse SAM 3.1 + Muse Spark to build grounding truth.

Two stages per image. (1) PROPOSE: SAM 3.1 segments generic animal concepts
("animal", "a cat" — short concepts ground reliably; descriptive prose does
not, verified by probing) and the union is deduplicated by IoU. (2)
CLASSIFY: one Muse Spark call per image with boxes assigns each box a cat
identity (or rejects it) using the full visual descriptions from the cat
profiles; the existing YOLO boxes are included as REFERENCE ONLY. Only
boxes with an accepted identity are written, as YOLO lines next to images/
in muse_labels/.

Usage (Python >= 3.10, stdlib + pyyaml only):
    python image_server/grounding_truth/label_with_sam.py 2026-10-28-16
    python image_server/grounding_truth/label_with_sam.py --start 2026-10-28-16-33 --end 2026-10-28-16-40
    python image_server/grounding_truth/label_with_sam.py --start 2026-10-28-16-33 --end 2026-10-28-16-33 --limit 5 --dry-run

Time bounds are YYYY-MM-DD-HH-MM (the end minute is fully included) or a bare
hour YYYY-MM-DD-HH. A range may span hours: every retrieved_images/<hour>/
folder overlapped by the range is processed, and each folder gets its own
muse_labels/ directory beside images/ and labels/.

Output per hour folder:
    muse_labels/
        <same stem as the image>.txt   YOLO lines: <class_id> xc yc w h
        classes.txt                    copy of the folder's classes (labelImg)
An empty .txt means the image was processed and SAM found no matching cat;
images are skipped (never overwritten) when their muse_labels/ file already
exists, unless --overwrite is passed.

Billing (Meta Model API, Standard mode, verified 2026-11-03 at
https://dev.meta.ai/docs/pricing-rate-limits): SAM 3.1 costs $2.50 per 1,000
images (billed per image, not per token); Muse Spark costs $1.25/$4.25 per
1M input/output tokens. A typical image with cats costs ~2 SAM requests +
1 Spark call (~$0.01). The script prints the running cost estimate and
writes a Markdown + JSON session report to
image_server/grounding_truth/reports/ after every run, including dry runs.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import datetime
import json
import logging
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request

try:
    import yaml
except ImportError:  # pragma: no cover - pyyaml is a declared dependency
    yaml = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fetch_hour_images import jpeg_size, to_yolo  # noqa: E402  (stdlib-only module)

logger = logging.getLogger(__name__)

GROUNDING_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(GROUNDING_DIR))
REPO_ENV = os.path.join(REPO_ROOT, ".env")
DEFAULT_RETRIEVED = os.path.join(os.path.dirname(GROUNDING_DIR), "retrieved_images")
DEFAULT_CATS = os.path.join(REPO_ROOT, "agent", "agent_data", "config", "cats.yaml")
DEFAULT_MODELS = os.path.join(REPO_ROOT, "agent", "supervisor_agent", "models.json")
REPORTS_DIR = os.path.join(GROUNDING_DIR, "reports")

SAM_PRICING_URL = "https://dev.meta.ai/docs/pricing-rate-limits"
SAM_USD_PER_1000_IMAGES = 2.50
SAM_FALLBACK_MODEL = "sam-3.1"

HOUR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}$")
MINUTE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}$")
PREFIXED_RE = re.compile(r"^(\d+)_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{6}\.jpg)$")
# SAM box token: <|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..[;c=..]|> (fields by key, c optional)
BOX_RE = re.compile(r"<\|box;([^|]*)\|>")


def load_env():
    """Read the repo-root .env so MODEL_API_KEY works from any directory."""
    if not os.path.isfile(REPO_ENV):
        logger.warning(f"{REPO_ENV} not found, using the environment only")
        return
    for line in open(REPO_ENV):
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip("\"'")


def load_sam_settings(model_config=None):
    """Return (model, base_url, api_key) for the segmentation role.

    Falls back to the agent role's base URL and to sam-3.1, mirroring how the
    agent resolves roles from models.json.
    """
    cfg_path = model_config or os.getenv("MODEL_CONFIG") or DEFAULT_MODELS
    with open(cfg_path, encoding="utf-8") as f:
        data = json.load(f)
    roles = data.get("roles", {})
    role_cfg = dict(roles.get("segmentation") or roles.get("agent") or {})
    model = role_cfg.get("model") or SAM_FALLBACK_MODEL
    base_url = role_cfg.get("base_url") or data.get("base_url", "")
    api_key = os.getenv(role_cfg.get("api_key_env") or data.get("api_key_env", "MODEL_API_KEY"), "")
    return str(model), base_url.rstrip("/"), api_key.strip().strip("\"'")


def load_cats(path=None):
    """Parse cats.yaml into [{name, tracking_ids, summary, description, cues}]."""
    if yaml is None:
        raise RuntimeError("pyyaml is required (pip install pyyaml)")
    with open(path or DEFAULT_CATS, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cats = []
    for entry in data.get("cats", []):
        tracking = entry.get("tracking_ids") or {}
        cues = list(entry.get("low_res_cues") or []) + list(entry.get("camera_cues") or [])
        cats.append({
            "name": entry.get("name", "?"),
            "tracking_ids": {int(v) for v in tracking.values() if isinstance(v, int)},
            "summary": (entry.get("summary") or "").strip(),
            "description": " ".join((entry.get("description") or "").split()),
            "cues": [c.strip() for c in cues if c and c.strip()],
        })
    return cats


# SAM grounding concepts, unioned per image. Verified by live probing:
# "animal" has the best recall, "a cat" agrees on clear cats; bare "cat",
# "cats" and "kitten" return nothing, and descriptive paragraphs return
# nothing. Identity is decided by the Spark stage, not by these concepts.
SAM_CONCEPTS = ["animal", "a cat"]
DEDUP_IOU = 0.5

SPARK_FALLBACK_MODEL = "muse-spark-1.3"
SPARK_FALLBACK_PRICING = {"input": 1.25, "output": 4.25}  # USD per 1M tokens


def short_concept(cat):
    """One-line cat summary for queries and reports."""
    summary = cat["summary"].rstrip(".")
    return f"{cat['name']}: {summary}" if summary else cat["name"]


def build_sam_prompt(concept):
    """SAM input_text: a bare grounding concept (prose does not ground)."""
    return concept


def build_spark_prompt(classes, boxes, ref_by_class, img_w, img_h):
    """Full Spark input_text: identify each proposed box.

    classes: [{class_id, name, cat}] from the hour folder's class mapping.
    boxes: [(x1, y1, x2, y2)] proposed regions in source pixels.
    ref_by_class: {class_id: [(x1, y1, x2, y2)]} old-detector boxes, included
    as REFERENCE ONLY (they may be wrong; the image is the evidence).
    """
    lines = [
        f"Identify each boxed region in this {img_w}x{img_h} home-camera frame. "
        f"Reply with ONLY a JSON object, no other text.",
        "Cats:",
    ]
    for q in classes:
        cat = q["cat"]
        cues = "; ".join(cat["cues"][:3])
        lines.append(f"- id {q['class_id']} {q['name']}: {cat['summary']} Cues: {cues}.")
    lines.append("Boxes (pixels, x1 y1 x2 y2):")
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        lines.append(f"- box {i}: [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")
    refs = [f"class {cid} at [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]"
            for cid, rbs in sorted(ref_by_class.items()) for x1, y1, x2, y2 in rbs]
    if refs:
        lines.append("Old-detector reference (may be wrong; context only, judge "
                     "from the image): " + "; ".join(refs) + ".")
    else:
        lines.append("Old-detector reference: none (it found nothing; context "
                     "only, it may have missed a cat).")
    lines.append("Rules: assign each box the id of the cat it shows, or -1 when "
                 "it shows no cat or you are unsure. "
                 'Format: {"boxes": [{"box": 0, "id": 1}]}')
    return "\n".join(lines)


def parse_spark_json(text, num_boxes, valid_ids):
    """Parse Spark's reply into [class_id or -1] per box. Raises ValueError."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
        entries = data["boxes"]
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"not a boxes-JSON object: {e}")
    ids = [-1] * num_boxes
    try:
        for entry in entries:
            idx, cid = int(entry["box"]), int(entry["id"])
            if 0 <= idx < num_boxes and (cid in valid_ids or cid == -1):
                ids[idx] = cid
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"bad boxes entry: {e}")
    return ids


def box_iou(a, b):
    """IoU of two (x1, y1, x2, y2) pixel boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def dedupe_boxes(boxes, iou_thresh=DEDUP_IOU):
    """Greedy union: keep a box only if IoU < thresh with everything kept."""
    kept = []
    for box in boxes:
        if all(box_iou(box, other) < iou_thresh for other in kept):
            kept.append(box)
    return kept


def parse_sam_boxes(output_text):
    """Parse SAM output_text into [{obj_id, x1, y1, x2, y2, w, h, conf}].

    Grammar (one line per frame, records comma-separated):
        <Nf>id<|box;...|>><|mask;...|>,id<|box;...|>><|mask;...|>
    Empty text means zero matches, which is a valid outcome. Malformed box
    tokens are skipped, never fatal.
    """
    boxes = []
    for line in (output_text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        for match in BOX_RE.finditer(line):
            fields = {}
            for part in match.group(1).split(";"):
                if "=" in part:
                    key, value = part.split("=", 1)
                    fields[key.strip()] = value.strip()
            try:
                box = {
                    "x1": int(fields["x1"]), "y1": int(fields["y1"]),
                    "x2": int(fields["x2"]), "y2": int(fields["y2"]),
                    "w": int(fields["w"]), "h": int(fields["h"]),
                }
            except (KeyError, ValueError):
                continue
            if "c" in fields:
                try:
                    box["conf"] = float(fields["c"])
                except ValueError:
                    pass
            # object id = integer just before this box token (after <Nf> or a comma)
            prefix = line[:match.start()]
            id_match = re.search(r"(?:^|>|,)(\d+)$", prefix)
            box["obj_id"] = int(id_match.group(1)) if id_match else None
            boxes.append(box)
    return boxes


def output_text_of(response):
    """Concatenate every output_text part of a Responses API response."""
    texts = []
    for item in (response.get("output") or []):
        for part in (item.get("content") or []):
            if part.get("type") == "output_text" and part.get("text"):
                texts.append(part["text"])
    if not texts and isinstance(response.get("output_text"), str):
        texts.append(response["output_text"])
    return "\n".join(texts)


def _image_block(image_bytes):
    return {"type": "input_image", "image_url": (
        "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii"))}


def build_sam_payload(model, prompt, image_bytes):
    """Request body for one SAM segmentation call.

    SAM has no boxes-only mode: every record always pairs one box with one
    mask, and billing is per image either way, so we request the compact
    one_bit mask encoding (smaller payload, faster transfer) and keep only
    the boxes for the YOLO labels.
    """
    return {
        "model": model,
        "metadata": {"mask_encoding": "one_bit"},
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": prompt},
                        _image_block(image_bytes)],
        }],
    }


def build_spark_payload(model, prompt, image_bytes):
    """Request body for one Spark classification call (no SAM metadata)."""
    return {
        "model": model,
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": prompt},
                        _image_block(image_bytes)],
        }],
    }


def call_responses(base_url, api_key, payload, timeout=240):
    """POST a payload to the Responses API. Returns (output_text, usage)."""
    req = urllib.request.Request(
        base_url + "/responses", json.dumps(payload).encode("utf-8"),
        {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"API HTTP {e.code}: {detail}")
    return output_text_of(body), (body.get("usage") or {})


def is_transient_api_error(exc: Exception) -> bool:
    """True for API failures worth retrying: HTTP 429/5xx or transport timeouts.

    ``call_responses`` converts HTTP errors to ``RuntimeError("API HTTP {code}: ...")``;
    anything reaching us as a raw ``URLError`` is transport-level (timeout, DNS,
    refused). Client errors (400, 401, 404, ...) fail fast — retrying is pointless.
    """
    if re.match(r"API HTTP (429|5\d\d)\b", str(exc)):
        return True
    return isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError))


def call_responses_with_retry(base_url, api_key, payload, timeout=240, attempts=3):
    """``call_responses`` with backoff retries on transient failures.

    One 504/timeout used to fail the whole image (108 lost labels in a single
    session); transient errors now retry twice (2 s, 4 s) before giving up.
    """
    for attempt in range(attempts):
        try:
            return call_responses(base_url, api_key, payload, timeout)
        except Exception as exc:
            if not is_transient_api_error(exc) or attempt + 1 >= attempts:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")  # pragma: no cover


def load_spark_settings(model_config=None):
    """Return (model, base_url, api_key, pricing) for the agent role."""
    cfg_path = model_config or os.getenv("MODEL_CONFIG") or DEFAULT_MODELS
    with open(cfg_path, encoding="utf-8") as f:
        data = json.load(f)
    roles = data.get("roles", {})
    role_cfg = dict(roles.get("agent") or {})
    model = role_cfg.get("model") or SPARK_FALLBACK_MODEL
    base_url = (role_cfg.get("base_url") or data.get("base_url", "")).rstrip("/")
    api_key = os.getenv(role_cfg.get("api_key_env")
                        or data.get("api_key_env", "MODEL_API_KEY"), "")
    pricing = ((data.get("pricing_usd_per_1m_tokens") or {}).get(model)
               or dict(SPARK_FALLBACK_PRICING))
    return (str(model), base_url, api_key.strip().strip("\"'"),
            {"input": float(pricing.get("input", 1.25)),
             "output": float(pricing.get("output", 4.25))})


def resolve_bounds(hour, start, end):
    """Validate hour/--start/--end; return (start_key, end_key) stamp bounds.

    Bounds are YYYY-MM-DD-HH-MM (end minute fully included) or a bare hour.
    Unlike the hour fetcher, a range may span hours: every retrieved hour
    folder overlapped by [start_key, end_key) is processed.
    """
    for label, b in (("--start", start), ("--end", end)):
        if b is not None and not (MINUTE_RE.match(b) or HOUR_RE.match(b)):
            raise ValueError(f"{label} must look like 2026-10-28-16-33, got {b!r}")
    if hour is not None and not HOUR_RE.match(hour):
        raise ValueError(f"hour must look like 2026-10-28-16, got {hour!r}")
    anchor = hour or start or end
    if anchor is None:
        raise ValueError("pass an hour (2026-10-28-16) or --start/--end")
    if hour is not None:
        for label, b in (("--start", start), ("--end", end)):
            if b is not None and not b.startswith(hour):
                raise ValueError(f"{label} {b} is outside hour {hour}")
    start_key = start or (hour + "-00" if hour else None)
    if end:
        if len(end) == 13:  # bare hour: include the whole hour
            end_dt = datetime.datetime.strptime(end, "%Y-%m-%d-%H") + datetime.timedelta(hours=1)
        else:
            end_dt = datetime.datetime.strptime(end, "%Y-%m-%d-%H-%M") + datetime.timedelta(minutes=1)
        end_key = end_dt.strftime("%Y-%m-%d-%H-%M")
    elif hour:
        end_key = (datetime.datetime.strptime(hour, "%Y-%m-%d-%H")
                   + datetime.timedelta(hours=1)).strftime("%Y-%m-%d-%H")
    else:
        end_key = None
    if start_key is None:
        start_key = anchor[:13]  # open start: from the anchor hour's first frame
    if end_key is not None and end_key <= start_key:
        raise ValueError(f"--end {end} is before --start {start}")
    return start_key, end_key


def discover_images(retrieved, start_key, end_key, out_dirname="muse_labels"):
    """[(hour, images_dir, labels_dir, out_dir, indexed_name, stamp)] in time order."""
    found = []
    for hour in sorted(os.listdir(retrieved)):
        if not HOUR_RE.match(hour):
            continue
        if hour < start_key[:13]:
            continue
        if end_key is not None and hour > end_key[:13]:
            continue
        images_dir = os.path.join(retrieved, hour, "images")
        if not os.path.isdir(images_dir):
            continue
        labels_dir = os.path.join(retrieved, hour, "labels")
        out_dir = os.path.join(retrieved, hour, out_dirname)
        for name in sorted(os.listdir(images_dir)):
            m = PREFIXED_RE.match(name)
            if not m:
                continue
            stamp = m.group(2)[:-4]  # strip .jpg -> comparable timestamp
            if stamp < start_key:
                continue
            if end_key is not None and stamp >= end_key:
                continue
            found.append((hour, images_dir, labels_dir, out_dir, name, stamp))
    found.sort(key=lambda r: r[5])
    return found


def load_hour_classes(hour_dir, cats):
    """Map a retrieved hour folder to queries: [{class_id, name, cat}].

    Prefers classes_map.json (tracking id -> stable class index); falls back
    to classes.txt name matching against the cat profiles.
    """
    by_tracking = {}
    for cat in cats:
        for tid in cat["tracking_ids"]:
            by_tracking.setdefault(tid, cat)
    map_path = os.path.join(hour_dir, "classes_map.json")
    if os.path.isfile(map_path):
        with open(map_path, encoding="utf-8") as f:
            mapping = json.load(f)
        queries = []
        for class_id, tid in enumerate(mapping.get("order", [])):
            cat = by_tracking.get(int(tid))
            if cat is not None:
                queries.append({"class_id": class_id, "name": cat["name"], "cat": cat})
        if queries:
            return queries
    names_path = os.path.join(hour_dir, "classes.txt")
    queries = []
    if os.path.isfile(names_path):
        with open(names_path, encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        for class_id, raw in enumerate(names):
            stem = raw.split("-")[0].strip().lower()
            for cat in cats:
                if cat["name"].lower() == stem:
                    queries.append({"class_id": class_id, "name": cat["name"], "cat": cat})
                    break
    return queries


def read_reference_labels(labels_dir, image_name, class_id, img_w, img_h):
    """Reference YOLO boxes of one class as [(x1,y1,x2,y2,None)] pixels.

    The existing labels carry no confidence column, so conf is always None.
    Missing file means no reference (valid: the old detector saw nothing).
    """
    stem = os.path.splitext(image_name)[0] + ".txt"
    path = os.path.join(labels_dir, stem)
    boxes = []
    if not os.path.isfile(path):
        return boxes
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
                cx, cy, w, h = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            if cid != class_id:
                continue
            boxes.append(((cx - w / 2) * img_w, (cy - h / 2) * img_h,
                          (cx + w / 2) * img_w, (cy + h / 2) * img_h, None))
    return boxes


def read_all_reference_labels(labels_dir, image_name, class_ids, img_w, img_h):
    """{class_id: [(x1, y1, x2, y2)]} pixels for every known class."""
    return {cid: [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in
                  read_reference_labels(labels_dir, image_name, cid, img_w, img_h)]
            for cid in class_ids}


def sam_boxes_to_yolo(boxes, img_w, img_h):
    """Convert parsed SAM boxes to [(x1,y1,x2,y2) pixels], dropping junk.

    Boxes are validated against SAM's own reported coordinate space, then
    rescaled to JPEG pixels when the aspect ratio matches (SAM may segment
    a resized copy); boxes from a mismatched aspect space are dropped.
    """
    kept = []
    for b in boxes:
        space_w, space_h = b.get("w") or img_w, b.get("h") or img_h
        yolo = to_yolo(b["x1"], b["y1"], b["x2"], b["y2"], space_w, space_h)
        if yolo is None:
            continue
        if space_w <= 0 or space_h <= 0 or img_w <= 0 or img_h <= 0:
            continue
        if abs((space_w / space_h) / (img_w / img_h) - 1.0) > 0.03:
            continue  # different framing: never guess across it
        cx, cy, w, h = yolo
        kept.append(((cx - w / 2) * img_w, (cy - h / 2) * img_h,
                     (cx + w / 2) * img_w, (cy + h / 2) * img_h))
    return kept


def estimate_cost_usd(sam_calls, input_tokens=0, output_tokens=0, spark_pricing=None):
    """Total USD: SAM per-image billing + Spark per-token billing."""
    pricing = spark_pricing or SPARK_FALLBACK_PRICING
    return (sam_calls * SAM_USD_PER_1000_IMAGES / 1000.0
            + input_tokens / 1_000_000 * pricing["input"]
            + output_tokens / 1_000_000 * pricing["output"])


def format_duration(seconds):
    """32 -> '32s', 95 -> '1m 35s', 3725 -> '1h 2m'."""
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


_USAGE_SHORT = {"input_tokens": "input", "output_tokens": "output", "total_tokens": "total"}


def _tally_usage(result, usage):
    for long_key, short_key in _USAGE_SHORT.items():
        if isinstance(usage.get(long_key), int):
            result["usage"][short_key] = result["usage"].get(short_key, 0) + usage[long_key]


def process_one(job, ctx):
    """Label one image: SAM proposes, Spark classifies. Never raises."""
    hour, images_dir, labels_dir, out_dir, name, stamp, classes = job
    started = time.monotonic()
    result = {"hour": hour, "image": name, "stamp": stamp, "status": "ok",
              "sam_calls": 0, "spark_calls": 0, "proposed": 0,
              "boxes": {}, "dropped": 0, "rejected": 0, "seconds": 0.0,
              "usage": {}, "error": None}
    try:
        img_path = os.path.join(images_dir, name)
        with open(img_path, "rb") as f:
            image_bytes = f.read()
        size = jpeg_size(img_path)
        if size is None:
            raise RuntimeError("unreadable JPEG header")
        img_w, img_h = size
        # Stage 1: propose regions with the union of grounding concepts.
        proposed = []
        for concept in SAM_CONCEPTS:
            output_text, usage = call_responses_with_retry(
                ctx["sam_base_url"], ctx["sam_api_key"],
                build_sam_payload(ctx["sam_model"], build_sam_prompt(concept),
                                  image_bytes))
            result["sam_calls"] += 1
            _tally_usage(result, usage)
            parsed = parse_sam_boxes(output_text)
            kept = sam_boxes_to_yolo(parsed, img_w, img_h)
            result["dropped"] += len(parsed) - len(kept)
            proposed.extend(kept)
        proposed = dedupe_boxes(proposed)
        result["proposed"] = len(proposed)
        out_lines = []
        if proposed:
            # Stage 2: one Spark call identifies every proposed box.
            valid_ids = [c["class_id"] for c in classes]
            ref = read_all_reference_labels(
                labels_dir, name, valid_ids, img_w, img_h)
            prompt = build_spark_prompt(classes, proposed, ref, img_w, img_h)
            with ctx["first_spark_lock"]:
                if ctx["first_spark_prompt"] is None:
                    ctx["first_spark_prompt"] = (f"{hour}/{name}", prompt)
            output_text, usage = call_responses_with_retry(
                ctx["spark_base_url"], ctx["spark_api_key"],
                build_spark_payload(ctx["spark_model"], prompt, image_bytes))
            result["spark_calls"] += 1
            _tally_usage(result, usage)
            try:
                ids = parse_spark_json(output_text, len(proposed), valid_ids)
            except ValueError:
                # one retry: the model sometimes wraps JSON in prose
                output_text, usage = call_responses_with_retry(
                    ctx["spark_base_url"], ctx["spark_api_key"],
                    build_spark_payload(ctx["spark_model"], prompt, image_bytes))
                result["spark_calls"] += 1
                _tally_usage(result, usage)
                ids = parse_spark_json(output_text, len(proposed), valid_ids)
            id_to_name = {c["class_id"]: c["name"] for c in classes}
            for (x1, y1, x2, y2), cid in zip(proposed, ids):
                if cid < 0:
                    result["rejected"] += 1
                    continue
                yolo = to_yolo(x1, y1, x2, y2, img_w, img_h)
                if yolo is None:
                    result["dropped"] += 1
                    continue
                cx, cy, w, h = yolo
                out_lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                result["boxes"][id_to_name[cid]] = \
                    result["boxes"].get(id_to_name[cid], 0) + 1
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, os.path.splitext(name)[0] + ".txt"),
                  "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + ("\n" if out_lines else ""))
    except Exception as e:  # per-image failure must not stop the session
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"[:300]
    result["seconds"] = time.monotonic() - started
    return result


def write_session_report(path_md, path_json, session):
    """Write the Markdown + JSON session report. Returns (path_md, path_json)."""
    s = session
    tokens = s["tokens"]
    md = [
        f"# SAM grounding-truth session — {s['session_id']}",
        "",
        f"- Started: {s['started']}  ·  finished: {s['finished']}  ·  "
        f"wall time: {s['wall_seconds']:.1f}s",
        f"- Range: {s['start_key']} .. {s['end_key'] or 'open'}"
        f"{' (DRY RUN - no API calls, no labels written)' if s['dry_run'] else ''}",
        f"- Hour folders: {', '.join(s['hours']) or 'none'}",
        f"- Models: {s['sam_model']} (propose) + {s['spark_model']} (classify)  ·  "
        f"workers: {s['workers']}  ·  limit: {s['limit'] or 'none'}",
        f"- Pricing: SAM ${SAM_USD_PER_1000_IMAGES:.2f}/1K images, Spark "
        f"${s['spark_pricing']['input']:.2f}/${s['spark_pricing']['output']:.2f} "
        f"per 1M in/out tokens ([official pricing]({SAM_PRICING_URL}), Standard)",
        "",
        "## Totals",
        "",
        f"- Images in range: {s['images_in_range']}  ·  processed: {s['processed']}  ·  "
        f"skipped (already labeled): {s['skipped']}  ·  failed: {s['failed']}",
        f"- SAM requests: {s['sam_calls']} (concepts {', '.join(s['concepts'])}, "
        f"unioned per image)  ·  Spark calls: {s['spark_calls']} "
        f"(one per image with proposed boxes)",
        f"- Regions proposed: {s['proposed']}  ·  boxes written: {s['boxes_total']} "
        f"({', '.join(f'{k}: {v}' for k, v in sorted(s['boxes_by_class'].items())) or 'none'})"
        f"  ·  rejected by Spark (-1/unsure): {s['rejected']}"
        f"  ·  dropped (degenerate/mismatched): {s['dropped']}",
        f"- Tokens: {tokens['total']} total "
        f"({tokens['input']} in / {tokens['output']} out, all from Spark; "
        f"SAM reports 0 tokens)",
        f"- Estimated cost: ${s['cost_usd']:.4f} "
        f"(SAM ${s['sam_cost_usd']:.4f} + Spark ${s['spark_cost_usd']:.4f}; "
        f"${s['cost_usd'] / max(s['processed'], 1):.4f} per processed image)",
        "",
        "## Classes",
        "",
    ]
    for q in s["queries"]:
        md.append(f"- class {q['class_id']} = {q['name']}: {q['concept']}")
    md += [
        "",
        "## Full prompts of the first requests",
        "",
        f"SAM image: {s['first_sam_image'] or 'n/a (no images in range)'}",
        "",
        "```text",
        s["first_sam_prompt"] or "(none)",
        "```",
        "",
        f"Spark image: {s['first_spark_image'] or 'n/a'}"
        f"{' (illustrated with reference boxes; dry run makes no SAM calls)' if s['dry_run'] else ''}",
        "",
        "```text",
        s["first_spark_prompt"] or "(none)",
        "```",
        "",
        "## Per-hour breakdown",
        "",
        "| Hour | In range | Processed | Skipped | Failed | Boxes |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for hour, row in sorted(s["per_hour"].items()):
        md.append(f"| {hour} | {row['in_range']} | {row['processed']} | "
                  f"{row['skipped']} | {row['failed']} | {row['boxes']} |")
    md += ["", "## Failures", ""]
    if s["failures"]:
        for f in s["failures"][:50]:
            md.append(f"- {f['hour']}/{f['image']}: {f['error']}")
        if len(s["failures"]) > 50:
            md.append(f"- ... and {len(s['failures']) - 50} more (see JSON report)")
    else:
        md.append("None.")
    md += [
        "",
        "## Notes",
        "",
        "- An empty muse_labels .txt means the image was processed and no box "
        "survived (nothing proposed, or everything rejected); it is not an error.",
        "- The labels/ boxes are Spark prompt context only (marked as possibly "
        "wrong); written geometry comes from SAM, written identity from Spark.",
        "- SAM boxes from a mismatched aspect space are dropped, never rescaled.",
        "",
    ]
    os.makedirs(os.path.dirname(path_md), exist_ok=True)
    with open(path_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)
    return path_md, path_json


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("hour", nargs="?", default=None,
                        help="YYYY-MM-DD-HH, e.g. 2026-10-28-16 (omit when --start/--end are given)")
    parser.add_argument("--start", default=None, help="label from this minute (YYYY-MM-DD-HH-MM)")
    parser.add_argument("--end", default=None, help="label through this minute, inclusive (YYYY-MM-DD-HH-MM)")
    parser.add_argument("--workers", type=int, default=10, help="parallel SAM requests (default 10)")
    parser.add_argument("--limit", type=int, default=None, help="label only the first N images in range (debug)")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve files, print the first prompt and cost estimate; no API calls, no labels")
    parser.add_argument("--overwrite", action="store_true",
                        help="relabel images that already have a muse_labels/ file (default skips them)")
    parser.add_argument("--retrieved", default=None, help="retrieved_images root (default image_server/retrieved_images)")
    parser.add_argument("--cats", default=None, help="cat profiles YAML (default agent/agent_data/config/cats.yaml)")
    parser.add_argument("--models", default=None, help="models.json for model names, base URLs and pricing (default agent/supervisor_agent/models.json)")
    parser.add_argument("--report-dir", default=None, help="session report folder (default grounding_truth/reports)")
    args = parser.parse_args()

    try:
        start_key, end_key = resolve_bounds(args.hour, args.start, args.end)
    except ValueError as e:
        sys.exit(str(e))
    if args.workers < 1:
        sys.exit("--workers must be >= 1")
    load_env()
    sam_model, sam_base_url, sam_api_key = load_sam_settings(args.models)
    spark_model, spark_base_url, spark_api_key, spark_pricing = load_spark_settings(args.models)
    if not sam_base_url or not spark_base_url:
        sys.exit("models.json must give a base_url for the segmentation and agent roles")
    if (not sam_api_key or not spark_api_key) and not args.dry_run:
        sys.exit("set MODEL_API_KEY in the repo .env (or pass --dry-run to preview for free)")

    retrieved = args.retrieved or DEFAULT_RETRIEVED
    cats = load_cats(args.cats)
    targets = discover_images(retrieved, start_key, end_key)
    if args.limit is not None:
        targets = targets[:max(args.limit, 0)]
    logger.info(f"{len(targets)} images in range {start_key} .. {end_key or 'open'}")

    # per-hour class queries (same cats everywhere in practice; resolved per folder)
    queries_by_hour = {}
    for hour, *_ in targets:
        if hour not in queries_by_hour:
            queries_by_hour[hour] = load_hour_classes(os.path.join(retrieved, hour), cats)
            if not queries_by_hour[hour]:
                logger.warning(f"{hour}: no classes match the cat profiles, folder skipped")
    jobs, skipped = [], 0
    per_hour = {}
    for hour, images_dir, labels_dir, out_dir, name, stamp in targets:
        row = per_hour.setdefault(hour, {"in_range": 0, "processed": 0, "skipped": 0, "failed": 0, "boxes": 0})
        row["in_range"] += 1
        queries = queries_by_hour.get(hour) or []
        if not queries:
            continue
        label_path = os.path.join(out_dir, os.path.splitext(name)[0] + ".txt")
        if os.path.isfile(label_path) and not args.overwrite and not args.dry_run:
            skipped += 1
            row["skipped"] += 1
            continue
        jobs.append((hour, images_dir, labels_dir, out_dir, name, stamp, queries))

    # full prompts of the first requests, printed before any API call.
    # The SAM prompt is deterministic; the Spark prompt needs proposed boxes,
    # so dry runs illustrate it with the first image's reference boxes.
    first_sam_prompt, first_sam_image = "", None
    first_spark_prompt, first_spark_image = "", None
    if jobs:
        hour, images_dir, labels_dir, _out, name, _stamp, classes = jobs[0]
        first_sam_image = f"{hour}/{name}"
        first_sam_prompt = "\n".join(
            f"concept {i}: {build_sam_prompt(c)}"
            for i, c in enumerate(SAM_CONCEPTS))
        print("=" * 72)
        print(f"FIRST SAM PROMPTS ({first_sam_image}, unioned):")
        print("=" * 72)
        print(first_sam_prompt)
        print("=" * 72)
        if args.dry_run:
            size = jpeg_size(os.path.join(images_dir, name)) or (0, 0)
            valid_ids = [c["class_id"] for c in classes]
            ref = read_all_reference_labels(labels_dir, name, valid_ids, *size)
            demo_boxes = [b for rbs in ref.values() for b in rbs] or [(0, 0, 1, 1)]
            first_spark_prompt = build_spark_prompt(classes, demo_boxes, ref, *size)
            first_spark_image = f"{hour}/{name} (reference boxes as illustration)"
            print("=" * 72)
            print(f"FIRST SPARK PROMPT ({first_spark_image}):")
            print("=" * 72)
            print(first_spark_prompt)
            print("=" * 72)

    if args.dry_run:
        est_sam = len(targets) * len(SAM_CONCEPTS)
        print(f"DRY RUN: {len(targets)} images, {est_sam} SAM requests planned "
              f"(~${est_sam * SAM_USD_PER_1000_IMAGES / 1000.0:.4f}) plus one Spark "
              f"call per image with proposed boxes "
              f"(~$0.005 each at ${spark_pricing['input']:.2f}/"
              f"${spark_pricing['output']:.2f} per 1M tokens; {SAM_PRICING_URL}). "
              f"Time estimate unavailable without API calls.")

    session_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    started = datetime.datetime.now()
    wall_start = time.monotonic()
    results = []
    ctx = {"sam_base_url": sam_base_url, "sam_api_key": sam_api_key,
           "sam_model": sam_model, "spark_base_url": spark_base_url,
           "spark_api_key": spark_api_key, "spark_model": spark_model,
           "first_spark_lock": threading.Lock(), "first_spark_prompt": None}
    if jobs and not args.dry_run:
        # pilot: first image runs synchronously to calibrate the estimate
        pilot = process_one(jobs.pop(0), ctx)
        results.append(pilot)
        total = len(results) + len(jobs)
        per_call = pilot["usage"] or {"input": 850, "output": 1000}
        per_call_src = "pilot" if pilot["usage"] else "typical"
        est_sam = total * len(SAM_CONCEPTS)
        est_cost = estimate_cost_usd(est_sam, total * per_call.get("input", 0),
                                     total * per_call.get("output", 0), spark_pricing)
        est_wall = pilot["seconds"] * (1 + (total - 1) / max(args.workers, 1))
        print(f"ESTIMATE for {total} images (pilot: {pilot['seconds']:.1f}s, "
              f"{per_call.get('input', 0)}/{per_call.get('output', 0)} in/out tokens, "
              f"{per_call_src}): ~{est_sam} SAM + up to {total} Spark calls, "
              f"~${est_cost:.4f}, ~{format_duration(est_wall)} wall at {args.workers} "
              f"workers (rough: assumes linear scaling; Spark assumes every image "
              f"needs classification).")
        done = 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_one, job, ctx): job for job in jobs}
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())
                done += 1
                if done % 25 == 0 or done == total:
                    logger.info(f"{done}/{total} images done")
    wall_seconds = time.monotonic() - wall_start
    if ctx["first_spark_prompt"] is not None and not args.dry_run:
        first_spark_image, first_spark_prompt = ctx["first_spark_prompt"]
        print("=" * 72)
        print(f"FIRST SPARK PROMPT ({first_spark_image}):")
        print("=" * 72)
        print(first_spark_prompt)
        print("=" * 72)

    sam_calls = sum(r["sam_calls"] for r in results)
    spark_calls = sum(r["spark_calls"] for r in results)
    proposed = sum(r["proposed"] for r in results)
    tokens = {"input": 0, "output": 0, "total": 0}
    boxes_by_class, boxes_total, dropped, rejected, failed = {}, 0, 0, 0, []
    for r in results:
        for key in tokens:
            tokens[key] += r["usage"].get(key, 0)
        for name, count in r["boxes"].items():
            boxes_by_class[name] = boxes_by_class.get(name, 0) + count
            boxes_total += count
            per_hour[r["hour"]]["boxes"] += count
        dropped += r["dropped"]
        rejected += r["rejected"]
        if r["status"] == "error":
            failed.append({"hour": r["hour"], "image": r["image"], "error": r["error"]})
            per_hour[r["hour"]]["failed"] += 1
        else:
            per_hour[r["hour"]]["processed"] += 1
    processed = len(results)
    sam_cost = sam_calls * SAM_USD_PER_1000_IMAGES / 1000.0
    spark_cost = (tokens["input"] / 1_000_000 * spark_pricing["input"]
                  + tokens["output"] / 1_000_000 * spark_pricing["output"])
    cost_usd = sam_cost + spark_cost

    # copy classes.txt beside the new labels for labelImg (per touched hour)
    if not args.dry_run:
        for hour in {r["hour"] for r in results if r["status"] == "ok"}:
            src = os.path.join(retrieved, hour, "classes.txt")
            dst = os.path.join(retrieved, hour, "muse_labels", "classes.txt")
            if os.path.isfile(src) and not os.path.isfile(dst):
                try:
                    with open(src, "rb") as f_in, open(dst, "wb") as f_out:
                        f_out.write(f_in.read())
                except OSError as e:
                    logger.warning(f"{hour}: cannot copy classes.txt: {e}")

    queries_all = []
    seen = set()
    for hour in sorted(queries_by_hour):
        for q in queries_by_hour[hour]:
            key = (q["class_id"], q["name"])
            if key not in seen:
                seen.add(key)
                queries_all.append({"class_id": q["class_id"], "name": q["name"],
                                    "concept": short_concept(q["cat"])})
    session = {
        "session_id": session_id,
        "started": started.strftime("%Y-%m-%d %H:%M:%S"),
        "finished": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "wall_seconds": round(wall_seconds, 1),
        "dry_run": bool(args.dry_run),
        "start_key": start_key, "end_key": end_key,
        "hours": sorted(queries_by_hour),
        "sam_model": sam_model, "spark_model": spark_model,
        "spark_pricing": spark_pricing, "concepts": list(SAM_CONCEPTS),
        "workers": args.workers, "limit": args.limit,
        "images_in_range": len(targets), "processed": processed,
        "skipped": skipped, "failed": len(failed),
        "sam_calls": sam_calls, "spark_calls": spark_calls, "proposed": proposed,
        "boxes_total": boxes_total, "boxes_by_class": boxes_by_class,
        "dropped": dropped, "rejected": rejected,
        "tokens": tokens, "cost_usd": round(cost_usd, 4),
        "sam_cost_usd": round(sam_cost, 4), "spark_cost_usd": round(spark_cost, 4),
        "pricing_usd_per_1000_images": SAM_USD_PER_1000_IMAGES,
        "pricing_url": SAM_PRICING_URL,
        "queries": queries_all,
        "first_sam_image": first_sam_image, "first_sam_prompt": first_sam_prompt,
        "first_spark_image": first_spark_image, "first_spark_prompt": first_spark_prompt,
        "per_hour": per_hour, "failures": failed,
        "results": sorted(results, key=lambda r: (r["hour"], r["image"])),
    }
    report_dir = args.report_dir or REPORTS_DIR
    md_path, json_path = write_session_report(
        os.path.join(report_dir, f"session-{session_id}.md"),
        os.path.join(report_dir, f"session-{session_id}.json"), session)

    print(f"Done: {processed} images processed, {skipped} skipped, {len(failed)} failed, "
          f"{sam_calls} SAM + {spark_calls} Spark calls, {proposed} proposed, "
          f"{boxes_total} boxes ({rejected} rejected), "
          f"{tokens['total']} tokens ({tokens['input']} in / {tokens['output']} out), "
          f"{wall_seconds:.1f}s wall, estimated cost ${cost_usd:.4f} "
          f"(SAM ${sam_cost:.4f} + Spark ${spark_cost:.4f}).")
    print(f"Session report: {md_path} (+ .json)")


if __name__ == "__main__":
    main()
