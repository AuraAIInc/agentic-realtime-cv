"""Evaluate one tracking run: label sets + raw/revised/validated vs GT, cost, time.

Ground truth: <hour>/labels/*.txt by default (validated YOLO labels).
--gt-source selects the label directory: labels | muse | spark.

Positional input is a single tracking-run directory (absolute path fine):

    python evaluation/evaluate.py /path/to/output/tracking_runs/<run_name>

The hour directory resolves automatically: labels-mode runs carry it in
run.json args; replay/live runs derive it from the run window
(<data-root>/YYYY-MM-DD-HH). --hour overrides. --data-root defaults to
this repo's image_server/retrieved_images tree.

Stages scored (IoU 0.5 and 0.75 unless --iou says otherwise):

- muse / spark: <hour>/muse_labels (SAM 3.1) and <hour>/spark_labels
  (Spark 1.3) against GT, when present. No cost attached (labeled offline).
- raw: run stream.jsonl (raw detector replay, incl. DB detections).
- revised: run revised.jsonl minus tombstones and interpolated guesses.
- validated: run validated.jsonl minus interpolated guesses.

Frames join on timestamp; only run frames covered by GT are scored.
The report opens with the headline (raw -> validated improvement, wall
time, supervisor $ cost), then the run's detection charts, then details
per stage, per class, and cost.

Outputs (into --out, default <repo>/output/evaluation_runs/):
<run>_eval.md, <run>_metrics.json, <run>_{original,revised,validated}.png
copied from the run dir when present, plus <run>_gt.png rendered from the
ground-truth labels over the evaluated frames — all embedded in the report.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import re
import shutil
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Repo root, so the GT chart render can lazily import agent.tracking.charts.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import Box, evaluate_frames  # noqa: E402

GT_SOURCES = {"muse": "muse_labels", "spark": "spark_labels", "labels": "labels"}
LABEL_STAGES = (("muse", "muse_labels"), ("spark", "spark_labels"))
PIPE_STAGES = (("raw", "stream.jsonl"), ("revised", "revised.jsonl"), ("validated", "validated.jsonl"))
# Detection strip charts written by the tracking run (agent/tracking/charts.py).
CHART_FILES = (
    ("original", "original.png", "Original detections"),
    ("revised", "revised.png", "Revised detections"),
    ("validated", "validated.png", "Validated detections"),
)
# Rendered by evaluate.py itself (not copied): ground truth over eval frames.
GT_CHART_KEY = "gt"
GT_CHART_TITLE = "Ground truth detections"
REPORT_CHART_ORDER = tuple([k for k, _f, _t in CHART_FILES] + [GT_CHART_KEY])
CHART_TITLES = {k: t for k, _f, t in CHART_FILES} | {GT_CHART_KEY: GT_CHART_TITLE}
MISSING = "—"
STEM_RE = re.compile(r"^\d+_(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d+)$")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_ROOT = os.path.join(_REPO, "image_server", "retrieved_images")
DEFAULT_OUT_DIR = os.path.join(_REPO, "output", "evaluation_runs")


def stem_to_time(stem: str) -> str | None:
    """'5502_2026-10-28-16-33-00-304643' -> '2026-10-28T16:33:00.304643'."""
    m = STEM_RE.match(stem)
    if not m:
        return None
    y, mo, d, h, mi, s, us = m.groups()
    return f"{y}-{mo}-{d}T{h}:{mi}:{s}.{us}"


def stem_to_datetime(stem: str) -> datetime.datetime | None:
    """Stem to datetime (fractional seconds zero-padded to microseconds)."""
    m = STEM_RE.match(stem)
    if not m:
        return None
    y, mo, d, h, mi, s, us = m.groups()
    try:
        return datetime.datetime(int(y), int(mo), int(d), int(h), int(mi), int(s), int(us.ljust(6, "0")[:6]))
    except ValueError:
        return None


def parse_time(value) -> datetime.datetime | None:
    """Run timestamps (datetime or ISO string) to datetime, else None."""
    if isinstance(value, datetime.datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def jpeg_size(path: str) -> tuple | None:
    """Read (width, height) from a JPEG file without third-party libs."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    i = 2
    while i + 8 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3):
            h, w = struct.unpack(">HH", data[i + 5 : i + 9])
            return w, h
        if marker == 0xFF:
            i += 1
            continue
        i += 2 + struct.unpack(">H", data[i + 2 : i + 4])[0]
    return None


def load_yolo_labels(label_dir: str, order: list, img_w: int, img_h: int) -> dict:
    """Return {datetime: [Box, ...]} with YOLO ids mapped via order."""
    gt: dict = {}
    for path in sorted(glob.glob(os.path.join(label_dir, "*.txt"))):
        ts = stem_to_datetime(os.path.splitext(os.path.basename(path))[0])
        if ts is None:  # e.g. classes.txt
            continue
        boxes = []
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cid, cx, cy, w, h = float(parts[0]), *map(float, parts[1:5])
                except ValueError:
                    continue
                if int(cid) < 0 or int(cid) >= len(order):
                    continue
                cls = order[int(cid)]
                x1 = (cx - w / 2) * img_w
                y1 = (cy - h / 2) * img_h
                x2 = (cx + w / 2) * img_w
                y2 = (cy + h / 2) * img_h
                boxes.append(Box(x1, y1, x2, y2, cls, 1.0))
        gt[ts] = boxes
    return gt


def load_jsonl(path: str) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_stage(run_dir: str, fname: str) -> dict:
    """Return {datetime: [Box, ...]} for one prediction stage.

    Tombstones and interpolated guesses (orig_cat_id None) never count as
    predictions; raw stream rows carry no orig_cat_id key and all count.
    Rows without a usable box are skipped.
    """
    path = os.path.join(run_dir, fname)
    preds: dict = {}
    if not os.path.exists(path):
        return preds
    for row in load_jsonl(path):
        if row.get("deleted"):
            continue
        if "orig_cat_id" in row and row["orig_cat_id"] is None:
            continue
        box = [row.get(k) for k in ("box_x1", "box_y1", "box_x2", "box_y2")]
        if not all(isinstance(v, (int, float)) for v in box):
            continue
        ts = parse_time(row.get("create_time"))
        if ts is None or row.get("cat_id") is None:
            continue
        preds.setdefault(ts, []).append(
            Box(box[0], box[1], box[2], box[3], int(row["cat_id"]), row.get("conf") or 0.0)
        )
    return preds


def repo_root_of_run(run_dir: str) -> str:
    """<repo>/output/tracking_runs/<run> -> <repo> (best effort)."""
    parts = os.path.normpath(os.path.abspath(run_dir)).split(os.sep)
    if len(parts) >= 4 and parts[-2] == "tracking_runs" and parts[-3] == "output":
        return os.sep.join(parts[:-3]) or os.sep
    return os.getcwd()


def resolve_hour_dir(run_dir: str, meta: dict, hour_override: str | None, data_root: str) -> str:
    """Locate the retrieved-images hour dir for a run.

    Precedence: --hour, labels-mode run.json hour_dir (resolved against the
    run's own repo when relative), else the run window's hour stamp under
    data_root.
    """
    if hour_override:
        return hour_override
    args = meta.get("args") or {}
    hour_arg = args.get("hour_dir") or args.get("hour")
    if hour_arg:
        if os.path.isabs(hour_arg):
            return hour_arg
        return os.path.join(repo_root_of_run(run_dir), hour_arg)
    window = meta.get("window") or {}
    start = parse_time(window.get("start"))
    if start is None:
        raise ValueError(f"{run_dir}/run.json has no usable window.start; pass --hour explicitly")
    return os.path.join(data_root, start.strftime("%Y-%m-%d-%H"))


def collect_cost(run_dir: str, meta: dict) -> dict:
    """Wall time + supervisor $/token/image cost from run artifacts."""
    cost = {
        "elapsed_s": meta.get("elapsed_s"),
        "started_at": meta.get("started_at"),
        "finished_at": meta.get("finished_at"),
        "validations": 0,
        "verdicts": {},
        "est_cost_usd": 0.0,
        "images_used": 0,
        "tool_calls": 0,
        "usage_tokens": {},
        "llm_ms": 0.0,
        "tools_ms": 0.0,
    }
    path = os.path.join(run_dir, "validations.jsonl")
    if not os.path.exists(path):
        return cost
    for row in load_jsonl(path):
        review = row.get("review") or {}
        cost["validations"] += 1
        cost["verdicts"][review.get("verdict")] = cost["verdicts"].get(review.get("verdict"), 0) + 1
        run = row.get("agent_run") or {}
        cost["est_cost_usd"] += run.get("est_cost_usd") or 0.0
        cost["images_used"] += run.get("images_used") or 0
        cost["tool_calls"] += run.get("tool_calls") or 0
        for k, v in (run.get("usage") or {}).items():
            if isinstance(v, (int, float)):
                cost["usage_tokens"][k] = cost["usage_tokens"].get(k, 0) + v
        cost["llm_ms"] += run.get("llm_ms") or 0.0
        cost["tools_ms"] += run.get("tools_ms") or 0.0
    cost["est_cost_usd"] = round(cost["est_cost_usd"], 6)
    return cost


def discover_charts(run_dir: str) -> dict:
    """Presence + absolute source path of the run's detection strip charts.

    The ground-truth chart is rendered by evaluate.py (not discovered), so
    it starts as absent until a render fills it in.
    """
    charts = {}
    for key, fname, _title in CHART_FILES:
        src = os.path.join(run_dir, fname)
        ok = os.path.isfile(src)
        charts[key] = {"present": ok, "src": src if ok else None, "file": None}
    charts[GT_CHART_KEY] = {"present": False, "src": None, "file": None}
    return charts


def copy_charts(charts: dict, out_dir: str, run_name: str) -> dict:
    """Copy present charts into out_dir as <run>_<chart>.png.

    Records the copied file name on each entry (None when the run has no
    such chart). Missing charts are skipped silently, never an error.
    """
    for key, fname, _title in CHART_FILES:
        entry = charts.setdefault(key, {"present": False, "src": None})
        src = entry.get("src")
        if not entry.get("present") or not src or not os.path.isfile(src):
            entry["present"] = False
            entry["file"] = None
            continue
        dest_name = f"{run_name}_{fname}"
        shutil.copy2(src, os.path.join(out_dir, dest_name))
        entry["file"] = dest_name
    return charts


def load_run_escalations(run_dir: str) -> list | None:
    """Run escalation dicts for chart overlays; None when unavailable."""
    path = os.path.join(run_dir, "escalations.jsonl")
    if not os.path.isfile(path):
        return None
    try:
        rows = load_jsonl(path)
    except (OSError, ValueError):
        return None
    return rows or None


def render_gt_chart(
    gt: dict,
    frame_set: list,
    img_w: int,
    order: list,
    names: dict,
    dest_path: str,
    escalations: list | None = None,
) -> bool:
    """Render the GT strip chart (same style as the run charts) to dest_path.

    Only evaluated frames are plotted. Returns False — never raises — when
    charting is unavailable (no matplotlib/agent install) or rendering fails,
    so a missing GT chart degrades to a skipped report section.
    """
    try:
        from agent.tracking.charts import plot_strip_chart
    except ImportError:
        return False
    norm_names = {}
    for k, v in (names or {}).items():
        try:
            norm_names[int(k)] = v
        except (TypeError, ValueError):
            norm_names[k] = v
    names = norm_names
    rows = []
    for t in frame_set:
        for b in gt.get(t, []):
            rows.append({"create_time": t, "box_x1": b.x1, "box_x2": b.x2, "cat_id": b.cls})
    try:
        plot_strip_chart(rows, dest_path, GT_CHART_TITLE, order, names, escalations, img_w)
    except Exception:
        return False
    return os.path.isfile(dest_path)


def score_frames(preds: dict, gt: dict, frame_set: list, order: list, iou_thrs: list) -> tuple:
    """(metrics_by_iou, preds_outside_gt) over the joined frame set."""
    frames = {t: (preds.get(t, []), gt[t]) for t in frame_set}
    metrics = {f"iou_{t}": evaluate_frames(frames, t, classes=order) for t in iou_thrs}
    outside = sum(len(v) for t, v in preds.items() if t not in gt)
    return metrics, outside


def _audit_time(value) -> datetime.datetime:
    return value if isinstance(value, datetime.datetime) else datetime.datetime.fromisoformat(str(value))


def load_audit(path: str | None, hour: str) -> dict:
    """The gt_audit.yaml entry for one hour ({} when no audit or no entry)."""
    if not path:
        return {}
    import yaml

    with open(path) as f:
        return ((yaml.safe_load(f) or {}).get("hours") or {}).get(hour) or {}


def apply_audit(gt: dict, audit: dict, view_id: int = 3) -> tuple[dict, dict]:
    """Apply window / drop_decoys / relabel corrections; returns (gt, change counts).

    GT keys may be datetimes or ISO strings; audit times are ISO strings.
    """
    if not audit:
        return gt, {}
    changes = {"frames_outside_window": 0, "decoy_boxes_dropped": 0, "boxes_relabeled": 0}
    window = audit.get("window")
    lo, hi = (_audit_time(window[0]), _audit_time(window[1])) if window else (None, None)
    relabels = [(_audit_time(r["from"]), _audit_time(r["to"]), r["cls_from"], r["cls_to"])
                for r in audit.get("relabel") or []]
    decoys = []
    if audit.get("drop_decoys"):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from agent.tracking.decoys import load_decoys, match_decoy

        decoys = load_decoys(view_id)
    out = {}
    for t, boxes in gt.items():
        when = _audit_time(t)
        if lo is not None and not (lo <= when < hi):
            changes["frames_outside_window"] += 1
            continue
        kept = []
        for b in boxes:
            if decoys and match_decoy((b.x1, b.y1, b.x2, b.y2), decoys):
                changes["decoy_boxes_dropped"] += 1
                continue
            for r_lo, r_hi, c_from, c_to in relabels:
                if r_lo <= when < r_hi and b.cls == c_from:
                    b = Box(b.x1, b.y1, b.x2, b.y2, c_to, b.conf)
                    changes["boxes_relabeled"] += 1
            kept.append(b)
        out[t] = kept
    return out, changes


def evaluate_run(
    run_dir: str,
    hour_dir: str,
    gt_source: str,
    iou_thrs: list,
    gt_chart_path: str | None = None,
    audit_path: str | None = None,
) -> dict:
    run_dir = os.path.normpath(os.path.abspath(run_dir))
    with open(os.path.join(run_dir, "run.json")) as f:
        meta = json.load(f)
    label_dir = os.path.join(hour_dir, GT_SOURCES[gt_source])
    with open(os.path.join(hour_dir, "classes_map.json")) as f:
        cmap = json.load(f)
    order = [int(v) for v in cmap["order"]]
    names = {str(k): v for k, v in cmap["names"].items()}

    images = sorted(glob.glob(os.path.join(hour_dir, "images", "*.jpg")))
    size = jpeg_size(images[0]) if images else None
    img_w, img_h = size if size else (800, 600)

    gt = load_yolo_labels(label_dir, order, img_w, img_h)
    gt, audit_changes = apply_audit(gt, load_audit(audit_path, os.path.basename(os.path.normpath(hour_dir))))
    run_frames = [parse_time(r.get("create_time")) for r in load_jsonl(os.path.join(run_dir, "images.jsonl"))]
    run_frames = [t for t in run_frames if t is not None]
    frame_set = [t for t in run_frames if t in gt]
    missing_gt = [t for t in run_frames if t not in gt]

    result = {
        "run": os.path.basename(run_dir),
        "run_dir": run_dir,
        "mode": meta.get("mode"),
        "window": meta.get("window"),
        "hour": os.path.basename(os.path.normpath(hour_dir)),
        "hour_dir": hour_dir,
        "gt_source": gt_source,
        "gt_dir": label_dir,
        "class_names": names,
        "image_size": [img_w, img_h],
        "run_frames": len(run_frames),
        "eval_frames": len(frame_set),
        "frames_without_gt": len(missing_gt),
        "audit": audit_changes,
        "stages": {},
        "label_sets": {},
        "cost": collect_cost(run_dir, meta),
        "charts": discover_charts(run_dir),
    }
    for stage, fname in PIPE_STAGES:
        preds = load_stage(run_dir, fname)
        metrics, outside = score_frames(preds, gt, frame_set, order, iou_thrs)
        result["stages"][stage] = {
            "file": fname,
            "present": os.path.exists(os.path.join(run_dir, fname)),
            "preds_outside_gt_frames": outside,
            "metrics": metrics,
        }
    for stage, dirname in LABEL_STAGES:
        label_path = os.path.join(hour_dir, dirname)
        if not os.path.isdir(label_path):
            result["label_sets"][stage] = {"dir": dirname, "present": False}
            continue
        preds = load_yolo_labels(label_path, order, img_w, img_h)
        metrics, outside = score_frames(preds, gt, frame_set, order, iou_thrs)
        result["label_sets"][stage] = {
            "dir": dirname,
            "present": True,
            "preds_outside_gt_frames": outside,
            "metrics": metrics,
        }
    if gt_chart_path:
        ok = render_gt_chart(
            gt, frame_set, img_w, order, names, gt_chart_path, load_run_escalations(run_dir)
        )
        result["charts"][GT_CHART_KEY] = {
            "present": ok,
            "src": gt_chart_path if ok else None,
            "file": os.path.basename(gt_chart_path) if ok else None,
        }
    return result


def fmt(x: float) -> str:
    return f"{x:.3f}"


def dfmt(x: float) -> str:
    return f"{x:+.3f}"


def md_table(headers: list, rows: list, aligns: list | None = None) -> str:
    """Render a padded GitHub-flavored markdown table.

    The separator row derives from the header count, so it cannot drift
    out of sync with the data columns. First column defaults to
    left-aligned, the rest to right-aligned (numeric).
    """
    str_rows = [[str(c) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for r in str_rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    if aligns is None:
        aligns = ["l"] + ["r"] * (len(headers) - 1)

    def pad(cell: str, i: int) -> str:
        return cell.rjust(widths[i]) if aligns[i] == "r" else cell.ljust(widths[i])

    header_line = "| " + " | ".join(pad(h, i) for i, h in enumerate(headers)) + " |"
    sep_cells = []
    for i, a in enumerate(aligns):
        dashes = "-" * max(widths[i], 3)
        sep_cells.append(f":{dashes}" if a == "l" else f"{dashes}:" if a == "r" else f":{dashes}:")
    sep_line = "| " + " | ".join(sep_cells) + " |"
    body = ["| " + " | ".join(pad(c, i) for i, c in enumerate(r)) + " |" for r in str_rows]
    return "\n".join([header_line, sep_line] + body)


def stage_cells(name: str, entry: dict, iou_key: str) -> list:
    if not entry.get("present"):
        return [name] + [MISSING] * 11
    m = entry["metrics"][iou_key]
    a = m["agnostic"]
    return [
        name,
        fmt(m["precision"]),
        fmt(m["recall"]),
        fmt(m["f1"]),
        fmt(a["precision"]),
        fmt(a["recall"]),
        fmt(a["f1"]),
        fmt(m["mean_matched_iou"]),
        fmt(m["class_accuracy_on_matched"]),
        str(m["tp"]),
        str(m["fp"]),
        str(m["fn"]),
    ]


def headline(res: dict, iou_thr: float) -> list:
    """Brief raw -> validated improvement + time + cost (report top)."""
    key = f"iou_{iou_thr}"
    raw = res["stages"]["raw"]
    val = res["stages"]["validated"]
    cost = res["cost"]
    lines = ["## Headline", ""]
    if raw.get("present") and val.get("present"):
        mr, mv = raw["metrics"][key], val["metrics"][key]
        ar, av = mr["agnostic"], mv["agnostic"]
        lines += [
            f"Raw detector -> supervisor-validated (IoU >= {iou_thr}):",
            "",
            f"- F1 {fmt(mr['f1'])} -> **{fmt(mv['f1'])}** ({dfmt(mv['f1'] - mr['f1'])}) | "
            f"identity on matched {fmt(mr['class_accuracy_on_matched'])} -> "
            f"**{fmt(mv['class_accuracy_on_matched'])}** "
            f"({dfmt(mv['class_accuracy_on_matched'] - mr['class_accuracy_on_matched'])})",
            f"- Recall {fmt(mr['recall'])} -> **{fmt(mv['recall'])}** "
            f"({dfmt(mv['recall'] - mr['recall'])}); localization F1 "
            f"{fmt(ar['f1'])} -> **{fmt(av['f1'])}** ({dfmt(av['f1'] - ar['f1'])})",
            "",
        ]
    else:
        lines += ["- Raw/validated stages incomplete; no headline delta.", ""]
    elapsed = cost.get("elapsed_s")
    lines.append(
        f"- Cost: **${cost.get('est_cost_usd', 0.0):.4f}** supervisor spend over "
        f"{cost.get('validations', 0)} validations ({cost.get('images_used', 0)} images, "
        f"{cost.get('tool_calls', 0)} tool calls)"
        + (f"; wall time **{elapsed:.1f} s**" if isinstance(elapsed, (int, float)) else "")
        + "."
    )
    lines += [""]
    return lines


def label_table(res: dict, iou_key: str, label: str) -> str:
    headers = [
        "label set",
        f"P@{label}",
        f"R@{label}",
        f"F1@{label}",
        "P(loc)",
        "R(loc)",
        "F1(loc)",
        "mean IoU",
        "cls-acc",
        "TP",
        "FP",
        "FN",
    ]
    rows = [stage_cells(stage, res["label_sets"][stage], iou_key) for stage, _ in LABEL_STAGES]
    return md_table(headers, rows)


def stage_table(res: dict, iou_key: str, label: str) -> str:
    headers = [
        "stage",
        f"P@{label}",
        f"R@{label}",
        f"F1@{label}",
        "P(loc)",
        "R(loc)",
        "F1(loc)",
        "mean IoU",
        "cls-acc",
        "TP",
        "FP",
        "FN",
    ]
    rows = [stage_cells(stage, res["stages"][stage], iou_key) for stage, _ in PIPE_STAGES]
    return md_table(headers, rows)


def delta_table(res: dict, iou_key: str) -> str:
    """Stage-to-stage deltas (revised - raw, validated - revised, validated - raw)."""
    headers = ["step", "ΔP", "ΔR", "ΔF1", "ΔP(loc)", "ΔR(loc)", "ΔF1(loc)", "Δmean IoU", "Δcls-acc"]
    rows = []
    for a_stage, b_stage in (("raw", "revised"), ("revised", "validated"), ("raw", "validated")):
        a = res["stages"][a_stage]
        b = res["stages"][b_stage]
        if not a["present"] or not b["present"]:
            rows.append([f"{a_stage} → {b_stage}"] + [MISSING] * 8)
            continue
        ma, mb = a["metrics"][iou_key], b["metrics"][iou_key]
        aa, ab = ma["agnostic"], mb["agnostic"]
        rows.append(
            [
                f"{a_stage} → {b_stage}",
                dfmt(mb["precision"] - ma["precision"]),
                dfmt(mb["recall"] - ma["recall"]),
                dfmt(mb["f1"] - ma["f1"]),
                dfmt(ab["precision"] - aa["precision"]),
                dfmt(ab["recall"] - aa["recall"]),
                dfmt(ab["f1"] - aa["f1"]),
                dfmt(mb["mean_matched_iou"] - ma["mean_matched_iou"]),
                dfmt(mb["class_accuracy_on_matched"] - ma["class_accuracy_on_matched"]),
            ]
        )
    return md_table(headers, rows)


def per_class_table(res: dict, iou_key: str, names: dict) -> str:
    headers = ["stage", "class", "P", "R", "F1", "TP", "FP", "FN"]
    rows = []
    for stage, _ in PIPE_STAGES:
        s = res["stages"][stage]
        if not s["present"]:
            continue
        for cid, mc in s["metrics"][iou_key]["per_class"].items():
            cname = names.get(str(cid), str(cid))
            rows.append(
                [
                    stage,
                    f"{cid} ({cname})",
                    fmt(mc["precision"]),
                    fmt(mc["recall"]),
                    fmt(mc["f1"]),
                    str(mc["tp"]),
                    str(mc["fp"]),
                    str(mc["fn"]),
                ]
            )
    return md_table(headers, rows, aligns=["l", "l", "r", "r", "r", "r", "r", "r"])


def cost_block(res: dict) -> list:
    cost = res["cost"]
    tokens = cost.get("usage_tokens") or {}
    total_tokens = sum(v for k, v in tokens.items() if "token" in k.lower() and isinstance(v, (int, float)))
    verdicts = ", ".join(f"{k}={v}" for k, v in sorted(cost.get("verdicts", {}).items()))
    return [
        "### Cost & time",
        "",
        f"- Wall time: {cost.get('elapsed_s')} s "
        f"({cost.get('started_at')} → {cost.get('finished_at')})",
        f"- Supervisor spend: **${cost.get('est_cost_usd', 0.0):.4f}** over "
        f"{cost.get('validations', 0)} validations ({verdicts or 'none'})",
        f"- Images sent: {cost.get('images_used', 0)}; tool calls: {cost.get('tool_calls', 0)}; "
        f"tokens: {int(total_tokens)} total {tokens if tokens else ''}",
        f"- LLM time: {cost.get('llm_ms', 0.0):.0f} ms; tools time: {cost.get('tools_ms', 0.0):.0f} ms",
        "",
    ]


def chart_lines(res: dict) -> list:
    """Detection-chart section: embedded copied charts, or a no-charts note.

    Prefers the copied `file` name (relative link next to the report);
    falls back to the absolute `src` path for results that were scored
    without copying. Results without chart info render no section.
    """
    charts = res.get("charts") or {}
    if not charts:
        return []
    lines = ["## Detection charts", ""]
    entries = [(k, CHART_TITLES[k]) for k in REPORT_CHART_ORDER if charts.get(k, {}).get("present")]
    if not entries:
        return lines + ["(no charts; the run used `--no-plot` or charting failed)", ""]
    for key, title in entries:
        link = charts[key].get("file") or charts[key].get("src") or ""
        lines += [f"### {title}", "", f"![{title}]({link})", ""]
    return lines


def render_report(res: dict, iou_thrs: list) -> str:
    names = res["class_names"]
    parts = [
        f"# Evaluation: `{res['run']}`",
        "",
        f"- Mode: {res['mode']} — window: {(res['window'] or {}).get('start')} → {(res['window'] or {}).get('end_exclusive')}",
        f"- Hour: `{res['hour']}` — ground truth: `{res['gt_dir']}`",
        f"- Classes: {', '.join(f'{k}={v}' for k, v in names.items())}; "
        f"image size {res['image_size'][0]}x{res['image_size'][1]}",
        f"- Frames: {res['eval_frames']} evaluated "
        f"({res['run_frames']} in run, {res['frames_without_gt']} without GT)",
        *([f"- GT audit applied: {res['audit']}"] if res.get("audit") else []),
        "",
    ]
    parts += headline(res, iou_thrs[0])
    parts += chart_lines(res)
    for t in iou_thrs:
        key = f"iou_{t}"
        gt_n = res["stages"]["raw"]["metrics"][key]["n_gt"]
        parts += [
            f"## IoU >= {t} (GT boxes: {gt_n})",
            "",
            "P/R/F1 are class-aware (right class + box overlap). "
            "P/R/F1(loc) are class-agnostic (box overlap only). "
            "cls-acc = fraction of spatially matched pairs with the right class.",
            "",
            "### Label sets vs GT (offline labelers, no pipeline cost)",
            "",
            label_table(res, key, str(t)),
            "",
            "### Pipeline stages",
            "",
            stage_table(res, key, str(t)),
            "",
            "### Stage deltas",
            "",
            delta_table(res, key),
            "",
            "### Per-class breakdown",
            "",
            per_class_table(res, key, names),
            "",
        ]
    parts += cost_block(res)
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate one tracking run vs YOLO GT (+ cost).")
    ap.add_argument("run_dir", help="Tracking run dir (absolute path fine).")
    ap.add_argument("--hour", default=None, help="retrieved_images/<hour> dir (default: auto from run).")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="retrieved_images tree for hour lookup.")
    ap.add_argument("--gt-source", choices=sorted(GT_SOURCES), default="labels")
    ap.add_argument("--iou", type=float, action="append", default=[0.5, 0.75])
    ap.add_argument("--out", default=None, help="Report dir (default: <repo>/output/evaluation_runs/).")
    ap.add_argument("--audit", default=None, help="Apply documented GT corrections (e.g. evaluation/gt_audit.yaml).")
    args = ap.parse_args()

    with open(os.path.join(args.run_dir, "run.json")) as f:
        meta = json.load(f)
    hour_dir = resolve_hour_dir(args.run_dir, meta, args.hour, args.data_root)
    if not os.path.isdir(hour_dir):
        ap.error(f"hour dir not found: {hour_dir} (pass --hour explicitly)")

    out_dir = args.out or DEFAULT_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    run_name = os.path.basename(os.path.normpath(os.path.abspath(args.run_dir)))
    gt_chart_path = os.path.join(out_dir, f"{run_name}_gt.png")
    res = evaluate_run(args.run_dir, hour_dir, args.gt_source, args.iou, gt_chart_path, audit_path=args.audit)
    res["charts"] = copy_charts(res.get("charts") or {}, out_dir, res["run"])
    with open(os.path.join(out_dir, f"{res['run']}_metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    report = render_report(res, args.iou)
    report_path = os.path.join(out_dir, f"{res['run']}_eval.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(report)
    copied = sorted(c["file"] for c in res["charts"].values() if c.get("file"))
    print(f"\nwrote {report_path} + {res['run']}_metrics.json" + (f" + {', '.join(copied)}" if copied else ""))
    return None


if __name__ == "__main__":
    main()
