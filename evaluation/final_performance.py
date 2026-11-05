"""Numbers + charts for documents/final-model-performance-20261104.md.

Scores the raw detector stream against the agent-validated output for each replay run,
over the full run window and over sliding 2-minute windows (step 30 s), using the audited
ground truth (evaluation/gt_audit.yaml) and IoU 0.5. The showcase window is the 2-minute
window with the largest F1 gain that has >= 300 evaluated frames and >= 300 GT boxes; it
is reported next to the full-window result, never instead of it.

    .venv/bin/python evaluation/final_performance.py
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import evaluate as E  # noqa: E402
from metrics import evaluate_frames  # noqa: E402

OUT = os.path.join(REPO, "documents", "assets", "final-performance-20261104")
RUNS = sorted(glob.glob(os.path.join(REPO, "output", "tracking_runs", "view3_*_replay_db_20261104-*")))
AUDIT = os.path.join(HERE, "gt_audit.yaml")
WIN_MIN, STEP_S, MIN_FRAMES, MIN_GT = 2, 30, 300, 300

# Validated palette (dataviz validate_palette.js, light + dark): raw = the Pi, validated = agent.
RAW, VAL = "#199e70", "#9085e9"
BG, INK, INK2, MUTED, GRID = "#1a1a19", "#f2f1ed", "#c3c2b7", "#8e8d84", "#2f2e2b"


def load_run(run: str) -> dict:
    meta = json.load(open(os.path.join(run, "run.json")))
    hour = E.resolve_hour_dir(run, meta, None, E.DEFAULT_DATA_ROOT)
    cmap = json.load(open(os.path.join(hour, "classes_map.json")))
    order = [int(v) for v in cmap["order"]]
    w, h = E.image_size(meta, hour)
    gt = E.load_yolo_labels(os.path.join(hour, "labels"), order, w, h)
    gt, _ = E.apply_audit(gt, E.load_audit(AUDIT, os.path.basename(os.path.normpath(hour))))
    frames = [E.parse_time(r["create_time"]) for r in E.load_jsonl(os.path.join(run, "images.jsonl"))]
    return {"name": os.path.basename(run), "meta": meta, "order": order, "gt": gt,
            "frames": [t for t in frames if t in gt],
            "raw": E.load_stage(run, "stream.jsonl"), "val": E.load_stage(run, "validated.jsonl"),
            "cost": E.collect_cost(run, meta)}


def score(r: dict, frames: list) -> dict:
    out = {}
    for stage in ("raw", "val"):
        m = evaluate_frames({f: (r[stage].get(f, []), r["gt"][f]) for f in frames}, 0.5, classes=r["order"])
        out[stage] = {k: m[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn", "n_gt", "class_accuracy_on_matched")}
    out["frames"] = len(frames)
    return out


def sweep(r: dict) -> list:
    fs = r["frames"]
    t, end = fs[0].replace(second=0, microsecond=0), fs[-1]
    rows = []
    while t + dt.timedelta(minutes=WIN_MIN) <= end + dt.timedelta(seconds=STEP_S):
        sub = [f for f in fs if t <= f < t + dt.timedelta(minutes=WIN_MIN)]
        if sub:
            s = score(r, sub)
            rows.append({"start": t.isoformat(), **s})
        t += dt.timedelta(seconds=STEP_S)
    return rows


def style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def dumbbell(ax, rows, title):
    """rows: [(label, raw, val, fmt)] -> one dumbbell per metric, raw → validated."""
    style(ax)
    for i, (lab, a, b, fmt) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([a, b], [y, y], color=MUTED, linewidth=2, zorder=1)
        ax.scatter([a], [y], s=90, color=RAW, zorder=2, edgecolors=BG, linewidths=2)
        ax.scatter([b], [y], s=90, color=VAL, zorder=3, edgecolors=BG, linewidths=2)
        ax.text(b + (0.02 if b >= a else -0.02), y + 0.22, fmt.format(b), color=INK, fontsize=11,
                ha="left" if b >= a else "right", fontweight="bold")
        ax.text(a + (-0.02 if b >= a else 0.02), y + 0.22, fmt.format(a), color=INK2, fontsize=10,
                ha="right" if b >= a else "left")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], color=INK2, fontsize=11)
    ax.set_xlim(0, 1.08)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    os.makedirs(OUT, exist_ok=True)
    report = {"runs": {}}
    for run in RUNS:
        r = load_run(run)
        full = score(r, r["frames"])
        sw = sweep(r)
        ok = [w for w in sw if w["frames"] >= MIN_FRAMES and w["raw"]["n_gt"] >= MIN_GT]
        best = max(ok, key=lambda w: w["val"]["f1"] - w["raw"]["f1"]) if ok else None
        report["runs"][r["name"]] = {"window": r["meta"]["window"], "full": full, "showcase": best,
                                     "sweep": sw, "cost": r["cost"]}
        print(r["name"][:28], "full F1 %.3f -> %.3f" % (full["raw"]["f1"], full["val"]["f1"]),
              "| showcase", best and best["start"][11:16], best and "%.3f -> %.3f" % (best["raw"]["f1"], best["val"]["f1"]))
    json.dump(report, open(os.path.join(OUT, "numbers.json"), "w"), indent=1, default=str)

    legend = [Line2D([0], [0], marker="o", color=BG, markerfacecolor=RAW, markersize=9, label="raw detector (Pi)"),
              Line2D([0], [0], marker="o", color=BG, markerfacecolor=VAL, markersize=9, label="after agent review")]

    # 1. headline run: full window vs showcase window
    head = report["runs"][next(k for k in report["runs"] if "1633" in k)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), facecolor=BG)
    for ax, (key, title) in zip(axes, (("full", "Full 8 minutes · 1,319 frames"),
                                       ("showcase", f"Showcase · {head['showcase']['start'][11:16]}–"
                                        f"{(dt.datetime.fromisoformat(head['showcase']['start']) + dt.timedelta(minutes=WIN_MIN)).strftime('%H:%M')}"
                                        f" · {head['showcase']['frames']} frames"))):
        s = head[key]
        dumbbell(ax, [("precision", s["raw"]["precision"], s["val"]["precision"], "{:.3f}"),
                      ("identity accuracy", s["raw"]["class_accuracy_on_matched"], s["val"]["class_accuracy_on_matched"], "{:.3f}"),
                      ("F1", s["raw"]["f1"], s["val"]["f1"], "{:.3f}")], title)
    axes[1].set_yticklabels([])
    fig.legend(handles=legend, loc="upper right", frameon=False, labelcolor=INK2, fontsize=10, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(OUT, "headline_dumbbell.png"), dpi=160, facecolor=BG)
    plt.close(fig)

    # 2. sweep for the headline run: F1 per 2-minute window, raw vs validated
    sw = [w for w in head["sweep"] if w["raw"]["n_gt"] > 0]
    xs = [dt.datetime.fromisoformat(w["start"]) for w in sw]
    fig, ax = plt.subplots(figsize=(12, 3.4), facecolor=BG)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.grid(axis="x", visible=False)
    b0 = dt.datetime.fromisoformat(head["showcase"]["start"])
    ax.axvspan(b0, b0 + dt.timedelta(seconds=1), color=VAL, alpha=0)  # keeps date axis stable
    ax.axvspan(b0 - dt.timedelta(seconds=10), b0 + dt.timedelta(seconds=10), color=VAL, alpha=0.18, linewidth=0)
    ax.plot(xs, [w["raw"]["f1"] for w in sw], color=RAW, linewidth=2, marker="o", markersize=5, label="raw detector (Pi)")
    ax.plot(xs, [w["val"]["f1"] for w in sw], color=VAL, linewidth=2, marker="o", markersize=5, label="after agent review")
    ax.annotate("showcase window", (b0, head["showcase"]["val"]["f1"]), xytext=(18, 18), textcoords="offset points",
                color=INK, fontsize=10, arrowprops={"arrowstyle": "-", "color": MUTED})
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 @ IoU 0.5", color=MUTED, fontsize=10)
    ax.set_xlabel("start of the 2-minute window (2026-10-28)", color=MUTED, fontsize=10)
    import matplotlib.dates as mdates

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(frameon=False, labelcolor=INK2, fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "window_sweep.png"), dpi=160, facecolor=BG)
    plt.close(fig)

    # 3. all runs, full window: precision and F1
    names = {"1633": "Oct 28 16:33 (8 min)", "0413": "Oct 25 04:13 (26 min)", "0200": "Oct 26 02:00 (31 min)"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.0), facecolor=BG)
    for ax, (metric, title) in zip(axes, (("precision", "Precision, full window"), ("f1", "F1, full window"))):
        rows = []
        for k, v in sorted(report["runs"].items(), key=lambda kv: "1633" not in kv[0]):
            lab = next(n for c, n in names.items() if c in k)
            rows.append((lab, v["full"]["raw"][metric], v["full"]["val"][metric], "{:.3f}"))
        dumbbell(ax, rows, title)
    axes[1].set_yticklabels([])
    fig.legend(handles=legend, loc="upper right", frameon=False, labelcolor=INK2, fontsize=10, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(OUT, "all_runs.png"), dpi=160, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    main()
