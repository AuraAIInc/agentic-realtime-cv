"""Programmatic evaluation: score one run and pull headline numbers.

Run from the repo root:
    .venv/bin/python evaluation/examples/programmatic.py [RUN_DIR] [OUT_DIR]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from evaluate import copy_charts, evaluate_run, render_report, resolve_hour_dir  # noqa: E402
import json  # noqa: E402

RUN = (
    "/Users/muyang/Documents/Personal/Github/agentic-realtime-cv-e2e"
    "/output/tracking_runs/view3_20261028-1633_1640_replay_db_20261104-110807-376138"
)


def main() -> None:
    run_dir = sys.argv[1] if len(sys.argv) > 1 else RUN
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/eval_out"
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(run_dir, "run.json")) as f:
        meta = json.load(f)
    from evaluate import DEFAULT_DATA_ROOT

    hour_dir = resolve_hour_dir(run_dir, meta, None, DEFAULT_DATA_ROOT)
    run_name = os.path.basename(os.path.normpath(os.path.abspath(run_dir)))
    res = evaluate_run(
        run_dir,
        hour_dir,
        gt_source="labels",
        iou_thrs=[0.5, 0.75],
        gt_chart_path=os.path.join(out_dir, f"{run_name}_gt.png"),
    )
    res["charts"] = copy_charts(res.get("charts") or {}, out_dir, res["run"])

    with open(os.path.join(out_dir, f"{res['run']}_metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    report = render_report(res, [0.5, 0.75])
    with open(os.path.join(out_dir, f"{res['run']}_eval.md"), "w") as f:
        f.write(report)

    # Headline numbers for dashboards / further aggregation.
    m_raw = res["stages"]["raw"]["metrics"]["iou_0.5"]
    m_val = res["stages"]["validated"]["metrics"]["iou_0.5"]
    print(f"run: {res['run']}")
    print(f"F1 raw -> validated: {m_raw['f1']:.3f} -> {m_val['f1']:.3f}")
    print(
        "identity raw -> validated: "
        f"{m_raw['class_accuracy_on_matched']:.3f} -> {m_val['class_accuracy_on_matched']:.3f}"
    )
    print(f"spend: ${res['cost']['est_cost_usd']:.4f}  wall: {res['cost']['elapsed_s']} s")
    print(f"wrote {out_dir}/{res['run']}_eval.md + _metrics.json")


if __name__ == "__main__":
    main()
