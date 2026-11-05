#!/usr/bin/env python3
"""End-to-end demo: one camera window through the whole pipeline, then a final report.

Runs entirely from this repo's committed demo data (``demo/data/2026-10-28-16``:
view 3, 2026-10-28 16:33:00-16:38:59, 989 frames, 686 raw YOLO boxes):

  1. Local image MCP server      image_server/mcp_image_server over the demo frames (127.0.0.1)
  2. Tracking pipeline           agent.main --replay: TrackCorrector (frame scope, T-codes)
                                 -> PeriodCorrector (visit scope, P-codes) -> escalations
  3. Supervisor agent            --validate: Muse Spark reviews every escalation (merged
                                 batches) with frame / crop / reference tools, verdicts are
                                 applied to the rows (validated.jsonl / validated.db)
  4. Memory                      --memory: short-term day memory (every period, episode,
                                 verdict) and long-term spatial-temporal memory (combined,
                                 LLM-labeled period activity), fresh store per demo run
  5. Evaluation                  evaluation/evaluate.py: raw -> revised -> validated vs the
                                 ground-truth labels, as given and with the documented audit
  6. Final report                <out>/DEMO_REPORT.md

The demo never touches the shared services in the repo-root .env (MySQL, the LAN image
server, the tracking MCP, the camera): it runs with an isolated env file
(``AGENT_ENV_FILE``) and takes only ``MODEL_API_KEY`` from the environment or the .env.
Without a key it runs offline: tracking, escalation, memory (stub LLM) and evaluation,
with no supervisor verdicts.

Usage (from the repo root, inside the project venv):

    python run_demo.py                       # full loop, ~10-15 min with a model key
    python run_demo.py --offline             # no LLM calls (~1 min)
    python run_demo.py --memory-dir output/demo_memory   # keep memory across demo runs
    python run_demo.py --report-only output/demo/<stamp> # rebuild the final report
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = REPO / "demo" / "data" / "2026-10-28-16"
VIEW_ID = 3
START, END = "2026-10-28-16-33", "2026-10-28-16-38"  # --end minute is included: 16:33:00-16:38:59
GT_AUDIT = REPO / "evaluation" / "gt_audit.yaml"
IOU = "0.5"
# Process-env variables that would point a child at shared services; never inherited.
SCRUB_PREFIXES = ("MYSQL_", "S3_AWS_", "SES_AWS_", "IMAGE_S3_", "CAMERA_MCP", "IMAGE_MCP", "TRACKING_MCP",
                  "MCP_AUTH_TOKEN", "AGENT_INVOKE_DUMP", "LOCAL_FRAME_DIRS", "MEMORY_", "TRACK_RUN_DIR")


def log(msg: str) -> None:
    print(f"\033[1;36m[demo]\033[0m {msg}", flush=True)


# ------------------------------------------------------------------ setup


def model_api_key() -> str | None:
    if os.environ.get("MODEL_API_KEY"):
        return os.environ["MODEL_API_KEY"]
    env = REPO / ".env"
    if env.exists():
        try:
            from dotenv import dotenv_values
        except ImportError:
            return None
        return (dotenv_values(env).get("MODEL_API_KEY") or "").strip() or None
    return None


def check_data() -> dict:
    manifest = json.loads((DATA / "manifest.json").read_text())
    for sub in ("images", "labels"):
        if not (DATA / sub).is_dir():
            raise SystemExit(f"missing {DATA / sub}; see demo/README.md")
    if not (DATA / "raw_detections.jsonl").exists():
        raise SystemExit(f"missing {DATA / 'raw_detections.jsonl'}; see demo/README.md")
    return manifest


def check_imports() -> None:
    missing = []
    for mod in ("dotenv", "yaml", "numpy", "matplotlib", "PIL", "mcp", "openai", "pydantic"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise SystemExit(f"missing Python packages {missing}; run: pip install -r requirements.txt")


def build_image_root(out: Path) -> Path:
    """Symlink the demo frames into the image server's YYYY-MM-DD/HH/MM/<stamp>.jpg layout."""
    root = out / "image_root"
    n = 0
    for img in sorted((DATA / "images").glob("*.jpg")):
        stamp = img.stem.split("_", 1)[1]  # <idx>_<YYYY-MM-DD-HH-MM-SS-ffffff>
        y, mo, d, h, mi = stamp.split("-")[:5]
        dest = root / f"{y}-{mo}-{d}" / h / mi / f"{stamp}.jpg"
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.symlink_to(img.resolve())
        except OSError:
            shutil.copy2(img, dest)  # filesystems without symlinks
        n += 1
    log(f"image root: {n} frames under {root}")
    return root


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def write_demo_env(out: Path, port: int, token: str, image_root: Path, memory_dir: Path) -> Path:
    """Isolated config for every child process (loaded via AGENT_ENV_FILE, override=True)."""
    lines = {
        # Supervisor MCP servers: the local image server only. Empty URL = server left out.
        "IMAGE_MCP_URL": f"http://127.0.0.1:{port}/mcp",
        "CAMERA_MCP_URL": "",
        "TRACKING_MCP_URL": "",
        "MCP_AUTH_TOKEN": token,
        # Image server: demo frames, local only (no S3 day ever qualifies).
        "IMAGE_ROOT": str(image_root),
        "IMAGE_MCP_HOST": "127.0.0.1",
        "IMAGE_MCP_PORT": str(port),
        "IMAGE_S3_CUTOFF_DATE": "2999-12-31",
        # local__get_frame / box crops read the demo frames straight from disk.
        "LOCAL_FRAME_DIRS": str(DATA / "images"),
        # Memory stores for this demo only (never agent/memory/data).
        "MEMORY_STORE_DIR": str(memory_dir),
        # No MySQL writes from the demo.
        "AGENT_INVOKE_DUMP": "false",
    }
    path = out / "demo.env"
    path.write_text("# Written by run_demo.py; no secrets (MODEL_API_KEY comes from the process env).\n"
                    + "".join(f"{k}={v}\n" for k, v in lines.items()))
    return path


def child_env(env_file: Path, api_key: str | None) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(SCRUB_PREFIXES)}
    env["AGENT_ENV_FILE"] = str(env_file)
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("MODEL_API_KEY", None)
    if api_key:
        env["MODEL_API_KEY"] = api_key
    return env


def start_image_server(env: dict, port: int, log_path: Path) -> subprocess.Popen:
    fh = open(log_path, "w")
    proc = subprocess.Popen([sys.executable, "-m", "mcp_image_server"], cwd=REPO / "image_server",
                            env=env, stdout=fh, stderr=subprocess.STDOUT)
    deadline = time.time() + 30
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"image server exited early; see {log_path}")
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                log(f"image MCP server up on 127.0.0.1:{port} (log {log_path.relative_to(REPO)})")
                return proc
        time.sleep(0.3)
    proc.terminate()
    raise SystemExit(f"image server did not open port {port}; see {log_path}")


def run_logged(cmd: list, env: dict, log_path: Path, cwd: Path = REPO) -> int:
    """Run a child, streaming its output to the console and a log file."""
    log("$ " + " ".join(str(c) for c in cmd))
    with open(log_path, "w") as fh:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            fh.write(line)
            sys.stdout.write(line)
        return proc.wait()


# ------------------------------------------------------------------ report


def _jsonl(path: Path) -> list:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _rel(path: Path, base: Path) -> str:
    return os.path.relpath(path, base)


def _fmt_pct(x) -> str:
    return "—" if x is None else f"{x:.3f}"


def _stage_row(name: str, m: dict | None) -> str:
    if not m:
        return f"| {name} | — | — | — | — | — |"
    return (f"| {name} | {_fmt_pct(m.get('precision'))} | {_fmt_pct(m.get('recall'))} | **{_fmt_pct(m.get('f1'))}** "
            f"| {_fmt_pct(m.get('class_accuracy_on_matched'))} | {m.get('tp', 0)}/{m.get('fp', 0)}/{m.get('fn', 0)} |")


def _eval_section(metrics_path: Path, title: str) -> list:
    if not metrics_path.exists():
        return [f"### {title}", "", "_Not available (evaluation step skipped or failed)._", ""]
    res = json.loads(metrics_path.read_text())
    key = f"iou_{IOU}"
    lines = [f"### {title}", "",
             f"{res.get('eval_frames')} frames scored against `{_rel(Path(res.get('gt_dir', '')), REPO)}`"
             + (f"; audit changes: {res['audit']}" if res.get("audit") else "") + ".", "",
             f"| Stage (IoU ≥ {IOU}) | Precision | Recall | F1 | Identity acc. | TP/FP/FN |",
             "|---|---|---|---|---|---|"]
    for label, lm in (res.get("label_sets") or {}).items():
        lines.append(_stage_row(f"{label} (offline label set)", (lm.get("metrics") or {}).get(key)))
    for stage in ("raw", "revised", "validated"):
        st = (res.get("stages") or {}).get(stage) or {}
        lines.append(_stage_row({"raw": "Raw YOLO", "revised": "TrackCorrector (revised)",
                                 "validated": "Supervisor-validated"}[stage], (st.get("metrics") or {}).get(key)))
    raw = (((res.get("stages") or {}).get("raw") or {}).get("metrics") or {}).get(key) or {}
    val = (((res.get("stages") or {}).get("validated") or {}).get("metrics") or {}).get(key) or {}
    if raw.get("f1") is not None and val.get("f1") is not None:
        lines += ["", f"Raw → validated: F1 {raw['f1']:.3f} → **{val['f1']:.3f}** ({val['f1'] - raw['f1']:+.3f}), "
                      f"identity {raw.get('class_accuracy_on_matched', 0):.3f} → "
                      f"**{val.get('class_accuracy_on_matched', 0):.3f}**."]
    return lines + [""]


def write_report(out: Path, meta: dict) -> Path:
    runs = sorted((out / "tracking_runs").glob("view*")) if (out / "tracking_runs").is_dir() else []
    run_dir = runs[-1] if runs else None
    eval_dir = out / "evaluation"
    lines = ["# End-to-end demo report", ""]
    run = json.loads((run_dir / "run.json").read_text()) if run_dir and (run_dir / "run.json").exists() else {}
    counts = run.get("counts", {})
    offline = meta.get("offline", False)

    lines += [
        f"- **Window:** view {VIEW_ID}, 2026-10-28 16:33:00 → 16:38:59 (`--start {START} --end {END}`)",
        f"- **Data:** `{_rel(DATA, REPO)}` — {meta.get('manifest', {}).get('raw_detections', {}).get('frames', '?')} frames, "
        f"{meta.get('manifest', {}).get('raw_detections', {}).get('boxes', '?')} raw YOLO boxes, ground-truth labels for every frame",
        f"- **Mode:** {'OFFLINE (no model key: no supervisor verdicts, stub memory LLM)' if offline else 'full loop (Muse Spark supervisor + memory LLM)'}",
        f"- **Generated:** {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"- **Tracking run:** `{_rel(run_dir, REPO) if run_dir else '—'}` (its own session report: "
        + (f"[report.md]({_rel(run_dir / 'report.md', out)})" if run_dir and (run_dir / "report.md").exists() else "—") + ")",
        "",
        "## Stage timings", "",
        "| Stage | Seconds | Status |", "|---|---|---|",
    ]
    for name, t in meta.get("stages", {}).items():
        lines.append(f"| {name} | {t.get('seconds', 0):.1f} | {t.get('status', '')} |")
    lines.append("")

    if not run_dir:
        lines += ["_No tracking run found; the pipeline step failed. See `logs/`._", ""]
        path = out / "DEMO_REPORT.md"
        path.write_text("\n".join(lines))
        return path

    # 1. Tracking
    periods = _jsonl(run_dir / "periods.jsonl")
    names = run.get("cat_names", {})
    lines += [
        "## 1. Detection → tracking", "",
        f"- Frames processed: {counts.get('messages_processed', 0)} (empty frames included; they close visits)",
        f"- Raw detections kept (conf > 0.5, known cat): {counts.get('detections_kept', 0)} of {counts.get('detections_raw', 0)}"
        f"; decoy boxes dropped: {counts.get('detections_decoy', 0)}",
        f"- TrackCorrector revised rows: {counts.get('revised', 0)} (incl. interpolations); tombstoned rows: {counts.get('deleted', 0)}",
        f"- PeriodCorrector visits (periods): {counts.get('periods', 0)}", "",
        "| Period | Start | End | Cat | Score | Detect rate | Mean conf |", "|---|---|---|---|---|---|---|",
    ]
    def num(x):
        return f"{x:.2f}" if isinstance(x, (int, float)) else "—"

    for p in periods:
        w = p.get("winner") or {}
        cid = p.get("coordinated_cat_id")
        cat = f"{names.get(str(cid), cid)} ({cid})" if cid is not None else "unassigned"
        lines.append(f"| {p.get('period_id')} | {str(p.get('start'))[11:19]} | {str(p.get('end'))[11:19]} | "
                     f"{cat} | {num(w.get('score'))} | {num(w.get('detect_rate'))} | {num(w.get('detect_conf'))} |")
    lines.append("")

    # 2. Escalations + verdicts
    escs = _jsonl(run_dir / "escalations.jsonl")
    vals = {(v.get("escalation") or {}).get("name"): v for v in _jsonl(run_dir / "validations.jsonl")}
    validated = _jsonl(run_dir / "validated.jsonl")
    touched_by = Counter()
    for r in validated:
        for name in str(r.get("validated_by") or "").split(","):
            if name:
                touched_by[name] += 1
    by_code = Counter(e.get("code") for e in escs)
    lines += [
        "## 2. Escalation → supervisor agent", "",
        f"{len(escs)} escalations ({', '.join(f'{c}×{n}' for c, n in sorted(by_code.items())) or 'none'}). "
        "T-codes come from the TrackCorrector (frame scope), P-codes from the PeriodCorrector (visit scope), "
        "H1 is the quiet-window heartbeat. Each one went to the supervisor agent "
        "(escalations within a minute of each other share one merged call).", "",
        "| Escalation | When | Why it was raised | Verdict | Rows touched | Rationale |", "|---|---|---|---|---|---|",
    ]
    for e in escs:
        v = vals.get(e.get("name")) or {}
        review = v.get("review") or {}
        verdict = review.get("verdict") or ("— (offline)" if offline else "—")
        rationale = (review.get("rationale") or "").replace("|", "/").replace("\n", " ")
        if len(rationale) > 260:
            rationale = rationale[:257] + "…"
        lines.append(f"| {e.get('name')} | {str(e.get('time'))[11:19]} | {(e.get('reason') or '').replace('|', '/')} | "
                     f"**{verdict}** | {touched_by.get(e.get('name'), 0)} | {rationale} |")
    lines.append("")
    agent_runs = {}
    for v in vals.values():
        ar = v.get("agent_run") or {}
        if ar.get("run_dir"):
            agent_runs[ar["run_dir"]] = ar
    if agent_runs:
        cost = sum(a.get("est_cost_usd") or 0 for a in agent_runs.values())
        toks = sum((a.get("usage") or {}).get("total_tokens", 0) for a in agent_runs.values())
        imgs = sum(a.get("images_used", 0) for a in agent_runs.values())
        tools = sum(a.get("tool_calls", 0) for a in agent_runs.values())
        lines += [f"Supervisor spend: **${cost:.4f}** over {len(agent_runs)} agent calls "
                  f"({len(vals)} verdicts, {imgs} images, {tools} tool calls, {toks:,} tokens). Traces:", ""]
        for rd in agent_runs:
            trace = Path(rd) / "trace.md"
            lines.append(f"- [{Path(rd).name}]({_rel(trace, out)})")
        lines.append("")
    verdict_counts = counts.get("validations_by_verdict") or {}
    if verdict_counts:
        lines += [f"Verdicts: {', '.join(f'{k}×{n}' for k, n in sorted(verdict_counts.items()))}.", ""]
    touched = [r for r in validated if r.get("validated_by")]
    if validated:
        lines += [f"Verdicts applied to the rows: {len(touched)} of {len(validated)} validated rows carry a "
                  "`validated_by` stamp (`validated.jsonl`, `validated.db`). Visit-level (P) verdicts stamp every "
                  "row of the visit they confirm or correct, so most rows are touched; frame-level (T) verdicts "
                  "touch single frames.", ""]

    # 3. Memory
    mem_dir = Path(meta.get("memory_dir", out / "memory"))
    lines += ["## 3. Memory", ""]
    if not meta.get("memory", True):
        lines += ["_Memory was off for this run (`--no-memory`)._", ""]
    else:
        st_md = sorted(mem_dir.glob(f"view_{VIEW_ID}_short_term_*.md"))
        lt = mem_dir / f"view_{VIEW_ID}_long_term.yaml"
        lines += [f"Store: `{_rel(mem_dir, REPO)}`.", ""]
        if st_md:
            text = st_md[-1].read_text(encoding="utf-8")
            lines += [f"**Short-term (day) memory** — [{st_md[-1].name}]({_rel(st_md[-1], out)}): every combined period "
                      "activity (any confidence), tracker episodes and their verdicts. Excerpt:", "", "```text"]
            lines += text.strip().splitlines()[:40] + ["```", ""]
        else:
            lines += ["_No short-term memory file was written._", ""]
        if lt.exists():
            try:
                import yaml

                tree = yaml.safe_load(lt.read_text(encoding="utf-8")) or {}
                spatial = tree.get("spatial") or {}
                temporal = tree.get("temporal") or {}
                n_sp = sum(len(v or []) for v in spatial.values()) if isinstance(spatial, dict) else 0
                n_tm = sum(len(v or []) for v in temporal.values()) if isinstance(temporal, dict) else 0
                lines += [f"**Long-term memory** — [{lt.name}]({_rel(lt, out)}): {n_sp} spatial units and {n_tm} temporal "
                          "units hold only high-confidence activity (detector and every member label ≥ 0.7); the session "
                          "merge at the end promotes the day's curated summaries.", ""]
                for level in ("L2", "L1"):
                    for unit in (spatial.get(level) or [])[:2]:
                        if unit.get("activity"):
                            lines.append(f"- `{level}/{unit.get('id')}`: {unit.get('activity')}")
                lines.append("")
            except Exception as exc:  # noqa: BLE001 - report never fails on memory parsing
                lines += [f"Long-term memory: [{lt.name}]({_rel(lt, out)}) (could not summarize: {exc})", ""]
        else:
            lines += ["_No long-term memory record met the confidence gate in this window._", ""]

    # 4. Evaluation
    lines += ["## 4. Evaluation vs ground truth", ""]
    stem = run_dir.name
    lines += _eval_section(eval_dir / "as_given" / f"{stem}_metrics.json", "Ground truth as given")
    audited = eval_dir / "audited" / f"{stem}_metrics.json"
    audit_changes = (json.loads(audited.read_text()).get("audit") or {}) if audited.exists() else {}
    if audited.exists() and not any(audit_changes.values()):
        lines += ["### With documented GT audit (evaluation/gt_audit.yaml)", "",
                  f"The audit makes no changes to this hour's labels ({audit_changes or 'no entry'}), "
                  "so the numbers are the same as above.", ""]
    else:
        lines += _eval_section(audited, "With documented GT audit (evaluation/gt_audit.yaml)")
    full = eval_dir / "as_given" / f"{stem}_eval.md"
    if full.exists():
        lines += [f"Full evaluation (per class, IoU 0.75, cost breakdown): [{full.name}]({_rel(full, out)}).", ""]

    # 5. Charts
    lines += ["## 5. Timelines", "",
              "Per-cat strips over the window; red lines/bands mark escalations.", ""]
    for key, title in (("original", "Raw YOLO detections"), ("revised", "After the TrackCorrector"),
                       ("validated", "After supervisor verdicts"), ("gt", "Ground truth")):
        img = eval_dir / "as_given" / f"{stem}_{key}.png"
        if not img.exists():
            img = run_dir / f"{key}.png"
        if img.exists():
            lines += [f"**{title}**", "", f"![{title}]({_rel(img, out)})", ""]

    # 6. Artifacts
    lines += ["## 6. Artifacts", "",
              f"- Tracking run: `{_rel(run_dir, REPO)}/` — stream / revised / periods / images / escalations / "
              "validations / validated JSONL, validated.db, charts, report.md, run.json",
              f"- Evaluation: `{_rel(eval_dir, REPO)}/`",
              f"- Memory: `{_rel(mem_dir, REPO)}/`",
              f"- Logs: `{_rel(out / 'logs', REPO)}/` (pipeline, evaluation, image server)", ""]
    path = out / "DEMO_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ------------------------------------------------------------------ main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="no LLM calls: skip the supervisor agent, stub memory LLM")
    ap.add_argument("--no-memory", action="store_true", help="run without short-/long-term memory")
    ap.add_argument("--memory-dir", default=None, help="memory store to use (default: fresh <out>/memory)")
    ap.add_argument("--out", default=None, help="output dir (default output/demo/<timestamp>)")
    ap.add_argument("--process-viz", action="store_true", help="also render per-frame process frames + MP4 (slow)")
    ap.add_argument("--report-only", metavar="OUT_DIR", help="rebuild DEMO_REPORT.md for an earlier demo output dir")
    args = ap.parse_args(argv)

    if args.report_only:
        out = Path(args.report_only).resolve()
        meta = json.loads((out / "demo_meta.json").read_text()) if (out / "demo_meta.json").exists() else {}
        print(write_report(out, meta))
        return 0

    check_imports()
    manifest = check_data()
    api_key = None if args.offline else model_api_key()
    offline = api_key is None
    if offline and not args.offline:
        log("MODEL_API_KEY not found (env or .env): running OFFLINE (no supervisor verdicts, stub memory LLM)")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out).resolve() if args.out else REPO / "output" / "demo" / stamp
    (out / "logs").mkdir(parents=True, exist_ok=True)
    memory_dir = Path(args.memory_dir).resolve() if args.memory_dir else out / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    meta = {"manifest": manifest, "offline": offline, "memory": not args.no_memory,
            "memory_dir": str(memory_dir), "stages": {}}
    log(f"output: {out}")

    def stage(name, fn):
        t0 = time.time()
        try:
            status = fn()
        except SystemExit as exc:
            status = f"failed: {exc}"
        except Exception as exc:  # noqa: BLE001 - one failed stage still gets a report
            status = f"failed: {exc!r}"
        meta["stages"][name] = {"seconds": time.time() - t0, "status": status or "ok"}
        (out / "demo_meta.json").write_text(json.dumps(meta, indent=2, default=str))
        log(f"{name}: {meta['stages'][name]['status']} ({meta['stages'][name]['seconds']:.1f} s)")
        return meta["stages"][name]["status"]

    port = free_port()
    token = secrets.token_urlsafe(24)
    image_root = build_image_root(out)
    env_file = write_demo_env(out, port, token, image_root, memory_dir)
    env = child_env(env_file, api_key)
    server = None

    try:
        def _server():
            nonlocal server
            server = start_image_server(env, port, out / "logs" / "image_server.log")
            return "ok"

        stage("1. image MCP server", _server)

        def _pipeline():
            images = sorted((DATA / "images").glob("*.jpg"))
            cmd = [sys.executable, "-m", "agent.main", "--view-id", str(VIEW_ID), "--start", START, "--end", END,
                   "--replay", str(DATA / "raw_detections.jsonl"), "--plot", "--report",
                   "--out", str(out / "tracking_runs")]
            cmd += ["--no-validate"] if offline else ["--validate"]
            if args.no_memory:
                cmd += ["--no-memory"]
            else:
                cmd += ["--memory"] + (["--bootstrap-frame", str(images[0])] if images and not offline else [])
            if args.process_viz:
                cmd += ["--process-viz"]
            rc = run_logged(cmd, env, out / "logs" / "pipeline.log")
            return "ok" if rc == 0 else f"failed (exit {rc}; see logs/pipeline.log)"

        stage("2-4. tracking + escalation + supervisor + memory", _pipeline)
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()

    runs = sorted((out / "tracking_runs").glob("view*")) if (out / "tracking_runs").is_dir() else []
    if runs:
        run_dir = runs[-1]

        def _eval(sub, extra):
            def go():
                cmd = [sys.executable, "evaluation/evaluate.py", str(run_dir), "--hour", str(DATA),
                       "--out", str(out / "evaluation" / sub), "--iou", "0.5", "--iou", "0.75", *extra]
                rc = run_logged(cmd, env, out / "logs" / f"evaluation_{sub}.log")
                return "ok" if rc == 0 else f"failed (exit {rc})"
            return go

        stage("5a. evaluation (GT as given)", _eval("as_given", []))
        stage("5b. evaluation (GT audited)", _eval("audited", ["--audit", str(GT_AUDIT)]))

    report = write_report(out, meta)
    log(f"final report: {report}")
    failed = [k for k, v in meta["stages"].items() if not str(v.get("status")).startswith("ok")]
    if failed:
        log(f"stages with problems: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
