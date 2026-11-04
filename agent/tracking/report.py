"""Session report for tracking runs: one markdown summary per run directory.

:func:`write_session_report` reads a finished run directory (``run.json`` +
JSONL logs + charts), the LLM traces the run produced (validation agent runs
under ``output/agent_runs/``, memory traces under ``output/memory_runs/``),
and the memory stores, and writes ``report.md`` next to them: run identity
and timing, key metrics, the detection charts, periods, every escalation
with its validation verdict, a link to each LLM trace, memory-store
summaries, image/token/cost totals, and the artifact list.

``TrackingRun.finish()`` writes the report automatically (unless
``--no-report``); regenerate for any old run with::

    python -m agent.tracking.report output/tracking_runs/<run_name>

Every section degrades gracefully: a missing input yields a one-line note,
never an exception.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

REPORT_FILE = "report.md"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
AGENT_RUNS_DIR = REPO_ROOT / "output" / "agent_runs"
MEMORY_RUNS_DIR = REPO_ROOT / "output" / "memory_runs"


# ------------------------------------------------------------------ readers


def _read_jsonl(path: Path) -> list:
    rows = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                try:
                    rows.append(json.loads(text))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return rows


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _rel(run_dir: Path, target) -> str:
    """Target path rendered relative to the run dir (for markdown links)."""
    try:
        return os.path.relpath(str(target), str(run_dir))
    except (ValueError, OSError):
        return str(target)


def _shorten(text, n: int = 160) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _one_line(text, n: int = 160) -> str:
    """Table-safe single line (pipes escaped)."""
    return _shorten(text, n).replace("|", "\\|")


def _fmt_dur(seconds) -> str:
    if seconds is None:
        return "n/a"
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return "n/a"
    if seconds < 0:
        return "n/a"
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins, secs = divmod(seconds, 60)
    if mins < 60:
        return f"{int(mins)}m {secs:04.1f}s"
    hours, mins = divmod(int(mins), 60)
    return f"{hours}h {mins:02d}m {secs:04.1f}s"


def _fmt_ts(value) -> str:
    if isinstance(value, datetime.datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    text = str(value or "")
    try:
        return datetime.datetime.fromisoformat(text).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return text[:19] or "n/a"


def _file_size(path: Path) -> str:
    try:
        n = path.stat().st_size
    except OSError:
        return "n/a"
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 / 1024:.1f} MB"


# ------------------------------------------------------------------ trace stats


def _parse_stats_block(md_text: str) -> dict:
    """``{key: raw value}`` from a trace.md / memory-trace ``## Stats`` list."""
    stats: dict[str, str] = {}
    in_stats = False
    for line in (md_text or "").splitlines():
        if line.startswith("## "):
            in_stats = line.strip() == "## Stats"
            continue
        if not in_stats or not line.startswith("- "):
            continue
        body = line[2:]
        if ":" in body:
            key, value = body.split(":", 1)
            stats[key.strip()] = value.strip()
    return stats


def _num(value):
    """int/float from a stats value, else None (handles "n/a", "5 (high_res: 0)")."""
    if isinstance(value, (int, float)):
        return value
    if not isinstance(value, str):
        return None
    head = value.split("(")[0].strip().split()[0] if value.strip() else ""
    if head.lower() in ("", "n/a", "-", "none"):
        return None
    try:
        return float(head) if "." in head else int(head)
    except ValueError:
        return None


def _trace_file_stats(trace_path: Path) -> dict:
    """Numeric summary parsed from a trace markdown's Stats block ({} when unreadable)."""
    try:
        raw = trace_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    stats = _parse_stats_block(raw)
    out = {
        "model": stats.get("model", ""),
        "kind": stats.get("kind", ""),
        "label": stats.get("label", ""),
        "turns": _num(stats.get("turns")),
        "tool_calls": _num(stats.get("tool_calls")),
        "images_used": _num(stats.get("images_used")),
        "llm_ms": _num(stats.get("llm_ms")),
        "tools_ms": _num(stats.get("tools_ms")),
        "total_ms": _num(stats.get("total_ms")),
        "input_tokens": _num(stats.get("input_tokens")),
        "output_tokens": _num(stats.get("output_tokens")),
        "cached_tokens": _num(stats.get("cached_tokens")),
        "total_tokens": _num(stats.get("total_tokens")),
        "est_cost_usd": _num(stats.get("est_cost_usd")),
        "stopped_reason": stats.get("stopped_reason", ""),
    }
    return {k: v for k, v in out.items() if v not in (None, "")}


# ------------------------------------------------------------------ discovery


def _run_time_bounds(run_dir: Path, meta: dict, *, end_slack: float = 60.0) -> tuple[float, float]:
    """Epoch (start, end) covering the run, with slack for trace matching.

    Start gets −60 s; end gets +``end_slack`` (60 s default, more for memory:
    old runs stamped run.json before memory shutdown, so promote traces
    trail it by minutes). New runs carry started_at/finished_at in run.json;
    old runs fall back to the oldest/newest file mtime in the run dir
    (stream.jsonl is opened at run start, run.json written at the end).
    ``created_at`` is finish-side only — it must never stand in for the
    start, or the window collapses.
    """
    try:
        mtimes = [p.stat().st_mtime for p in run_dir.iterdir()]
    except OSError:
        mtimes = []

    def _epoch(value):
        try:
            return datetime.datetime.fromisoformat(str(value or "")).timestamp()
        except (ValueError, TypeError, OSError):
            return None

    start = _epoch(meta.get("started_at")) or (min(mtimes) if mtimes else None)
    if start is None:
        start = _epoch(meta.get("created_at"))
    end = _epoch(meta.get("finished_at")) or _epoch(meta.get("created_at")) or (max(mtimes) if mtimes else None)
    now = datetime.datetime.now().timestamp()
    start = start if start is not None else now
    end = end if end is not None else now
    return start - 60.0, max(end + end_slack, start)


def _mtime_in(path: Path, bounds) -> bool:
    try:
        return bounds[0] <= path.stat().st_mtime <= bounds[1]
    except OSError:
        return False


def _collect_validation_runs(run_dir: Path, ops: list, bounds) -> list[dict]:
    """One entry per validation agent call: {names, run_dir, trace, stats}.

    Primary source is each op's recorded ``agent_run`` stats; a time-window
    scan of ``output/agent_runs/*-validation/`` backfills runs whose ops
    predate the recording (old runs) or whose trace moved.
    """
    entries: dict[str, dict] = {}
    order: list[str] = []
    for op in ops:
        esc = op.get("escalation") or {}
        name = esc.get("name") or esc.get("code") or "?"
        agent_run = op.get("agent_run") or {}
        key = str(agent_run.get("run_dir") or agent_run.get("trace") or f"missing:{name}")
        entry = entries.get(key)
        if entry is None:
            entry = {"names": [], "run_dir": agent_run.get("run_dir"), "trace": agent_run.get("trace"), "stats": {}}
            entries[key] = entry
            order.append(key)
        entry["names"].append(name)
        if agent_run and not entry["stats"]:
            entry["stats"] = {
                "model": agent_run.get("model", ""),
                "turns": agent_run.get("turns"),
                "tool_calls": agent_run.get("tool_calls"),
                "images_used": agent_run.get("images_used"),
                "llm_ms": agent_run.get("llm_ms"),
                "tools_ms": agent_run.get("tools_ms"),
                "total_ms": agent_run.get("total_ms"),
                "stopped_reason": agent_run.get("stopped_reason", ""),
            }
            usage = agent_run.get("usage") or {}
            for tok in ("input_tokens", "output_tokens", "cached_tokens", "total_tokens"):
                if usage.get(tok) is not None:
                    entry["stats"][tok] = usage[tok]
            if agent_run.get("est_cost_usd") is not None:
                entry["stats"]["est_cost_usd"] = agent_run.get("est_cost_usd")
            entry["stats"] = {k: v for k, v in entry["stats"].items() if v not in (None, "")}
    # Backfill: validation dirs in the window that no op claimed.
    claimed = {str(e["trace"]) for e in entries.values() if e.get("trace")}
    claimed |= {str(e["run_dir"]) for e in entries.values() if e.get("run_dir")}
    try:
        candidates = sorted(AGENT_RUNS_DIR.iterdir()) if AGENT_RUNS_DIR.is_dir() else []
    except OSError:
        candidates = []
    for cand in candidates:
        if not cand.is_dir() or "-validation" not in cand.name:
            continue
        trace = cand / "trace.md"
        if str(trace) in claimed or str(cand) in claimed:
            continue
        if not _mtime_in(trace if trace.exists() else cand, bounds):
            continue
        key = f"scan:{cand.name}"
        entries[key] = {"names": [], "run_dir": str(cand), "trace": str(trace) if trace.exists() else None, "stats": {}}
        order.append(key)
    # Parse stats from trace files where the op carried none.
    for key in order:
        entry = entries[key]
        if not entry["stats"] and entry.get("trace"):
            entry["stats"] = _trace_file_stats(Path(entry["trace"]))
        # Attribute scanned runs to member names via the trace's case headers.
        if not entry["names"] and entry.get("trace"):
            entry["names"] = _trace_case_names(Path(entry["trace"]))
    # Drop the "missing" placeholders that never resolved to a real run.
    return [entries[k] for k in order if not k.startswith("missing:") or entries[k]["stats"] or entries[k]["trace"]]


def _trace_case_names(trace_path: Path) -> list[str]:
    """Escalation names reviewed in a validation trace.

    Merged batches carry "=== Case N: <name> ===" headers; every prompt
    (solo or merged member) also opens with "[<name>]" on its first line,
    which is the fallback for solo calls.
    """
    import re

    names: list[str] = []
    try:
        text = trace_path.read_text(encoding="utf-8")
    except OSError:
        return names
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("=== Case ") and ": " in line and line.endswith(" ==="):
            name = line.split(": ", 1)[1][: -len(" ===")].strip()
            if name and name not in names:
                names.append(name)
    for match in re.finditer(r"^\[(E\d+-[A-Z]+\d*)\]", text, re.MULTILINE):
        if match.group(1) not in names:
            names.append(match.group(1))
    return names


def _collect_memory_traces(bounds) -> list[dict]:
    """Memory LLM traces written during the run: {path, kind, label, stats, images}."""
    try:
        cands = sorted(MEMORY_RUNS_DIR.glob("*.md")) if MEMORY_RUNS_DIR.is_dir() else []
    except OSError:
        return []
    out = []
    for md in cands:
        if not _mtime_in(md, bounds):
            continue
        stats = _trace_file_stats(md)
        stem = md.stem
        images = sorted(p.name for p in MEMORY_RUNS_DIR.glob(stem + "*.jpg")) + sorted(
            p.name for p in MEMORY_RUNS_DIR.glob(stem + "*.png")
        )
        # Filenames look like <stamp>-<seq>-<kind>-<label>.md; the Stats block
        # is authoritative, the filename is the fallback.
        kind = stats.get("kind") or ""
        label = stats.get("label") or ""
        if not kind or not label:
            tail = stem.split("-", 4)
            if len(tail) == 5:
                kind = kind or tail[3]
                label = label or tail[4]
        out.append({"path": md, "kind": kind or "?", "label": label or md.name, "stats": stats, "images": images})
    return out


# ------------------------------------------------------------------ memory stores


def _memory_paths(view_id, days: list[str]) -> dict:
    """Long-term + per-day short-term store paths (via config when available)."""
    try:
        from agent.memory import MemoryConfig

        cfg = MemoryConfig.from_yaml()
        store_dir = Path(cfg.store_dir)
        if not store_dir.is_absolute():
            store_dir = REPO_ROOT / store_dir
        long_term = store_dir / f"view_{view_id}_long_term.yaml"
        short_terms = [(d, store_dir / f"view_{view_id}_short_term_{d}.md", store_dir / f"view_{view_id}_short_term_{d}.json") for d in days]
        return {"long_term": long_term, "short_terms": short_terms, "trace_dir": MEMORY_RUNS_DIR}
    except Exception:
        store_dir = REPO_ROOT / "agent" / "memory" / "data"
        return {
            "long_term": store_dir / f"view_{view_id}_long_term.yaml",
            "short_terms": [
                (d, store_dir / f"view_{view_id}_short_term_{d}.md", store_dir / f"view_{view_id}_short_term_{d}.json") for d in days
            ],
            "trace_dir": MEMORY_RUNS_DIR,
        }


def _long_term_summary(path: Path) -> dict:
    """Unit counts per level from the long-term YAML ({} when unreadable)."""
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    out: dict[str, int] = {}
    for label, items in ((data.get("spatial") or {}).items()):
        out[f"spatial.{label}"] = len(items or [])
    for name, items in ((data.get("temporal") or {}).items()):
        out[f"temporal.{name}"] = len(items or [])
    out["bootstrapped"] = bool(data.get("bootstrapped", False))
    return out


def _window_days(meta: dict) -> list[str]:
    days: list[str] = []
    try:
        start = datetime.datetime.fromisoformat(str((meta.get("window") or {}).get("start", "")))
        end = datetime.datetime.fromisoformat(str((meta.get("window") or {}).get("end", "")))
        day = start.date()
        while day <= end.date() and len(days) < 366:
            days.append(day.isoformat())
            day += datetime.timedelta(days=1)
    except ValueError:
        pass
    return days


# ------------------------------------------------------------------ sections


def _reconstruct_command(meta: dict) -> str:
    args = meta.get("args") or {}
    parts = ["python -m agent.main"]
    if args.get("view_id") is not None:
        parts.append(f"--view-id {args['view_id']}")
    for key in ("start", "end"):
        if args.get(key):
            parts.append(f"--{key} {args[key]}")
    if args.get("live"):
        parts.append("--live")
    elif args.get("replay"):
        parts.append(f"--replay {args['replay']}")
    elif args.get("replay_db"):
        parts.append("--replay-db")
    for flag in ("memory", "validate", "plot", "process_viz"):
        if args.get(flag):
            parts.append(f"--{flag}".replace("_", "-"))
    for key in ("min_conf", "cat_idx", "fps", "port", "out", "bootstrap_frame"):
        if args.get(key) not in (None, ""):
            parts.append(f"--{key.replace('_', '-')} {args[key]}")
    if args.get("no_video"):
        parts.append("--no-video")
    return " ".join(parts)


def _sum_stat(entries: list[dict], key: str) -> float | None:
    total = 0.0
    seen = False
    for entry in entries:
        value = _num((entry.get("stats") or {}).get(key))
        if value is not None:
            total += value
            seen = True
    return total if seen else None


def _fmt_num(value, decimals: int = 1) -> str:
    if value is None:
        return "n/a"
    if decimals == 0:
        return f"{value:,.0f}"
    return f"{value:,.{decimals}f}"


def _fmt_cost(value) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.4f}"


def _header_lines(meta: dict, run_dir: Path) -> list[str]:
    window = meta.get("window") or {}
    cats = meta.get("cat_ids") or []
    names = meta.get("cat_names") or {}
    who = ", ".join(f"{names.get(str(c), names.get(c, '?'))} (id {c})" for c in cats) or "n/a"
    elapsed = meta.get("elapsed_s")
    if elapsed is None:
        try:
            elapsed = (
                datetime.datetime.fromisoformat(str(meta.get("finished_at", "")))
                - datetime.datetime.fromisoformat(str(meta.get("started_at", "")))
            ).total_seconds()
        except (ValueError, TypeError):
            elapsed = None
    started_note = " (~estimated from file times)" if meta.get("_started_estimated") else ""
    return [
        f"# Session report — {meta.get('run_name', run_dir.name)}",
        "",
        f"- mode: {meta.get('mode', 'n/a')} · view {meta.get('view_id', 'n/a')} · cats: {who}",
        f"- window: {window.get('start', '?')} → {window.get('end', '?')} (rule: {window.get('rule', 'n/a')})",
        f"- started: {_fmt_ts(meta.get('started_at') or meta.get('created_at'))}{started_note} · "
        f"finished: {_fmt_ts(meta.get('finished_at') or meta.get('created_at'))} · "
        f"wall time: {_fmt_dur(elapsed)}",
        f"- run dir: `{run_dir}`",
        "",
        "```",
        _reconstruct_command(meta),
        "```",
        "",
    ]


def _metrics_lines(meta: dict, counts: dict, n_validated: int | None) -> list[str]:
    rows = [
        ("messages seen / processed", f"{counts.get('messages_seen', 'n/a')} / {counts.get('messages_processed', 'n/a')}"),
        ("detections raw / kept", f"{counts.get('detections_raw', 'n/a')} / {counts.get('detections_kept', 'n/a')}"),
        ("detections dropped (low-conf / unknown-cat / bad-box)",
         f"{counts.get('detections_low_conf', 0)} / {counts.get('detections_unknown_cat', 0)} / {counts.get('detections_bad_box', 0)}"),
        ("stream rows (original)", f"{counts.get('stream', 'n/a')}"),
        ("revised rows (+ tombstones)", f"{counts.get('revised', 'n/a')} (+{counts.get('deleted', 0)})"),
        ("periods closed", f"{counts.get('periods', 'n/a')}"),
        ("escalations", f"{counts.get('escalations_total', counts.get('escalations', 'n/a'))}"),
        ("validations journaled", f"{counts.get('validations_total', counts.get('validations', 'n/a'))}"),
        ("validated rows", f"{n_validated if n_validated is not None else 'n/a (no --validate)'}"),
        ("escalations by code", ", ".join(f"{k}×{v}" for k, v in sorted((counts.get("escalations_by_code") or {}).items())) or "n/a"),
        ("verdicts", ", ".join(f"{k}×{v}" for k, v in sorted((counts.get("validations_by_verdict") or {}).items())) or "n/a"),
    ]
    lines = ["## Key metrics", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    return lines + [""]


def _chart_lines(run_dir: Path) -> list[str]:
    lines = ["## Detection charts", ""]
    charts = [("original.png", "Original detections"), ("revised.png", "Revised detections"), ("validated.png", "Validated detections")]
    if not any((run_dir / name).exists() for name, _ in charts):
        return lines + ["(no charts; the run used `--no-plot` or charting failed)", ""]
    for name, title in charts:
        if (run_dir / name).exists():
            lines += [f"### {title}", "", f"![{title}]({name})", ""]
    return lines


def _period_lines(periods: list, names: dict) -> list[str]:
    lines = ["## Periods", ""]
    if not periods:
        return lines + ["(no periods closed)", ""]
    lines += ["| Period | Span | Cat | Detections | Mean conf | Detect rate | Score |", "|---|---|---|---|---|---|---|"]
    for p in periods:
        winner = p.get("winner") or {}
        cat = p.get("coordinated_cat_id")
        label = f"{names.get(str(cat), names.get(cat, '?'))} ({cat})" if cat is not None else "unassigned"
        span = f"{_fmt_ts(p.get('start'))} → {_fmt_ts(p.get('end'))}"
        lines.append(
            f"| {p.get('period_id', '?')} | {span} | {label} | {winner.get('detect_length', 'n/a')} | "
            f"{winner.get('detect_conf', 'n/a')} | {winner.get('detect_rate', 'n/a')} | {winner.get('score', 'n/a')} |"
        )
    return lines + [""]


def _escalation_lines(run_dir: Path, escalations: list, ops: list, val_runs: list[dict]) -> list[str]:
    lines = ["## Escalations & validation outputs", ""]
    if not escalations:
        return lines + ["(no escalations emitted)", ""]
    by_name = {}
    for op in ops:
        esc = op.get("escalation") or {}
        if esc.get("name"):
            by_name[esc["name"]] = op
    trace_for: dict[str, str] = {}
    for entry in val_runs:
        if entry.get("trace"):
            link = _rel(run_dir, entry["trace"])
            for name in entry["names"]:
                trace_for.setdefault(name, link)
    lines += ["| Escalation | Scope | Time | Reason | Verdict | Trace |", "|---|---|---|---|---|---|"]
    for esc in escalations:
        name = esc.get("name") or f"E{esc.get('seq', '?')}-{esc.get('code', '?')}"
        op = by_name.get(name, {})
        review = op.get("review") or {}
        verdict = review.get("verdict", "n/a" if ops else "not validated (--no-validate)")
        trace = f"[trace]({trace_for[name]})" if name in trace_for else "n/a"
        lines.append(
            f"| {name} | {esc.get('scope', '?')} | {_fmt_ts(esc.get('time'))} | "
            f"{_one_line(esc.get('reason'), 120)} | {verdict} | {trace} |"
        )
    lines += [""]
    if any((op.get("review") or {}).get("rationale") for op in ops):
        lines += ["### Verdict rationales", ""]
        for esc in escalations:
            name = esc.get("name") or "?"
            review = (by_name.get(name) or {}).get("review") or {}
            if review.get("rationale"):
                lines.append(f"- **{name}** ({review.get('verdict')}): {_one_line(review['rationale'], 400)}")
        lines += [""]
    return lines


def _tracing_lines(run_dir: Path, val_runs: list[dict], mem_traces: list[dict]) -> list[str]:
    lines = ["## LLM tracing", ""]
    lines += ["### Validation agent calls", ""]
    if not val_runs:
        lines += ["(no validation agent runs found — `--no-validate`, or traces predate this run)", ""]
    else:
        lines += ["| Call | Cases | Turns | Tool calls | Images | LLM time | Tokens in/out | Cost | Trace |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for i, entry in enumerate(val_runs, 1):
            stats = entry.get("stats") or {}
            cases = ", ".join(entry["names"]) or "?"
            link = f"[trace]({_rel(run_dir, entry['trace'])})" if entry.get("trace") else "n/a"
            tokens = f"{stats.get('input_tokens', '?')}/{stats.get('output_tokens', '?')}"
            lines.append(
                f"| {i} | {_one_line(cases, 60)} | {stats.get('turns', '?')} | {stats.get('tool_calls', '?')} | "
                f"{stats.get('images_used', '?')} | {_fmt_dur((stats.get('llm_ms') or 0) / 1000) if stats.get('llm_ms') is not None else 'n/a'} | "
                f"{tokens} | {_fmt_cost(stats.get('est_cost_usd'))} | {link} |"
            )
        lines += [""]
    lines += ["### Memory LLM calls", ""]
    if not mem_traces:
        lines += ["(no memory traces in this run's window — `--no-memory`, or the stub client)", ""]
    else:
        lines += ["| Kind | Label | Images | LLM time | Tokens in/out | Cost | Trace |", "|---|---|---|---|---|---|---|"]
        for trace in mem_traces:
            stats = trace["stats"]
            link = f"[trace]({_rel(run_dir, trace['path'])})"
            tokens = f"{stats.get('input_tokens', '?')}/{stats.get('output_tokens', '?')}"
            lines.append(
                f"| {trace['kind']} | {_one_line(trace['label'], 40)} | {len(trace['images'])} | "
                f"{_fmt_dur((stats.get('llm_ms') or 0) / 1000) if stats.get('llm_ms') is not None else 'n/a'} | "
                f"{tokens} | {_fmt_cost(stats.get('est_cost_usd'))} | {link} |"
            )
        lines += [""]
    return lines


def _images_cost_lines(val_runs: list[dict], mem_traces: list[dict]) -> list[str]:
    val_images = _sum_stat(val_runs, "images_used") or 0
    mem_images = sum(len(t["images"]) for t in mem_traces)
    val_ms = _sum_stat(val_runs, "llm_ms")
    mem_ms = _sum_stat(mem_traces, "llm_ms")
    val_tokens = (_sum_stat(val_runs, "input_tokens"), _sum_stat(val_runs, "output_tokens"))
    mem_tokens = (_sum_stat(mem_traces, "input_tokens"), _sum_stat(mem_traces, "output_tokens"))
    val_cost = _sum_stat(val_runs, "est_cost_usd")
    mem_cost = _sum_stat(mem_traces, "est_cost_usd")
    total_cost = (val_cost or 0) + (mem_cost or 0) if val_cost is not None or mem_cost is not None else None
    activity_calls = sum(1 for t in mem_traces if t["kind"] == "activity")
    return [
        "## Images, tokens & cost",
        "",
        "| Resource | Validation | Memory | Total |",
        "|---|---|---|---|",
        f"| LLM calls | {len(val_runs)} | {len(mem_traces)} | {len(val_runs) + len(mem_traces)} |",
        f"| Images analyzed | {_fmt_num(val_images, 0)} | {_fmt_num(mem_images, 0)} | {_fmt_num(val_images + mem_images, 0)} |",
        f"| LLM time | {_fmt_dur((val_ms or 0) / 1000) if val_ms is not None else 'n/a'} | "
        f"{_fmt_dur((mem_ms or 0) / 1000) if mem_ms is not None else 'n/a'} | "
        f"{_fmt_dur(((val_ms or 0) + (mem_ms or 0)) / 1000) if val_ms is not None or mem_ms is not None else 'n/a'} |",
        f"| Input tokens | {_fmt_num(val_tokens[0], 0)} | {_fmt_num(mem_tokens[0], 0)} | "
        f"{_fmt_num((val_tokens[0] or 0) + (mem_tokens[0] or 0), 0) if val_tokens[0] is not None or mem_tokens[0] is not None else 'n/a'} |",
        f"| Output tokens | {_fmt_num(val_tokens[1], 0)} | {_fmt_num(mem_tokens[1], 0)} | "
        f"{_fmt_num((val_tokens[1] or 0) + (mem_tokens[1] or 0), 0) if val_tokens[1] is not None or mem_tokens[1] is not None else 'n/a'} |",
        f"| Est. cost | {_fmt_cost(val_cost)} | {_fmt_cost(mem_cost)} | {_fmt_cost(total_cost)} |",
        "",
        f"_Validation images = frames the reviewer fetched via `images__get_image`; memory images = key frames judged "
        f"for activity ({activity_calls} activity calls) plus bootstrap frames._",
        "",
    ]


def _memory_store_lines(run_dir: Path, meta: dict) -> list[str]:
    lines = ["## Memory stores", ""]
    view_id = meta.get("view_id")
    if view_id is None:
        return lines + ["(unknown view; no store lookup attempted)", ""]
    paths = _memory_paths(view_id, _window_days(meta))
    long_term = paths["long_term"]
    if long_term.exists():
        summary = _long_term_summary(long_term)
        units = ", ".join(f"{k}×{v}" for k, v in sorted(summary.items()) if k != "bootstrapped") or "empty"
        lines += [
            f"- long-term: [{_rel(run_dir, long_term)}]({_rel(run_dir, long_term)}) "
            f"({_file_size(long_term)}; {units}; bootstrapped: {summary.get('bootstrapped', '?')})"
        ]
    else:
        lines.append(f"- long-term: `{_rel(run_dir, long_term)}` (not written — `--no-memory` or nothing qualified)")
    run_name = meta.get("run_name", "")
    for day, md_path, json_path in paths["short_terms"]:
        bits = []
        session = {}
        if json_path.exists():
            data = _read_json(json_path)
            session = ((data.get("sessions") or {}).get(run_name)) or {}
            periods = session.get("periods") or []
            episodes = session.get("episodes") or []
            acts = [p.get("activity") for p in periods if p.get("activity")]
            bits.append(f"{len(periods)} periods, {len(episodes)} episodes")
            if acts:
                bits.append("activities: " + ", ".join(dict.fromkeys(acts)))
        if md_path.exists() or json_path.exists():
            links = " / ".join(
                f"[{_rel(run_dir, p)}]({_rel(run_dir, p)})" for p in (md_path, json_path) if p.exists()
            )
            lines.append(f"- short-term {day}: {links}" + (f" ({'; '.join(bits)})" if bits else ""))
    if len(lines) == 2:
        lines.append("(no short-term day files overlap this run's window)")
    lines += [""]
    return lines


def _validated_lines(validated_rows: list, ops: list) -> list[str]:
    lines = ["## Validated output", ""]
    if not validated_rows:
        return lines + ["(no validated rows — `--no-validate`, or validation produced no output)", ""]
    by_verdict = Counter(r.get("verdict", "?") for r in validated_rows)
    touched = sum(1 for r in validated_rows if r.get("validated_by") not in (None, "", "pass-through"))
    lines += [
        f"- rows: {len(validated_rows)} ({touched} touched by a verdict, {len(validated_rows) - touched} pass-through)",
        "- by verdict: " + ", ".join(f"{k}×{v}" for k, v in sorted(by_verdict.items())),
    ]
    merges = [
        (op.get("escalation") or {}).get("name")
        for op in ops
        if (op.get("review") or {}).get("verdict") in ("merge", "merge_visits")
    ]
    if merges:
        lines.append(f"- merges applied: {', '.join(m for m in merges if m)}")
    drops = sum(1 for op in ops if (op.get("review") or {}).get("verdict") in ("drop", "drop_box", "cut_at_gap_start", "shorten_gap"))
    if drops:
        lines.append(f"- drop/trim verdicts: {drops}")
    return lines + [""]


def _artifact_lines(run_dir: Path, meta: dict) -> list[str]:
    lines = ["## Artifacts", ""]
    try:
        files = sorted((p for p in run_dir.iterdir() if p.is_file()), key=lambda p: p.name)
    except OSError:
        return lines + ["(run dir unreadable)", ""]
    lines += ["| File | Size |", "|---|---|"]
    for path in files:
        if path.name == REPORT_FILE:
            continue
        lines.append(f"| [{path.name}]({path.name}) | {_file_size(path)} |")
    frames = run_dir / "process_frames"
    if frames.is_dir():
        try:
            n = sum(1 for _ in frames.iterdir())
        except OSError:
            n = "?"
        lines.append(f"| [process_frames/](process_frames/) | {n} frames |")
    declared = meta.get("artifacts") or []
    missing = [a for a in declared if a != REPORT_FILE and not (run_dir / a.rstrip("/")).exists()]
    if missing:
        lines.append("")
        lines.append(f"_Declared but missing: {', '.join(missing)}_")
    return lines + [""]


# ------------------------------------------------------------------ writer


def write_session_report(run_dir: str | Path, *, run_name: str | None = None) -> Path:
    """Write ``report.md`` into a finished run directory; returns its path.

    Reads ``run.json`` plus the JSONL logs, charts, LLM traces and memory
    stores; every unreadable input degrades its section to a one-line note.
    """
    run_dir = Path(run_dir)
    meta = _read_json(run_dir / "run.json")
    if run_name:
        meta.setdefault("run_name", run_name)
    meta.setdefault("run_name", run_dir.name)
    counts = meta.get("counts") or {}
    names = meta.get("cat_names") or {}

    ops = _read_jsonl(run_dir / "validations.jsonl")
    escalations = _read_jsonl(run_dir / "escalations.jsonl")
    periods = _read_jsonl(run_dir / "periods.jsonl")
    validated_rows = [r for r in _read_jsonl(run_dir / "validated.jsonl") if isinstance(r, dict)]

    bounds = _run_time_bounds(run_dir, meta)
    mem_bounds = _run_time_bounds(run_dir, meta, end_slack=600.0)
    if meta.get("elapsed_s") is None:
        span = (bounds[1] - bounds[0]) - 120.0  # bounds carry ±60 s slack
        if span > 0:
            meta["elapsed_s"] = round(span, 1)
    if not meta.get("started_at"):
        # Display-only estimate for old runs (never written back to run.json).
        meta["started_at"] = datetime.datetime.fromtimestamp(bounds[0] + 60.0).isoformat()
        meta["_started_estimated"] = True
    val_runs = _collect_validation_runs(run_dir, ops, bounds)
    mem_traces = _collect_memory_traces(mem_bounds)

    lines: list[str] = []
    lines += _header_lines(meta, run_dir)
    lines += _metrics_lines(meta, counts, len(validated_rows) if validated_rows or (run_dir / "validated.jsonl").exists() else None)
    lines += _chart_lines(run_dir)
    lines += _period_lines(periods, names)
    lines += _escalation_lines(run_dir, escalations, ops, val_runs)
    lines += _tracing_lines(run_dir, val_runs, mem_traces)
    lines += _images_cost_lines(val_runs, mem_traces)
    lines += _memory_store_lines(run_dir, meta)
    lines += _validated_lines(validated_rows, ops)
    lines += _artifact_lines(run_dir, meta)

    out = run_dir / REPORT_FILE
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("session report written: %s", out)
    return out


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="agent.tracking.report",
        description="Regenerate report.md for a finished tracking run directory.",
    )
    parser.add_argument("run_dir", help="run directory (or run name under output/tracking_runs/)")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        cand = REPO_ROOT / "output" / "tracking_runs" / args.run_dir
        if cand.is_dir():
            run_dir = cand
    if not run_dir.is_dir():
        parser.error(f"run directory not found: {args.run_dir}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] %(message)s")
    write_session_report(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["REPORT_FILE", "write_session_report", "main"]
