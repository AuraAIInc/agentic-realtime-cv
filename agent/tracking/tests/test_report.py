"""Session report tests: sections, links, graceful degradation."""

import datetime
import json
from pathlib import Path

from agent.tracking import REPORT_FILE, write_session_report


def _write_jsonl(path: Path, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _trace_md(*, model="muse-spark-1.3", extra=()) -> str:
    lines = [
        "# Agent run",
        "",
        "## Stats",
        "",
        f"- model: {model}",
        "- turns: 2",
        "- tool_calls: 3",
        "- llm_ms: 1200.5",
        "- tools_ms: 300.0",
        "- total_ms: 1500.5",
        "- input_tokens: 1000",
        "- output_tokens: 200",
        "- cached_tokens: 100",
        "- total_tokens: 1200",
        "- est_cost_usd: 0.002500",
        "- images_used: 2 (high_res: 0)",
        *extra,
        "",
    ]
    return "\n".join(lines)


def _run_meta(run_dir: Path, **overrides) -> dict:
    now = datetime.datetime.now()
    meta = {
        "run_name": run_dir.name,
        "mode": "replay",
        "args": {"view_id": 999, "start": "2026-10-27-18-00", "end": "2026-10-27-18-25",
                 "replay_db": True, "memory": True, "validate": True, "plot": True},
        "view_id": 999,
        "cat_ids": [25],
        "cat_names": {"25": "Qb"},
        "window": {"start": "2026-10-27T18:00", "end": "2026-10-27T18:25",
                   "end_exclusive": "2026-10-27T18:26:00", "rule": "start <= t < end + 60s"},
        "counts": {"messages_seen": 10, "messages_processed": 10, "detections_raw": 4, "detections_kept": 4,
                   "stream": 4, "revised": 6, "periods": 1, "escalations": 1, "validations": 1,
                   "escalations_total": 1, "escalations_by_code": {"T7": 1},
                   "validations_total": 1, "validations_by_verdict": {"keep_synthesis": 1}},
        "artifacts": ["stream.jsonl", "run.json"],
        "started_at": (now - datetime.timedelta(seconds=90)).isoformat(),
        "finished_at": now.isoformat(),
        "elapsed_s": 90.0,
        "created_at": now.isoformat(),
        "run_dir": str(run_dir),
    }
    meta.update(overrides)
    return meta


def _seed_run(tmp_path: Path, monkeypatch, *, with_traces=True) -> Path:
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps(_write_meta(run_dir)), encoding="utf-8")
    _write_jsonl(run_dir / "escalations.jsonl", [
        {"escalation": True, "seq": 1, "name": "E1-T7", "time": "2026-10-27T18:00:49",
         "end_time": None, "scope": "frame", "code": "T7", "reason": "gap with no detection",
         "cat_id": 25, "period_id": 1, "metrics": {}},
    ])
    agent_run = None
    if with_traces:
        agent_dir = tmp_path / "agent_runs" / "20261103-210000-000000-validation"
        agent_dir.mkdir(parents=True)
        (agent_dir / "trace.md").write_text(_trace_md(), encoding="utf-8")
        (agent_dir / "step01_get_image_0.jpg").write_bytes(b"fake-jpg")
        agent_run = {"run_dir": str(agent_dir), "trace": str(agent_dir / "trace.md"),
                     "model": "muse-spark-1.3", "turns": 2, "tool_calls": 3, "images_used": 2,
                     "high_res_used": 0, "llm_ms": 1200.5, "tools_ms": 300.0, "total_ms": 1500.5,
                     "usage": {"input_tokens": 1000, "output_tokens": 200, "cached_tokens": 100, "total_tokens": 1200},
                     "est_cost_usd": 0.0025, "stopped_reason": "answered"}
        monkeypatch.setattr("agent.tracking.report.AGENT_RUNS_DIR", tmp_path / "agent_runs")
        mem_dir = tmp_path / "memory_runs"
        mem_dir.mkdir()
        (mem_dir / "20261103-210100-000000-001-activity-period-1.md").write_text(
            _trace_md(extra=("- kind: activity", "- label: period 1")), encoding="utf-8")
        (mem_dir / "20261103-210100-000000-001-activity-period-1-0.jpg").write_bytes(b"fake-jpg")
        monkeypatch.setattr("agent.tracking.report.MEMORY_RUNS_DIR", mem_dir)
    else:
        monkeypatch.setattr("agent.tracking.report.AGENT_RUNS_DIR", tmp_path / "agent_runs_empty")
        monkeypatch.setattr("agent.tracking.report.MEMORY_RUNS_DIR", tmp_path / "memory_runs_empty")
    _write_jsonl(run_dir / "validations.jsonl", [
        {"escalation": {"escalation": True, "seq": 1, "name": "E1-T7", "time": "2026-10-27T18:00:49",
                        "scope": "frame", "code": "T7", "reason": "gap", "cat_id": 25, "period_id": 1},
         "review": {"verdict": "keep_synthesis", "rationale": "cat visible in gap frames", "cut_time": None},
         "agent_run": agent_run},
    ])
    _write_jsonl(run_dir / "periods.jsonl", [
        {"period_id": 1, "start": "2026-10-27 18:00:00.000000", "end": "2026-10-27 18:01:18.000000",
         "coordinated_cat_id": 25, "winner": {"detect_length": 4, "detect_conf": 0.55,
         "detect_rate": 0.34, "period_area_count": 1, "score": 12.2}, "view_id": 999},
    ])
    _write_jsonl(run_dir / "validated.jsonl", [
        {"create_time": "2026-10-27T18:00:01", "cat_id": 25, "period_id": 1, "view_id": 999,
         "validated_by": "E1-T7", "verdict": "keep_synthesis"},
        {"create_time": "2026-10-27T18:00:02", "cat_id": 25, "period_id": 1, "view_id": 999,
         "validated_by": "pass-through", "verdict": "pass"},
    ])
    (run_dir / "original.png").write_bytes(b"fake-png")
    return run_dir


def _write_meta(run_dir: Path) -> dict:
    return _run_meta(run_dir)


def test_report_covers_all_sections_and_links(tmp_path, monkeypatch):
    run_dir = _seed_run(tmp_path, monkeypatch, with_traces=True)
    out = write_session_report(run_dir)
    assert out == run_dir / REPORT_FILE
    text = out.read_text(encoding="utf-8")
    for section in ("# Session report", "## Key metrics", "## Detection charts", "## Periods",
                    "## Escalations & validation outputs", "## LLM tracing", "## Images, tokens & cost",
                    "## Memory stores", "## Validated output", "## Artifacts"):
        assert section in text, section
    assert "![Original detections](original.png)" in text
    assert "E1-T7" in text and "keep_synthesis" in text
    assert "cat visible in gap frames" in text  # rationale
    assert "trace.md" in text  # validation + memory trace links
    assert "activity" in text  # memory trace kind row
    assert "90.0s" in text or "1m 30" in text  # wall time
    assert "view_999_long_term.yaml" in text  # memory store lookup (missing -> noted)
    assert "keep_synthesis×1" in text  # validated verdict distribution
    assert "run.json" in text  # artifact list


def test_report_header_notes_frame_size_when_present(tmp_path, monkeypatch):
    run_dir = _seed_run(tmp_path, monkeypatch, with_traces=False)
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert "(detection coords)" not in write_session_report(run_dir).read_text(encoding="utf-8")
    meta["img_size"] = [800, 600]
    (run_dir / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    text = write_session_report(run_dir).read_text(encoding="utf-8")
    assert "frame: 800x600 (detection coords)" in text


def test_report_degrades_without_traces_charts_or_validate(tmp_path, monkeypatch):
    run_dir = _seed_run(tmp_path, monkeypatch, with_traces=False)
    (run_dir / "original.png").unlink()
    (run_dir / "validated.jsonl").unlink()
    (run_dir / "validations.jsonl").unlink()
    out = write_session_report(run_dir)
    text = out.read_text(encoding="utf-8")
    assert "(no charts" in text
    assert "not validated (--no-validate)" in text
    assert "(no validation agent runs found" in text
    assert "(no memory traces in this run's window" in text
    assert "(no validated rows" in text


def test_report_reads_stats_from_scanned_trace_when_op_has_none(tmp_path, monkeypatch):
    run_dir = _seed_run(tmp_path, monkeypatch, with_traces=False)
    agent_dir = tmp_path / "agent_runs_empty" / "20261103-210000-000000-validation"
    agent_dir.mkdir(parents=True)
    (agent_dir / "trace.md").write_text(
        "=== Case 1 of 1: E1-T7 ===\n" + _trace_md(), encoding="utf-8")
    out = write_session_report(run_dir)
    text = out.read_text(encoding="utf-8")
    # The scanned run is attributed to E1-T7 via the case header and parsed for stats.
    assert "E1-T7" in text
    assert "1000/200" in text  # tokens in/out from the scanned trace
