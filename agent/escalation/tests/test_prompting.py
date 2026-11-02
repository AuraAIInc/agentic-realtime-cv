"""Every escalation code gets a standalone reviewer prompt; emit() logs it."""

import datetime
import json
import logging

from agent.escalation import Escalation, build_prompt, emit

T0 = datetime.datetime(2026, 10, 28, 16, 35, 21, 976102)
T1 = datetime.datetime(2026, 10, 28, 16, 34, 42, 769881)


def _esc(code, **kw):
    args = {"time": T0, "scope": "frame", "code": code, "reason": f"{code} reason", "seq": 6}
    args.update(kw)
    return Escalation(**args)


CASES = {
    # code: (kwargs, must-contain substrings)
    "T1": (
        {"cat_id": 25, "metrics": {"original_cat_id": 26, "candidates": {25: "distance", 26: "iou"}, "chosen_cat_id": 25}},
        ["cat 26", "cat 25", "nearness", "2026-10-28T16:35:21.976102"],
    ),
    "T2": (
        {"cat_id": 25, "metrics": {"original_cat_id": 26, "rule": "distance"}},
        ["cat 26", "cat 25", "nearby"],
    ),
    "T3": (
        {"cat_id": 25, "metrics": {"killed_cat_id": 26, "iou": 0.82, "conf_margin": 0.04}},
        ["cat 26", "0.82", "0.04"],
    ),
    "T4": (
        {"cat_id": 25, "period_id": 3, "metrics": {"base_iou": 0.0, "wiped_rows": 107}},
        ["0.00", "107", "period 3"],
    ),
    "T5": (
        {"cat_id": 25, "metrics": {"target_conf": 0.32, "target_time": T1}},
        ["0.32", "2026-10-28T16:34:42.769881"],
    ),
    "T6": (
        {"cat_id": 26, "period_id": 3, "metrics": {"distance_px": 370.5, "box_conf": 0.51, "old_period_id": 2, "retracted_rows": 0}},
        ["370", "visit 2", "visit 3"],
    ),
    "T7": (
        {"cat_id": 26, "period_id": 3, "metrics": {"gap_start": T1, "gap_seconds": 5.17}},
        ["5.2 s", "2026-10-28T16:34:42.769881"],
    ),
    "P1": (
        {
            "scope": "period",
            "end_time": T0,
            "cat_id": 25,
            "period_id": 4,
            "metrics": {"scores": [1.0, 8.5, 7.9], "margin": 0.6},
        },
        ["visit 4", "0.6", "8.5"],
    ),
    "P2": (
        {"scope": "period", "end_time": T0, "cat_id": 25, "period_id": 4, "metrics": {"score": 2.1, "detect_length": 3}},
        ["visit 4", "2.1", "3 original frames"],
    ),
    "P3": (
        {"scope": "period", "end_time": T0, "cat_id": 25, "period_id": 4, "metrics": {"wanted_cat_id": 26}},
        ["visit 4", "cat 26", "cat 25"],
    ),
    "P4": (
        {"scope": "period", "end_time": T0, "period_id": 4, "metrics": {"seconds": 42.0, "rows": 120}},
        ["visit 4", "42 s", "120 frames"],
    ),
    "P5": (
        {"scope": "period", "end_time": T0, "cat_id": 26, "period_id": 1, "metrics": {"detect_conf": 0.54, "detect_rate": 0.14}},
        ["visit 1", "0.54", "0.14"],
    ),
    "P6": (
        {
            "scope": "period",
            "end_time": T0,
            "cat_id": 25,
            "period_id": 4,
            "metrics": {"prev_period_id": 3, "gap_seconds": 39.2, "distance_px": 152.0},
        },
        ["visits 3 and 4", "39.2 s", "152 pixels"],
    ),
    "P7": (
        {
            "scope": "period",
            "end_time": T0,
            "cat_id": 25,
            "period_id": 4,
            "metrics": {
                "prev_period_id": 3,
                "gap_seconds": 120.0,
                "overlap_iou": 0.75,
                "new_displacement_px": 4.0,
                "prev_displacement_px": 6.0,
                "new_head_frames": 8,
                "prev_trail_frames": 10,
            },
        },
        ["visits 3 and 4", "0.75", "120-second"],
    ),
}

SECTIONS = ["Conflict:", "Validate:", "Context:", "Alternatives:", "Resources"]


def test_every_code_has_standalone_prompt():
    assert set(CASES) == {"T1", "T2", "T3", "T4", "T5", "T6", "T7", "P1", "P2", "P3", "P4", "P5", "P6", "P7"}
    for code, (kw, needles) in CASES.items():
        prompt = build_prompt(_esc(code, **kw))
        assert prompt.startswith(f"[E6-{code}]"), code
        for section in SECTIONS:
            assert section in prompt, (code, section)
        for needle in needles:
            assert needle in prompt, (code, needle)
        assert "small YOLO" in prompt, code  # model caveat: hint, not evidence
        assert "images__get_image" in prompt, code
        assert "  - 2026-10-28T" in prompt, code  # key-frame bullets with reasons


def test_t5_empty_buffer_variant():
    prompt = build_prompt(_esc("T5", cat_id=25, period_id=2, metrics={"buffer_rows": 12}))
    assert "guessed frames (12 rows" in prompt


def test_unknown_code_falls_back():
    prompt = build_prompt(_esc("T9"))
    assert "T9 reason" in prompt
    assert "Validate:" in prompt


def test_emit_logs_prompt_after_escalation(caplog, capsys):
    with caplog.at_level(logging.INFO, logger="agent.escalation.escalation"):
        emit([], _esc("T2", cat_id=25, metrics={"original_cat_id": 26, "rule": "distance"}))
    assert json.loads(capsys.readouterr().out.strip())["code"] == "T2"
    texts = [r.getMessage() for r in caplog.records]
    prompt_logs = [t for t in texts if "Escalation prompt E6-T2" in t]
    assert len(prompt_logs) == 1
    assert "Validate:" in prompt_logs[0]
