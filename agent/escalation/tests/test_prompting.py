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
        {
            "cat_id": 25,
            "metrics": {
                "original_cat_id": 26,
                "candidates": {25: "distance", 26: "iou_last"},
                "chosen_cat_id": 25,
                "box": [400.0, 20.0, 490.0, 200.0],
                "candidate_refs": {
                    "25": {"rule": "distance", "source_time": "2026-10-28T16:34:42.769881",
                           "source_box": [350.0, 25.0, 440.0, 205.0], "source_distance_px": 50.0},
                    "26": {"rule": "iou_last", "source_time": "2026-10-28T16:34:40.000000",
                           "source_box": [395.0, 18.0, 488.0, 198.0]},
                },
            },
        },
        ["cat 26", "cat 25", "nearness", "overlap with latest", "2026-10-28T16:35:21.976102",
         "first match it found", "50 px away", "Box [400.0, 20.0, 490.0, 200.0]"],
    ),
    "T2": (
        {
            "cat_id": 25,
            "metrics": {
                "original_cat_id": 26,
                "rule": "distance",
                "box": [400.0, 20.0, 490.0, 200.0],
                "source_time": "2026-10-28T16:34:42.769881",
                "source_box": [350.0, 25.0, 440.0, 205.0],
                "source_distance_px": 50.0,
                "source_threshold_px": 160.0,
            },
        },
        ["cat 26", "cat 25", "nearby", "spatial continuity", "Box [400.0, 20.0, 490.0, 200.0]",
         "source of the correction", "2026-10-28T16:34:42.769881", "50 px away", "under 160 px"],
    ),
    "T3": (
        {
            "cat_id": 25,
            "metrics": {
                "killed_cat_id": 26, "iou": 0.82, "conf_margin": 0.04,
                "killed_box": [11.0, 12.0, 101.0, 102.0], "kept_box": [10.0, 10.0, 100.0, 100.0],
                "killed_conf": 0.60, "kept_conf": 0.64,
            },
        },
        ["cat 26", "0.82", "0.04", "Survivor", "dedup rule",
         "Box [10.0, 10.0, 100.0, 100.0]", "Box [11.0, 12.0, 101.0, 102.0]"],
    ),
    "T4": (
        {
            "cat_id": 25,
            "period_id": 3,
            "metrics": {
                "base_iou": 0.0,
                "wiped_rows": 107,
                "base_time": "2026-10-28T16:34:42.769881",
                "base_box": [10.0, 20.0, 100.0, 200.0],
                "box": [400.0, 20.0, 490.0, 200.0],
                "had_open_visit": True,
            },
        },
        ["0.00", "107", "period_id 3", "Task:", "anchor", "last committed position",
         "Box [10.0, 20.0, 100.0, 200.0]", "Box [400.0, 20.0, 490.0, 200.0]",
         "mints a new period"],
    ),
    "T5": (
        {
            "cat_id": 25,
            "metrics": {
                "target_conf": 0.32, "target_time": T1,
                "target_box": [10.0, 20.0, 100.0, 200.0],
                "base_time": "2026-10-28T16:34:30.000000", "base_box": [300.0, 300.0, 400.0, 400.0],
            },
        },
        ["0.32", "2026-10-28T16:34:42.769881", "Box [10.0, 20.0, 100.0, 200.0]",
         "best of the weaklings", "anchor"],
    ),
    "T6": (
        {
            "cat_id": 26,
            "period_id": 3,
            "metrics": {
                "distance_px": 370.5,
                "box_conf": 0.51,
                "old_period_id": 2,
                "retracted_rows": 2,
                "pre_time": "2026-10-28T16:34:42.769881",
                "pre_box": [10.0, 20.0, 100.0, 200.0],
                "post_box": [400.0, 20.0, 490.0, 200.0],
                "retracted_frames": [{"t": "2026-10-28T16:35:00.000000", "box": [200.0, 20.0, 290.0, 200.0]}],
            },
        },
        ["370", "visit 2", "visit 3", "Task:", "sureness score", "Box [10.0, 20.0, 100.0, 200.0]",
         "Box [400.0, 20.0, 490.0, 200.0]", "Interpolated box [200.0, 20.0, 290.0, 200.0]",
         "pre-jump frame", "post-jump frame", "period_id 2 and 3"],
    ),
    "T7": (
        {
            "cat_id": 26,
            "period_id": 3,
            "metrics": {
                "gap_start": T1,
                "gap_seconds": 5.17,
                "interp_frames": [
                    {"t": "2026-10-28T16:34:42.769881", "box": [10.0, 20.0, 100.0, 200.0]},
                    {"t": "2026-10-28T16:35:21.976102", "box": [12.0, 22.0, 102.0, 202.0]},
                ],
            },
        },
        ["5.2 s", "2026-10-28T16:34:42.769881", "Interpolated box [10.0, 20.0, 100.0, 200.0]",
         "period_id 3", "first real sighting after", "cut the whole span"],
    ),
    "P1": (
        {
            "scope": "period",
            "end_time": T0,
            "cat_id": 25,
            "period_id": 4,
            "metrics": {
                "scores": [1.0, 8.5, 7.9],
                "margin": 0.6,
                "runner_up_cat_id": 26,
                "votes": {"25": {"score": 8.5, "frames": 12}, "26": {"score": 7.9, "frames": 9}},
                "key_frames": [
                    {"t": "2026-10-28T16:35:21.976102", "box": [10.0, 20.0, 100.0, 200.0], "real": True},
                ],
            },
        },
        ["visit 4", "0.6", "8.5", "vote points", "cat 25: 8.5 pts (12 frames)",
         "Box [10.0, 20.0, 100.0, 200.0]", "period_id 4"],
    ),
    "P2": (
        {
            "scope": "period",
            "end_time": T0,
            "cat_id": 25,
            "period_id": 4,
            "metrics": {
                "score": 2.1,
                "detect_length": 3,
                "key_frames": [
                    {"t": "2026-10-28T16:35:21.976102", "box": [10.0, 20.0, 100.0, 200.0], "real": True},
                    {"t": "2026-10-28T16:35:22.976102", "box": None, "real": False},
                ],
            },
        },
        ["visit 4", "2.1", "3 original frames", "vote points", "x0.2 penalty",
         "Box [10.0, 20.0, 100.0, 200.0]", "period_id 4", "real sighting",
         "reassign to another cat"],
    ),
    "P3": (
        {
            "scope": "period",
            "end_time": T0,
            "cat_id": 25,
            "period_id": 4,
            "metrics": {
                "wanted_cat_id": 26,
                "taken_by_period_id": 5,
                "taken_by_span": ["2026-10-28T16:35:00.000000", "2026-10-28T16:36:00.000000"],
                "key_frames": [
                    {"t": "2026-10-28T16:35:21.976102", "box": [10.0, 20.0, 100.0, 200.0], "real": True},
                ],
            },
        },
        ["visit 4", "cat 26", "cat 25", "visit 5", "higher score claims first",
         "period_id 5", "Box [10.0, 20.0, 100.0, 200.0]"],
    ),
    "P4": (
        {
            "scope": "period",
            "end_time": T0,
            "period_id": 4,
            "metrics": {
                "seconds": 42.0,
                "rows": 120,
                "detect_counts": {"26": 40, "25": 4},
                "key_frames": [
                    {"t": "2026-10-28T16:35:21.976102", "box": [10.0, 20.0, 100.0, 200.0], "real": True},
                ],
            },
        },
        ["visit 4", "42 s", "120 frames", "cat 26 x40", "vote points",
         "Box [10.0, 20.0, 100.0, 200.0]", "period_id 4"],
    ),
    "P5": (
        {
            "scope": "period",
            "end_time": T0,
            "cat_id": 26,
            "period_id": 1,
            "metrics": {
                "detect_conf": 0.54,
                "detect_rate": 0.14,
                "total_frames": 100,
                "detect_counts": {"26": 12, "25": 2},
                "winner_box": [10.0, 20.0, 100.0, 200.0],
                "key_frames": [
                    {"t": "2026-10-28T16:33:06.797209", "box": [10.0, 20.0, 100.0, 200.0], "real": True},
                    {"t": "2026-10-28T16:33:34.000000", "box": None, "real": False},
                ],
            },
        },
        ["visit 1", "0.54", "of 100 frames in total", "cat 26: 12", "cat 25: 2",
         "Box [10.0, 20.0, 100.0, 200.0]", "period_id 1", "one continuous tracked visit",
         "real sighting", "guessed frame"],
    ),
    "P6": (
        {
            "scope": "period",
            "end_time": T0,
            "cat_id": 25,
            "period_id": 4,
            "metrics": {
                "prev_period_id": 3,
                "gap_seconds": 39.2,
                "distance_px": 152.0,
                "prev_start": T1,
                "prev_end_box": [10.0, 20.0, 100.0, 200.0],
                "new_start_box": [120.0, 30.0, 210.0, 210.0],
            },
        },
        ["visits 3 and 4", "39.2-second", "152 px apart", "visit 3 ran", "2026-10-28T16:34:42.769881",
         "Box [10.0, 20.0, 100.0, 200.0]", "Box [120.0, 30.0, 210.0, 210.0]", "period_id 3 and 4"],
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
                "prev_start": T1,
                "prev_end_box": [10.0, 20.0, 100.0, 200.0],
                "new_start_box": [12.0, 22.0, 102.0, 202.0],
            },
        },
        ["visits 3 and 4", "0.75", "120-second", "visit 3 ran", "stillness",
         "Box [10.0, 20.0, 100.0, 200.0]", "period_id 3 and 4"],
    ),
    "P8": (
        {
            "scope": "period",
            "end_time": T0,
            "cat_id": 25,
            "period_id": 4,
            "metrics": {
                "tracked_cat_id": 26,
                "voted_cat_id": 25,
                "tracked_frames": 1096,
                "relabeled_frames": 772,
                "votes": {"25": {"score": 9.0, "frames": 772}, "26": {"score": 6.2, "frames": 153}},
                "key_frames": [
                    {"t": "2026-10-28T16:35:21.976102", "box": [10.0, 20.0, 100.0, 200.0], "real": True},
                ],
            },
        },
        ["visit 4", "labeled cat 26", "vote awards it to cat 25", "cat 25: 9.0 pts (772 frames)",
         "moved 772 frames", "vote points", "Box [10.0, 20.0, 100.0, 200.0]", "period_id 4",
         "split the mixed visit", "judge only the cat inside the listed box"],
    ),
    "H1": (
        {
            "time": T1,
            "scope": "period",
            "end_time": T0,
            "metrics": {
                "quiet_seconds": 312.0,
                "messages": 48,
                "cat_ids": [25, 26],
                "kind": "periodic",
            },
        },
        ["Quiet window spot-check", "312 seconds", "48 frames ingested",
         "cat 25", "cat 26", "routine audit", "keeps everything as-is",
         "2026-10-28T16:34:42.769881", "2026-10-28T16:35:21.976102"],
    ),
}

SECTIONS = ["Conflict:", "Validate:", "Context:", "Alternatives:", "Resources"]


def test_every_code_has_standalone_prompt():
    assert set(CASES) == {"T1", "T2", "T3", "T4", "T5", "T6", "T7", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "H1"}
    for code, (kw, needles) in CASES.items():
        prompt = build_prompt(_esc(code, **kw))
        assert prompt.startswith(f"[E6-{code}]"), code
        sections = ["Task:" if (code in ("T3", "T4", "T5", "T6", "T7", "P4") and s == "Conflict:") else s for s in SECTIONS]
        for section in sections:
            assert section in prompt, (code, section)
        if code in ("T3", "T4", "T5", "T6", "T7", "P4"):
            assert "Conflict:" not in prompt
        for needle in needles:
            assert needle in prompt, (code, needle)
        assert "small YOLO" in prompt, code  # model caveat: hint, not evidence
        assert "images__get_image" in prompt, code
        assert "  - 2026-10-28T" in prompt, code  # key-frame bullets with reasons


def test_t5_empty_buffer_variant():
    prompt = build_prompt(_esc("T5", cat_id=25, period_id=2, metrics={"buffer_rows": 12}))
    assert "guessed frames (12 rows" in prompt


def test_t7_without_interp_frames_falls_back_to_samples():
    prompt = build_prompt(_esc("T7", cat_id=26, period_id=3, metrics={"gap_start": T1, "gap_seconds": 5.17}))
    assert "Task: double-check 5.2 seconds" in prompt
    assert "Interpolated box" not in prompt
    assert "  - 2026-10-28T16:34:42.769881" in prompt
    assert "first real sighting after" in prompt


def test_t6_without_boxes_falls_back_to_plain_frames():
    prompt = build_prompt(
        _esc("T6", cat_id=26, period_id=3,
             metrics={"distance_px": 370.5, "box_conf": 0.51, "old_period_id": 2, "retracted_rows": 0})
    )
    assert "Task: validate whether splitting visit 2 into visit 3" in prompt
    assert "Box [" not in prompt
    assert "a minute earlier; last known position" in prompt


def test_p5_without_counts_falls_back_to_rate_text():
    prompt = build_prompt(
        _esc("P5", scope="period", end_time=T0, cat_id=26, period_id=1, metrics={"detect_conf": 0.54, "detect_rate": 0.14})
    )
    assert "original-frame rate 0.14" in prompt
    assert "Box [" not in prompt
    assert "period_id 1" in prompt


def test_t2_without_source_falls_back_to_plain_frames():
    prompt = build_prompt(_esc("T2", cat_id=25, metrics={"original_cat_id": 26, "rule": "distance"}))
    assert "spatial continuity" in prompt
    assert "no visible basis" in prompt
    assert "source of the correction" not in prompt
    assert "Box [" not in prompt


def test_p6_without_span_falls_back_to_plain_gap():
    prompt = build_prompt(
        _esc("P6", scope="period", end_time=T0, cat_id=25, period_id=4,
             metrics={"prev_period_id": 3, "gap_seconds": 39.2, "distance_px": 152.0})
    )
    assert "visit 3 ended at" in prompt and "start is unknown" in prompt
    assert "152 px apart" in prompt
    assert "Box [" not in prompt


def test_p2_without_key_frames_falls_back_to_plain_samples():
    prompt = build_prompt(
        _esc("P2", scope="period", end_time=T0, cat_id=25, period_id=4, metrics={"score": 2.1, "detect_length": 3})
    )
    assert "vote points" in prompt and "x0.2 penalty" in prompt
    assert "Box [" not in prompt


def test_p2_long_weak_visit_notes_no_penalty():
    prompt = build_prompt(
        _esc("P2", scope="period", end_time=T0, cat_id=25, period_id=4, metrics={"score": 2.1, "detect_length": 12})
    )
    assert "No x0.2 penalty" in prompt


def test_t4_without_boxes_falls_back_to_plain_sighting():
    prompt = build_prompt(_esc("T4", cat_id=25, period_id=3, metrics={"base_iou": 0.0, "wiped_rows": 107}))
    assert "anchor (base) is the cat's last committed position" in prompt
    assert "Box [" not in prompt


def test_t4_closed_visit_notes_nothing_to_close():
    prompt = build_prompt(
        _esc("T4", cat_id=25, period_id=3,
             metrics={"base_iou": 0.0, "wiped_rows": 107, "had_open_visit": False})
    )
    assert "nothing left to close" in prompt


def test_t1_without_refs_falls_back_to_plain_dispute():
    prompt = build_prompt(
        _esc("T1", cat_id=25, metrics={"original_cat_id": 26, "candidates": {25: "distance", 26: "iou_last"}, "chosen_cat_id": 25})
    )
    assert "first match it found" in prompt
    assert "overlap with latest" in prompt
    assert "Box [" not in prompt


def test_t3_without_boxes_falls_back_to_plain_photo():
    prompt = build_prompt(_esc("T3", cat_id=25, metrics={"killed_cat_id": 26, "iou": 0.82, "conf_margin": 0.04}))
    assert "dedup rule" in prompt
    assert "Box [" not in prompt


def test_t5_without_boxes_falls_back_to_plain_target():
    prompt = build_prompt(_esc("T5", cat_id=25, metrics={"target_conf": 0.32, "target_time": T1}))
    assert "best of the weaklings" in prompt
    assert "Box [" not in prompt


def test_p1_without_votes_falls_back_to_score_vector():
    prompt = build_prompt(
        _esc("P1", scope="period", end_time=T0, cat_id=25, period_id=4, metrics={"scores": [1.0, 8.5, 7.9], "margin": 0.6})
    )
    assert "[1.0, 8.5, 7.9] per-cat scores" in prompt
    assert "vote points" in prompt
    assert "Box [" not in prompt


def test_p3_without_taker_falls_back_to_generic_other():
    prompt = build_prompt(
        _esc("P3", scope="period", end_time=T0, cat_id=25, period_id=4, metrics={"wanted_cat_id": 26})
    )
    assert "another simultaneous visit" in prompt
    assert "Box [" not in prompt


def test_p4_without_counts_falls_back_to_plain_unknown():
    prompt = build_prompt(
        _esc("P4", scope="period", end_time=T0, period_id=4, metrics={"seconds": 42.0, "rows": 120})
    )
    assert "too few coherent sightings" in prompt
    assert "Box [" not in prompt


def test_p7_without_span_falls_back_to_plain_gap():
    prompt = build_prompt(
        _esc("P7", scope="period", end_time=T0, cat_id=25, period_id=4,
             metrics={"prev_period_id": 3, "gap_seconds": 120.0, "overlap_iou": 0.75,
                      "new_displacement_px": 4.0, "prev_displacement_px": 6.0,
                      "new_head_frames": 8, "prev_trail_frames": 10})
    )
    assert "start is unknown" in prompt
    assert "stillness" in prompt
    assert "Box [" not in prompt


def test_h1_end_of_run_variant():
    prompt = build_prompt(
        _esc("H1", time=T1, scope="period", end_time=T0,
             metrics={"quiet_seconds": 125.0, "messages": 30, "cat_ids": [25], "kind": "end_of_run"})
    )
    assert "Quiet run spot-check" in prompt
    assert "whole run stayed quiet" in prompt
    assert "125 seconds (30 frames ingested)" in prompt
    assert "cat 25" in prompt


def test_cat_profiles_block_from_mapping():
    from agent.escalation.prompting import cat_profile_block

    profiles = {
        25: {"name": "Qb", "summary": "White short-hair.", "low_res_cues": ["bright blob", "thin tail"]},
        26: {"name": "Qbo", "summary": "Cream long-hair."},
    }
    block = cat_profile_block([25, "26", 99, None], profiles)
    assert block.startswith("Cats:\n")
    assert "- cat 25 (Qb): White short-hair. Look for: bright blob; thin tail." in block
    assert "- cat 26 (Qbo): Cream long-hair." in block
    assert "99" not in block
    assert cat_profile_block([99], profiles) == ""
    assert "Qb" in cat_profile_block(None, profiles)  # all known


def test_build_prompt_inserts_cats_before_resources():
    profiles = {25: {"name": "Qb", "summary": "White short-hair."}, 26: {"name": "Qbo", "summary": "Cream long-hair."}}
    prompt = build_prompt(_esc("T2", cat_id=25, metrics={"original_cat_id": 26, "rule": "distance"}), profiles=profiles)
    assert prompt.index("Cats:\n- cat 26 (Qbo)") < prompt.index("\nResources")
    assert "- cat 25 (Qb)" in prompt
    prompt = build_prompt(_esc("T2", cat_id=25, metrics={"original_cat_id": 26, "rule": "distance"}), profiles={})
    assert "Cats:" not in prompt


def test_p5_carries_both_cat_profiles_and_three_actions():
    profiles = {25: {"name": "Qb", "summary": "White short-hair."}, 26: {"name": "Qbo", "summary": "Cream long-hair."}}
    prompt = build_prompt(
        _esc(
            "P5",
            scope="period",
            end_time=T0,
            cat_id=26,
            period_id=1,
            metrics={"detect_conf": 0.54, "detect_rate": 0.14, "runner_up_cat_id": 25},
        ),
        profiles=profiles,
    )
    assert prompt.index("Cats:\n- cat 26 (Qbo)") < prompt.index("\nResources")
    assert "- cat 25 (Qb)" in prompt
    assert "1) IDENTITY" in prompt and "QUOTE the cues" in prompt
    assert "consistent sightings prove a real visit, not the right label" in prompt
    assert "change the label to the right cat (say which)" in prompt
    assert "local__find_references + local__get_reference_image and compare side-by-side" in prompt
    assert "CROPPED frames via local__get_frame with box=" in prompt


def test_p2_requires_identity_ruling_first():
    prompt = build_prompt(
        _esc(
            "P2",
            scope="period",
            end_time=T0,
            cat_id=25,
            period_id=3,
            metrics={"score": 2.2, "detect_length": 2},
        )
    )
    assert "1) IDENTITY" in prompt and "QUOTE the cues" in prompt
    assert "Thin evidence makes identity MORE uncertain, not less" in prompt
    assert "local__find_references + local__get_reference_image" in prompt
    assert "CROPPED frames via local__get_frame with box=" in prompt


def test_p8_requires_verified_cut_placement():
    prompt = build_prompt(
        _esc(
            "P8",
            scope="period",
            end_time=T0,
            cat_id=26,
            period_id=2,
            metrics={"tracked_cat_id": 26, "voted_cat_id": 25, "votes": {"25": {"score": 9.0, "frames": 10}}},
        )
    )
    assert "LAST frame where the first cat is still stably visible" in prompt
    assert "VERIFY by fetching one frame just before and one just after" in prompt
    assert "no clean handover, do not split" in prompt
    assert "local__find_references + local__get_reference_image" in prompt
    assert "1) BOX IDENTITY" in prompt and "QUOTE" in prompt
    assert "Full-frame views do NOT count" in prompt
    assert "local__get_frame(timestamp=..., box=[x1, y1, x2, y2])" in prompt
    assert "the vote is a lead, not evidence" in prompt
    assert "unanimous box-cropped evidence for cat 25 at 3+ spread samples" in prompt
    assert "AFTER the latest timestamp the first cat is placed" in prompt
    assert "State that window in your rationale" in prompt
    assert "under ~10 s" in prompt
    assert "Crop ONLY with the listed (time, box) pairs" in prompt
    assert "never invent boxes" in prompt
    assert "local__get_track_box(period_id=2, timestamp=...)" in prompt


def test_p6_offers_different_cats_with_all_profiles():
    profiles = {25: {"name": "Qb", "summary": "White short-hair."}, 26: {"name": "Qbo", "summary": "Cream long-hair."}}
    prompt = build_prompt(
        _esc(
            "P6",
            scope="period",
            end_time=T0,
            cat_id=25,
            period_id=4,
            metrics={"prev_period_id": 3, "gap_seconds": 39.2, "distance_px": 152.0},
        ),
        profiles=profiles,
    )
    assert "or are they different cats?" in prompt
    assert "or different cats (say which cat the new visit is)" in prompt
    assert "- cat 25 (Qb)" in prompt and "- cat 26 (Qbo)" in prompt


def test_unknown_code_falls_back():
    prompt = build_prompt(_esc("T9"))
    assert "T9 reason" in prompt
    assert "Validate:" in prompt


def test_emit_logs_prompt_after_escalation(caplog):
    with caplog.at_level(logging.INFO, logger="agent.escalation.escalation"):
        emit([], _esc("T2", cat_id=25, metrics={"original_cat_id": 26, "rule": "distance"}))
    texts = [r.getMessage() for r in caplog.records]
    assert any(t.startswith("Escalation E6-T2:") and '"code": "T2"' in t for t in texts)
    prompt_logs = [t for t in texts if "Escalation prompt E6-T2" in t]
    assert len(prompt_logs) == 1
    assert "Validate:" in prompt_logs[0]


def test_p6_different_cats_names_the_cat_from_appearance():
    prompt = build_prompt(
        _esc(
            "P6",
            scope="period",
            end_time=T0,
            cat_id=25,
            period_id=6,
            metrics={"prev_period_id": 4, "gap_seconds": 56.2, "distance_px": 252},
        )
    )
    assert "name the new visit's cat from its APPEARANCE" in prompt
    assert "never which cat is in the box" in prompt
    assert "The camera labeled visit 6 cat 25" in prompt
