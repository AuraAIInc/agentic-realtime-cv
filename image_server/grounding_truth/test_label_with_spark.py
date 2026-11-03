"""Offline tests for label_with_spark (no API calls, no network).

Run from the image_server/ folder:
    python -m pytest grounding_truth/test_label_with_spark.py -q
(pytest from the repo root works too.)
"""

import importlib.util
import json
import os

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "label_with_spark", os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_with_spark.py")
)
label_with_spark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(label_with_spark)


def _cat(name="Qb"):
    return {"name": name, "tracking_ids": {25},
            "summary": "Short-haired white cat.",
            "description": "Compact short-haired cat, nearly white.",
            "cues": ["cool white colour", "smooth outline", "thin tail"]}


def test_schema_requires_no_additional_properties():
    schema = label_with_spark.SparkBoxes.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["SparkBox"]["additionalProperties"] is False
    assert schema["required"] == ["boxes"]


def test_validate_good_and_bad_replies():
    good = label_with_spark.SparkBoxes.model_validate_json(
        '{"boxes": [{"id": 1, "x1": 10, "y1": 20, "x2": 30, "y2": 40}]}')
    assert [(b.id, b.x1) for b in good.boxes] == [(1, 10)]
    with pytest.raises(ValueError):
        label_with_spark.SparkBoxes.model_validate_json("not json")
    with pytest.raises(ValueError):
        label_with_spark.SparkBoxes.model_validate_json('{"boxes": [{"id": 0}]}')


def test_prompt_carries_descriptions_and_reference_only():
    classes = [{"class_id": 0, "name": "Qb", "cat": _cat()}]
    prompt = label_with_spark.build_spark_detect_prompt(
        classes, {0: [(12, 22, 32, 42)]}, 800, 600)
    assert "Short-haired white cat" in prompt
    assert "cool white colour" in prompt
    assert "may be wrong or missing; context only" in prompt
    assert "class 0 at [12,22,32,42]" in prompt
    assert "0-1000 scale" in prompt


def test_prompt_without_reference_says_so():
    classes = [{"class_id": 1, "name": "Qbo", "cat": _cat("Qbo")}]
    prompt = label_with_spark.build_spark_detect_prompt(classes, {}, 800, 600)
    assert "found nothing" in prompt


def test_payload_carries_strict_schema():
    payload = label_with_spark.build_spark_detect_payload(
        "muse-spark-1.3-contributor", "Detect...", b"\xff\xd8")
    assert payload["model"] == "muse-spark-1.3-contributor"
    fmt = payload["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True
    assert set(fmt["schema"]["properties"]) == {"boxes"}
    content = payload["input"][0]["content"]
    assert content[1]["image_url"].startswith("data:image/jpeg;base64,")


def test_boxes_to_yolo_filters_junk():
    parsed = label_with_spark.SparkBoxes.model_validate_json(json.dumps({"boxes": [
        {"id": 0, "x1": 0, "y1": 0, "x2": 500, "y2": 1000},   # kept
        {"id": 9, "x1": 0, "y1": 0, "x2": 10, "y2": 10},       # unknown id
        {"id": 0, "x1": 0, "y1": 0, "x2": 1001, "y2": 10},     # out of range
        {"id": 1, "x1": 5, "y1": 5, "x2": 5, "y2": 9},         # degenerate
    ]}))
    kept, dropped = label_with_spark.spark_boxes_to_yolo(parsed, [0, 1], 800, 600)
    assert kept == [(0, 0.25, 0.5, 0.5, 1.0)]
    assert dropped == 3


def test_cost_math():
    pricing = {"input": 0.10, "output": 0.20}
    assert label_with_spark.estimate_cost_usd(0, 0, 0, pricing) == 0.0
    assert label_with_spark.estimate_cost_usd(5, 1_000_000, 1_000_000, pricing) == pytest.approx(0.30)


def test_discover_uses_spark_labels_dir(tmp_path):
    import label_with_sam as sam_mod

    root = tmp_path / "retrieved_images"
    (root / "2026-10-28-16" / "images").mkdir(parents=True)
    (root / "2026-10-28-16" / "images" / "0001_2026-10-28-16-33-00-000001.jpg").touch()
    found = sam_mod.discover_images(str(root), "2026-10-28-16-33", "2026-10-28-16-34",
                                    label_with_spark.OUT_DIRNAME)
    assert found and found[0][3].endswith("spark_labels")


def test_session_report_writes_md_and_json(tmp_path):
    session = {
        "session_id": "t", "started": "s", "finished": "f", "wall_seconds": 1.0,
        "dry_run": True, "start_key": "a", "end_key": "b", "hours": ["h"],
        "model": "muse-spark-1.3-contributor", "pricing": {"input": 0.10, "output": 0.20},
        "workers": 10, "limit": None,
        "images_in_range": 2, "processed": 0, "skipped": 0, "failed": 0,
        "calls": 0, "boxes_total": 0, "boxes_by_class": {}, "dropped": 0,
        "tokens": {"input": 0, "output": 0, "total": 0}, "cost_usd": 0.0,
        "queries": [{"class_id": 0, "name": "Qb", "concept": "Qb: white"}],
        "first_prompt_image": "h/img.jpg", "first_prompt": "Detect...",
        "per_hour": {"h": {"in_range": 2, "processed": 0, "skipped": 0, "failed": 0, "boxes": 0}},
        "failures": [], "results": [],
    }
    md, js = label_with_spark.write_session_report(
        str(tmp_path / "r.md"), str(tmp_path / "r.json"), session)
    body = open(md).read()
    assert "DRY RUN" in body and "CONTRIBUTOR" in body and "Detect..." in body
    assert json.load(open(js))["session_id"] == "t"
