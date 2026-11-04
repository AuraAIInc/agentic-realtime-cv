"""Offline tests for label_with_sam (no API calls, no network).

Run from the image_server/ folder:
    python -m pytest grounding_truth/test_label_with_sam.py -q
(pytest from the repo root works too.)
"""

import importlib.util
import json
import os

_SPEC = importlib.util.spec_from_file_location(
    "label_with_sam", os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_with_sam.py")
)
label_with_sam = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(label_with_sam)

# Exact image example from the Meta SAM reading-segmentation docs (537x561).
DOCS_IMAGE_LINE = (
    "<0f>0<|box;x1=353;y1=382;x2=453;y2=427;w=537;h=561|>"
    "<|mask;x=0;y=0;data=46,101,~...|>,"
    "1<|box;x1=260;y1=382;x2=340;y2=427;w=537;h=561|>"
    "<|mask;x=0;y=0;data=46,81,~...|>,"
    "2<|box;x1=516;y1=117;x2=536;y2=150;w=537;h=561|>"
    "<|mask;x=0;y=0;data=34,21,~...|>"
)


def _cat(name="Qb"):
    return {"name": name, "tracking_ids": {25},
            "summary": "Short-haired white cat.",
            "description": "Compact short-haired cat, nearly white.",
            "cues": ["cool white colour", "smooth outline", "thin tail"]}


def test_parse_docs_example():
    boxes = label_with_sam.parse_sam_boxes(DOCS_IMAGE_LINE)
    assert [(b["obj_id"], b["x1"], b["y1"], b["x2"], b["y2"], b["w"], b["h"])
            for b in boxes] == [
        (0, 353, 382, 453, 427, 537, 561),
        (1, 260, 382, 340, 427, 537, 561),
        (2, 516, 117, 536, 150, 537, 561),
    ]


def test_parse_empty_is_zero_matches():
    assert label_with_sam.parse_sam_boxes("") == []
    assert label_with_sam.parse_sam_boxes("   \n  ") == []


def test_parse_confidence_and_malformed():
    text = ("<0f>1<|box;x1=10;y1=20;x2=30;y2=40;w=800;h=600;c=0.96|>"
            "<|mask;x=0;y=0;data=1,1,~x|>,"
            "2<|box;x1=oops|><|mask;x=0;y=0;data=1,1,~x|>")
    boxes = label_with_sam.parse_sam_boxes(text)
    assert len(boxes) == 1
    assert boxes[0]["obj_id"] == 1
    assert boxes[0]["conf"] == 0.96


def test_sam_prompt_is_bare_concept():
    assert label_with_sam.build_sam_prompt("animal") == "animal"


def test_spark_prompt_carries_descriptions_and_reference_only():
    classes = [{"class_id": 0, "name": "Qb", "cat": _cat()}]
    prompt = label_with_sam.build_spark_prompt(
        classes, [(10, 20, 30, 40)], {0: [(12, 22, 32, 42)]}, 800, 600)
    assert "Short-haired white cat" in prompt  # profile summary attached
    assert "cool white colour" in prompt  # profile cues attached
    assert "box 0: [10,20,30,40]" in prompt  # proposed box to identify
    assert "may be wrong; context only" in prompt  # reference-only note
    assert "class 0 at [12,22,32,42]" in prompt  # reference box in pixels
    assert '{"boxes": [{"box": 0, "id": 1}]}' in prompt  # strict format


def test_spark_prompt_without_reference_says_so():
    classes = [{"class_id": 1, "name": "Qbo", "cat": _cat("Qbo")}]
    prompt = label_with_sam.build_spark_prompt(classes, [(1, 2, 3, 4)], {}, 800, 600)
    assert "found nothing" in prompt


def test_parse_spark_json():
    assert label_with_sam.parse_spark_json('{"boxes": [{"box": 0, "id": 1}]}', 1, [0, 1]) == [1]
    assert label_with_sam.parse_spark_json('{"boxes": [{"box": 0, "id": -1}]}', 1, [0, 1]) == [-1]
    assert label_with_sam.parse_spark_json('```json\n{"boxes": []}\n```', 2, [0, 1]) == [-1, -1]
    try:
        label_with_sam.parse_spark_json("not json", 1, [0, 1])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_iou_and_dedupe():
    a = (0, 0, 100, 100)
    assert label_with_sam.box_iou(a, a) == 1.0
    assert label_with_sam.box_iou(a, (200, 200, 300, 300)) == 0.0
    assert label_with_sam.dedupe_boxes([a, (5, 5, 105, 105)]) == [a]
    assert len(label_with_sam.dedupe_boxes([a, (200, 200, 300, 300)])) == 2


def test_output_text_of_response():
    body = {"output": [
        {"type": "reasoning", "content": []},
        {"type": "message", "content": [
            {"type": "output_text", "text": "<0f>"},
            {"type": "output_refusal", "refusal": "x"},
        ]},
    ]}
    assert label_with_sam.output_text_of(body) == "<0f>"
    assert label_with_sam.output_text_of({"output_text": "abc"}) == "abc"


def test_yolo_conversion_and_space_guard():
    boxes = [{"x1": 0, "y1": 0, "x2": 400, "y2": 300, "w": 800, "h": 600}]
    assert label_with_sam.sam_boxes_to_yolo(boxes, 800, 600) == [(0, 0, 400, 300)]
    # same-aspect resized space rescales fine (normalized coords are identical)
    assert label_with_sam.sam_boxes_to_yolo(boxes, 1600, 1200) == [(0, 0, 800, 600)]
    # mismatched aspect space is dropped, never guessed across
    assert label_with_sam.sam_boxes_to_yolo(boxes, 800, 800) == []
    # degenerate box dropped
    bad = [{"x1": 5, "y1": 5, "x2": 5, "y2": 9, "w": 800, "h": 600}]
    assert label_with_sam.sam_boxes_to_yolo(bad, 800, 600) == []


def test_bounds_single_hour_and_span():
    assert label_with_sam.resolve_bounds("2026-10-28-16", None, None) == (
        "2026-10-28-16-00", "2026-10-28-17")
    assert label_with_sam.resolve_bounds(None, "2026-10-28-16-33", "2026-10-28-16-40") == (
        "2026-10-28-16-33", "2026-10-28-16-41")
    # end minute fully included; ranges may span hours
    assert label_with_sam.resolve_bounds(None, "2026-10-28-16-58", "2026-10-28-17-02") == (
        "2026-10-28-16-58", "2026-10-28-17-03")


def test_discover_and_hour_classes(tmp_path):
    root = tmp_path / "retrieved_images"
    for hour in ("2026-10-28-16", "2026-10-28-17"):
        (root / hour / "images").mkdir(parents=True)
        (root / hour / "labels").mkdir(parents=True)
    (root / "2026-10-28-16" / "images" / "0001_2026-10-28-16-33-00-000001.jpg").touch()
    (root / "2026-10-28-16" / "images" / "0002_2026-10-28-16-34-00-000002.jpg").touch()
    (root / "2026-10-28-17" / "images" / "0003_2026-10-28-17-00-00-000003.jpg").touch()
    (root / "2026-10-28-16" / "classes_map.json").write_text(json.dumps(
        {"order": [25, 26], "names": {"25": "Qb-2", "26": "Qbo-2"}}))
    cats = [_cat("Qb"), dict(_cat("Qbo"), tracking_ids={26})]
    found = label_with_sam.discover_images(str(root), "2026-10-28-16-33", "2026-10-28-16-35")
    assert [r[4] for r in found] == ["0001_2026-10-28-16-33-00-000001.jpg",
                                     "0002_2026-10-28-16-34-00-000002.jpg"]
    queries = label_with_sam.load_hour_classes(str(root / "2026-10-28-16"), cats)
    assert [(q["class_id"], q["name"]) for q in queries] == [(0, "Qb"), (1, "Qbo")]


def test_payload_requests_compact_masks():
    payload = label_with_sam.build_sam_payload("sam-3.1", "Segment...", b"\xff\xd8")
    assert payload["model"] == "sam-3.1"
    assert payload["metadata"] == {"mask_encoding": "one_bit"}
    content = payload["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "Segment..."}
    assert content[1]["image_url"].startswith("data:image/jpeg;base64,")


def test_cost_math():
    std = {"input": 1.25, "output": 4.25}
    assert label_with_sam.estimate_cost_usd(0, 0, 0, std) == 0.0
    assert label_with_sam.estimate_cost_usd(1000, 0, 0, std) == 2.5
    assert label_with_sam.estimate_cost_usd(1000, 1_000_000, 1_000_000, std) == 2.5 + 1.25 + 4.25


def test_classify_defaults_to_contributor():
    model, _, _, pricing = label_with_sam.load_spark_settings()
    assert model == "muse-spark-1.3-contributor"
    assert pricing == {"input": 0.10, "output": 0.20}


def test_classify_standard_flag_uses_regular_spark():
    model, _, _, pricing = label_with_sam.load_spark_settings(contributor=False)
    assert model == "muse-spark-1.3"
    assert pricing == {"input": 1.25, "output": 4.25}
    assert label_with_sam.standard_spark_pricing() == {"input": 1.25, "output": 4.25}


def test_missing_model_pricing_logs_and_continues(tmp_path, caplog):
    cfg = tmp_path / "models.json"
    cfg.write_text(json.dumps({"base_url": "https://x", "roles": {"agent": {"model": "muse-spark-1.3"}},
                               "pricing_usd_per_1m_tokens": {}}))
    model, _, _, pricing = label_with_sam.load_spark_settings(str(cfg))
    assert model == "muse-spark-1.3-contributor" and pricing is None
    assert label_with_sam.standard_spark_pricing(str(cfg)) is None
    assert "no input/output entry for muse-spark-1.3-contributor" in caplog.text
    assert label_with_sam.estimate_cost_usd(10, 100, 100, None) is None
    assert label_with_sam.fmt_usd(None) == "n/a"


def test_contributor_report_shows_standard_equivalent(tmp_path):
    session = {
        "session_id": "t", "started": "s", "finished": "f", "wall_seconds": 1.0,
        "dry_run": False, "start_key": "a", "end_key": "b", "hours": ["h"],
        "sam_model": "sam-3.1", "spark_model": "muse-spark-1.3-contributor",
        "spark_tier": "contributor",
        "spark_pricing": {"input": 0.10, "output": 0.20},
        "spark_pricing_standard": {"input": 1.25, "output": 4.25},
        "concepts": ["animal"],
        "workers": 10, "limit": None,
        "images_in_range": 1, "processed": 1, "skipped": 0, "failed": 0,
        "sam_calls": 2, "spark_calls": 1, "proposed": 1,
        "boxes_total": 1, "boxes_by_class": {"Qb": 1}, "dropped": 0, "rejected": 0,
        "tokens": {"input": 1000, "output": 2000, "total": 3000},
        "cost_usd": 0.0055, "sam_cost_usd": 0.005, "spark_cost_usd": 0.0005,
        "cost_standard_usd": 0.01476, "spark_cost_standard_usd": 0.00976,
        "queries": [{"class_id": 0, "name": "Qb", "concept": "Qb: white"}],
        "first_sam_image": "h/img.jpg", "first_sam_prompt": "animal",
        "first_spark_image": "h/img.jpg", "first_spark_prompt": "Identify...",
        "per_hour": {"h": {"in_range": 1, "processed": 1, "skipped": 0, "failed": 0, "boxes": 1}},
        "failures": [], "results": [],
    }
    md, js = label_with_sam.write_session_report(
        str(tmp_path / "r.md"), str(tmp_path / "r.json"), session)
    body = open(md).read()
    assert "Contributor billed" in body
    assert "Standard equivalent: $0.0148" in body
    assert "quote this in project reports" in body
    assert "may be used to train future Meta models" in body
    assert json.load(open(js))["spark_tier"] == "contributor"


def test_session_report_writes_md_and_json(tmp_path):
    session = {
        "session_id": "t", "started": "s", "finished": "f", "wall_seconds": 1.0,
        "dry_run": True, "start_key": "a", "end_key": "b", "hours": ["h"],
        "sam_model": "sam-3.1", "spark_model": "muse-spark-1.3",
        "spark_pricing": {"input": 1.25, "output": 4.25}, "concepts": ["animal"],
        "workers": 10, "limit": None,
        "images_in_range": 2, "processed": 0, "skipped": 0, "failed": 0,
        "sam_calls": 0, "spark_calls": 0, "proposed": 0,
        "boxes_total": 0, "boxes_by_class": {}, "dropped": 0, "rejected": 0,
        "tokens": {"input": 0, "output": 0, "total": 0},
        "cost_usd": 0.0, "sam_cost_usd": 0.0, "spark_cost_usd": 0.0,
        "queries": [{"class_id": 0, "name": "Qb", "concept": "Qb: white"}],
        "first_sam_image": "h/img.jpg", "first_sam_prompt": "animal",
        "first_spark_image": "h/img.jpg", "first_spark_prompt": "Identify...",
        "per_hour": {"h": {"in_range": 2, "processed": 0, "skipped": 0, "failed": 0, "boxes": 0}},
        "failures": [], "results": [],
    }
    md, js = label_with_sam.write_session_report(
        str(tmp_path / "r.md"), str(tmp_path / "r.json"), session)
    body = open(md).read()
    assert "DRY RUN" in body
    assert "animal" in body and "Identify..." in body
    assert json.load(open(js))["session_id"] == "t"


def test_transient_classifier():
    assert label_with_sam.is_transient_api_error(RuntimeError("API HTTP 504: timed out"))
    assert label_with_sam.is_transient_api_error(RuntimeError("API HTTP 429: slow down"))
    assert label_with_sam.is_transient_api_error(RuntimeError("API HTTP 500: boom"))
    assert not label_with_sam.is_transient_api_error(RuntimeError("API HTTP 400: bad schema"))
    assert not label_with_sam.is_transient_api_error(RuntimeError("API HTTP 401: no key"))
    assert not label_with_sam.is_transient_api_error(RuntimeError("unreadable JPEG header"))
    import urllib.error

    assert label_with_sam.is_transient_api_error(urllib.error.URLError("timed out"))
    assert label_with_sam.is_transient_api_error(TimeoutError())


def test_retry_wrapper_retries_transient_then_succeeds(monkeypatch):
    calls = []

    def flaky(base_url, api_key, payload, timeout=240):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("API HTTP 504: request timed out")
        return ("ok", {"input": 1})

    monkeypatch.setattr(label_with_sam, "call_responses", flaky)
    monkeypatch.setattr(label_with_sam.time, "sleep", lambda s: None)
    assert label_with_sam.call_responses_with_retry("u", "k", {}) == ("ok", {"input": 1})
    assert len(calls) == 3


def test_retry_wrapper_fails_fast_on_client_error(monkeypatch):
    import pytest

    def bad_request(base_url, api_key, payload, timeout=240):
        raise RuntimeError("API HTTP 400: bad schema")

    sleeps = []
    monkeypatch.setattr(label_with_sam, "call_responses", bad_request)
    monkeypatch.setattr(label_with_sam.time, "sleep", sleeps.append)
    with pytest.raises(RuntimeError, match="API HTTP 400"):
        label_with_sam.call_responses_with_retry("u", "k", {})
    assert sleeps == []
