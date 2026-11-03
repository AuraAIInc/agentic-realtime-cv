"""MetaModelClient: request shape + parsing, via an injected fake transport.

No network: the fake stands in for ``openai.OpenAI`` so these tests pin the
real client's wiring (model role, strict-JSON format, vision payload, error
mapping) while staying offline. Behavioral tests use ``StubLLMClient``.
"""

from __future__ import annotations

import json

import pytest

from agent.memory import LLMClientError, MetaModelClient, StubLLMClient, load_model_settings


class _FakeResponses:
    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = self.script.pop(0)
        if isinstance(text, Exception):
            raise text

        class _Resp:
            output_text = text

        return _Resp()


class _FakeClient:
    def __init__(self, script):
        self.responses = _FakeResponses(script)


def _client(*outputs):
    return MetaModelClient(client=_FakeClient(list(outputs)))


def test_settings_come_from_memory_role():
    model, base_url, key_env = load_model_settings("memory")
    assert model == "muse-spark-1.3"
    assert base_url == "https://api.meta.ai/v1"
    assert key_env == "MODEL_API_KEY"


def test_store_update_shape_and_parsing():
    c = _client('{"contradicts_stable": true, "note": "x"}')
    out = c.store_update({"stable": "old", "weight": 5}, "new obs")
    assert out == {"contradicts_stable": True, "note": "x"}
    call = c._client.responses.calls[0]
    assert call["model"] == "muse-spark-1.3"
    assert "OBSERVATION: new obs" in call["input"]
    fmt = call["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True
    assert set(fmt["schema"]["required"]) == {
        "contradicts_stable", "note",
    }


def test_json_fences_stripped_and_bad_output_maps_to_error():
    c = _client('```json\n{"contradicts_stable": false, "note": "n"}\n```')
    assert c.store_update({}, "o")["note"] == "n"
    with pytest.raises(LLMClientError, match="non-JSON"):
        _client("not json at all").store_update({}, "o")
    with pytest.raises(LLMClientError, match="Meta Model API call failed"):
        _client(RuntimeError("transport down")).summarize("L1", [], {})


def test_summarize_carries_word_cap():
    c = _client('{"activity": "a", "appearance": "p", "behavior": "b"}')
    out = c.summarize("L1", [{"id": "r0c0", "text": "hi", "visits": 3}], {"visits": 3})
    assert out == {"text": "a", "appearance": "p", "behavior": "b"}
    assert "100 words" in c._client.responses.calls[0]["instructions"]


def test_no_keywords_anywhere_in_requests_or_results():
    c = _client(
        '{"contradicts_stable": false, "note": "n"}',
        '{"activity": "t", "appearance": "a", "behavior": "b"}',
    )
    assert c.store_update({}, "o") == {"contradicts_stable": False, "note": "n"}
    assert c.summarize("L1", [], {}) == {"text": "t", "appearance": "a", "behavior": "b"}
    blob = json.dumps([call for call in c._client.responses.calls])
    assert "keyword" not in blob.lower()


def test_bootstrap_vision_payload():
    c = _client(json.dumps({
        "full_view": {"environment": "e", "background": "b", "cat_contrast": "c"},
        "zones": [
            {"id": "r0c0", "environment": "e", "background": "b", "cat_contrast": "c"},
            {"noid": True},
        ],
    }))
    res = c.bootstrap(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
    assert res["full_view"]["environment"] == "e"
    assert [z["id"] for z in res["zones"]] == ["r0c0"]  # id-less zones dropped
    call = c._client.responses.calls[0]
    parts = call["input"][0]["content"]
    assert parts[0]["type"] == "input_text"
    assert parts[1]["type"] == "input_image"
    assert parts[1]["image_url"].startswith("data:image/png;base64,")
    assert "r0c0" in call["instructions"]  # zone ids enumerated


def _activity_kwargs(**kw):
    args = {
        "images": [b"\xff\xd8fake-1", b"\xff\xd8fake-2"],
        "cat_name": "Qb",
        "cat_id": 25,
        "span": "16:34 to 16:40",
        "boxes": [[1, 2, 3, 4], [1, 2, 3, 4]],
        "keyframe_names": ["a.jpg", "b.jpg"],
        "candidates": [{"name": "resting", "description": "lying still"}, {"name": "walking", "description": "moving"}],
    }
    args.update(kw)
    return args


def test_predict_activity_shape_enum_and_clamp():
    c = _client('{"activity": "resting", "activity_confidence": 1.7, "appearance": "white", "background": "rug"}')
    out = c.predict_activity(**_activity_kwargs(), label="period 2")
    assert out == {"activity": "resting", "activity_confidence": 1.0, "appearance": "white", "background": "rug"}
    call = c._client.responses.calls[0]
    assert call["text"]["format"]["name"] == "memory_activity"
    assert call["text"]["format"]["schema"]["properties"]["activity"]["enum"] == ["resting", "walking"]
    content = call["input"][0]["content"]
    assert content[0]["type"] == "input_text" and "Qb (cat 25)" in content[0]["text"]
    assert "16:34 to 16:40" in content[0]["text"] and "[1, 2, 3, 4]" in content[0]["text"]
    assert [p["type"] for p in content[1:]] == ["input_image", "input_image"]
    assert "resting: lying still" in call["instructions"]


def test_predict_activity_unknown_when_missing_or_bad_confidence():
    c = _client('{"activity": "", "activity_confidence": "high", "appearance": "", "background": ""}')
    out = c.predict_activity(**_activity_kwargs())
    assert out["activity"] == "unknown" and out["activity_confidence"] == 0.0


def test_predict_activity_traces_with_kind_and_images(tmp_path):
    c = _client('{"activity": "walking", "activity_confidence": 0.8, "appearance": "a", "background": "b"}')
    c.trace_dir = tmp_path
    c.predict_activity(**_activity_kwargs(), label="period 2")
    (md,) = tmp_path.glob("*.md")
    assert "-activity-period-2.md" in md.name
    assert "memory_activity" in md.read_text()
    assert sorted(p.suffix for p in tmp_path.glob("*.jpg")) == [".jpg", ".jpg"]


def test_stub_predict_activity_is_unknown():
    out = StubLLMClient().predict_activity(**_activity_kwargs())
    assert out == {"activity": "unknown", "activity_confidence": 0.0, "appearance": "", "background": ""}
