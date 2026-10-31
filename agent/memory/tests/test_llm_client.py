"""MetaModelClient: request shape + parsing, via an injected fake transport.

No network: the fake stands in for ``openai.OpenAI`` so these tests pin the
real client's wiring (model role, strict-JSON format, vision payload, error
mapping) while staying offline. Behavioral tests use ``StubLLMClient``.
"""

from __future__ import annotations

import json

import pytest

from agent.memory import LLMClientError, MetaModelClient, load_model_settings


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
    c = _client('{"keywords": ["sofa"], "contradicts_stable": true, "note": "x"}')
    out = c.store_update({"stable": "old", "weight": 5}, "new obs")
    assert out == {"keywords": ["sofa"], "contradicts_stable": True, "note": "x"}
    call = c._client.responses.calls[0]
    assert call["model"] == "muse-spark-1.3"
    assert "OBSERVATION: new obs" in call["input"]
    fmt = call["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True
    assert set(fmt["schema"]["required"]) == {
        "keywords", "contradicts_stable", "note",
    }


def test_json_fences_stripped_and_bad_output_maps_to_error():
    c = _client('```json\n{"keywords": [], "contradicts_stable": false, "note": "n"}\n```')
    assert c.store_update({}, "o")["note"] == "n"
    with pytest.raises(LLMClientError, match="non-JSON"):
        _client("not json at all").store_update({}, "o")
    with pytest.raises(LLMClientError, match="Meta Model API call failed"):
        _client(RuntimeError("transport down")).summarize("L1", [], {})


def test_summarize_carries_word_cap():
    c = _client('{"text": "t", "keywords": ["k"]}')
    c.summarize("L1", [{"id": "r0c0", "text": "hi", "visits": 3}], {"visits": 3})
    assert "100 words" in c._client.responses.calls[0]["instructions"]


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
