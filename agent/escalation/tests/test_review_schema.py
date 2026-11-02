"""Each escalation code has a pydantic review-output schema; emit() logs it."""

import json
import logging

import pytest
from pydantic import ValidationError

from agent.escalation import REVIEW_SCHEMAS, Escalation, emit, review_model_for, review_schema_json

ALL_CODES = {"T1", "T2", "T3", "T4", "T5", "T6", "T7", "P1", "P2", "P3", "P4", "P5", "P6", "P7"}


def test_every_code_has_schema_with_verdict_and_shared_fields():
    assert set(REVIEW_SCHEMAS) == ALL_CODES
    for code, model in REVIEW_SCHEMAS.items():
        schema = json.loads(review_schema_json(code))
        assert "verdict" in schema["properties"], code
        assert schema["properties"]["verdict"].get("enum"), code
        for field in ("escalation", "confidence", "explanation", "frames_checked", "needs_human"):
            assert field in schema["properties"], (code, field)
        assert "confidence" in schema["required"], code


def test_reviews_validate_and_reject_bad_verdicts():
    good = {"escalation": "E6-T2", "confidence": 0.9, "explanation": "saw cat 25 clearly", "verdict": "confirm"}
    assert review_model_for("T2").model_validate(good).verdict == "confirm"
    with pytest.raises(ValidationError):
        review_model_for("T2").model_validate({**good, "verdict": "maybe"})
    with pytest.raises(ValidationError):
        review_model_for("P6").model_validate({"escalation": "E9-P6", "verdict": "merge"})


def test_unknown_code_falls_back_to_base_model():
    assert review_model_for("T9").__name__ == "EscalationReview"
    assert "verdict" not in json.loads(review_schema_json("T9"))["properties"]


def test_emit_logs_schema_after_prompt(caplog, capsys):
    import datetime

    esc = Escalation(time=datetime.datetime(2026, 10, 28, 16, 35, 21), scope="frame", code="T2", reason="r", seq=6)
    with caplog.at_level(logging.INFO, logger="agent.escalation.escalation"):
        emit([], esc)
    assert json.loads(capsys.readouterr().out.strip())["name"] == "E6-T2"
    texts = [r.getMessage() for r in caplog.records]
    assert any(t.startswith("Escalation prompt E6-T2:") for t in texts)
    schemas = [t for t in texts if t.startswith("Escalation output schema E6-T2:")]
    assert len(schemas) == 1
    assert '"confirm"' in schemas[0] and '"verdict"' in schemas[0]
