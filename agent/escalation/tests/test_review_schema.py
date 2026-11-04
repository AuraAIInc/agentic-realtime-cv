"""Each escalation code has a pydantic review-output schema; emit() logs it."""

import json
import logging

import pytest
from pydantic import ValidationError

from agent.escalation import REVIEW_SCHEMAS, Escalation, emit, review_model_for, review_schema_json

ALL_CODES = {"T1", "T2", "T3", "T4", "T5", "T6", "T7", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "H1"}


def test_every_code_has_minimal_schema():
    assert set(REVIEW_SCHEMAS) == ALL_CODES
    for code, model in REVIEW_SCHEMAS.items():
        schema = json.loads(review_schema_json(code))
        assert "verdict" in schema["properties"], code
        assert schema["properties"]["verdict"].get("enum"), code
        assert schema["required"] == ["verdict", "rationale"], code
        assert "rationale" in schema["properties"], code
        allowed = {"verdict", "rationale"}
        if code in ("T1", "T2", "P1", "P4", "P5"):
            allowed.add("chosen_cat_id")
        if code == "T7":
            allowed.add("cut_time")
        assert set(schema["properties"]) == allowed, code


def test_reviews_validate_and_reject_bad_verdicts():
    assert (
        review_model_for("T2").model_validate({"verdict": "confirm", "rationale": "coat matches"}).verdict
        == "confirm"
    )
    assert (
        review_model_for("T2")
        .model_validate({"verdict": "reassign_other", "chosen_cat_id": 26, "rationale": "nearby cat"})
        .chosen_cat_id
        == 26
    )
    assert review_model_for("T7").model_validate(
        {"verdict": "shorten_gap", "cut_time": "2026-10-28T16:34:00", "rationale": "cat left early"}
    ).cut_time
    assert (
        review_model_for("H1").model_validate({"verdict": "flag_issue", "rationale": "wrong cat"}).verdict
        == "flag_issue"
    )
    assert (
        review_model_for("P8").model_validate({"verdict": "relabel_to_vote", "rationale": "vote wins"}).verdict
        == "relabel_to_vote"
    )
    with pytest.raises(ValidationError):
        review_model_for("P8").model_validate({"verdict": "keep", "rationale": "x"})
    with pytest.raises(ValidationError):
        review_model_for("H1").model_validate({"verdict": "maybe", "rationale": "x"})
    with pytest.raises(ValidationError):
        review_model_for("T2").model_validate({"verdict": "maybe", "rationale": "x"})
    with pytest.raises(ValidationError):
        review_model_for("T2").model_validate({"verdict": "confirm", "confidence": 0.9})
    with pytest.raises(ValidationError):
        review_model_for("T2").model_validate({"verdict": "confirm"})  # rationale required
    with pytest.raises(ValidationError):
        review_model_for("P6").model_validate({})


def test_unknown_code_falls_back_to_base_model():
    assert review_model_for("T9").__name__ == "EscalationReview"
    assert set(json.loads(review_schema_json("T9"))["properties"]) == {"verdict", "rationale"}


def test_emit_logs_schema_after_prompt(caplog):
    import datetime

    esc = Escalation(time=datetime.datetime(2026, 10, 28, 16, 35, 21), scope="frame", code="T2", reason="r", seq=6)
    with caplog.at_level(logging.INFO, logger="agent.escalation.escalation"):
        emit([], esc)
    texts = [r.getMessage() for r in caplog.records]
    records = [t for t in texts if t.startswith("Escalation E6-T2:")]
    assert len(records) == 1
    assert json.loads(records[0].split(": ", 1)[1])["name"] == "E6-T2"
    assert any(t.startswith("Escalation prompt E6-T2:") for t in texts)
    schemas = [t for t in texts if t.startswith("Escalation output schema E6-T2:")]
    assert len(schemas) == 1
    assert '"confirm"' in schemas[0] and '"verdict"' in schemas[0]
