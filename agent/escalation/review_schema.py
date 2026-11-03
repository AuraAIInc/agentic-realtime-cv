"""Expected LLM review output for each escalation code, as pydantic models.

Kept minimal on purpose: the ``verdict``, a short ``rationale`` for it, plus
``chosen_cat_id`` / ``cut_time`` where a verdict names one.
``review_schema_json(code)`` renders the model's JSON schema so it can
be pasted into the review request; ``emit()`` logs it right after the prompt.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EscalationReview(BaseModel):
    """One parsed verdict. Subclasses narrow ``verdict`` to the code's options."""

    model_config = ConfigDict(extra="forbid")

    verdict: str = Field(description="The decision (allowed values depend on the escalation code).")
    rationale: str = Field(description="One or two sentences explaining the evidence behind the verdict.")


class T1Review(EscalationReview):
    """One box fits several cats."""

    verdict: Literal["confirm_assigned", "reassign_candidate", "reject_box"] = Field(
        description="confirm_assigned: the tracker's cat is right. reassign_candidate: another candidate cat is "
        "right (say which in chosen_cat_id). reject_box: no cat is clearly there, drop the box."
    )
    chosen_cat_id: int | None = Field(default=None, description="Required when verdict is reassign_candidate.")


class T2Review(EscalationReview):
    """Box re-assigned by nearness alone, no overlap."""

    verdict: Literal["confirm", "keep_original", "reassign_other", "drop"] = Field(
        description="confirm: the new cat is right. keep_original: the camera's first guess was right. "
        "reassign_other: a different nearby cat is right (say which in chosen_cat_id). drop: no cat there."
    )
    chosen_cat_id: int | None = Field(default=None, description="Required when verdict is reassign_other.")


class T3Review(EscalationReview):
    """Duplicate boxes killed on a slim certainty gap."""

    verdict: Literal["confirm_survivor", "keep_killed", "keep_both"] = Field(
        description="confirm_survivor: the kept box is right. keep_killed: the discarded box should have won. "
        "keep_both: two cats genuinely overlap, keep both boxes."
    )


class T4Review(EscalationReview):
    """New sighting missed the track; history was wiped."""

    verdict: Literal["confirm_reset", "link_continuation", "drop_box_as_outlier"] = Field(
        description="confirm_reset: a genuinely new appearance, wiping history was right. link_continuation: same cat, "
        "the new box continues the old track. drop_box_as_outlier: the new box is a one-off error, keep the old track."
    )


class T5Review(EscalationReview):
    """Committed to a position with no solid sighting."""

    verdict: Literal["commit_target", "wait_for_stronger", "close_visit"] = Field(
        description="commit_target: the weak box is really the cat, keep it. wait_for_stronger: do not commit yet, "
        "wait for a stronger sighting. close_visit: there is no cat here, close the visit."
    )


class T6Review(EscalationReview):
    """Box jumped across the frame; visit was split."""

    verdict: Literal["confirm_split", "merge_visits", "drop_box"] = Field(
        description="confirm_split: two different cats (or a false box), the split was right. merge_visits: one "
        "fast-moving cat, join the visits back together. drop_box: the jumped box is false, keep the old visit."
    )


class T7Review(EscalationReview):
    """Long run of guessed frames with no sighting."""

    verdict: Literal["keep_synthesis", "cut_at_gap_start", "shorten_gap"] = Field(
        description="keep_synthesis: the cat stayed put unseen, the guessed run is fine. cut_at_gap_start: the cat "
        "left, end the visit where sightings stopped. shorten_gap: partly present — say the last frame where the cat "
        "is actually visible in cut_time."
    )
    cut_time: str | None = Field(
        default=None, description="ISO timestamp of the last frame showing the cat. Required when verdict is shorten_gap."
    )


class P1Review(EscalationReview):
    """Two cats nearly tied for one visit."""

    verdict: Literal["confirm_winner", "award_runner_up", "split_visit"] = Field(
        description="confirm_winner: the awarded cat is right for the whole visit. award_runner_up: the runner-up cat "
        "is right (say which in chosen_cat_id). split_visit: different cats at different times, split it."
    )
    chosen_cat_id: int | None = Field(default=None, description="Required when verdict is award_runner_up.")


class P2Review(EscalationReview):
    """Visit awarded on very thin evidence."""

    verdict: Literal["keep", "drop", "merge_neighbor"] = Field(
        description="keep: a real appearance, keep the visit. drop: noise, delete the visit. merge_neighbor: part of "
        "a neighboring visit of the same cat, merge them."
    )


class P3Review(EscalationReview):
    """Visit got its second-choice cat."""

    verdict: Literal["confirm_reassign", "restore_wanted", "mark_unknown"] = Field(
        description="confirm_reassign: the second-choice cat is right. restore_wanted: this visit really is the "
        "first-choice cat (the overlapping visit is the wrong one). mark_unknown: cannot tell, leave unassigned."
    )


class P4Review(EscalationReview):
    """Long visit nobody could label."""

    verdict: Literal["assign_cat", "split_fragments", "confirm_unknown"] = Field(
        description="assign_cat: you can tell which cat it is (say which in chosen_cat_id). split_fragments: mixed "
        "cats, needs a human (rows kept as-is). confirm_unknown: genuinely unidentifiable."
    )
    chosen_cat_id: int | None = Field(default=None, description="Required when verdict is assign_cat.")


class P5Review(EscalationReview):
    """Visit verdict rests on weak, patchy sightings."""

    verdict: Literal["keep", "drop", "trim_core", "reassign"] = Field(
        description="keep: a real visit, keep it. drop: noise padded with guesses, delete it. trim_core: keep only "
        "the high-certainty core (flagged for a human; rows kept as-is). reassign: another cat (say which in "
        "chosen_cat_id)."
    )
    chosen_cat_id: int | None = Field(default=None, description="Required when verdict is reassign.")


class P6Review(EscalationReview):
    """Two back-to-back visits that might be one."""

    verdict: Literal["merge", "keep_separate"] = Field(
        description="merge: the cat never left, join the two visits into one. keep_separate: the cat left and came "
        "back, two real visits."
    )


class P7Review(EscalationReview):
    """One long nap split into two visits."""

    verdict: Literal["merge", "keep_separate"] = Field(
        description="merge: one continuous presence (sleeping cat), join the visits. keep_separate: the cat left and "
        "returned to the same spot, two real visits."
    )


REVIEW_SCHEMAS = {
    "T1": T1Review,
    "T2": T2Review,
    "T3": T3Review,
    "T4": T4Review,
    "T5": T5Review,
    "T6": T6Review,
    "T7": T7Review,
    "P1": P1Review,
    "P2": P2Review,
    "P3": P3Review,
    "P4": P4Review,
    "P5": P5Review,
    "P6": P6Review,
    "P7": P7Review,
}


def review_model_for(code: str) -> type[EscalationReview]:
    """Pydantic model for a code's expected review output (base model if unknown)."""
    return REVIEW_SCHEMAS.get(code, EscalationReview)


def review_schema_json(code: str) -> str:
    """Compact JSON schema text to hand the LLM alongside the prompt."""
    return json.dumps(review_model_for(code).model_json_schema(), separators=(",", ":"))
