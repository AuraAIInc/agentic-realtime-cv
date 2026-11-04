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
        "left, end the visit where sightings stopped and delete the interpolated gap rows; frames after the gap "
        "become a new visit. shorten_gap: partly present — say the last frame where the cat is actually visible in "
        "cut_time; rows after it are deleted and frames after the gap become a new visit."
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

    verdict: Literal["confirm", "drop", "reassign"] = Field(
        description="confirm: the identity ruling favors the labeled cat (cues quoted) AND a real appearance. drop: "
        "noise, delete the visit. reassign: another cat's visit (say which in chosen_cat_id)."
    )
    chosen_cat_id: int | None = Field(default=None, description="Required when verdict is reassign.")


class P3Review(EscalationReview):
    """Visit got its second-choice cat."""

    verdict: Literal["confirm", "reassign", "drop"] = Field(
        description="confirm: the assigned cat is right for this visit. reassign: another profile cat is right "
        "(say which in chosen_cat_id). drop: not a cat at all, delete the visit."
    )
    chosen_cat_id: int | None = Field(default=None, description="Required when verdict is reassign.")


class P4Review(EscalationReview):
    """Long visit nobody could label."""

    verdict: Literal["assign_cat", "split_fragments", "confirm_unknown"] = Field(
        description="assign_cat: you can tell which cat it is (say which in chosen_cat_id). split_fragments: mixed "
        "cats, needs a human (rows kept as-is). confirm_unknown: genuinely unidentifiable."
    )
    chosen_cat_id: int | None = Field(default=None, description="Required when verdict is assign_cat.")


class P5Review(EscalationReview):
    """Visit verdict rests on weak, patchy sightings."""

    verdict: Literal["keep", "drop", "reassign"] = Field(
        description="keep: the identity ruling favors the labeled cat (cues quoted) AND a real visit — confirm the "
        "original detection. drop: noise padded with guesses, delete the labels. reassign: a different cat wearing "
        "the wrong label (say which in chosen_cat_id)."
    )
    chosen_cat_id: int | None = Field(default=None, description="Required when verdict is reassign.")


class P6Review(EscalationReview):
    """Two back-to-back visits that might be one."""

    verdict: Literal["merge", "keep_separate", "different_cats"] = Field(
        description="merge: the cat never left, join the two visits into one. keep_separate: the cat left and came "
        "back, two real visits of the same cat. different_cats: the visits are different cats — the new visit is "
        "actually chosen_cat_id (the previous visit keeps its label)."
    )
    chosen_cat_id: int | None = Field(
        default=None,
        description="Required when verdict is different_cats: the cat whose appearance matches the new visit's frames "
        "(quote the cues in the rationale), not the cat implied by other visits.",
    )


class P7Review(EscalationReview):
    """One long nap split into two visits."""

    verdict: Literal["merge", "keep_separate"] = Field(
        description="merge: one continuous presence (sleeping cat), join the visits. keep_separate: the cat left and "
        "returned to the same spot, two real visits."
    )


class P8Review(EscalationReview):
    """Tracker label disagrees with the period vote."""

    verdict: Literal["relabel_to_vote", "keep_tracked", "split_mixed"] = Field(
        description="relabel_to_vote: box-cropped evidence favors the voted cat at EVERY checked sample (3+ spread "
        "frames) — relabel the whole visit. keep_tracked: the tracker's cat is right, the vote was wrong. "
        "split_mixed: the visit mixes two cats with a clean handover — "
        "split at cut_time; frames up to the cut keep the tracked label, frames after take chosen_cat_id as a new "
        "visit. cut_time must be the LAST frame where the first cat is stably visible (verify both sides first); "
        "with no clean handover, do not split."
    )
    chosen_cat_id: int | None = Field(
        default=None, description="Cat for frames after the cut. Required when verdict is split_mixed."
    )
    cut_time: str | None = Field(
        default=None,
        description="ISO timestamp of the last frame of the first cat. Required when verdict is split_mixed.",
    )


class H1Review(EscalationReview):
    """Quiet-window spot-check: the tracker was confident, nothing was flagged."""

    verdict: Literal["confirm", "flag_issue"] = Field(
        description="confirm: tracking looks right across the spot-check window. flag_issue: something looks wrong "
        "(wrong cat, missed cat, bad boxes) — describe it in the rationale; rows stay as-is for a follow-up."
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
    "P8": P8Review,
    "H1": H1Review,
}


def review_model_for(code: str) -> type[EscalationReview]:
    """Pydantic model for a code's expected review output (base model if unknown)."""
    return REVIEW_SCHEMAS.get(code, EscalationReview)


def review_schema_json(code: str) -> str:
    """Compact JSON schema text to hand the LLM alongside the prompt."""
    return json.dumps(review_model_for(code).model_json_schema(), separators=(",", ":"))
