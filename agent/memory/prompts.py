"""Prompt templates for the memory LLM calls (phase 3).

Pure string builders: no I/O, no SDK imports. Four prompts share one style
guide: store-update (merge one observation), promote-summary (compress
children into a parent), and bootstrap-vision (first-frame JSON).
"""

from __future__ import annotations

STYLE_GUIDE = """Write plain, concrete language a non-expert can read.
Use the present tense for stable facts ("Qb naps here most afternoons").
Name cats by lowercase id (qb, qbo) only when the input identifies them.
Never invent cats, places, or events not present in the input.
Output the required JSON object only: no markdown, no code fences, no preamble."""

STORE_UPDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "keywords": {"type": "array", "items": {"type": "string"}},
        "contradicts_stable": {"type": "boolean"},
        "note": {"type": "string"},
    },
    "required": ["keywords", "contradicts_stable", "note"],
    "additionalProperties": False,
}

PROMOTE_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["text", "keywords"],
    "additionalProperties": False,
}

BOOTSTRAP_VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "full_view": {
            "type": "object",
            "properties": {
                "environment": {"type": "string"},
                "background": {"type": "string"},
                "cat_contrast": {"type": "string"},
            },
            "required": ["environment", "background", "cat_contrast"],
            "additionalProperties": False,
        },
        "zones": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "environment": {"type": "string"},
                    "background": {"type": "string"},
                    "cat_contrast": {"type": "string"},
                },
                "required": ["id", "environment", "background", "cat_contrast"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["full_view", "zones"],
    "additionalProperties": False,
}


def store_update_instructions() -> str:
    """Merge one observation against a unit's established stable text."""
    return (
        "You maintain one unit of a cat-tracking spatial-temporal memory.\n"
        f"{STYLE_GUIDE}\n"
        "Compare OBSERVATION against the unit's STABLE text and respond with JSON:\n"
        '{"keywords": [lowercase search terms from the observation],\n'
        ' "contradicts_stable": bool,\n'
        ' "note": "one plain sentence, at most 40 words, restating the observation"}.\n'
        "The established stable pattern outranks any single new input: set "
        "contradicts_stable true only when the observation is genuinely "
        "incompatible with STABLE (different routine, different place use), not "
        "for extra detail, rephrasing, or a first observation."
    )


def store_update_input(unit: dict, observation: str) -> str:
    stable = unit.get("stable") or "(no established pattern yet)"
    weight = unit.get("weight", 0)
    return f"STABLE (weight {weight}): {stable}\nOBSERVATION: {observation}"


def promote_summary_instructions(level: str, word_cap: int) -> str:
    """Compress child units into one parent summary (dominant patterns only)."""
    return (
        f"You compress {level} child summaries into one parent memory unit.\n"
        f"{STYLE_GUIDE}\n"
        "Respond with JSON: {\"text\": parent summary, \"keywords\": [terms]}.\n"
        "Dominant patterns only: every child below the support gate was already "
        "removed, so summarize what repeats across the remaining children and "
        "their rolled-up visit counts. Never let a one-off detail into the "
        "parent text; when over the word cap, drop the lowest-support detail "
        "first.\n"
        f"Hard cap: at most {word_cap} words in text."
    )


def promote_summary_input(
    level: str, children: list[dict], stats: dict, word_cap: int
) -> str:
    lines = [f"LEVEL: {level} (cap {word_cap} words)", f"ROLLED-UP STATS: {stats}"]
    for child in children:
        lines.append(
            f"- {child.get('id')}: visits={child.get('visits', 0)} "
            f"text={child.get('text', '')}"
        )
    return "\n".join(lines)


def bootstrap_vision_instructions(zone_ids: list[str]) -> str:
    """Describe the first full-view frame as strict environment JSON."""
    zones = ", ".join(zone_ids)
    return (
        "You bootstrap a cat-tracking spatial memory from its first camera frame.\n"
        f"{STYLE_GUIDE}\n"
        "Respond with JSON: {\"full_view\": {\"environment\", \"background\", "
        "\"cat_contrast\"}, \"zones\": [{\"id\", \"environment\", \"background\", "
        "\"cat_contrast\"}, ...]}.\n"
        f"Cover the whole view plus exactly these zones: {zones}.\n"
        "environment: what the area is (sofa, doorway, food bowl, floor).\n"
        "background: colors, materials, lighting relevant to tracking.\n"
        "cat_contrast: how well a cat would stand out here, and any occluders.\n"
        "Describe only what is visible; mark anything unclear as unknown."
    )


def bootstrap_vision_input() -> str:
    return "Describe this first camera frame as the bootstrap JSON object."
