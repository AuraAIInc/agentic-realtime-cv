"""Reviewer prompts: one short standalone validation prompt per escalation.

Each ``build_<code>_prompt`` turns an :class:`Escalation` into a few terse
lines a human or LLM reviewer can answer fast: conflict, the validation
question, minimal context (first guess, proposal, why uncertain), alternatives,
and key frames listed directly with pick reasons. Kept short on purpose —
these prompts feed an LLM reviewer, so every extra sentence costs generation
time. Tracking terms get a few-word gloss where they appear.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.escalation.escalation import Escalation

MODEL_CAVEAT = "Note: proposal is from a small YOLO on a weak CPU — a hint, not evidence. Trust the images and cat profiles."


def _f(v, fmt: str = ".2f") -> str:
    """Format a metric number, tolerating missing values."""
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return "?"


def _ts(v) -> str:
    if isinstance(v, datetime.datetime):
        return v.isoformat(timespec="microseconds")
    return str(v)


def _head(esc: Escalation, title: str) -> str:
    name = esc.name or esc.code
    return f"[{name}] {title}."


def _frame(ts, why: str) -> str:
    """One key-frame bullet: timestamp plus why it was picked."""
    return f"  - {_ts(ts)} — {why}"


def _minute_samples(start, end, cap: int = 6) -> list:
    """Evenly spaced sample times across a span, roughly one per minute."""
    if not isinstance(start, datetime.datetime) or not isinstance(end, datetime.datetime):
        return []
    total = (end - start).total_seconds()
    if total <= 0:
        return []
    n = min(cap, max(1, int(total // 60)))
    if n == 1:
        return [start + (end - start) / 2]
    return [start + (end - start) * i / (n + 1) for i in range(1, n + 1)]


def _dedupe(frames: list[str]) -> list[str]:
    seen, out = set(), []
    for line in frames:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out


def build_t1_prompt(esc: Escalation) -> str:
    m = esc.metrics
    cands = m.get("candidates") or {}
    cand_txt = ", ".join(f"{c} ({'overlap' if r == 'iou' else 'nearness'})" for c, r in sorted(cands.items()))
    return (
        f"{_head(esc, 'One box fits several cats')}\n"
        f"Conflict: camera guessed cat {m.get('original_cat_id')}, but the box fits {len(cands)} cats' paths; tracker kept the first match.\n"
        f"Validate: which cat is in the photo — {esc.cat_id} or another candidate?\n"
        f"Context: proposal cat {m.get('chosen_cat_id')}. Fits: [{cand_txt}]. Winner picked by order, not evidence. {MODEL_CAVEAT}\n"
        f"Alternatives: assign another candidate, or drop the box.\n"
        f"Resources (images__get_image):\n"
        f"{_frame(esc.time, 'the disputed photo')}"
    )


def build_t2_prompt(esc: Escalation) -> str:
    m = esc.metrics
    return (
        f"{_head(esc, 'Identity switched on nearness alone, no overlap')}\n"
        f"Conflict: box moved from cat {m.get('original_cat_id')} to cat {esc.cat_id} by center nearness only, zero box overlap.\n"
        f"Validate: is the photo really cat {esc.cat_id}?\n"
        f"Context: first guess {m.get('original_cat_id')}; proposal {esc.cat_id}. Nearness fires with no overlap, so neighbors and false "
        f"alarms get stolen. {MODEL_CAVEAT}\n"
        f"Alternatives: keep {m.get('original_cat_id')}, pick another nearby cat, or drop.\n"
        f"Resources (images__get_image):\n"
        f'{_frame(esc.time, "the disputed photo; compare both cats' profiles")}'
    )


def build_t3_prompt(esc: Escalation) -> str:
    m = esc.metrics
    return (
        f"{_head(esc, 'Duplicate boxes, winner picked by a hair')}\n"
        f"Conflict: two boxes on the same spot (overlap {_f(m.get('iou'))}); cat {m.get('killed_cat_id')}'s box was discarded on a "
        f"{_f(m.get('conf_margin'))} certainty gap.\n"
        f"Validate: is the surviving cat {esc.cat_id} box right, or should the discarded one have won?\n"
        f"Context: camera reported cats {m.get('killed_cat_id')} and {esc.cat_id}; proposal keeps {esc.cat_id}. A "
        f"{_f(m.get('conf_margin'))} gap is detector noise. {MODEL_CAVEAT}\n"
        f"Alternatives: keep cat {m.get('killed_cat_id')}'s box, or keep both if two cats overlap.\n"
        f"Resources (images__get_image):\n"
        f"{_frame(esc.time, 'the disputed photo; one cat or two?')}"
    )


def build_t4_prompt(esc: Escalation) -> str:
    m = esc.metrics
    return (
        f"{_head(esc, 'Track thrown away: new sighting matched nothing')}\n"
        f"Conflict: new box barely overlaps cat {esc.cat_id}'s path (overlap {_f(m.get('base_iou'))}); tracker wiped {m.get('wiped_rows')} "
        f"history frames and closed the visit.\n"
        f"Validate: new appearance (wipe right) or same cat jumping (history wrongly lost)?\n"
        f"Context: proposal discards history and restarts. Low overlap can be a new cat, a fast move, or one bad box — wiping is "
        f"irreversible. {MODEL_CAVEAT}\n"
        f"Alternatives: keep old track, treat box as outlier, or link as continuation.\n"
        f"Resources: period {esc.period_id} (tracking__get_period_detail); frames via images__get_image:\n"
        f"{_frame(esc.time, 'the unmatched new sighting; also check frames just before it')}"
    )


def build_t5_prompt(esc: Escalation) -> str:
    m = esc.metrics
    if "target_conf" in m:
        conflict = (
            f"no box overlaps the path and none tops 0.5 certainty (best {_f(m.get('target_conf'))} at {_ts(m.get('target_time'))})"
        )
        correction = f"use the weak {_f(m.get('target_conf'))}-certainty box as the position"
        alts = "wait for a stronger sighting, or close the visit"
        frames = "\n".join(
            [
                _frame(m.get("target_time"), "the weak box the tracker committed to"),
                _frame(esc.time, "when the fallback decision was made"),
            ]
        )
    else:
        conflict = f"history holds only empty or guessed frames ({m.get('buffer_rows')} rows, no real sighting)"
        correction = "keep the visit open with no confirmed position"
        alts = "close the visit instead of stretching over empty frames"
        frames = _frame(esc.time, "decision time; is any cat visible?")
    return (
        f"{_head(esc, 'Committing with no solid sighting')}\n"
        f"Conflict: cat {esc.cat_id} has an open visit but {conflict}.\n"
        f"Validate: commit to this weak evidence, or is the track running on noise?\n"
        f"Context: proposal: {correction}. Weak and guessed boxes often drift off the cat. {MODEL_CAVEAT}\n"
        f"Alternatives: {alts}.\n"
        f"Resources (images__get_image):\n"
        f"{frames}"
    )


def build_t6_prompt(esc: Escalation) -> str:
    m = esc.metrics
    before = esc.time - datetime.timedelta(seconds=60) if isinstance(esc.time, datetime.datetime) else None
    return (
        f"{_head(esc, 'Cat seemingly teleported; visit split')}\n"
        f"Conflict: cat {esc.cat_id} jumped {_f(m.get('distance_px'), '.0f')} px in one step; tracker split visit {m.get('old_period_id')} "
        f"into visit {esc.period_id}, deleting {m.get('retracted_rows')} guessed frames.\n"
        f"Validate: one fast cat (split wrong) or two cats / false box (split right)?\n"
        f"Context: old path vs new box (certainty {_f(m.get('box_conf'))}); proposal keeps them separate. A sprint and a glitch look "
        f"identical. {MODEL_CAVEAT}\n"
        f"Alternatives: merge into one visit, or drop the jumped box.\n"
        f"Resources: visits {m.get('old_period_id')}, {esc.period_id} (tracking__get_period_detail); frames via images__get_image:\n"
        f"{_frame(esc.time, 'post-jump frame')}\n"
        f"{_frame(before, 'a minute earlier; last known position')}"
    )


def build_t7_prompt(esc: Escalation) -> str:
    m = esc.metrics
    start, end = m.get("gap_start"), esc.time
    lines = _dedupe(
        [
            _frame(start, "last real sighting; gap starts here"),
            *(_frame(t, "no sighting, tracker guessed — is the cat visible?") for t in _minute_samples(start, end)),
            _frame(end, "gap end; still no fresh sighting"),
        ]
    )
    return (
        f"{_head(esc, 'Seconds of guessed frames, no sighting')}\n"
        f"Conflict: cat {esc.cat_id} has {_f(m.get('gap_seconds'), '.1f')} seconds of frames where the camera saw nothing; all positions "
        f"since {_ts(start)} were guessed.\n"
        f"Validate: cat present but missed (guessing right), or gone while the track imagined it?\n"
        f"Context: proposal keeps the guessed run through {_ts(end)}. Guessing assumes presence. {MODEL_CAVEAT}\n"
        f"Alternatives: cut the visit at {_ts(start)}, or shorten to the last visible frame.\n"
        f"Resources: visit {esc.period_id} (tracking__get_period_detail); key frames ~1/min via images__get_image:\n" + "\n".join(lines)
    )


def _period_frames(esc: Escalation, focus: str) -> str:
    return "\n".join(
        _dedupe(
            [
                _frame(esc.time, "visit start"),
                *(_frame(t, focus) for t in _minute_samples(esc.time, esc.end_time)),
                _frame(esc.end_time, "visit end"),
            ]
        )
    )


def build_p1_prompt(esc: Escalation) -> str:
    m = esc.metrics
    scores = m.get("scores") or []
    order = sorted(range(len(scores)), key=lambda i: -scores[i]) if scores else []
    top = _f(scores[order[0]], ".1f") if order else "?"
    runner = _f(scores[order[1]], ".1f") if len(order) > 1 else "?"
    return (
        f"{_head(esc, 'Two cats nearly tied for this visit')}\n"
        f"Conflict: visit {esc.period_id} scored cat {esc.cat_id} {top} vs runner-up {runner} (margin {_f(m.get('margin'), '.1f')} pts).\n"
        f"Validate: cat {esc.cat_id} throughout, the runner-up, or different cats at different times?\n"
        f"Context: votes [{', '.join(_f(s, '.1f') for s in scores)}] per cat; proposal awards all to {esc.cat_id}. Winner may have one lucky "
        f"frame. {MODEL_CAVEAT}\n"
        f"Alternatives: award runner-up, or split the visit.\n"
        f"Resources: visit {esc.period_id} (tracking__get_period_detail); key frames ~1/min via images__get_image:\n"
        f"{_period_frames(esc, 'which cat is visible? linger on missing/low-certainty stretches')}"
    )


def build_p2_prompt(esc: Escalation) -> str:
    m = esc.metrics
    return (
        f"{_head(esc, 'Visit awarded on thin evidence')}\n"
        f"Conflict: visit {esc.period_id} -> cat {esc.cat_id} on {_f(m.get('score'), '.1f')} pts from {m.get('detect_length')} original "
        f"frames (frames actually seen, not guessed).\n"
        f"Validate: real appearance or noise that should never have become a visit?\n"
        f"Context: proposal keeps it as cat {esc.cat_id}. Short low-score visits are often stray boxes. {MODEL_CAVEAT}\n"
        f"Alternatives: drop as noise, or merge into a neighboring visit.\n"
        f"Resources: visit {esc.period_id} (tracking__get_period_detail); key frames ~1/min via images__get_image:\n"
        f"{_period_frames(esc, 'is a cat really there? evidence is thin')}"
    )


def build_p3_prompt(esc: Escalation) -> str:
    m = esc.metrics
    return (
        f"{_head(esc, 'Visit got second-choice cat; first choice was taken')}\n"
        f"Conflict: visit {esc.period_id} voted cat {m.get('wanted_cat_id')}, but a simultaneous visit claimed it (one cat, one place), so it "
        f"got cat {esc.cat_id}.\n"
        f"Validate: really cat {esc.cat_id} — or is the other visit the mislabeled one?\n"
        f"Context: proposal assigns {esc.cat_id}. The swap was a scheduling fix, not new evidence. {MODEL_CAVEAT}\n"
        f"Alternatives: restore {m.get('wanted_cat_id')} here and reassign the other visit, or mark one unknown.\n"
        f"Resources: visit {esc.period_id} + overlapping visits (tracking__get_period_detail); key frames ~1/min via images__get_image:\n"
        f"{_period_frames(esc, 'identify the cat; compare the overlapping visit at the same times')}"
    )


def build_p4_prompt(esc: Escalation) -> str:
    m = esc.metrics
    return (
        f"{_head(esc, 'Long visit nobody could label')}\n"
        f"Conflict: visit {esc.period_id} ran {_f(m.get('seconds'), '.0f')} seconds / {m.get('rows')} frames but scored too low for every cat.\n"
        f"Validate: which cat (if any) is this — and why did voting fail?\n"
        f"Context: proposal none (left unknown). A long visit usually IS a cat; occlusion, bad boxes, or a missing profile may hide it. "
        f"{MODEL_CAVEAT}\n"
        f"Alternatives: assign best-guess cat, split into fragments, or confirm unidentifiable.\n"
        f"Resources: visit {esc.period_id} (tracking__get_period_detail); key frames ~1/min via images__get_image:\n"
        f"{_period_frames(esc, 'identify the cat; focus where voting went blind')}"
    )


def build_p5_prompt(esc: Escalation) -> str:
    m = esc.metrics
    return (
        f"{_head(esc, 'Verdict rests on weak, patchy sightings')}\n"
        f"Conflict: visit {esc.period_id} (cat {esc.cat_id}): avg certainty {_f(m.get('detect_conf'))}, original-frame rate "
        f"{_f(m.get('detect_rate'))} (share of frames actually seen; rest guessed).\n"
        f"Validate: real visit or noise padded with guesses?\n"
        f"Context: proposal keeps cat {esc.cat_id}. Mostly-guessed visits may not reflect reality. {MODEL_CAVEAT}\n"
        f"Alternatives: drop, trim to the high-certainty core, or reassign.\n"
        f"Resources: visit {esc.period_id} (tracking__get_period_detail); key frames ~1/min via images__get_image:\n"
        f"{_period_frames(esc, 'is the cat visible? weigh real sightings over guesses')}"
    )


def build_p6_prompt(esc: Escalation) -> str:
    m = esc.metrics
    lines = _dedupe(
        [
            _frame(esc.time, f"visit {m.get('prev_period_id')} ends; last confirmed position"),
            *(_frame(t, "gap, untracked — did the cat stay or leave?") for t in _minute_samples(esc.time, esc.end_time)),
            _frame(esc.end_time, f"visit {esc.period_id} starts; first confirmed position"),
        ]
    )
    return (
        f"{_head(esc, 'Two back-to-back visits, might be one')}\n"
        f"Conflict: visit {esc.period_id} starts {_f(m.get('gap_seconds'), '.1f')} seconds after visit {m.get('prev_period_id')} ended, "
        f"{_f(m.get('distance_px'), '.0f')} pixels away.\n"
        f"Validate: merge visits {m.get('prev_period_id')} and {esc.period_id} into one cat {esc.cat_id} visit?\n"
        f"Context: proposal keeps separate. Short gap + nearby boxes suggests continuity, but the cat may have left and returned. "
        f"{MODEL_CAVEAT}\n"
        f"Alternatives: merge, or confirm separate visits.\n"
        f"Resources: visits {m.get('prev_period_id')}, {esc.period_id} (tracking__get_period_detail); gap frames ~1/min via "
        f"images__get_image:\n" + "\n".join(lines)
    )


def build_p7_prompt(esc: Escalation) -> str:
    m = esc.metrics
    lines = _dedupe(
        [
            _frame(esc.time, f"visit {m.get('prev_period_id')} ends, cat still — nap's last frame"),
            *(_frame(t, "quiet gap, untracked — cat still lying there?") for t in _minute_samples(esc.time, esc.end_time)),
            _frame(esc.end_time, f"visit {esc.period_id} starts, cat still — nap's first frame"),
        ]
    )
    return (
        f"{_head(esc, 'One nap split into two visits')}\n"
        f"Conflict: visit {esc.period_id} starts where visit {m.get('prev_period_id')} ended (overlap {_f(m.get('overlap_iou'))}) after a "
        f"{_f(m.get('gap_seconds'), '.0f')}-second gap.\n"
        f"Validate: merge visits {m.get('prev_period_id')} and {esc.period_id} into one cat {esc.cat_id} presence?\n"
        f"Context: proposal keeps separate. Both ends barely move ({_f(m.get('prev_displacement_px'), '.0f')}/"
        f"{_f(m.get('new_displacement_px'), '.0f')} px drift); one nap vs two naps in the same spot look identical. {MODEL_CAVEAT}\n"
        f"Alternatives: merge, or confirm the cat left and returned.\n"
        f"Resources: visits {m.get('prev_period_id')}, {esc.period_id} (tracking__get_period_detail); gap frames ~1/min via "
        f"images__get_image:\n" + "\n".join(lines)
    )


_BUILDERS = {
    "T1": build_t1_prompt,
    "T2": build_t2_prompt,
    "T3": build_t3_prompt,
    "T4": build_t4_prompt,
    "T5": build_t5_prompt,
    "T6": build_t6_prompt,
    "T7": build_t7_prompt,
    "P1": build_p1_prompt,
    "P2": build_p2_prompt,
    "P3": build_p3_prompt,
    "P4": build_p4_prompt,
    "P5": build_p5_prompt,
    "P6": build_p6_prompt,
    "P7": build_p7_prompt,
}


def build_prompt(esc: Escalation) -> str:
    """Dispatch to the per-code builder; unknown codes get a generic prompt."""
    builder = _BUILDERS.get(esc.code)
    if builder is None:
        return (
            f"{_head(esc, 'Unrecognized escalation code')}\n"
            f"Conflict: {esc.reason}\n"
            f"Validate: is the tracker's decision at {_ts(esc.time)} correct?\n"
            f"Resources (images__get_image):\n"
            f"{_frame(esc.time, 'the decision frame')}"
        )
    return builder(esc)
