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
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.escalation.escalation import Escalation

MODEL_CAVEAT = "Note: proposal is from a small YOLO on a weak CPU — a hint, not evidence. Trust the images and cat profiles."

_BOX_FOCUS = (
    "Each period holds exactly one cat, but another cat may share the frame under its own period — "
    "judge only the cat inside the listed box, not the most prominent cat in the image. "
    "Full-frame views do NOT count as box evidence: fetch every judged frame CROPPED to its listed box via "
    "local__get_frame(timestamp=..., box=[x1, y1, x2, y2]) (or images__get_image(box=...) when that server is up)."
)

_SCORE_EXPLAIN = (
    "pts = vote points (up to ~13): +3 best avg certainty, +2 most original frames, +4.2 widest area, +1 each "
    "for >20 frames / rate>0.5 / conf>0.5 / rate>0.3 — then x0.2 if under 10 original frames."
)


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


_RULE_MEANING = {"iou_last": "overlap with latest", "iou_buffer": "overlap with history", "distance": "nearness"}


def build_t1_prompt(esc: Escalation) -> str:
    m = esc.metrics
    cands = m.get("candidates") or {}
    refs = m.get("candidate_refs") or {}
    box = m.get("box")
    box_txt = f", Box {_box_txt(box)}" if box else ""
    order = sorted({str(c) for c in list(cands) + list(refs)})
    cand_bits = []
    ref_frames = []
    for c in order:
        rule = (refs.get(c) or {}).get("rule")
        if rule is None:
            for k, v in cands.items():
                if str(k) == c:
                    rule = v
                    break
        bit = f"cat {c} ({_RULE_MEANING.get(rule, rule)})"
        ref = refs.get(c) or {}
        if ref.get("source_time") and ref.get("source_box"):
            bit += f": latest box {_box_txt(ref['source_box'])} at {_ts(ref['source_time'])}"
            if rule == "distance" and ref.get("source_distance_px") is not None:
                bit += f", {_f(ref['source_distance_px'], '.0f')} px away"
            ref_frames.append(
                _frame(ref["source_time"], f"cat {c}'s latest position ({_RULE_MEANING.get(rule, rule)}). Box {_box_txt(ref['source_box'])}")
            )
        cand_bits.append(bit)
    frames = "\n".join([_frame(esc.time, f"the disputed photo{box_txt}")] + ref_frames)
    return (
        f"{_head(esc, 'One box fits several cats')}\n"
        f"Conflict: camera guessed cat {m.get('original_cat_id')}, but the box fits {len(order)} cats' paths; "
        f"tracker kept the first match it found (cat order, not best fit).\n"
        f"Validate: which cat is in the photo — {m.get('chosen_cat_id')} or another candidate?\n"
        f"Context: the tracker checks a stray box against every cat's recent path — by box overlap with the latest position, "
        f"overlap with older history, or center nearness — and stops at the first hit. Here: {'; '.join(cand_bits)}. "
        f"Proposal: cat {m.get('chosen_cat_id')}. {MODEL_CAVEAT}\n"
        f"Alternatives: assign another candidate, or drop the box.\n"
        f"Resources (images__get_image):\n"
        f"{frames}"
    )


def build_t2_prompt(esc: Escalation) -> str:
    m = esc.metrics
    orig, new = m.get("original_cat_id"), esc.cat_id
    box = m.get("box")
    src_t, src_box = m.get("source_time"), m.get("source_box")
    box_txt = f", Box {_box_txt(box)}" if box else ""
    if src_t is not None and src_box is not None:
        basis = (
            f"the nearest cat-{new} position on record was box {_box_txt(src_box)} at {_ts(src_t)}, "
            f"{_f(m.get('source_distance_px'), '.0f')} px away (the rule fires under "
            f"{_f(m.get('source_threshold_px'), '.0f')} px) — that nearness is the entire basis for the switch"
        )
        frames = "\n".join(
            [
                _frame(esc.time, f"the disputed photo; compare both cats' profiles{box_txt}"),
                _frame(
                    src_t,
                    f"source of the correction: cat {new}'s nearest recent position, "
                    f"{_f(m.get('source_distance_px'), '.0f')} px away. Box {_box_txt(src_box)}",
                ),
            ]
        )
    else:
        basis = f"no prior cat-{new} position was recorded, so the switch has no visible basis"
        frames = _frame(esc.time, f"the disputed photo; compare both cats' profiles{box_txt}")
    return (
        f"{_head(esc, 'Identity switched on nearness alone, no overlap')}\n"
        f"Conflict: the camera said cat {orig}, but the tracker re-labeled the box to cat {new} by nearness only — "
        f"the boxes do not overlap at all.\n"
        f"Validate: is the photo really cat {new}?\n"
        f"Context: the tracker assumes cats move smoothly (spatial continuity): a new box near a cat's last known position is "
        f"treated as that cat even when the detector said otherwise. Here the detector's first guess was cat {orig}{box_txt} at "
        f"{_ts(esc.time)}; {basis}. Nearness fires with no overlap, so neighbors and false alarms get stolen. {MODEL_CAVEAT}\n"
        f"Alternatives: keep {orig}, pick another nearby cat, or drop the box.\n"
        f"Resources (images__get_image):\n"
        f"{frames}"
    )


def build_t3_prompt(esc: Escalation) -> str:
    m = esc.metrics
    kept, killed = esc.cat_id, m.get("killed_cat_id")
    kept_box, killed_box = m.get("kept_box"), m.get("killed_box")
    if kept_box is not None and killed_box is not None:
        boxes = (
            f"Survivor: cat {kept} Box {_box_txt(kept_box)} certainty {_f(m.get('kept_conf'))}; discarded: cat {killed} "
            f"Box {_box_txt(killed_box)} certainty {_f(m.get('killed_conf'))}."
        )
        frame = _frame(
            esc.time,
            f"both boxes are on this one frame — kept: cat {kept} Box {_box_txt(kept_box)}; killed: cat {killed} "
            f"Box {_box_txt(killed_box)}. One cat or two?",
        )
    else:
        boxes = f"camera reported cats {killed} and {kept}."
        frame = _frame(esc.time, "the disputed photo; one cat or two?")
    return (
        f"{_head(esc, 'Duplicate boxes, winner picked by a hair')}\n"
        f"Task: validate keeping cat {kept}'s box and discarding cat {killed}'s — two boxes on the same spot (overlap "
        f"{_f(m.get('iou'))}, where 1.0 means identical).\n"
        f"Validate: look at the photo. Is the surviving box right, should the discarded one have won, or are two cats genuinely "
        f"overlapping?\n"
        f"Context: dedup rule — when two boxes land on the same spot the tracker keeps the higher-certainty one. {boxes} The "
        f"{_f(m.get('conf_margin'))} certainty gap is detector noise, not a real preference. Proposal keeps cat {kept}. {MODEL_CAVEAT}\n"
        f"Alternatives: keep cat {killed}'s box instead, or keep both if two cats overlap.\n"
        f"Resources (images__get_image):\n"
        f"{frame}"
    )


def build_t4_prompt(esc: Escalation) -> str:
    m = esc.metrics
    pid = esc.period_id
    base_t, base_box, box = m.get("base_time"), m.get("base_box"), m.get("box")
    had_open = m.get("had_open_visit")
    if base_box is not None and box is not None:
        anchor = f"the anchor was box {_box_txt(base_box)} at {_ts(base_t)}; the new sighting is box {_box_txt(box)}"
        lines = "\n".join(
            [
                _frame(base_t, f"the anchor: last committed position before the gap. Box {_box_txt(base_box)}"),
                _frame(esc.time, f"the unmatched new sighting, overlap {_f(m.get('base_iou'))} vs the anchor. Box {_box_txt(box)}"),
            ]
        )
    else:
        anchor = "the anchor (last committed position) and the new sighting"
        lines = _frame(esc.time, "the unmatched new sighting; also check frames just before it")
    if had_open is False:
        close = "No visit was still committing, so there is nothing left to close."
    else:
        close = "The old visit then closes on the following frames, and the next commit mints a new period."
    cont = m.get("continued_period_id")
    cont_txt = (
        f" The restart continued as visit {cont} (period_id {cont}) — linking would join it back to visit {pid}."
        if isinstance(cont, int)
        else ""
    )
    return (
        f"{_head(esc, 'Track thrown away: new sighting matched nothing')}\n"
        f"Task: validate whether discarding cat {esc.cat_id}'s old anchor and restarting from the new box at {_ts(esc.time)} was "
        f"right — overlap {_f(m.get('base_iou'))}, {m.get('wiped_rows')} buffered rows wiped.\n"
        f"Validate: look at the frames. Is this a genuinely new appearance (restart right), the same cat reappearing elsewhere "
        f"(history wrongly lost), or one bad box (restart over a glitch)?\n"
        f"Context: the anchor (base) is the cat's last committed position. Nothing had been seen for a while — no pending "
        f"detections — but the visit was still alive, so when the new box arrived somewhere the anchor cannot reach, the tracker "
        f"concluded it cannot continue the old track. Here {anchor}. The wipe is safe because with nothing pending, buffered rows "
        f"hold only empties and guesses, never real sightings; the new box is kept as the fresh start. {close}{cont_txt} {MODEL_CAVEAT}\n"
        f"Alternatives: keep the old track and treat the new box as an outlier, or link the new box as a continuation.\n"
        f"Resources: visit {pid} (this is period_id {pid} — look it up in tracking__get_period_detail or the database); frames via "
        f"images__get_image (boxes are the tracked positions for reference):\n"
        f"{lines}"
    )


def build_t5_prompt(esc: Escalation) -> str:
    m = esc.metrics
    target_box, base_box = m.get("target_box"), m.get("base_box")
    base_ref = ""
    if base_box is not None:
        base_ref = f" The anchor (last committed position) is box {_box_txt(base_box)} at {_ts(m.get('base_time'))} — nothing overlaps it."
        base_frame = _frame(m.get("base_time"), f"the anchor nothing overlaps. Box {_box_txt(base_box)}")
    else:
        base_frame = None
    if "target_conf" in m:
        task = (
            f"cat {esc.cat_id} has an open visit but no trustworthy box: nothing overlaps the anchor and none tops 0.5 certainty "
            f"(best {_f(m.get('target_conf'))} at {_ts(m.get('target_time'))})"
        )
        correction = f"use the weak {_f(m.get('target_conf'))}-certainty box as the position"
        alts = "wait for a stronger sighting, or close the visit"
        target_line = f"the weak box the tracker committed to{f', Box {_box_txt(target_box)}' if target_box else ''}"
        frames = "\n".join([x for x in [_frame(m.get("target_time"), target_line), base_frame,
                                        _frame(esc.time, "when the fallback decision was made")] if x is not None])
    else:
        task = f"cat {esc.cat_id} has an open visit but history holds only empty or guessed frames ({m.get('buffer_rows')} rows, no real sighting)"
        correction = "keep the visit open with no confirmed position"
        alts = "close the visit instead of stretching over empty frames"
        frames = _frame(esc.time, "decision time; is any cat visible?")
    return (
        f"{_head(esc, 'Committing with no solid sighting')}\n"
        f"Task: validate committing cat {esc.cat_id}'s position with no solid sighting — {task}.\n"
        f"Validate: look at the frames. Should the tracker commit to this weak evidence, or is the track running on noise?\n"
        f"Context: each frame the tracker must pick one box per cat as its current position (the target): normally a box "
        f"overlapping the anchor or above 0.5 certainty. Here neither exists, so it fell back to the best of the weaklings. "
        f"Proposal: {correction}.{base_ref} Weak and guessed boxes often drift off the cat. {MODEL_CAVEAT}\n"
        f"Alternatives: {alts}.\n"
        f"Resources (images__get_image):\n"
        f"{frames}"
    )


def build_t6_prompt(esc: Escalation) -> str:
    m = esc.metrics
    old_pid, new_pid = m.get("old_period_id"), esc.period_id
    dist = _f(m.get("distance_px"), ".0f")
    pre_t, pre_box, post_box = m.get("pre_time"), m.get("pre_box"), m.get("post_box")
    mids = [f for f in (m.get("retracted_frames") or []) if isinstance(f, dict) and f.get("t")]
    if pre_t is not None and post_box is not None:
        lines = _dedupe(
            [_frame(pre_t, f"pre-jump frame; last confirmed position. Box {_box_txt(pre_box)}")]
            + [_frame(f["t"], f"intermediate frame; interpolated guess. Interpolated box {_box_txt(f.get('box'))}") for f in mids]
            + [_frame(esc.time, f"post-jump frame; new sighting {dist} px away. Box {_box_txt(post_box)}")]
        )
        evidence = (
            f"last confirmed position was box {_box_txt(pre_box)} at {_ts(pre_t)} (that is the old path end); "
            f"the new sighting is box {_box_txt(post_box)} at {_ts(esc.time)}"
        )
    else:
        before = esc.time - datetime.timedelta(seconds=60) if isinstance(esc.time, datetime.datetime) else None
        lines = [_frame(esc.time, "post-jump frame"), _frame(before, "a minute earlier; last known position")]
        evidence = "the old path versus the new sighting"
    return (
        f"{_head(esc, 'Cat seemingly teleported; visit split')}\n"
        f"Task: validate whether splitting visit {old_pid} into visit {new_pid} at {_ts(esc.time)} was right — "
        f"cat {esc.cat_id} jumped {dist} px in one step, and the tracker deleted {m.get('retracted_rows')} guessed frames.\n"
        f"Validate: look at the frames below. Is this one fast-moving cat (the split was wrong, merge the visits back), "
        f"or two different cats or a false box (the split was right)?\n"
        f"Context: {evidence} (certainty {_f(m.get('box_conf'))} is the camera's 0-to-1 sureness score for the new box); "
        f"proposal keeps visits {old_pid} and {new_pid} separate. A sprinting cat and a box glitch look identical. {MODEL_CAVEAT}\n"
        f"Alternatives: merge visits {old_pid} and {new_pid} into one, or drop the jumped box as false.\n"
        f"Resources: visits {old_pid}, {new_pid} (these are period_id {old_pid} and {new_pid} — look them up in "
        f"tracking__get_period_detail or the database); frames via images__get_image (boxes are the tracked positions for reference):\n"
        + "\n".join(lines)
    )


def _box_txt(box) -> str:
    try:
        return "[" + ", ".join(_f(v, ".1f") for v in box) + "]"
    except TypeError:
        return "?"


def build_t7_prompt(esc: Escalation) -> str:
    m = esc.metrics
    start, end = m.get("gap_start"), esc.time
    interp = [f for f in (m.get("interp_frames") or []) if isinstance(f, dict) and f.get("t")]
    if interp:
        frames = []
        for i, f in enumerate(interp):
            if i == 0:
                why = "first guessed frame (last real sighting is just before it)"
            elif i == len(interp) - 1:
                why = "gap end; still no fresh sighting"
            else:
                why = "mid-gap guess"
            frames.append(_frame(f["t"], f"{why}. Interpolated box {_box_txt(f.get('box'))}"))
        lines = _dedupe(frames)
    else:
        lines = _dedupe(
            [
                _frame(start, "gap starts; guess starts here"),
                *(_frame(t, "no sighting, tracker guessed — is the cat visible?") for t in _minute_samples(start, end)),
                _frame(end, "gap end; still no fresh sighting"),
            ]
        )
    pid = esc.period_id
    return (
        f"{_head(esc, 'Seconds of guessed frames, no sighting')}\n"
        f"Task: double-check {_f(m.get('gap_seconds'), '.1f')} seconds of positions from {_ts(start)} to {_ts(end)} — "
        f"every box in this span was guessed, none was seen.\n"
        f"Validate: cat {esc.cat_id} present but missed (keep the guesses), or gone while the track imagined it (cut them)?\n"
        f"Context: proposal keeps the guessed run. Guessing assumes presence. {MODEL_CAVEAT}\n"
        f"Alternatives: cut the whole span {_ts(start)} to {_ts(end)}, or shorten to the last frame where the cat is visible. "
        f"Either way the visit splits there: frames after {_ts(end)} become a new visit.\n"
        f"Resources: visit {pid} (this is period_id {pid} — use it to look up rows in tracking__get_period_detail or the database); "
        f"key frames via images__get_image (boxes are the interpolated guesses for reference):\n"
        + "\n".join(lines)
        + f"\nAlso check the first real sighting after {_ts(end)} in visit {pid} — the box this guessed run connects to "
        f"(scan forward from gap end)."
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
    pid = esc.period_id
    votes = m.get("votes") or {}
    if votes:
        ranked = sorted(votes, key=lambda c: -float(votes[c].get("score", 0)))
        table = "; ".join(f"cat {c}: {_f(votes[c].get('score'), '.1f')} pts ({int(votes[c].get('frames', 0))} frames)" for c in ranked)
        runner_id = m.get("runner_up_cat_id", ranked[1] if len(ranked) > 1 else "?")
    else:
        scores = m.get("scores") or []
        table = f"[{', '.join(_f(s, '.1f') for s in scores)}] per-cat scores"
        runner_id = m.get("runner_up_cat_id", "?")
    key_frames = [f for f in (m.get("key_frames") or []) if isinstance(f, dict) and f.get("t")]
    frames = _labeled_key_frames(key_frames) if key_frames else _period_frames(esc, "which cat is visible? linger on missing/low-certainty stretches")
    return (
        f"{_head(esc, 'Two cats nearly tied for this visit')}\n"
        f"Conflict: visit {pid} is a close identity vote — {table} (margin {_f(m.get('margin'), '.1f')} pts).\n"
        f"Validate: watching the frames, is this cat {esc.cat_id} throughout, cat {runner_id}, or different cats at different times?\n"
        f"Context: {_SCORE_EXPLAIN} Proposal awards the whole visit to cat {esc.cat_id}, but the winner may just have one lucky "
        f"high-certainty frame. {MODEL_CAVEAT}\n"
        f"Alternatives: award cat {runner_id}, or split the visit if two cats appear at different times.\n"
        f"Resources: visit {pid} (this is period_id {pid} — look it up in tracking__get_period_detail or the database); key frames "
        f"via images__get_image (boxes are the tracked positions for reference):\n"
        f"{frames}"
    )


def _labeled_key_frames(key_frames: list) -> str:
    lines = []
    for i, f in enumerate(key_frames):
        label = "visit start" if i == 0 else ("visit end" if i == len(key_frames) - 1 else "mid-visit check")
        lines.append(_key_frame_line(f["t"], label, f.get("box"), f.get("real")))
    return "\n".join(_dedupe(lines))


def build_p2_prompt(esc: Escalation) -> str:
    m = esc.metrics
    pid = esc.period_id
    n = m.get("detect_length")
    if isinstance(n, (int, float)) and n < 10:
        penalty = f"Here {int(n)} frames triggered the x0.2 penalty."
    elif isinstance(n, (int, float)):
        penalty = "No x0.2 penalty (10+ frames) — the raw vote itself was weak."
    else:
        penalty = ""
    key_frames = [f for f in (m.get("key_frames") or []) if isinstance(f, dict) and f.get("t")]
    frames = _labeled_key_frames(key_frames) if key_frames else _period_frames(esc, "is a cat really there? evidence is thin")
    return (
        f"{_head(esc, 'Visit awarded on thin evidence')}\n"
        f"Conflict: visit {pid} -> cat {esc.cat_id} on {_f(m.get('score'), '.1f')} pts from {n} original "
        f"frames (frames actually seen, not guessed).\n"
        f"Validate: rule in this order.\n"
        f"1) IDENTITY — which cat is in these few frames? Compare against BOTH cat profiles below and QUOTE the cues "
        f"that match one cat and rule the other out; if the pose hides the cues, fetch both cats' reference images "
        f"with local__find_references + local__get_reference_image and compare side-by-side. Thin evidence makes "
        f"identity MORE uncertain, not less — never confirm a label you have not appearance-checked. "
        f"Judge the cat inside the visit's box (fetch CROPPED frames via local__get_frame with box=...) — a second "
        f"cat is often co-present in the same frames.\n"
        f"2) EXISTENCE — real appearance (confirm), noise that should never have become a visit (drop), "
        f"or another cat's visit (reassign)?\n"
        f"Context: {_SCORE_EXPLAIN} {penalty} Proposal keeps it "
        f"as cat {esc.cat_id}. Short low-score visits are often stray boxes. {MODEL_CAVEAT}\n"
        f"Alternatives: drop as noise, or reassign to another cat.\n"
        f"Resources: visit {pid} (this is period_id {pid} — look it up in tracking__get_period_detail or the database); key frames "
        f"via images__get_image (boxes are the tracked positions for reference):\n"
        f"{frames}"
    )


def build_p3_prompt(esc: Escalation) -> str:
    m = esc.metrics
    pid, wanted = esc.period_id, m.get("wanted_cat_id")
    taker, span = m.get("taken_by_period_id"), m.get("taken_by_span") or []
    if taker is not None:
        taker_txt = f"visit {taker}" + (f" ({_ts(span[0])} to {_ts(span[1])})" if len(span) == 2 else "")
    else:
        taker_txt = "another simultaneous visit"
    key_frames = [f for f in (m.get("key_frames") or []) if isinstance(f, dict) and f.get("t")]
    frames = _labeled_key_frames(key_frames) if key_frames else _period_frames(esc, "identify the cat; compare the overlapping visit at the same times")
    return (
        f"{_head(esc, 'Visit got second-choice cat; first choice was taken')}\n"
        f"Conflict: visit {pid} ({_ts(esc.time)} to {_ts(esc.end_time)}) voted cat {wanted}, but {taker_txt} already claimed "
        f"that cat — one cat cannot be in two places at once, and the higher score claims first — so this visit got cat {esc.cat_id}.\n"
        f"Validate: watching both visits' frames, is visit {pid} truly cat {esc.cat_id} (confirm), another profile cat such as "
        f"cat {wanted} (reassign), or not a cat at all (drop)?\n"
        f"Context: proposal assigns cat {esc.cat_id}. The swap was a scheduling fix, not new identity evidence — either visit could be "
        f"the wrongly labeled one. Every visit is either one of the profile cats or not a cat. {MODEL_CAVEAT}\n"
        f"Alternatives: confirm cat {esc.cat_id}, reassign to the cat actually visible, or drop as not-a-cat.\n"
        f"Resources: visit {pid} (this is period_id {pid} — look it up in tracking__get_period_detail or the database)"
        + (f" plus visit {taker} (period_id {taker})" if taker is not None else " plus the overlapping visits")
        + "; key frames via images__get_image (boxes are the tracked positions for reference):\n"
        f"{frames}"
    )


def build_p4_prompt(esc: Escalation) -> str:
    m = esc.metrics
    pid = esc.period_id
    counts = m.get("detect_counts") or {}
    total = m.get("rows")
    if counts and total:
        seen = sum(int(v) for v in counts.values())
        parts = ", ".join(f"cat {c} x{int(counts[c])}" for c in sorted(counts, key=lambda c: -int(counts[c])))
        evidence = f"of {total} frames, the camera labeled {seen} ({parts}) and nothing in the other {total - seen}"
    else:
        evidence = "too few coherent sightings to score any cat"
    key_frames = [f for f in (m.get("key_frames") or []) if isinstance(f, dict) and f.get("t")]
    frames = _labeled_key_frames(key_frames) if key_frames else _period_frames(esc, "identify the cat; focus where voting went blind")
    return (
        f"{_head(esc, 'Long visit nobody could label')}\n"
        f"Task: identify visit {pid} ({_ts(esc.time)} to {_ts(esc.end_time)}, {_f(m.get('seconds'), '.0f')} seconds / {total} frames) — "
        f"it scored below every cat's threshold, so it got no identity.\n"
        f"Validate: watching the frames, which cat (if any) is this — and why did voting fail?\n"
        f"Context: visits win a cat only by vote points ({_SCORE_EXPLAIN}); here {evidence}. A long visit usually IS a cat — "
        f"occlusion, bad boxes, or a missing profile may have hidden it. There is no proposal; it was left unknown. {MODEL_CAVEAT}\n"
        f"Alternatives: assign the best-guess cat, split into labeled fragments, or confirm unidentifiable.\n"
        f"Resources: visit {pid} (this is period_id {pid} — look it up in tracking__get_period_detail or the database); key frames "
        f"via images__get_image (boxes are the tracked positions for reference):\n"
        f"{frames}"
    )


def _key_frame_line(t, label, box, real) -> str:
    if box is not None:
        ref = f"Box {_box_txt(box)} ({'real sighting' if real else 'guessed, no sighting'})"
    else:
        ref = "no box committed (guessed frame)"
    return _frame(t, f"{label}. {ref}")


def build_p5_prompt(esc: Escalation) -> str:
    m = esc.metrics
    pid = esc.period_id
    total = m.get("total_frames")
    counts = m.get("detect_counts") or {}
    key_frames = [f for f in (m.get("key_frames") or []) if isinstance(f, dict) and f.get("t")]
    if total and counts:
        seen = sum(int(v) for v in counts.values())
        parts = ", ".join(f"cat {c}: {int(counts[c])}" for c in sorted(counts, key=lambda c: -int(counts[c])))
        evidence = (
            f"of {total} frames in total, the camera actually spotted a cat in {seen} ({parts}); "
            f"the other {total - seen} were filled in by guessing"
        )
    else:
        evidence = f"original-frame rate {_f(m.get('detect_rate'))} (share of frames actually seen; rest guessed)"
    winner_box = m.get("winner_box")
    location = f", typically around box {_box_txt(winner_box)} (median of its sightings)" if winner_box else ""
    if key_frames:
        frames = _labeled_key_frames(key_frames)
    else:
        frames = _period_frames(esc, "is the cat visible? weigh real sightings over guesses")
    return (
        f"{_head(esc, 'Verdict rests on weak, patchy sightings')}\n"
        f"Conflict: visit {pid} (cat {esc.cat_id}) runs {_ts(esc.time)} to {_ts(esc.end_time)} with avg certainty "
        f"{_f(m.get('detect_conf'))}; {evidence}.\n"
        f"Validate: rule in this order.\n"
        f"1) IDENTITY — which cat is actually in these frames, cat {esc.cat_id} or another? Fetch frames across the visit, "
        f"compare what you see against BOTH cat profiles below, and QUOTE the cues that match one cat and rule the other "
        f"out (fur length/outline, color tone, tail, face). Do this first: consistent sightings prove a real visit, not the "
        f"right label — never confirm a label you have not appearance-checked. If the pose hides the tail/face/fur cues "
        f"(curled or sleeping cat), fetch BOTH cats' reference images with local__find_references + "
        f"local__get_reference_image and compare side-by-side before ruling. "
        f"Judge the cat inside the visit's box (fetch CROPPED frames via local__get_frame with box=...) — a second "
        f"cat is often co-present in the same frames.\n"
        f"2) EXISTENCE — real continuous visit, or noise padded with guesses?\n"
        f"Context: proposal keeps cat {esc.cat_id}{location}. Mostly-guessed visits may not reflect reality. {MODEL_CAVEAT}\n"
        f"Alternatives: change the label to the right cat (say which), confirm the original detection (only when the identity "
        f"ruling favors cat {esc.cat_id}), or drop the labels if no cat is really there.\n"
        f"Resources: visit {pid} (this is period_id {pid} — one continuous tracked visit from {_ts(esc.time)} to "
        f"{_ts(esc.end_time)}; look it up in tracking__get_period_detail or the database); key frames via images__get_image "
        f"(boxes are the tracked positions for reference):\n"
        f"{frames}"
    )


def build_p6_prompt(esc: Escalation) -> str:
    m = esc.metrics
    prev, new = m.get("prev_period_id"), esc.period_id
    prev_start = m.get("prev_start")
    prev_box, new_box = m.get("prev_end_box"), m.get("new_start_box")
    if prev_start is not None:
        span = f"visit {prev} ran {_ts(prev_start)} to {_ts(esc.time)}; visit {new} starts at {_ts(esc.end_time)}"
    else:
        span = f"visit {prev} ended at {_ts(esc.time)} (its start is unknown); visit {new} starts at {_ts(esc.end_time)}"
    prev_ref = f" Box {_box_txt(prev_box)}" if prev_box else ""
    new_ref = f" Box {_box_txt(new_box)}" if new_box else ""
    lines = _dedupe(
        [
            _frame(esc.time, f"visit {prev} ends; last confirmed position.{prev_ref}"),
            *(_frame(t, "gap, untracked — did the cat stay or leave?") for t in _minute_samples(esc.time, esc.end_time)),
            _frame(esc.end_time, f"visit {new} starts; first confirmed position.{new_ref}"),
        ]
    )
    return (
        f"{_head(esc, 'Two back-to-back visits, might be one')}\n"
        f"Conflict: {span} — a {_f(m.get('gap_seconds'), '.1f')}-second gap. The last box of visit {prev} and the first box of "
        f"visit {new} are {_f(m.get('distance_px'), '.0f')} px apart (center to center).\n"
        f"Validate: merge visits {prev} and {new} into one cat {esc.cat_id} visit — or are they different cats?\n"
        f"Context: proposal keeps separate. Short gap + nearby boxes suggests continuity, but the cat may have left and returned, "
        f"or the new visit may be a different cat wearing the wrong label. {MODEL_CAVEAT} {_BOX_FOCUS}\n"
        f"Alternatives: merge, confirm separate visits of the same cat, or different cats (say which cat the new visit is).\n"
        f"If you answer different cats: name the new visit's cat from its APPEARANCE in visit {new}'s own frames — compare "
        f"against BOTH cat profiles below and QUOTE the cues (fur length/outline, color tone, tail, face). Track bookkeeping "
        f"(\"another visit already has this cat\") decides that the visits differ, never which cat is in the box. The camera "
        f"labeled visit {new} cat {esc.cat_id}; keep that label unless the appearance rules it out.\n"
        f"Resources: visits {prev}, {new} (these are period_id {prev} and {new} — look them up in tracking__get_period_detail or "
        f"the database); gap frames ~1/min via images__get_image (boxes are the tracked positions for reference):\n"
        + "\n".join(lines)
    )


def build_p7_prompt(esc: Escalation) -> str:
    m = esc.metrics
    prev, new = m.get("prev_period_id"), esc.period_id
    prev_start = m.get("prev_start")
    prev_box, new_box = m.get("prev_end_box"), m.get("new_start_box")
    if prev_start is not None:
        span = f"visit {prev} ran {_ts(prev_start)} to {_ts(esc.time)}; visit {new} starts at {_ts(esc.end_time)}"
    else:
        span = f"visit {prev} ended at {_ts(esc.time)} (its start is unknown); visit {new} starts at {_ts(esc.end_time)}"
    prev_ref = f" Box {_box_txt(prev_box)}" if prev_box else ""
    new_ref = f" Box {_box_txt(new_box)}" if new_box else ""
    lines = _dedupe(
        [
            _frame(esc.time, f"visit {prev} ends, cat still — nap's last frame.{prev_ref}"),
            *(_frame(t, "quiet gap, untracked — cat still lying there?") for t in _minute_samples(esc.time, esc.end_time)),
            _frame(esc.end_time, f"visit {new} starts, cat still — nap's first frame.{new_ref}"),
        ]
    )
    return (
        f"{_head(esc, 'One nap split into two visits')}\n"
        f"Conflict: {span} — a {_f(m.get('gap_seconds'), '.0f')}-second gap. The nap's last box and first box overlap "
        f"{_f(m.get('overlap_iou'))} (1.0 means identical), and both ends barely move: box centers wandered only "
        f"{_f(m.get('prev_displacement_px'), '.0f')} px over {m.get('prev_trail_frames')} frames before the gap and "
        f"{_f(m.get('new_displacement_px'), '.0f')} px over {m.get('new_head_frames')} frames after — a stillness check that "
        f"flags a sleeping cat.\n"
        f"Validate: merge visits {prev} and {new} into one continuous cat {esc.cat_id} presence?\n"
        f"Context: proposal keeps separate. One long nap and two separate naps in the same spot look identical to the tracker. "
        f"{MODEL_CAVEAT} {_BOX_FOCUS}\n"
        f"Alternatives: merge, or confirm the cat left and came back to the same spot.\n"
        f"Resources: visits {prev}, {new} (these are period_id {prev} and {new} — look them up in tracking__get_period_detail or "
        f"the database); gap frames ~1/min via images__get_image (boxes are the tracked positions for reference):\n"
        + "\n".join(lines)
    )


def build_p8_prompt(esc: Escalation) -> str:
    m = esc.metrics
    pid = esc.period_id
    tracked, voted = m.get("tracked_cat_id"), m.get("voted_cat_id")
    votes = m.get("votes") or {}
    if votes:
        ranked = sorted(votes, key=lambda c: -float(votes[c].get("score", 0)))
        table = "; ".join(f"cat {c}: {_f(votes[c].get('score'), '.1f')} pts ({int(votes[c].get('frames', 0))} frames)" for c in ranked)
    else:
        table = "no vote breakdown recorded"
    relabeled = m.get("relabeled_frames")
    if isinstance(relabeled, (int, float)) and relabeled:
        steal = f"Continuity moved {int(relabeled)} frames: the camera labeled them cat {voted} but the tracker filed them under cat {tracked}. "
    else:
        steal = ""
    key_frames = [f for f in (m.get("key_frames") or []) if isinstance(f, dict) and f.get("t")]
    frames = _labeled_key_frames(key_frames) if key_frames else _period_frames(esc, "which cat is visible? check start, middle and end")
    tail_from = m.get("tail_from")
    if tail_from:
        conflict = (
            f"Conflict: visit {pid} is labeled cat {tracked} and the whole-visit vote agrees ({table}), but from "
            f"{str(tail_from)[11:19]} the camera's last {int(m.get('tail_frames') or 0)} sightings name cat {voted}: the "
            f"track may have carried on after the cats swapped. Most likely verdict if the crops confirm a swap: "
            f"split_mixed with chosen_cat_id={voted} (find the true handover — it can come BEFORE {str(tail_from)[11:19]}, "
            f"because the camera mislabels too); keep_tracked if the tail is still cat {tracked}. relabel_to_vote would "
            f"relabel the WHOLE visit to cat {voted} — only if the start is cat {voted} too.\n"
        )
    else:
        conflict = (
            f"Conflict: visit {pid} is labeled cat {tracked} frame by frame, but the whole-visit vote awards it to cat {voted} "
            f"({table}).\n"
        )
    return (
        f"{_head(esc, 'Tracker label disagrees with the period vote')}\n"
        f"{conflict}"
        f"Validate: rule in this order.\n"
        f"1) BOX IDENTITY — at spread samples (start, middle, end at minimum), which cat is INSIDE the tracked box? "
        f"Fetch each sample CROPPED to its listed box via local__get_frame(timestamp=..., box=[x1, y1, x2, y2]) and QUOTE "
        f"the cues that match one cat and rule the other out. Full-frame views do NOT count — the other cat is often "
        f"co-present and more salient than the boxed one.\n"
        f"2) VERDICT — box evidence unanimous for the voted cat {voted} at every sample: relabel_to_vote. Unanimous for "
        f"the tracked cat {tracked}: keep_tracked. Different cats at different times with a clean handover: split_mixed.\n"
        f"Context: the tracker assumes smooth motion, so one early label can drag a whole visit along even when most frames "
        f"disagree; the vote recounts the camera's original per-frame labels instead ({_SCORE_EXPLAIN}). {steal}But the camera "
        f"mislabels cats routinely, so the vote is a lead, not evidence: the proposal to relabel the visit to cat {voted} "
        f"stands or falls on box-cropped image evidence, never on the point totals. {MODEL_CAVEAT} {_BOX_FOCUS}\n"
        f"Alternatives: keep the tracker's cat {tracked}, relabel the whole visit to cat {voted}, or split the mixed visit "
        f"at the frame where the cat changes (say when in cut_time and which cat resumes after in chosen_cat_id). "
        f"relabel_to_vote requires unanimous box-cropped evidence for cat {voted} at 3+ spread samples — never relabel a "
        f"whole visit from full-frame views. If you "
        f"split: first BOUND the handover — the cut must fall AFTER the latest timestamp the first cat is placed "
        f"(your crops AND the 'Already decided' priors) and BEFORE the earliest timestamp placing the second cat. "
        f"State that window in your rationale (e.g. 'first cat last at 16:35:21 per E10, second cat first at 16:35:25 "
        f"per my crop → cut inside (16:35:21, 16:35:25]'). Then narrow it: fetch crops BETWEEN your bracketing samples "
        f"until the gap is under ~10 s — never place a cut across an unsampled gap wider than that. Place cut_time at "
        f"the LAST frame where the first cat is still stably visible — not the first ambiguous "
        f"frame — then VERIFY by fetching one frame just before and one just after your cut and confirming each side shows "
        f"the expected cat. If the track oscillates between the cats with no clean handover, do not split: pick keep or relabel. "
        f"When the pose hides the cues, fetch both cats' reference images (local__find_references + "
        f"local__get_reference_image) and compare side-by-side.\n"
        f"Resources: visit {pid} (this is period_id {pid} — look it up in tracking__get_period_detail or the database); key frames "
        f"via images__get_image (boxes are the tracked positions for reference). Crop ONLY with the listed (time, box) "
        f"pairs — for bisection pick the listed times nearest your target and say which. For a time with no listed box, "
        f"fetch the exact tracked box first via local__get_track_box(period_id={pid}, timestamp=...) and crop THAT; "
        f"never invent boxes or reuse one sample's box at another time:\n"
        f"{frames}"
    )


def build_h1_prompt(esc: Escalation) -> str:
    m = esc.metrics
    quiet = _f(m.get("quiet_seconds"), ".0f")
    msgs = m.get("messages")
    msgs_txt = f" ({msgs} frames ingested)" if isinstance(msgs, (int, float)) else ""
    if (m.get("kind") or "") == "end_of_run":
        title = "Quiet run spot-check: nothing was flagged"
        conflict = (
            f"the whole run stayed quiet — no escalation from {_ts(esc.time)} to {_ts(esc.end_time)} "
            f"({quiet} seconds{msgs_txt})"
        )
    else:
        title = "Quiet window spot-check: nothing flagged for a while"
        conflict = (
            f"no escalation from {_ts(esc.time)} to {_ts(esc.end_time)} "
            f"({quiet} seconds of confident tracking{msgs_txt})"
        )
    cats = m.get("cat_ids") or []
    cats_txt = f" View cats under review: {', '.join(f'cat {c}' for c in cats)}." if cats else ""
    frames = _period_frames(esc, "spot-check: right cat, right boxes, nothing missed?")
    return (
        f"{_head(esc, title)}\n"
        f"Conflict: {conflict} — this is a routine audit, not a suspected error.\n"
        f"Validate: watching the frames, is the tracking right across the window — identities, boxes, presence?\n"
        f"Context: proposal keeps everything as-is.{cats_txt} The tracker saw nothing uncertain, so there is no "
        f"specific suspect; a confident stretch can still hide a steady mislabel or a missed cat. {MODEL_CAVEAT}\n"
        f"Alternatives: confirm, or flag the issue in words (rows stay as-is for a follow-up).\n"
        f"Resources: key frames ~1/min via images__get_image (boxes are the tracked positions for reference):\n"
        f"{frames}"
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
    "P8": build_p8_prompt,
    "H1": build_h1_prompt,
}


@lru_cache(maxsize=1)
def _load_profiles_cached() -> dict:
    """All cat profiles keyed by every known id (canonical + per-view tracking ids)."""
    try:
        from agent.tracking.cat_config import load_cat_profiles

        cats = load_cat_profiles()
    except Exception:
        return {}
    out = {}
    for cat in cats or []:
        try:
            out[int(cat.get("cat_id"))] = cat
            for tid in (cat.get("tracking_ids") or {}).values():
                out[int(tid)] = cat
        except (TypeError, ValueError):
            continue
    return out


def cat_profile_block(cat_ids, profiles=None) -> str:
    """'Cats:' appearance block for the given ids (None = all known, capped); '' when none resolve."""
    src = profiles if profiles is not None else _load_profiles_cached()
    if cat_ids is None:
        ids = sorted(src)[:6]
    else:
        ids = []
        for c in cat_ids:
            try:
                ids.append(int(c))
            except (TypeError, ValueError):
                continue
        ids = list(dict.fromkeys(ids))
    lines = []
    for cid in ids:
        cat = src.get(cid)
        if cat is None:
            continue
        cues = "; ".join((cat.get("low_res_cues") or [])[:4])
        text = f"- cat {cid} ({cat.get('name', '?')}): {cat.get('summary', 'no description')}"
        if cues:
            text += f" Look for: {cues}."
        lines.append(text)
    return "Cats:\n" + "\n".join(lines) if lines else ""


def _involved_cats(code: str, esc: Escalation) -> list | None:
    """Cat ids the reviewer must tell apart (None = all known profiles)."""
    m = esc.metrics or {}
    if code == "T1":
        refs = m.get("candidate_refs") or {}
        cands = {str(c) for c in list(refs) + list(m.get("candidates") or {})}
        chosen = str(m.get("chosen_cat_id"))
        return [chosen] + sorted(c for c in cands if c != chosen)
    if code == "T2":
        return [m.get("original_cat_id"), esc.cat_id]
    if code == "T3":
        return [esc.cat_id, m.get("killed_cat_id")]
    if code in ("T4", "T5", "T6", "T7"):
        return [esc.cat_id]
    if code == "P1":
        votes = m.get("votes") or {}
        if votes:
            return sorted(votes, key=lambda c: -float(votes[c].get("score", 0)))
        return [esc.cat_id, m.get("runner_up_cat_id")]
    if code == "P2":
        return None  # reassign can name any cat, so show every known profile
    if code == "P3":
        return [esc.cat_id, m.get("wanted_cat_id")]
    if code == "P4":
        return None
    if code == "P5":
        runner_up = m.get("runner_up_cat_id")
        if runner_up is not None:
            return [esc.cat_id, runner_up]
        counts = m.get("detect_counts") or {}
        rest = sorted((c for c in counts if str(c) != str(esc.cat_id)), key=lambda c: -int(counts[c]))
        return [esc.cat_id] + rest
    if code == "P6":
        return None  # the new visit may be any cat; the reviewer needs every profile
    if code == "P7":
        return [esc.cat_id]
    if code == "P8":
        return None  # split_mixed can name any cat for the second half; the reviewer needs every profile
    if code == "H1":
        cats = (m.get("cat_ids") or []) if isinstance(m, dict) else []
        return list(cats) if cats else None
    return [esc.cat_id]


def build_prompt(esc: Escalation, profiles=None) -> str:
    """Dispatch to the per-code builder; unknown codes get a generic prompt.

    A 'Cats:' appearance block for the involved cats is inserted before
    Resources (``profiles`` overrides the cats.yaml lookup; {} disables it).
    """
    builder = _BUILDERS.get(esc.code)
    if builder is None:
        prompt = (
            f"{_head(esc, 'Unrecognized escalation code')}\n"
            f"Conflict: {esc.reason}\n"
            f"Validate: is the tracker's decision at {_ts(esc.time)} correct?\n"
            f"Resources (images__get_image):\n"
            f"{_frame(esc.time, 'the decision frame')}"
        )
        cats = [esc.cat_id]
    else:
        prompt = builder(esc)
        cats = _involved_cats(esc.code, esc)
    block = cat_profile_block(cats, profiles)
    if block and "\nResources" in prompt:
        return prompt.replace("\nResources", f"\n{block}\nResources", 1)
    return prompt + (f"\n{block}" if block else "")
