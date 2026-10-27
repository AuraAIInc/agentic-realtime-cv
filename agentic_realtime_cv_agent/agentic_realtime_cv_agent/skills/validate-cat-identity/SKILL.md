---
name: validate-cat-identity
description: Decide whether a detection is a cat and which household cat it is, climbing a cheapest-evidence-first ladder. Use for validate_detection tasks from the CV pipeline.
---

# Validate a detection

You get one detection the CV pipeline is unsure about: a stored frame, a box, the detector's
cat candidates with confidences, and why it was flagged. The input already holds:

- `context`: what code found without images: the visit summary (frame count, detector
  agreement, the visit's cat), and other cats detected at the same moment (any view).
- `cats`: profiles of the candidate cats (summary, low-res cues, camera cues, how to tell apart).
- `crop`: one padded crop of the box (rung 2 below is already paid for).

Ids: the pipeline uses tracker ids (on view 3: Qb = 25, Qbo = 26). Answer with the
tracker id the pipeline used, and give the name.

## The ladder: stop at the first rung that gives a confident answer

1. **Metadata (in `context`, no images).**
   - Exclusion: a cat detected elsewhere at the same moment with high confidence cannot be
     this one. With two cats, ruling one out decides it, if the crop shows a cat at all.
   - Visit consensus: many frames, high detector agreement, no identity flip: this frame
     is noise; accept the visit's cat.
2. **The crop vs the written profiles.** Check `camera_cues` and `low_res_cues` of the
   candidates only. Name each cue you actually see. A verdict that cites no visible cue
   is not allowed.
   - Strongest quick cues: Qbo has a dark brown plume tail tip and a fuzzy outline;
     Qb is a smooth, compact, bright-white shape with no dark tail tip.
3. **Sharper view, only if the deciding cue is a high-res one or the box is under ~60 px.**
   Use `images__get_image` with the visit's `best_frame` (stored frames). Use
   `camera__capture_high_res(box)` only for a live frame; it pauses detection for ~3 s.
4. **Owner photos.** `local__find_references(cat_id, source="profile", pose, lighting)` for
   each remaining candidate, then `local__make_contact_sheet` with the query crop and one
   top reference per cat (one image, not three). Ask: which reference matches, on which features?
5. **Camera references.** If owner photos look too different from the overhead camera,
   `local__find_references(source="scene", view_id, box, lighting)`: same angle and
   resolution usually beats a studio photo.
6. **Human.** Still unsure: `action = send_to_human`. Never force a call.

## Rules

- Spend images only when the previous rung failed. You have a small image budget; if a tool
  says the budget is reached, decide from what you have or send to a human.
- Priors (where a cat usually sleeps) can raise or lower confidence, never override what
  the image shows.
- Night frames (purple or grey IR): colour cues are unreliable; rely on shape, outline
  softness and the dark tail tip.
- `is_cat=false` only when you can see the box holds no cat (cushion, bag, toy, blanket).
- `decided_at_step` = the rung number that settled it.

## Answer

Return only the JSON object the schema asks for. `action`:
`accept` (detector right), `relabel` (a cat, but a different one), `reject_not_cat`,
`send_to_human`.
