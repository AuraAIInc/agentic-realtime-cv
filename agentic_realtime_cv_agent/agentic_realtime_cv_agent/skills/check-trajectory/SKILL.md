---
name: check-trajectory
description: Check that one visit (period) is a physically plausible path of one animal, looking only at the few frames around what the code flagged. Use for check_trajectory tasks.
---

# Check a visit's continuity

A real cat cannot vanish from one spot and appear in another. Code already ran the physics
(`analysis` in the input): it rebuilt the visit's own boxes from raw detections and flagged
what is implausible. Visits with no flags never reach you. Your job is to look at the flagged
moments and decide, per segment, whether it is the same animal and whether it is a cat.

## Read the flags first

| Flag | Usually means |
| --- | --- |
| teleport | two different objects merged into one track |
| impossible_entry / impossible_exit | false positive, or the start/end was missed |
| long_gap | occlusion (fine) or a swap (check both sides) |
| size_jump | wrong object or partial occlusion |
| identity_flip | the detector switched cats; a split point candidate |
| other_cat_close | swap risk at that moment |
| static | a sleeping cat or a decoy: needs a look, never auto-reject |

## Look once

`analysis.keyframes` are already chosen to test the doubts: before/after each jump or flip,
spread over space, best frame per motion segment. Call `local__make_contact_sheet` ONCE with
those keyframes (`image_name` + `box`, caption = role), in tile order. Ask yourself only:
are tiles 1..n the same animal, and which tiles (if any) are not a cat?

- If tiles change animal at a flag, split there (`suggest_split_at` = the first time of the
  new animal) and mark the new segment `needs_identity=true` unless you can tell which cat
  it is from `cats` cues with confidence.
- Identity against the cat profiles is a separate question; answer it only when the
  sheet makes it obvious (dark plume tail = Qbo, smooth white = Qb).
- Do not request more images unless the sheet is unreadable (budget is small).

## Answer

Return only the JSON object the schema asks for. Segments cover the whole visit in time
order. `action`: `accept` (continuous, same animal), `split` (two or more animals),
`relabel` (one animal but the visit's cat id is wrong), `reject_not_cat`, `send_to_human`.
Splits and relabels are only logged; they are not written to the tracking database.
