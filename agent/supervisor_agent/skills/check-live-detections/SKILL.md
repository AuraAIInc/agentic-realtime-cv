---
name: check-live-detections
description: Answer "what is on camera now, and is the model right" in about three turns: a wide frame, zoomed crops, and a verdict table grounded on the image and cat profiles. Use for live camera questions in chat.
surface: chat
---

# Check the live camera against its detections

The on-device detector is small and wrong often. Treat its boxes and labels as
untrusted hints about *where to look*, never as evidence of *what is there*.
Every verdict below must come from what you see in the raw image, checked
against the cat profiles.

1. `camera__get_status` (note `sim`, `capture_size`, the loop model). If `sim` is
   true, the frames are a replay: do not cross-reference them with `tracking__`
   live data, which comes from a different moment.
2. `camera__capture_frame(max_side=800)`: the raw frame, no drawings. List every
   cat-like shape you see yourself first, then compare with the loop's boxes:
   which boxes match something real, which real cats have no box.
3. For disputed boxes and unboxed cats: fetch what settles it in ONE turn —
   `camera__get_track_frame` or `images__get_image` crops, or one
   `local__make_contact_sheet` for several — plus `local__get_cat_profiles(view_id)`
   for the distinguishing cues. When cues stay ambiguous, compare against one
   reference image per candidate (`local__find_references` +
   `local__get_reference_image`).
4. Answer as a small table: box → what is actually there → model label →
   verdict (true positive / false positive / missed cat / wrong id).

## Rules

- Decide from the raw pixels and the profiles. A detector label (cat_id, name,
  confidence) is a claim to verify, not a vote: at low confidence, or whenever
  the visible cues point the other way, the verdict is wrong id / uncertain —
  never the label.
- Cite the cues you actually see for every identity verdict (coat tone and
  length, tail shape and tip, paws, head). A verdict that cites no visible cue
  is not allowed.
- Two simultaneous boxes with the same cat_id means at least one label is wrong.
- A box that stays still for many minutes with high agreement may be an object
  (ball, toy, cushion), not a cat: look at it before trusting the votes.
- Only the loop's cat model identifies cats. Generic models (COCO labels,
  `cat_id: -1`) can say "object vs cat" at best, never which cat.
- If a tool error says a feature is unavailable (crop, resize, evicted frame),
  do not retry the same arguments: work around it or say what is missing.
- A zoom that returns `zoom_factor: 1.0` added no detail: do not count it as
  verification.
- If a zoom comes back without the target in it, you mis-aimed: re-aim from the
  returned `roi_display` (e.g. target visible just above the crop's top edge →
  shift the box up by that much) instead of guessing a fresh box. After two
  misses in a row, stop zooming and say what is missing.
