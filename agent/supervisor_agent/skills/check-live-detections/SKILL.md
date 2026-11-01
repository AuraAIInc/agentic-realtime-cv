---
name: check-live-detections
description: Answer "what is on camera now, and is the model right" in about three turns: one annotated frame, one round of zoomed crops, then a verdict table. Use for live camera questions in chat.
surface: chat
---

# Check the live camera against its detections

1. `camera__get_status` (note `sim`, `capture_size`, the loop model). If `sim` is
   true, the frames are a replay: do not cross-reference them with `tracking__`
   live data, which comes from a different moment.
2. `camera__capture_frame(draw_detections=true, max_side=800)`: one frame with
   the loop's boxes for that same instant. Compare each box with what you see.
3. For boxes that look wrong, and for any visible cat with no box: fetch what
   settles it in ONE turn — `camera__get_track_frame` or `images__get_image`
   crops of the disputed boxes, or one `local__make_contact_sheet` for several
   — plus `local__get_cat_profiles(view_id)` for the distinguishing cues.
4. Answer as a small table: box → what is actually there → model label →
   verdict (true positive / false positive / missed cat / wrong id).

## Rules

- Two simultaneous boxes with the same cat_id means at least one label is wrong.
- A box that stays still for many minutes with high agreement may be an object
  (ball, toy, cushion), not a cat: look at it before trusting the votes.
- Only the loop's cat model identifies cats. Generic models (COCO labels,
  `cat_id: -1`) can say "object vs cat" at best, never which cat.
- If a tool error says a feature is unavailable (crop, resize, evicted frame),
  do not retry the same arguments: work around it or say what is missing.
