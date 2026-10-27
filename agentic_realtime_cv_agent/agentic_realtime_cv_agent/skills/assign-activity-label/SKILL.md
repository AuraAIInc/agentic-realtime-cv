---
name: assign-activity-label
description: Give one cat crop exactly one activity label and any matching context labels from labels.yaml, or propose a new label when nothing fits. Use for assign_label tasks.
---

# Assign activity and context labels

The input holds:

- `labels`: the current label set (activity: pick exactly one; context: any number).
- `rules`: what code already decided without images: motion state at that time
  (moving / resting), zone overlaps, whether another cat is close, which activities are
  still possible (`candidate_activities`), and context labels already decided by rule.
- `crop`: the cat, cropped with some context around it.

## Steps

1. Choose only from `rules.candidate_activities`. Code removed the ones motion and zones rule
   out (a moving cat is not sleeping; eating needs the food-bowl zone).
2. Moving vs playing cannot be told from one still. If both are candidates and
   `rules.extra_frames` is given, call `local__make_contact_sheet` once with those frames.
3. Context labels: keep the ones in `rules.context_decided`; add a context label that needs
   an image only if you can see it (e.g. inside the tent).
4. Cite what you see for each label (posture, head position, eyes, paws, toy).

## When nothing fits

Pick `unknown` and fill `suggestion` with a short lower_snake_case name, one-line
description and the nearest existing label (e.g. `running`, nearest `walking`). Before
inventing a name, reuse an obviously equivalent one (zoomies = running). The runner files
the crop under `label_suggestions/<name>/` for a human to review. Do not add labels
yourself.

## Answer

Return only the JSON object the schema asks for. `action`: `labelled`, `suggested_new`
(activity unknown, suggestion filled), or `send_to_human` (cannot tell from the evidence).
