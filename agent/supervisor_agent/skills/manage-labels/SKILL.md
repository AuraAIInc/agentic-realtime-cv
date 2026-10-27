---
name: manage-labels
description: Turn a user's plain-language request ("label the cat when it sits in the tent") into a new label, and a new zone if it is place-based, confirming with the user before saving. Use for label commands.
---

# Add a label by command

1. `local__get_label_set` to see what exists. If an existing label already means the same
   thing, say so and stop.
2. Draft the entry:
   - name: lower_snake_case (`in_tent`, `on_sofa`)
   - kind: `activity` (what the cat does; exactly one per crop) or `context` (where / with
     what; any number). Places are almost always context.
   - description: one line a person would agree with.
   - needs / rule: place-based → `rule: {zone: <zone>, min_overlap: 0.5}` (0.6 for small
     enclosures like a tent) and add `needs: [image]` if being inside vs beside can only be
     told by looking.
3. Place-based and the zone is missing from `local__get_zones(view_id)` (view 3 is the live
   living-room camera unless the user says otherwise):
   - `camera__capture_frame(max_side=800)` to see the room, propose a box in the view's
     pixels for the place, and call `local__save_zone(confirm=false)`.
   - Show the user the box and ask them to confirm or correct it. Save with `confirm=true`
     only after they agree.
4. Call `local__add_label(confirm=false)`, show the draft (and any problems or similar
   labels it reports), and ask the user to confirm. Save with `confirm=true` only after a yes.
5. After saving, say the new label-set version and that labelling picks it up on the next
   task. Offer a backfill over recent visits with an image budget stated up front; do not
   start it unasked.

Never save without an explicit yes from the user in this conversation.
