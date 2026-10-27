# How the Track Corrector and Period Corrector Work

YOLO gives us one independent guess per image: "cat 1 is here, conf 0.62". Those guesses
flicker (missed frames), jump (one bad box), and swap labels (cat 1 called cat 0 for a
few frames). Two stages clean this up:

```
YOLO per image ──► TrackCorrector ──► revised detections ──► PeriodCorrector ──► periods
 {cat_id: box}     "which boxes are       (with period_id)     "which cat is this     (start, end,
                    the same animal?"                           track, really?"        cat_id, score)
```

- **[TrackCorrector](../existing_work_before_hackathon/model/track_corrector.py)** works frame by frame with a short delay (15 s).
  It fixes boxes and labels locally and groups consecutive boxes into **periods**: one
  continuous stretch of one animal on screen.
- **[PeriodCorrector](../existing_work_before_hackathon/model/period_corrector.py)** waits until a group of periods has ended,
  then votes on the whole period to decide which cat it was. It also enforces that one
  cat can't be in two overlapping periods.

Both are created per camera view (`TrackCorrector(view_id)`). The view's `cat_ids` (from
the `view` table) fix which cats exist. Internally each cat is addressed by its index
(`cat_idx`) into that list. Live, both run inside [detection_ingest_server.py](../existing_work_before_hackathon/detection_ingest_server.py) on every image.

---

## 1. TrackCorrector

### Data it keeps per cat

| State | Meaning |
| --- | --- |
| `buffer[cat_idx]` | Deque of the last ~15 s of rows for this cat, one per image: `[time, x1, y1, x2, y2, original_cat_id, conf, cat_id]`. Empty frames have `None` boxes. Rows here can still be rewritten. |
| `base` | The last row already committed (left the buffer with a box). The anchor in the past. |
| `target` | The row in the buffer the tracker trusts most. The anchor in the future. |
| `period_id` | ID of the current continuous period. A new one is allocated from the `period` table. |
| `detecting` | Whether a period is currently open. |

### What happens on each image (`process_images`)

Input is `{cat_id: [x1, y1, x2, y2, original_cat_id, conf]}` for detections above 0.5
confidence (the filter happens in the caller).

**Step 1: clean up labels (`add_new_detection_to_buffer`)**

1. **Duplicate boxes.** If two labels' boxes overlap with IoU > 0.7, it's one animal
   labeled twice, so keep the more confident one (`remove_multi_labeling`).
2. **Keep labels that agree with history.** A detection whose box overlaps (IoU > 0.5) any
   box already in *its own* cat's buffer is "reasonable" and keeps its label.
3. **Relabel the rest.** For any other detection, check whether it fits another cat's recent
   track (`match_buffer_history`). It fits if any of these hold:
   - IoU > 0.2 with that cat's last detection (or `base`)
   - that cat wasn't detected in this image and the box center is within 160 px (¼ width)
   - IoU > 0.8 with any box in that cat's buffer

   If it fits, the label is swapped. This is what fixes "cat 1 called cat 0 for a few frames".
4. **Jump reset.** If a cat had no live detection, still has a `base`, and the new box
   barely overlaps that base (IoU < 0.2), the old track is treated as stale. The buffer is
   wiped and the period is closed.
5. Append one row per cat to its buffer: the box if detected, otherwise an empty row.

**Step 2: pick a target (`select_target`)**

The target is the buffer row to interpolate toward. The first rule that matches wins:
1. the first box with IoU > 0.7 against `base` (continuing the same position)
2. the first box with conf > 0.5, or any box that overlaps `base`
3. otherwise the highest-confidence box

**Step 3: drop noise**

If neither the buffer nor the current period has ever reached conf 0.4, all of the
buffer's boxes are erased. A few weak boxes in a row don't start a period.

**Step 4: smooth and fill gaps (`interplate_buffer`)**

Every buffer row between `base` and `target` is rewritten by linear interpolation in time
from base box to target box. Effects:
- **Missed frames get filled**: empty rows between two detections get an interpolated box,
  with `original_cat_id = None` so you can still tell they were synthesized.
- **Jitter is smoothed.** If base and target differ a lot (IoU < 0.7), real boxes are blended
  with the interpolation, weighted by their confidence, so strong detections keep more of
  their own position.

**Step 5: commit old rows (`collect_buffer`)**

Rows older than `min(now − 15 s, target time)` are final:
- Each committed row with a box becomes the new `base` and gets the current `period_id`.
  A **new period** starts if none was open, or if the box is > 300 px from the last
  committed confident box. That's a teleport, so it's probably a different animal. On a
  teleport, trailing interpolated rows of the old period are retracted (yielded as `(False, row)`).
- Committed rows are popped from the buffer and yielded as `(True, row)`:
  - with a box: 9 fields, `[time, x1, y1, x2, y2, original_cat_id, conf, cat_id, period_id]`
  - empty: 8 fields (no `period_id`). An empty row coming out **closes** the period.
- If nothing has been detected for 60 s (`max_undetected_gap`), rows are flushed even
  without a base and the period is closed.

With `save_db=True`, committed rows also go to the `revised_detection` table (batched).

**Day boundary.** When the date changes, `finish_up()` flushes pending DB writes and resets
all state.

### Key parameters

| Parameter | Value | Effect |
| --- | --- | --- |
| `buffer_range` | 15 s | Delay before a row is final; how far ahead the tracker can "look" |
| `max_undetected_gap` | 60 s | Gap after which a period is closed |
| `min_conf` | 0.4 | Below this, a buffer is treated as noise |
| teleport distance | 300 px | Starts a new period |
| relabel distance | 160 px | Max center distance for a label swap when the cat wasn't seen |

---

## 2. PeriodCorrector

The tracker assigns each period to the label it was tracked under. That can be wrong for
the whole period, for example if YOLO mostly called cat 1 "cat 0". The PeriodCorrector
re-decides each period's cat by voting over the whole period.

### Grouping periods (`process_detection`)

It consumes the tracker's output stream:
- **9-field row**: a detection in period `P`, tracked as cat `C`. If `P` is new for `C`, the
  previous period of `C` is closed and `P` is opened.
- **8-field row (empty)**: closes `C`'s open period.
- `unprocess_detection` handles the tracker's retractions.

Periods that overlap in time form a **group**. When **no period is open anymore** (every
cat has left the screen), the group is complete and gets coordinated. The same happens at
the end of the day (`finish_up`).

### Scoring a period (`get_period_stats` + `get_stats_score`)

For each candidate cat, compute stats from the **original YOLO labels** inside the period
(not the tracker's relabels):

| Stat | Meaning |
| --- | --- |
| `detect_length` | number of frames YOLO labeled as that cat |
| `detect_rate` | `detect_length / period length` |
| `detect_conf` | mean YOLO confidence for that label |
| `period_area_count` | number of 50 × 50 px grid cells where that label's mean conf > 0.5. Rewards a label that holds up across positions, not just one spot. |

Score per cat:

| Condition | Points |
| --- | --- |
| highest `detect_conf` among cats | +3 |
| highest `detect_length` | +2 |
| `detect_length` > 20 | +1 |
| `detect_rate` > 0.5 | +1 |
| `detect_rate` > 0.3 | +1 |
| `detect_conf` > 0.5 | +1 |
| highest `period_area_count` | +4.2 |
| `detect_length` < 10 | whole score × 0.2 |

### Assigning cats (`coordinate_period_groups`)

The assignment is greedy, best score first:
1. Put every period in a max-heap by its best score.
2. Pop the best. If its top cat is free during that time range, assign it.
3. If that cat already has an **overlapping** period, zero out that cat's score for this
   period and push it back to try its next-best cat.
4. If a period's best remaining score is < 1, it gets **no cat** (`None`). It's treated as
   too weak to call.

Results are written to the `period` row that the tracker allocated: start, end,
`coordinated_cat_id`, the chosen cat's stats and score, plus all cats' stats for debugging.

---

## Reading the visualization with this in mind

In each frame from [run_tracking_viz.py](../existing_work_before_hackathon/run_tracking_viz.py) (see [TRACKING_VIZ.md](TRACKING_VIZ.md)):

- **Shaded area**: the 15 s buffer. Everything inside can still change.
- **Green thick line**: `base`. **Purple thick line**: `target`. Boxes between them are
  being interpolated.
- **Cat-colored bars left of the shaded area**: committed revised detections.
- **Blue bars to the right**: raw YOLO detections still to come. Comparing them with
  the committed bars shows what the tracker filled in, smoothed, or relabeled.
- `Period: [...]` lines printed by the script are PeriodCorrector output. They appear only
  when a whole group of overlapping periods has ended.

## Known quirks

- `remove_revised_detection_from_db` loops with `for k, q in self.bulk_revised_detection_insert`,
  which unpacks each 10-field row into two names. If a teleport retraction happens while
  rows are still batched, this raises. Also, the actual `DELETE` is commented out, so
  retracted rows already flushed to the DB stay there.
- `PeriodCorrector.detections` only resets at day end, and every group coordination scans
  it. It gets slower over the day.
- Even with `save_db=False`, both classes still write to the `period` table: they allocate
  IDs and update period rows. `run_tracking_viz.py` stubs those writes out.
