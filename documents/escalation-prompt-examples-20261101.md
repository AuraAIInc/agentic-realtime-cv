# Escalation prompt examples

One real example per escalation code, taken from tracking run `view3_20261028-1633_1640_replay_db_20261101-232410-056120`
(view 3, window 2026-10-28T16:33 .. 2026-10-28T16:40, 12 escalations). Each section shows the original escalation record, the full reviewer prompt generated for it, and the expected JSON output schema. Schemas are pretty-printed here; the logged form is the same JSON without whitespace.

Run command:
```
python -m agent.main --view-id 3 --start 2026-10-28-16-33 --end 2026-10-28-16-40 \
    --replay-db --process-viz --cat-idx 1 --plot
```

## T2 — weak-rule relabel (E5-T2)

### Escalation record

```json
{
  "escalation": true,
  "seq": 5,
  "name": "E5-T2",
  "time": "2026-10-28T16:34:36.154433",
  "end_time": null,
  "scope": "frame",
  "code": "T2",
  "reason": "detection labeled cat 25 relabeled to cat 26 by center distance alone, no IoU support",
  "cat_id": 26,
  "period_id": null,
  "metrics": {
    "original_cat_id": 25,
    "rule": "distance",
    "box": [
      296.0,
      369.0,
      371.0,
      531.0
    ],
    "source_time": "2026-10-28T16:34:35.793508",
    "source_box": [
      212.0,
      512.0,
      308.0,
      599.0
    ],
    "source_distance_px": 128.5787696316931,
    "source_threshold_px": 160.0
  }
}
```

### Reviewer prompt

```
[E5-T2] Identity switched on nearness alone, no overlap.
Conflict: the camera said cat 25, but the tracker re-labeled the box to cat 26 by nearness only — the boxes do not overlap at all.
Validate: is the photo really cat 26?
Context: the tracker assumes cats move smoothly (spatial continuity): a new box near a cat's last known position is treated as that cat even when the detector said otherwise. Here the detector's first guess was cat 25, Box [296.0, 369.0, 371.0, 531.0] at 2026-10-28T16:34:36.154433; the nearest cat-26 position on record was box [212.0, 512.0, 308.0, 599.0] at 2026-10-28T16:34:35.793508, 129 px away (the rule fires under 160 px) — that nearness is the entire basis for the switch. Nearness fires with no overlap, so neighbors and false alarms get stolen. Note: proposal is from a small YOLO on a weak CPU — a hint, not evidence. Trust the images and cat profiles.
Alternatives: keep 25, pick another nearby cat, or drop the box.
Cats:
- cat 25 (Qb): Short-haired white cat with faint silver shading, round face and blue-green eyes. Look for: Cool white to light silver colour, no warm cream or brown tones; Smooth, sharp body outline (short fur, no fluffy halo); Thin tail with grey rings or a grey tip, often carried upright when walking; Round head that looks large relative to a compact body.
- cat 26 (Qbo): Long-haired cream and golden-beige cat with darker brown paws, face and plume tail. Look for: Warm cream / beige colour, clearly warmer than Qb; Fuzzy, soft body outline with a fluffy halo; Big bushy plume tail, wider than the leg, grey-brown; Darker brown lower legs and paws against a pale body.
Resources (images__get_image):
  - 2026-10-28T16:34:36.154433 — the disputed photo; compare both cats' profiles, Box [296.0, 369.0, 371.0, 531.0]
  - 2026-10-28T16:34:35.793508 — source of the correction: cat 26's nearest recent position, 129 px away. Box [212.0, 512.0, 308.0, 599.0]
```

### Expected output schema

```json
{
  "additionalProperties": false,
  "description": "Box re-assigned by nearness alone, no overlap.",
  "properties": {
    "verdict": {
      "description": "confirm: the new cat is right. keep_original: the camera's first guess was right. reassign_other: a different nearby cat is right (say which in chosen_cat_id). drop: no cat there.",
      "enum": [
        "confirm",
        "keep_original",
        "reassign_other",
        "drop"
      ],
      "title": "Verdict",
      "type": "string"
    },
    "chosen_cat_id": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Required when verdict is reassign_other.",
      "title": "Chosen Cat Id"
    }
  },
  "required": [
    "verdict"
  ],
  "title": "T2Review",
  "type": "object"
}
```

## T4 — jump reset (E8-T4)

### Escalation record

```json
{
  "escalation": true,
  "seq": 8,
  "name": "E8-T4",
  "time": "2026-10-28T16:35:21.976102",
  "end_time": null,
  "scope": "frame",
  "code": "T4",
  "reason": "cat 25 new box missed the base (IoU 0.00); wiped 107 buffered rows and closed the period",
  "cat_id": 25,
  "period_id": 3,
  "metrics": {
    "base_iou": 0.0,
    "wiped_rows": 107,
    "base_time": "2026-10-28T16:34:42.769881",
    "base_box": [
      289.0,
      380.0,
      376.0,
      541.0
    ],
    "box": [
      145.0,
      357.0,
      234.0,
      462.0
    ],
    "had_open_visit": true
  }
}
```

### Reviewer prompt

```
[E8-T4] Track thrown away: new sighting matched nothing.
Task: validate whether discarding cat 25's old anchor and restarting from the new box at 2026-10-28T16:35:21.976102 was right — overlap 0.00, 107 buffered rows wiped.
Validate: look at the frames. Is this a genuinely new appearance (restart right), the same cat reappearing elsewhere (history wrongly lost), or one bad box (restart over a glitch)?
Context: the anchor (base) is the cat's last committed position. Nothing had been seen for a while — no pending detections — but the visit was still alive, so when the new box arrived somewhere the anchor cannot reach, the tracker concluded it cannot continue the old track. Here the anchor was box [289.0, 380.0, 376.0, 541.0] at 2026-10-28T16:34:42.769881; the new sighting is box [145.0, 357.0, 234.0, 462.0]. The wipe is safe because with nothing pending, buffered rows hold only empties and guesses, never real sightings; the new box is kept as the fresh start. The old visit then closes on the following frames, and the next commit mints a new period. Note: proposal is from a small YOLO on a weak CPU — a hint, not evidence. Trust the images and cat profiles.
Alternatives: keep the old track and treat the new box as an outlier, or link the new box as a continuation.
Cats:
- cat 25 (Qb): Short-haired white cat with faint silver shading, round face and blue-green eyes. Look for: Cool white to light silver colour, no warm cream or brown tones; Smooth, sharp body outline (short fur, no fluffy halo); Thin tail with grey rings or a grey tip, often carried upright when walking; Round head that looks large relative to a compact body.
Resources: visit 3 (this is period_id 3 — look it up in tracking__get_period_detail or the database); frames via images__get_image (boxes are the tracked positions for reference):
  - 2026-10-28T16:34:42.769881 — the anchor: last committed position before the gap. Box [289.0, 380.0, 376.0, 541.0]
  - 2026-10-28T16:35:21.976102 — the unmatched new sighting, overlap 0.00 vs the anchor. Box [145.0, 357.0, 234.0, 462.0]
```

### Expected output schema

```json
{
  "additionalProperties": false,
  "description": "New sighting missed the track; history was wiped.",
  "properties": {
    "verdict": {
      "description": "confirm_reset: a genuinely new appearance, wiping history was right. link_continuation: same cat, the new box continues the old track. drop_box_as_outlier: the new box is a one-off error, keep the old track.",
      "enum": [
        "confirm_reset",
        "link_continuation",
        "drop_box_as_outlier"
      ],
      "title": "Verdict",
      "type": "string"
    }
  },
  "required": [
    "verdict"
  ],
  "title": "T4Review",
  "type": "object"
}
```

## T6 — teleport split (E2-T6)

### Escalation record

```json
{
  "escalation": true,
  "seq": 2,
  "name": "E2-T6",
  "time": "2026-10-28T16:34:05.908104",
  "end_time": null,
  "scope": "frame",
  "code": "T6",
  "reason": "cat 26 box jumped 362 px; split period 1 -> 2 and retracted 10 interpolated rows",
  "cat_id": 26,
  "period_id": 2,
  "metrics": {
    "distance_px": 362.1670885102621,
    "box_conf": 0.5224,
    "old_period_id": 1,
    "retracted_rows": 10,
    "pre_time": "2026-10-28T16:34:01.879598",
    "pre_box": [
      210.0,
      509.0,
      311.0,
      599.0
    ],
    "post_box": [
      339.0,
      172.0,
      396.0,
      244.0
    ],
    "retracted_frames": [
      {
        "t": "2026-10-28T16:34:02.239640",
        "box": [
          221.52919171524132,
          478.88110381367187,
          318.596754230973,
          567.2723793882893
        ]
      },
      {
        "t": "2026-10-28T16:34:03.323620",
        "box": [
          256.2401788653163,
          388.20201335184805,
          341.46833491125494,
          471.74989537064107
        ]
      },
      {
        "t": "2026-10-28T16:34:04.463979",
        "box": [
          292.7565228896271,
          292.8066029937649,
          365.52949182649843,
          371.2591811952123
        ]
      },
      {
        "t": "2026-10-28T16:34:05.548031",
        "box": [
          327.46981560906204,
          202.12148945539613,
          388.40259168038966,
          275.7303523936665
        ]
      }
    ]
  }
}
```

### Reviewer prompt

```
[E2-T6] Cat seemingly teleported; visit split.
Task: validate whether splitting visit 1 into visit 2 at 2026-10-28T16:34:05.908104 was right — cat 26 jumped 362 px in one step, and the tracker deleted 10 guessed frames.
Validate: look at the frames below. Is this one fast-moving cat (the split was wrong, merge the visits back), or two different cats or a false box (the split was right)?
Context: last confirmed position was box [210.0, 509.0, 311.0, 599.0] at 2026-10-28T16:34:01.879598 (that is the old path end); the new sighting is box [339.0, 172.0, 396.0, 244.0] at 2026-10-28T16:34:05.908104 (certainty 0.52 is the camera's 0-to-1 sureness score for the new box); proposal keeps visits 1 and 2 separate. A sprinting cat and a box glitch look identical. Note: proposal is from a small YOLO on a weak CPU — a hint, not evidence. Trust the images and cat profiles.
Alternatives: merge visits 1 and 2 into one, or drop the jumped box as false.
Cats:
- cat 26 (Qbo): Long-haired cream and golden-beige cat with darker brown paws, face and plume tail. Look for: Warm cream / beige colour, clearly warmer than Qb; Fuzzy, soft body outline with a fluffy halo; Big bushy plume tail, wider than the leg, grey-brown; Darker brown lower legs and paws against a pale body.
Resources: visits 1, 2 (these are period_id 1 and 2 — look them up in tracking__get_period_detail or the database); frames via images__get_image (boxes are the tracked positions for reference):
  - 2026-10-28T16:34:01.879598 — pre-jump frame; last confirmed position. Box [210.0, 509.0, 311.0, 599.0]
  - 2026-10-28T16:34:02.239640 — intermediate frame; interpolated guess. Interpolated box [221.5, 478.9, 318.6, 567.3]
  - 2026-10-28T16:34:03.323620 — intermediate frame; interpolated guess. Interpolated box [256.2, 388.2, 341.5, 471.7]
  - 2026-10-28T16:34:04.463979 — intermediate frame; interpolated guess. Interpolated box [292.8, 292.8, 365.5, 371.3]
  - 2026-10-28T16:34:05.548031 — intermediate frame; interpolated guess. Interpolated box [327.5, 202.1, 388.4, 275.7]
  - 2026-10-28T16:34:05.908104 — post-jump frame; new sighting 362 px away. Box [339.0, 172.0, 396.0, 244.0]
```

### Expected output schema

```json
{
  "additionalProperties": false,
  "description": "Box jumped across the frame; visit was split.",
  "properties": {
    "verdict": {
      "description": "confirm_split: two different cats (or a false box), the split was right. merge_visits: one fast-moving cat, join the visits back together. drop_box: the jumped box is false, keep the old visit.",
      "enum": [
        "confirm_split",
        "merge_visits",
        "drop_box"
      ],
      "title": "Verdict",
      "type": "string"
    }
  },
  "required": [
    "verdict"
  ],
  "title": "T6Review",
  "type": "object"
}
```

## T7 — long synthesis (E1-T7)

### Escalation record

```json
{
  "escalation": true,
  "seq": 1,
  "name": "E1-T7",
  "time": "2026-10-28T16:34:01.519600",
  "end_time": null,
  "scope": "frame",
  "code": "T7",
  "reason": "cat 26 committed 5.1 s of interpolated-only frames with no real detection",
  "cat_id": 26,
  "period_id": 1,
  "metrics": {
    "gap_start": "2026-10-28T16:33:56.407317",
    "gap_seconds": 5.112283,
    "interp_frames": [
      {
        "t": "2026-10-28T16:33:56.407317",
        "box": [
          210.0,
          509.9382551134164,
          310.0617448865836,
          599.0
        ]
      },
      {
        "t": "2026-10-28T16:33:57.491689",
        "box": [
          210.0,
          509.7523330867797,
          310.2476669132203,
          599.0
        ]
      },
      {
        "t": "2026-10-28T16:33:58.571850",
        "box": [
          210.0,
          509.5671330611299,
          310.4328669388701,
          599.0
        ]
      },
      {
        "t": "2026-10-28T16:33:59.355506",
        "box": [
          210.0,
          509.4327705806287,
          310.5672294193713,
          599.0
        ]
      },
      {
        "t": "2026-10-28T16:34:00.435444",
        "box": [
          210.0,
          509.24760878965475,
          310.75239121034525,
          599.0
        ]
      },
      {
        "t": "2026-10-28T16:34:01.519600",
        "box": [
          210.0,
          509.0617237975023,
          310.9382762024977,
          599.0
        ]
      }
    ]
  }
}
```

### Reviewer prompt

```
[E1-T7] Seconds of guessed frames, no sighting.
Task: double-check 5.1 seconds of positions from 2026-10-28T16:33:56.407317 to 2026-10-28T16:34:01.519600 — every box in this span was guessed, none was seen.
Validate: cat 26 present but missed (keep the guesses), or gone while the track imagined it (cut them)?
Context: proposal keeps the guessed run. Guessing assumes presence. Note: proposal is from a small YOLO on a weak CPU — a hint, not evidence. Trust the images and cat profiles.
Alternatives: cut the whole span 2026-10-28T16:33:56.407317 to 2026-10-28T16:34:01.519600, or shorten to the last frame where the cat is visible.
Cats:
- cat 26 (Qbo): Long-haired cream and golden-beige cat with darker brown paws, face and plume tail. Look for: Warm cream / beige colour, clearly warmer than Qb; Fuzzy, soft body outline with a fluffy halo; Big bushy plume tail, wider than the leg, grey-brown; Darker brown lower legs and paws against a pale body.
Resources: visit 1 (this is period_id 1 — use it to look up rows in tracking__get_period_detail or the database); key frames via images__get_image (boxes are the interpolated guesses for reference):
  - 2026-10-28T16:33:56.407317 — first guessed frame (last real sighting is just before it). Interpolated box [210.0, 509.9, 310.1, 599.0]
  - 2026-10-28T16:33:57.491689 — mid-gap guess. Interpolated box [210.0, 509.8, 310.2, 599.0]
  - 2026-10-28T16:33:58.571850 — mid-gap guess. Interpolated box [210.0, 509.6, 310.4, 599.0]
  - 2026-10-28T16:33:59.355506 — mid-gap guess. Interpolated box [210.0, 509.4, 310.6, 599.0]
  - 2026-10-28T16:34:00.435444 — mid-gap guess. Interpolated box [210.0, 509.2, 310.8, 599.0]
  - 2026-10-28T16:34:01.519600 — gap end; still no fresh sighting. Interpolated box [210.0, 509.1, 310.9, 599.0]
Also check the first real sighting after 2026-10-28T16:34:01.519600 in visit 1 — the box this guessed run connects to (scan forward from gap end).
```

### Expected output schema

```json
{
  "additionalProperties": false,
  "description": "Long run of guessed frames with no sighting.",
  "properties": {
    "verdict": {
      "description": "keep_synthesis: the cat stayed put unseen, the guessed run is fine. cut_at_gap_start: the cat left, end the visit where sightings stopped. shorten_gap: partly present \u2014 say the last frame where the cat is actually visible in cut_time.",
      "enum": [
        "keep_synthesis",
        "cut_at_gap_start",
        "shorten_gap"
      ],
      "title": "Verdict",
      "type": "string"
    },
    "cut_time": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "ISO timestamp of the last frame showing the cat. Required when verdict is shorten_gap.",
      "title": "Cut Time"
    }
  },
  "required": [
    "verdict"
  ],
  "title": "T7Review",
  "type": "object"
}
```

## P2 — weak winner (E11-P2)

### Escalation record

```json
{
  "escalation": true,
  "seq": 11,
  "name": "E11-P2",
  "time": "2026-10-28T16:34:41.265746",
  "end_time": "2026-10-28T16:34:42.769881",
  "scope": "period",
  "code": "P2",
  "reason": "period 3 winner cat 25 is weak: score 2.2, 2 original frames",
  "cat_id": 25,
  "period_id": 3,
  "metrics": {
    "score": 2.2399999999999998,
    "detect_length": 2,
    "key_frames": [
      {
        "t": "2026-10-28T16:34:41.265746",
        "box": [
          290.0,
          389.0,
          373.0,
          542.0
        ],
        "real": true
      },
      {
        "t": "2026-10-28T16:34:41.625705",
        "box": [
          289.7606870393947,
          386.84618335455264,
          373.71793888181577,
          541.7606870393947
        ],
        "real": false
      },
      {
        "t": "2026-10-28T16:34:42.051547",
        "box": [
          289.47757282424783,
          384.2981554182304,
          374.5672815272565,
          541.4775728242478
        ],
        "real": false
      },
      {
        "t": "2026-10-28T16:34:42.410626",
        "box": [
          289.23884491751073,
          382.14960425759654,
          375.2834652474678,
          541.2388449175107
        ],
        "real": false
      },
      {
        "t": "2026-10-28T16:34:42.769881",
        "box": [
          289.0,
          380.0,
          376.0,
          541.0
        ],
        "real": true
      }
    ]
  }
}
```

### Reviewer prompt

```
[E11-P2] Visit awarded on thin evidence.
Conflict: visit 3 -> cat 25 on 2.2 pts from 2 original frames (frames actually seen, not guessed).
Validate: real appearance or noise that should never have become a visit?
Context: pts = vote points (up to ~13): +3 best avg certainty, +2 most original frames, +4.2 widest area, +1 each for >20 frames / rate>0.5 / conf>0.5 / rate>0.3 — then x0.2 if under 10 original frames. Here 2 frames triggered the x0.2 penalty. Proposal keeps it as cat 25. Short low-score visits are often stray boxes. Note: proposal is from a small YOLO on a weak CPU — a hint, not evidence. Trust the images and cat profiles.
Alternatives: drop as noise, or merge into a neighboring visit.
Cats:
- cat 25 (Qb): Short-haired white cat with faint silver shading, round face and blue-green eyes. Look for: Cool white to light silver colour, no warm cream or brown tones; Smooth, sharp body outline (short fur, no fluffy halo); Thin tail with grey rings or a grey tip, often carried upright when walking; Round head that looks large relative to a compact body.
Resources: visit 3 (this is period_id 3 — look it up in tracking__get_period_detail or the database); key frames via images__get_image (boxes are the tracked positions for reference):
  - 2026-10-28T16:34:41.265746 — visit start. Box [290.0, 389.0, 373.0, 542.0] (real sighting)
  - 2026-10-28T16:34:41.625705 — mid-visit check. Box [289.8, 386.8, 373.7, 541.8] (guessed, no sighting)
  - 2026-10-28T16:34:42.051547 — mid-visit check. Box [289.5, 384.3, 374.6, 541.5] (guessed, no sighting)
  - 2026-10-28T16:34:42.410626 — mid-visit check. Box [289.2, 382.1, 375.3, 541.2] (guessed, no sighting)
  - 2026-10-28T16:34:42.769881 — visit end. Box [289.0, 380.0, 376.0, 541.0] (real sighting)
```

### Expected output schema

```json
{
  "additionalProperties": false,
  "description": "Visit awarded on very thin evidence.",
  "properties": {
    "verdict": {
      "description": "keep: a real appearance, keep the visit. drop: noise, delete the visit. merge_neighbor: part of a neighboring visit of the same cat, merge them.",
      "enum": [
        "keep",
        "drop",
        "merge_neighbor"
      ],
      "title": "Verdict",
      "type": "string"
    }
  },
  "required": [
    "verdict"
  ],
  "title": "P2Review",
  "type": "object"
}
```

## P5 — low quality (E3-P5)

### Escalation record

```json
{
  "escalation": true,
  "seq": 3,
  "name": "E3-P5",
  "time": "2026-10-28T16:33:06.797209",
  "end_time": "2026-10-28T16:34:01.879598",
  "scope": "period",
  "code": "P5",
  "reason": "period 1 cat 26 is low quality: mean conf 0.54, original-frame rate 0.29",
  "cat_id": 26,
  "period_id": 1,
  "metrics": {
    "detect_conf": 0.5375,
    "detect_rate": 0.2876,
    "total_frames": 153,
    "detect_counts": {
      "26": 44
    },
    "winner_box": [
      209.0,
      509.0,
      310.0,
      599.0
    ],
    "key_frames": [
      {
        "t": "2026-10-28T16:33:06.797209",
        "box": [
          209.0,
          505.0,
          311.0,
          599.0
        ],
        "real": true
      },
      {
        "t": "2026-10-28T16:33:17.681195",
        "box": [
          209.0,
          509.0,
          309.77274282814693,
          599.0
        ],
        "real": false
      },
      {
        "t": "2026-10-28T16:33:28.925564",
        "box": [
          210.18141894539593,
          509.0,
          309.36283789079187,
          599.0
        ],
        "real": false
      },
      {
        "t": "2026-10-28T16:33:39.690269",
        "box": [
          210.0,
          509.0,
          309.0,
          599.0
        ],
        "real": false
      },
      {
        "t": "2026-10-28T16:33:50.995666",
        "box": [
          210.0,
          510.0,
          310.0,
          599.0
        ],
        "real": true
      },
      {
        "t": "2026-10-28T16:34:01.879598",
        "box": [
          210.0,
          509.0,
          311.0,
          599.0
        ],
        "real": true
      }
    ]
  }
}
```

### Reviewer prompt

```
[E3-P5] Verdict rests on weak, patchy sightings.
Conflict: visit 1 (cat 26) runs 2026-10-28T16:33:06.797209 to 2026-10-28T16:34:01.879598 with avg certainty 0.54; of 153 frames in total, the camera actually spotted a cat in 44 (cat 26: 44); the other 109 were filled in by guessing.
Validate: real visit or noise padded with guesses?
Context: proposal keeps cat 26, typically around box [209.0, 509.0, 310.0, 599.0] (median of its sightings). Mostly-guessed visits may not reflect reality. Note: proposal is from a small YOLO on a weak CPU — a hint, not evidence. Trust the images and cat profiles.
Alternatives: drop, trim to the high-certainty core, or reassign.
Cats:
- cat 26 (Qbo): Long-haired cream and golden-beige cat with darker brown paws, face and plume tail. Look for: Warm cream / beige colour, clearly warmer than Qb; Fuzzy, soft body outline with a fluffy halo; Big bushy plume tail, wider than the leg, grey-brown; Darker brown lower legs and paws against a pale body.
Resources: visit 1 (this is period_id 1 — one continuous tracked visit from 2026-10-28T16:33:06.797209 to 2026-10-28T16:34:01.879598; look it up in tracking__get_period_detail or the database); key frames via images__get_image (boxes are the tracked positions for reference):
  - 2026-10-28T16:33:06.797209 — visit start. Box [209.0, 505.0, 311.0, 599.0] (real sighting)
  - 2026-10-28T16:33:17.681195 — mid-visit check. Box [209.0, 509.0, 309.8, 599.0] (guessed, no sighting)
  - 2026-10-28T16:33:28.925564 — mid-visit check. Box [210.2, 509.0, 309.4, 599.0] (guessed, no sighting)
  - 2026-10-28T16:33:39.690269 — mid-visit check. Box [210.0, 509.0, 309.0, 599.0] (guessed, no sighting)
  - 2026-10-28T16:33:50.995666 — mid-visit check. Box [210.0, 510.0, 310.0, 599.0] (real sighting)
  - 2026-10-28T16:34:01.879598 — visit end. Box [210.0, 509.0, 311.0, 599.0] (real sighting)
```

### Expected output schema

```json
{
  "additionalProperties": false,
  "description": "Visit verdict rests on weak, patchy sightings.",
  "properties": {
    "verdict": {
      "description": "keep: a real visit, keep it. drop: noise padded with guesses, delete it. trim_core: keep only the high-certainty core (flagged for a human; rows kept as-is). reassign: another cat (say which in chosen_cat_id).",
      "enum": [
        "keep",
        "drop",
        "trim_core",
        "reassign"
      ],
      "title": "Verdict",
      "type": "string"
    },
    "chosen_cat_id": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Required when verdict is reassign.",
      "title": "Chosen Cat Id"
    }
  },
  "required": [
    "verdict"
  ],
  "title": "P5Review",
  "type": "object"
}
```

## P6 — merge candidate (E9-P6)

### Escalation record

```json
{
  "escalation": true,
  "seq": 9,
  "name": "E9-P6",
  "time": "2026-10-28T16:34:42.769881",
  "end_time": "2026-10-28T16:35:21.976102",
  "scope": "period",
  "code": "P6",
  "reason": "period 4 may connect to previous period 3 of cat 25: gap 39.2 s, box distance 152 px",
  "cat_id": 25,
  "period_id": 4,
  "metrics": {
    "prev_period_id": 3,
    "gap_seconds": 39.206221,
    "distance_px": 151.82226450688975,
    "prev_start": "2026-10-28T16:34:41.265746",
    "prev_end_box": [
      289.0,
      380.0,
      376.0,
      541.0
    ],
    "new_start_box": [
      145.0,
      357.0,
      234.0,
      462.0
    ]
  }
}
```

### Reviewer prompt

```
[E9-P6] Two back-to-back visits, might be one.
Conflict: visit 3 ran 2026-10-28T16:34:41.265746 to 2026-10-28T16:34:42.769881; visit 4 starts at 2026-10-28T16:35:21.976102 — a 39.2-second gap. The last box of visit 3 and the first box of visit 4 are 152 px apart (center to center).
Validate: merge visits 3 and 4 into one cat 25 visit?
Context: proposal keeps separate. Short gap + nearby boxes suggests continuity, but the cat may have left and returned. Note: proposal is from a small YOLO on a weak CPU — a hint, not evidence. Trust the images and cat profiles.
Alternatives: merge, or confirm separate visits.
Cats:
- cat 25 (Qb): Short-haired white cat with faint silver shading, round face and blue-green eyes. Look for: Cool white to light silver colour, no warm cream or brown tones; Smooth, sharp body outline (short fur, no fluffy halo); Thin tail with grey rings or a grey tip, often carried upright when walking; Round head that looks large relative to a compact body.
Resources: visits 3, 4 (these are period_id 3 and 4 — look them up in tracking__get_period_detail or the database); gap frames ~1/min via images__get_image (boxes are the tracked positions for reference):
  - 2026-10-28T16:34:42.769881 — visit 3 ends; last confirmed position. Box [289.0, 380.0, 376.0, 541.0]
  - 2026-10-28T16:35:02.372991 — gap, untracked — did the cat stay or leave?
  - 2026-10-28T16:35:21.976102 — visit 4 starts; first confirmed position. Box [145.0, 357.0, 234.0, 462.0]
```

### Expected output schema

```json
{
  "additionalProperties": false,
  "description": "Two back-to-back visits that might be one.",
  "properties": {
    "verdict": {
      "description": "merge: the cat never left, join the two visits into one. keep_separate: the cat left and came back, two real visits.",
      "enum": [
        "merge",
        "keep_separate"
      ],
      "title": "Verdict",
      "type": "string"
    }
  },
  "required": [
    "verdict"
  ],
  "title": "P6Review",
  "type": "object"
}
```

_Codes not seen in this run (no example): T1, T3, T5, P1, P3, P4, P7._
