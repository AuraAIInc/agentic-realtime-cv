# Detection accuracy report

## Run `view3_20261028-1633_1640_replay_db_20261102-002728-578281`

- Hour: `2026-10-28-16` — ground truth: `image_server/retrieved_images/2026-10-28-16/muse_labels`
- Classes: 25=Qb-2, 26=Qbo-2; image size 800x600
- Frames: 1318 evaluated (1319 in run, 1 without GT)

### IoU >= 0.5 (GT boxes: 1816)

P/R/F1 are class-aware (right class + box overlap). P/R/F1(loc) are class-agnostic (box overlap only). cls-acc = fraction of spatially matched pairs with the right class.

| stage | P@0.5 | R@0.5 | F1@0.5 | P(loc) | R(loc) | F1(loc) | mean IoU | cls-acc | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | 0.928 | 0.519 | 0.666 | 0.997 | 0.558 | 0.715 | 0.771 | 0.931 | 943 | 73 | 873 |
| revised | 0.170 | 0.119 | 0.140 | 0.979 | 0.687 | 0.807 | 0.785 | 0.172 | 216 | 1058 | 1600 |
| validated | 0.170 | 0.119 | 0.140 | 0.979 | 0.687 | 0.807 | 0.786 | 0.172 | 216 | 1058 | 1600 |

| step | ΔP | ΔR | ΔF1 | ΔP(loc) | ΔR(loc) | ΔF1(loc) | Δmean IoU | Δcls-acc |
|---|---|---|---|---|---|---|---|---|
| raw → revised | -0.759 | -0.400 | -0.526 | -0.018 | +0.129 | +0.092 | +0.014 | -0.758 |
| revised → validated | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.001 | +0.000 |

| stage | class | P | R | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| raw | 25 (Qb-2) | 0.980 | 0.602 | 0.746 | 802 | 16 | 530 |
| raw | 26 (Qbo-2) | 0.712 | 0.291 | 0.413 | 141 | 57 | 343 |
| revised | 25 (Qb-2) | 0.167 | 0.001 | 0.001 | 1 | 5 | 1331 |
| revised | 26 (Qbo-2) | 0.170 | 0.444 | 0.245 | 215 | 1053 | 269 |
| validated | 25 (Qb-2) | 0.250 | 0.002 | 0.003 | 2 | 6 | 1330 |
| validated | 26 (Qbo-2) | 0.169 | 0.442 | 0.245 | 214 | 1052 | 270 |

### IoU >= 0.75 (GT boxes: 1816)

P/R/F1 are class-aware (right class + box overlap). P/R/F1(loc) are class-agnostic (box overlap only). cls-acc = fraction of spatially matched pairs with the right class.

| stage | P@0.75 | R@0.75 | F1@0.75 | P(loc) | R(loc) | F1(loc) | mean IoU | cls-acc | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | 0.809 | 0.453 | 0.581 | 0.824 | 0.461 | 0.591 | 0.780 | 0.982 | 822 | 194 | 994 |
| revised | 0.104 | 0.073 | 0.086 | 0.733 | 0.514 | 0.605 | 0.843 | 0.141 | 133 | 1141 | 1683 |
| validated | 0.105 | 0.074 | 0.087 | 0.733 | 0.514 | 0.605 | 0.843 | 0.142 | 134 | 1140 | 1682 |

| step | ΔP | ΔR | ΔF1 | ΔP(loc) | ΔR(loc) | ΔF1(loc) | Δmean IoU | Δcls-acc |
|---|---|---|---|---|---|---|---|---|
| raw → revised | -0.705 | -0.379 | -0.494 | -0.091 | +0.053 | +0.013 | +0.063 | -0.841 |
| revised → validated | +0.001 | +0.001 | +0.001 | +0.000 | +0.000 | +0.000 | +0.000 | +0.001 |

| stage | class | P | R | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| raw | 25 (Qb-2) | 0.912 | 0.560 | 0.694 | 746 | 72 | 586 |
| raw | 26 (Qbo-2) | 0.384 | 0.157 | 0.223 | 76 | 122 | 408 |
| revised | 25 (Qb-2) | 0.167 | 0.001 | 0.001 | 1 | 5 | 1331 |
| revised | 26 (Qbo-2) | 0.104 | 0.273 | 0.151 | 132 | 1136 | 352 |
| validated | 25 (Qb-2) | 0.250 | 0.002 | 0.003 | 2 | 6 | 1330 |
| validated | 26 (Qbo-2) | 0.104 | 0.273 | 0.151 | 132 | 1134 | 352 |

## Run `view3_20261028-1633_1640_replay_db_20261102-012059-312907`

- Hour: `2026-10-28-16` — ground truth: `image_server/retrieved_images/2026-10-28-16/muse_labels`
- Classes: 25=Qb-2, 26=Qbo-2; image size 800x600
- Frames: 1318 evaluated (1319 in run, 1 without GT)

### IoU >= 0.5 (GT boxes: 1816)

P/R/F1 are class-aware (right class + box overlap). P/R/F1(loc) are class-agnostic (box overlap only). cls-acc = fraction of spatially matched pairs with the right class.

| stage | P@0.5 | R@0.5 | F1@0.5 | P(loc) | R(loc) | F1(loc) | mean IoU | cls-acc | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | 0.928 | 0.519 | 0.666 | 0.997 | 0.558 | 0.715 | 0.771 | 0.931 | 943 | 73 | 873 |
| revised | 0.170 | 0.119 | 0.140 | 0.979 | 0.687 | 0.807 | 0.785 | 0.172 | 216 | 1058 | 1600 |
| validated | 0.170 | 0.119 | 0.140 | 0.979 | 0.687 | 0.807 | 0.786 | 0.172 | 216 | 1058 | 1600 |

| step | ΔP | ΔR | ΔF1 | ΔP(loc) | ΔR(loc) | ΔF1(loc) | Δmean IoU | Δcls-acc |
|---|---|---|---|---|---|---|---|---|
| raw → revised | -0.759 | -0.400 | -0.526 | -0.018 | +0.129 | +0.092 | +0.014 | -0.758 |
| revised → validated | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.001 | +0.000 |

| stage | class | P | R | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| raw | 25 (Qb-2) | 0.980 | 0.602 | 0.746 | 802 | 16 | 530 |
| raw | 26 (Qbo-2) | 0.712 | 0.291 | 0.413 | 141 | 57 | 343 |
| revised | 25 (Qb-2) | 0.167 | 0.001 | 0.001 | 1 | 5 | 1331 |
| revised | 26 (Qbo-2) | 0.170 | 0.444 | 0.245 | 215 | 1053 | 269 |
| validated | 25 (Qb-2) | 0.250 | 0.002 | 0.003 | 2 | 6 | 1330 |
| validated | 26 (Qbo-2) | 0.169 | 0.442 | 0.245 | 214 | 1052 | 270 |

### IoU >= 0.75 (GT boxes: 1816)

P/R/F1 are class-aware (right class + box overlap). P/R/F1(loc) are class-agnostic (box overlap only). cls-acc = fraction of spatially matched pairs with the right class.

| stage | P@0.75 | R@0.75 | F1@0.75 | P(loc) | R(loc) | F1(loc) | mean IoU | cls-acc | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | 0.809 | 0.453 | 0.581 | 0.824 | 0.461 | 0.591 | 0.780 | 0.982 | 822 | 194 | 994 |
| revised | 0.104 | 0.073 | 0.086 | 0.733 | 0.514 | 0.605 | 0.843 | 0.141 | 133 | 1141 | 1683 |
| validated | 0.105 | 0.074 | 0.087 | 0.733 | 0.514 | 0.605 | 0.843 | 0.142 | 134 | 1140 | 1682 |

| step | ΔP | ΔR | ΔF1 | ΔP(loc) | ΔR(loc) | ΔF1(loc) | Δmean IoU | Δcls-acc |
|---|---|---|---|---|---|---|---|---|
| raw → revised | -0.705 | -0.379 | -0.494 | -0.091 | +0.053 | +0.013 | +0.063 | -0.841 |
| revised → validated | +0.001 | +0.001 | +0.001 | +0.000 | +0.000 | +0.000 | +0.000 | +0.001 |

| stage | class | P | R | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| raw | 25 (Qb-2) | 0.912 | 0.560 | 0.694 | 746 | 72 | 586 |
| raw | 26 (Qbo-2) | 0.384 | 0.157 | 0.223 | 76 | 122 | 408 |
| revised | 25 (Qb-2) | 0.167 | 0.001 | 0.001 | 1 | 5 | 1331 |
| revised | 26 (Qbo-2) | 0.104 | 0.273 | 0.151 | 132 | 1136 | 352 |
| validated | 25 (Qb-2) | 0.250 | 0.002 | 0.003 | 2 | 6 | 1330 |
| validated | 26 (Qbo-2) | 0.104 | 0.273 | 0.151 | 132 | 1134 | 352 |

## Run `view3_20261028-1633_1640_replay_db_20261102-221606-604774`

- Hour: `2026-10-28-16` — ground truth: `image_server/retrieved_images/2026-10-28-16/muse_labels`
- Classes: 25=Qb-2, 26=Qbo-2; image size 800x600
- Frames: 1318 evaluated (1319 in run, 1 without GT)

### IoU >= 0.5 (GT boxes: 1816)

P/R/F1 are class-aware (right class + box overlap). P/R/F1(loc) are class-agnostic (box overlap only). cls-acc = fraction of spatially matched pairs with the right class.

| stage | P@0.5 | R@0.5 | F1@0.5 | P(loc) | R(loc) | F1(loc) | mean IoU | cls-acc | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | 0.928 | 0.519 | 0.666 | 0.997 | 0.558 | 0.715 | 0.771 | 0.931 | 943 | 73 | 873 |
| revised | 0.170 | 0.119 | 0.140 | 0.979 | 0.687 | 0.807 | 0.785 | 0.172 | 216 | 1058 | 1600 |
| validated | 0.170 | 0.119 | 0.140 | 0.979 | 0.687 | 0.807 | 0.786 | 0.172 | 216 | 1058 | 1600 |

| step | ΔP | ΔR | ΔF1 | ΔP(loc) | ΔR(loc) | ΔF1(loc) | Δmean IoU | Δcls-acc |
|---|---|---|---|---|---|---|---|---|
| raw → revised | -0.759 | -0.400 | -0.526 | -0.018 | +0.129 | +0.092 | +0.014 | -0.758 |
| revised → validated | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.001 | +0.000 |

| stage | class | P | R | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| raw | 25 (Qb-2) | 0.980 | 0.602 | 0.746 | 802 | 16 | 530 |
| raw | 26 (Qbo-2) | 0.712 | 0.291 | 0.413 | 141 | 57 | 343 |
| revised | 25 (Qb-2) | 0.167 | 0.001 | 0.001 | 1 | 5 | 1331 |
| revised | 26 (Qbo-2) | 0.170 | 0.444 | 0.245 | 215 | 1053 | 269 |
| validated | 25 (Qb-2) | 0.250 | 0.002 | 0.003 | 2 | 6 | 1330 |
| validated | 26 (Qbo-2) | 0.169 | 0.442 | 0.245 | 214 | 1052 | 270 |

### IoU >= 0.75 (GT boxes: 1816)

P/R/F1 are class-aware (right class + box overlap). P/R/F1(loc) are class-agnostic (box overlap only). cls-acc = fraction of spatially matched pairs with the right class.

| stage | P@0.75 | R@0.75 | F1@0.75 | P(loc) | R(loc) | F1(loc) | mean IoU | cls-acc | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | 0.809 | 0.453 | 0.581 | 0.824 | 0.461 | 0.591 | 0.780 | 0.982 | 822 | 194 | 994 |
| revised | 0.104 | 0.073 | 0.086 | 0.733 | 0.514 | 0.605 | 0.843 | 0.141 | 133 | 1141 | 1683 |
| validated | 0.105 | 0.074 | 0.087 | 0.733 | 0.514 | 0.605 | 0.843 | 0.142 | 134 | 1140 | 1682 |

| step | ΔP | ΔR | ΔF1 | ΔP(loc) | ΔR(loc) | ΔF1(loc) | Δmean IoU | Δcls-acc |
|---|---|---|---|---|---|---|---|---|
| raw → revised | -0.705 | -0.379 | -0.494 | -0.091 | +0.053 | +0.013 | +0.063 | -0.841 |
| revised → validated | +0.001 | +0.001 | +0.001 | +0.000 | +0.000 | +0.000 | +0.000 | +0.001 |

| stage | class | P | R | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| raw | 25 (Qb-2) | 0.912 | 0.560 | 0.694 | 746 | 72 | 586 |
| raw | 26 (Qbo-2) | 0.384 | 0.157 | 0.223 | 76 | 122 | 408 |
| revised | 25 (Qb-2) | 0.167 | 0.001 | 0.001 | 1 | 5 | 1331 |
| revised | 26 (Qbo-2) | 0.104 | 0.273 | 0.151 | 132 | 1136 | 352 |
| validated | 25 (Qb-2) | 0.250 | 0.002 | 0.003 | 2 | 6 | 1330 |
| validated | 26 (Qbo-2) | 0.104 | 0.273 | 0.151 | 132 | 1134 | 352 |

## Run `view3_20261028-1633_1640_replay_db_20261102-231613-423659`

- Hour: `2026-10-28-16` — ground truth: `image_server/retrieved_images/2026-10-28-16/muse_labels`
- Classes: 25=Qb-2, 26=Qbo-2; image size 800x600
- Frames: 1318 evaluated (1319 in run, 1 without GT)

### IoU >= 0.5 (GT boxes: 1816)

P/R/F1 are class-aware (right class + box overlap). P/R/F1(loc) are class-agnostic (box overlap only). cls-acc = fraction of spatially matched pairs with the right class.

| stage | P@0.5 | R@0.5 | F1@0.5 | P(loc) | R(loc) | F1(loc) | mean IoU | cls-acc | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | 0.928 | 0.519 | 0.666 | 0.997 | 0.558 | 0.715 | 0.771 | 0.931 | 943 | 73 | 873 |
| revised | 0.170 | 0.119 | 0.140 | 0.979 | 0.687 | 0.807 | 0.785 | 0.172 | 216 | 1058 | 1600 |
| validated | 0.169 | 0.117 | 0.138 | 0.979 | 0.679 | 0.802 | 0.788 | 0.172 | 213 | 1047 | 1603 |

| step | ΔP | ΔR | ΔF1 | ΔP(loc) | ΔR(loc) | ΔF1(loc) | Δmean IoU | Δcls-acc |
|---|---|---|---|---|---|---|---|---|
| raw → revised | -0.759 | -0.400 | -0.526 | -0.018 | +0.129 | +0.092 | +0.014 | -0.758 |
| revised → validated | -0.000 | -0.002 | -0.001 | -0.000 | -0.008 | -0.005 | +0.003 | -0.000 |

| stage | class | P | R | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| raw | 25 (Qb-2) | 0.980 | 0.602 | 0.746 | 802 | 16 | 530 |
| raw | 26 (Qbo-2) | 0.712 | 0.291 | 0.413 | 141 | 57 | 343 |
| revised | 25 (Qb-2) | 0.167 | 0.001 | 0.001 | 1 | 5 | 1331 |
| revised | 26 (Qbo-2) | 0.170 | 0.444 | 0.245 | 215 | 1053 | 269 |
| validated | 25 (Qb-2) | 0.250 | 0.002 | 0.003 | 2 | 6 | 1330 |
| validated | 26 (Qbo-2) | 0.169 | 0.436 | 0.243 | 211 | 1041 | 273 |

### IoU >= 0.75 (GT boxes: 1816)

P/R/F1 are class-aware (right class + box overlap). P/R/F1(loc) are class-agnostic (box overlap only). cls-acc = fraction of spatially matched pairs with the right class.

| stage | P@0.75 | R@0.75 | F1@0.75 | P(loc) | R(loc) | F1(loc) | mean IoU | cls-acc | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | 0.809 | 0.453 | 0.581 | 0.824 | 0.461 | 0.591 | 0.780 | 0.982 | 822 | 194 | 994 |
| revised | 0.104 | 0.073 | 0.086 | 0.733 | 0.514 | 0.605 | 0.843 | 0.141 | 133 | 1141 | 1683 |
| validated | 0.106 | 0.074 | 0.087 | 0.741 | 0.514 | 0.607 | 0.843 | 0.142 | 134 | 1126 | 1682 |

| step | ΔP | ΔR | ΔF1 | ΔP(loc) | ΔR(loc) | ΔF1(loc) | Δmean IoU | Δcls-acc |
|---|---|---|---|---|---|---|---|---|
| raw → revised | -0.705 | -0.379 | -0.494 | -0.091 | +0.053 | +0.013 | +0.063 | -0.841 |
| revised → validated | +0.002 | +0.001 | +0.001 | +0.008 | +0.000 | +0.003 | +0.000 | +0.001 |

| stage | class | P | R | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| raw | 25 (Qb-2) | 0.912 | 0.560 | 0.694 | 746 | 72 | 586 |
| raw | 26 (Qbo-2) | 0.384 | 0.157 | 0.223 | 76 | 122 | 408 |
| revised | 25 (Qb-2) | 0.167 | 0.001 | 0.001 | 1 | 5 | 1331 |
| revised | 26 (Qbo-2) | 0.104 | 0.273 | 0.151 | 132 | 1136 | 352 |
| validated | 25 (Qb-2) | 0.250 | 0.002 | 0.003 | 2 | 6 | 1330 |
| validated | 26 (Qbo-2) | 0.105 | 0.273 | 0.152 | 132 | 1120 | 352 |

## Cross-run comparison (validated stage, IoU >= 0.5)

| run | P | R | F1 | P(loc) | R(loc) | mean IoU | cls-acc |
|---|---|---|---|---|---|---|---|
| `view3_20261028-1633_1640_replay_db_20261102-002728-578281` | 0.170 | 0.119 | 0.140 | 0.979 | 0.687 | 0.786 | 0.172 |
| `view3_20261028-1633_1640_replay_db_20261102-012059-312907` | 0.170 | 0.119 | 0.140 | 0.979 | 0.687 | 0.786 | 0.172 |
| `view3_20261028-1633_1640_replay_db_20261102-221606-604774` | 0.170 | 0.119 | 0.140 | 0.979 | 0.687 | 0.786 | 0.172 |
| `view3_20261028-1633_1640_replay_db_20261102-231613-423659` | 0.169 | 0.117 | 0.138 | 0.979 | 0.679 | 0.788 | 0.172 |
