# #20 `nearest_smooth` on the bench

- **Issue**: #20 (bench on other matches, ADR 0003)
- **Method**: the organisers' baseline on the Match's arrays: the side of the player nearest the
  ball, `D` where the ball is not tracked, then a centred majority vote of ± `window` frames.
  `window` chosen on the four training matches.
- **Command**:
  `python -m possession.bench train nearest_smooth` (degradation v1, seeds 0, 1, 2)
- **Metrics**: `experiments/metrics/bench/nearest_smooth.json`

## Training grid (mean over 4 matches × 3 seeds)

| window | spellF1 | macroF1 |
|---|---|---|
| 0 | 0.083 | 0.438 |
| 12 | 0.165 | 0.447 |
| 25 | 0.216 | 0.456 |
| **50** | **0.221** | **0.460** |
| 75 | 0.207 | 0.455 |
| 100 | 0.189 | 0.448 |
| 125 | 0.170 | 0.439 |
| 175 | 0.138 | 0.424 |
| 250 | 0.106 | 0.400 |

## Validation (J03WOY, window 50)

| seed | spellF1 | macroF1 | spells (pred / true) |
|---|---|---|---|
| 0 | 0.199 | 0.494 | 348 / 456 |
| 1 | 0.181 | 0.424 | 389 / 456 |
| 2 | 0.172 | 0.446 | 395 / 456 |
| **mean** | **0.184** | **0.455** | |

Test not run.

## Verdict

Kept as the bench's reference point. On the training matches the bench picks the window that is
best on BAR-ATH's full ground truth (50, published spellF1 0.2433), at a similar level, which is a
first sign the bench transfers. macroF1 is much lower here than on BAR-ATH (0.67): the DFL calls
44% of frames dead, and the baseline's `D` only comes from the hidden ball.
