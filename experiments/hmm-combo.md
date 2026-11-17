# hmm_combo: control confidence, P(alive) and cleaned tracking in one HMM

**Question.** Each of the 17 November features was within noise on its own (`hmm_conf` +0.007,
`stoppages` as a method below the HMM, the cleaning invisible on the bench). Do they add up when
the HMM reads them together?

**Method** (`possession/experiments/hmm_combo.py`). `hmm_touch_ms`'s features and settings, plus:

- **conf**: `hmm_conf`'s control features (hand-set scales, `controls.HAND`);
- **alive**: P(alive) from `possession/stoppage.py`'s features, its logit and its means over ±1 s
  and ±4 s. **Cross-fitted**: on a match the alive model was fitted on, the value comes from the
  model fitted without that match (`possession/models/hmm_combo_alive.json` keeps both), so the HMM
  never sees an in-sample P(alive);
- **clean**: `hmm_touch_ms`'s features and the controls read `track_xy_clean` with the per-frame
  sides of `track_team_fixed` (keepers fixed). `hmm_viterbi.ball_features` now accepts per-frame
  sides; with one side per slot it is unchanged (the submission CSV is byte-identical). On the
  bench this part is a no-op (checked: identical features), so it is judged on BAR-ATH only.

Bar fixed before running: at least +0.015 spellF1 on validation over `hmm_touch_ms`, seeds 0-2.

## Bench validation (J03WOY, degradation v2, seeds 0-2)

| | spellF1 | macroF1 | per seed |
|---|---|---|---|
| `hmm_touch_ms` | 0.473 | 0.838 | 0.481 / 0.487 / 0.451 |
| `hmm_conf_ms_hand` (#33, conf only) | 0.488 | 0.841 | |
| `hmm_combo_noconf` (alive + clean) | 0.487 | 0.846 | 0.490 / 0.499 / 0.471 |
| `hmm_combo_noalive` (conf + clean) | 0.490 | 0.841 | 0.490 / 0.498 / 0.483 |
| **`hmm_combo`** | **0.495** | **0.848** | 0.479 / 0.516 / 0.491 |

+0.022 over `hmm_touch_ms`: clears the bar, though one validation match and a seed spread of 0.04
make it a weak one. Each group adds about +0.005-0.008 on top of the other; neither is decisive
alone. Alive/dead F1 0.944-0.956 per seed.

## BAR-ATH (label-free)

| | possessions | median | dead share | ST agreement |
|---|---|---|---|---|
| `hmm_final_ms` (submitted) | 408 | 4.98 s | 35.5% | 0.814 |
| `hmm_combo` | 362 | 4.64 s | 39.7% | **0.832** |
| `hmm_combo_raw` (same models, raw tracking) | 364 | 4.64 s | 39.7% | 0.832 |

`hmm_combo` is fitted on train only, `hmm_final_ms` on train + validation, so the rows are not
like for like. The cleaning and keeper fix change BAR-ATH by two possessions and nothing in the
agreement: as the HMM uses them, they hardly matter.

## `hmm_combo_final`: refitted on train + validation, test run once

`hmm_combo`'s method and settings refitted on the five non-test matches, seeds 0-2, alive models
cross-fitted over the same five (`possession/experiments/hmm_combo_final.py`).

| bench **test** (J03WPY, J03WR9 × seeds 0-2) | spellF1 | macroF1 | per match and seed |
|---|---|---|---|
| `hmm_final_ms` (submitted) | 0.497 | 0.860 | 0.504 0.491 / 0.493 0.491 / 0.468 0.535 |
| `hmm_conf_ms` (#33) | 0.503 | 0.860 | |
| **`hmm_combo_final`** | **0.515** | **0.870** | 0.513 0.507 / 0.501 0.517 / 0.500 0.549 |

Ahead of `hmm_final_ms` on all six match-and-seed pairs, by 0.008 to 0.032. On BAR-ATH: 353
possessions (408), median 4.64 s, dead share 40.6% (35.5%), Smart Tagging agreement **0.837**
(0.814).

## `hmm_combo_final_islands`: plus the dead-island rule, test run once

`hmm_combo_final`, then `hmm.drop_dead_islands` at 3 s (75 frames, set by hand in
`dead-islands.md`): play under 3 s between two stoppages becomes dead ball. Nothing refitted.

| | bench **test** spellF1 | macroF1 | BAR-ATH possessions | median | dead share | ST agreement |
|---|---|---|---|---|---|---|
| `hmm_combo_final` | **0.515** | **0.870** | 353 | 4.64 s | 40.6% | **0.837** |
| `hmm_combo_final_islands` | 0.512 | 0.869 | 327 | 5.08 s | 41.2% | 0.836 |

The rule costs 0.003 on test (0.000-0.004 per match and seed), as on train: the bench has few long
breaks, so it can only show the cost. On BAR-ATH it removes 26 short possessions, mostly inside
long breaks, without changing the agreement. A draw the bench cannot settle.

## Swapping the dead-ball evidence

`hmm_combo.make_features(alive=..., extra=(...))`: P(alive) can be switched off and other evidence
appended (each source: match -> (X, names)). Anything appended must exist on the bench matches too,
or the HMM cannot learn its weight.

## Verdict

- `hmm_combo_final`: the strongest candidate, on validation and on the single test run. Replacing
  the submission is #5's decision.
- The cleaned tracking stays in (it is right, and costs nothing), but is not where the gain is.
