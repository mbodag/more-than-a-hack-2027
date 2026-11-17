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

## Verdict

- `hmm_combo`: best on validation so far. To become a candidate it needs the `hmm_final` treatment
  (refit on train + validation, seeds 0-2) and one test run: not done.
- The cleaned tracking stays in (it is right, and costs nothing), but is not where the gain is.
