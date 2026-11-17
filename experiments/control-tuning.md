# Control confidence tuned on the bench, and `hmm_conf_ms`

- **Issue**: none yet (follows `control-confidence-probe.md` and #31's `hmm_touch_ms`)
- **Method**:
  1. The ten scales of the control confidence (`possession/controls.py` `HAND`: where each rule
     score is 0 and where it is 1) tuned on the **training matches** (degradation v2, seeds 0-2) by
     random search (400 samples) and local refinement (300 steps). Target: ROC AUC of the
     confidence for "the DFL's possession just after the control is the controller's side" (live
     play only). Saved to `possession/models/control_params.json`.
  2. `hmm_conf_ms`: `hmm_touch_ms` (ball speed in m/s, best on validation) plus `hmm_conf`'s
     control features, with the tuned scales. `hmm_conf_ms_hand`: the same with the hand-set ones.
     Only `w` searched; the rest at `hmm_touch`'s settings.
- **Commands**: `python scripts/tune_controls.py` (12 s);
  `python -m possession.bench train hmm_conf_ms` (and `hmm_conf_ms_hand`);
  `python -m possession.bench test hmm_conf_ms`, `python -m possession.bench test hmm_touch_ms`
- **Metrics**: `experiments/metrics/control_tuning.json`, `experiments/metrics/bench/hmm_conf_ms*.json`,
  `experiments/metrics/bench/hmm_touch_ms.json` (adds `test`)

## 1. Tuning the confidence

| | train AUC | **val AUC** | val precision at ≥ 0.5 (share kept) | at ≥ 0.9 |
|---|---|---|---|---|
| hand-set | 0.761 | 0.748 | 0.803 (61%) | 0.895 (41%) |
| tuned | 0.801 | **0.798** | 0.900 (44%) | 0.935 (28%) |

The gain holds on validation (+0.050 AUC against +0.040 on train). What the search changed:

| score | hand-set (0 → 1) | tuned (0 → 1) | reading |
|---|---|---|---|
| close | 2.5 → 0.5 m | 1.2 → 0.7 m | much stricter: beyond 1.2 m is not a control |
| clear of opponents | -0.5 → +1.5 m | -1.0 → +4.0 m* | graded over a far wider margin |
| kept it | 1 → 10 frames | 1.3 → 3.8 frames | a few frames are enough |
| moving together | 4 → 1 m/s | 7.6 → 0.1 m/s | graded over a far wider range |
| kicked | 2 → 8 m/s | 0 → 17 m/s | graded over a far wider range |

\* at the edge of the search range.

## 2. In the HMM: validation (J03WOY, degradation v2)

| | train | seed 0 | seed 1 | seed 2 | **val spellF1** | macroF1 |
|---|---|---|---|---|---|---|
| `hmm_touch_ms` (reference) | 0.493 | 0.481 | 0.487 | 0.451 | 0.473 | 0.838 |
| `hmm_conf_ms_hand` | 0.511 | 0.477 | 0.502 | 0.484 | 0.488 | 0.841 |
| **`hmm_conf_ms`** (tuned) | 0.505 | 0.489 | 0.490 | 0.473 | **0.484** | 0.840 |

Both beat the reference on validation; the tuned one on every seed, the hand-set one on two of
three. Tuning the confidence did **not** help the HMM (0.484 against 0.488): its logistic models
re-weigh the features anyway, so a better-ranked score mostly repeats what they already learn.

## 3. Test (J03WPY, J03WR9), run once each

| | J03WPY s0 | J03WR9 s0 | J03WPY s1 | J03WR9 s1 | J03WPY s2 | J03WR9 s2 | **spellF1** | macroF1 |
|---|---|---|---|---|---|---|---|---|
| `hmm_touch_ms` | 0.506 | 0.498 | 0.487 | 0.500 | 0.461 | 0.534 | 0.498 | 0.861 |
| `hmm_conf_ms` | 0.486 | 0.524 | 0.499 | 0.515 | 0.454 | 0.537 | **0.503** | 0.860 |

+0.005 on test, better on 4 of 6 match-seeds; macroF1 unchanged. `hmm_conf_ms` predicts fewer
possessions (about 262 against 298 per match; the truth has 396), so it splits less.
`hmm_touch_ms` was run on test here for the first time, as the like-for-like reference
(`hmm_final`'s 0.479 is `hmm_touch` refitted on train + validation, without the m/s fix).

## Verdict

**The confidence tuning works as a control detector** (validation AUC 0.75 → 0.80) and is kept
as `control_params.json`. **In the HMM it is a small, consistent-looking gain within noise**:
+0.011 on validation, +0.005 on test, against `hmm_touch_ms`. Not enough on its own to replace
`hmm_touch_ms` as the candidate. If the submission is refitted with the m/s fix anyway (#31's
note), adding these features costs nothing and has not hurt anywhere it was measured.

Ideas left: tune the scales against the HMM's spellF1 directly instead of AUC; it under-predicts
possessions, so a smaller `switch` penalty may suit it.
