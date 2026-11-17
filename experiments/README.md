# Experiments

There is no leaderboard, so this folder is our record of what we tried and how it looked.

## One file per experiment

`experiments/<issue>-<slug>.md`, e.g. `experiments/12-hysteresis-control.md`. Separate files mean
parallel sessions never edit the same file. Each record has:

- **Issue and commit**: the issue number and the commit that produced the result, so it can be
  regenerated.
- **Method**: what changed against the previous best, in a few lines.
- **Command**: the exact command that produced the CSV.
- **Proxy metrics**: the label-free checks (see [CONTEXT.md](../CONTEXT.md#scoring)), e.g. number of
  spells, spell-length distribution, flips per minute, dead-ball share, agreement with Smart
  Tagging.
- **Validation**: whether `evaluation/validate_submission.py` passed.
- **Verdict**: kept, dropped or superseded, and why.

## What gets committed

- These Markdown records, and `experiments/metrics/<experiment>.json`, the proxy report the
  runner writes on every run (`python -m possession run <experiment>`, step 7). Its thresholds are
  provisional until #5; a warning in it is a prompt to look, never a verdict.
- **Never** experiment CSVs (they go in the gitignored `outputs/`). Only the final
  `<team_name>.csv` is committed.
- **Never** a score against the public 5-minute ground truth used to choose between experiments.
  That is a pipeline sanity check only (RULES.md).

## Summary

| Issue | Experiment | Key proxy metrics | Verdict |
|---|---|---|---|
| — | `baseline_smooth50` (organisers') | published full-match spellF1 0.2433 | the floor to beat |
| #20 | `nearest_smooth` (the same baseline, window from the bench) | bench val spellF1 0.184, macroF1 0.455 (degradation v1); 0.176 / 0.444 (v2) | the bench's reference point |
| #22 | `players_fill` (ball gaps read from the team shape) | bench val spellF1 0.253, macroF1 0.521 (v2); BAR-ATH: Smart Tagging agreement 0.727 vs 0.714, sub-second possessions 15% vs 21% | candidate; the bench overstates its gain on BAR-ATH |
| — | `players_fill_keepers` (the same, on cleaned tracking with keepers put on the right side) | bench: identical to `players_fill` (its sides are already right); BAR-ATH: Smart Tagging agreement 0.734 vs 0.727, possessions 535 vs 560, sub-second possessions 15.1% vs 14.6% | candidate; see `players-fill-keepers.md` |
| #26 | `hmm_viterbi` (3-state HMM, Viterbi per half) | bench val spellF1 0.335, macroF1 0.774 (v2) | superseded by `hmm_context` |
| #26 | `hmm_context` (+ windowed control evidence, gap context) | bench val 0.374 / 0.790 | superseded by `hmm_touch` |
| #26 | `hmm_touch` (+ side of the last and next control) | bench val 0.453 / 0.815 | the method; later variants within noise |
| #26 | `hmm_final` (`hmm_touch` refitted on train + val) | bench **test** 0.479 / 0.838 | candidate; BAR-ATH proxies pending |
| #24 | `stoppages` (dead balls from the players on every frame) | bench val 0.324 / 0.774; alive/dead F1 0.912, as `hmm_touch` | dropped as a method; see `stoppages.md` |
| — | `oracle_stoppages`, `oracle_team` (bench only) | on `hmm_touch`: true dead balls 0.518, true sides 0.847 | the gains left are in open-play changes of possession |
| — | `hmm_touch_ms` (ball speed from positions, m/s, on every match) | bench val 0.473 / 0.838; BAR-ATH 424 possessions, ST agreement 0.808 | fixes a unit mismatch; see `hmm_final_ms` |
| — | `hmm_final_ms` (`hmm_final` refitted with m/s ball speed) | bench **test** 0.497 / 0.860 vs 0.479 / 0.838; BAR-ATH: ST agreement 0.814 vs 0.803, possessions 408 vs 384 | recommended candidate in place of `hmm_final` |
| — | control confidence probe (`possession/controls.py`, not an experiment) | train, clean: confidence ≥ 0.4 precision 0.763 vs 0.717 for the 2 m rule at the same side recall (0.88); ≥ 0.9 reaches 0.881 | fed to the HMM as `hmm_conf`; see `control-confidence-probe.md` |
| — | `hmm_conf` (`hmm_touch` + confidence-weighted last/next control, pass-bys) | bench val 0.460 / 0.817 vs 0.453 / 0.815; train 0.478 vs 0.464; seed 0 worse | not kept: within noise |
| — | control confidence tuned on train (`scripts/tune_controls.py`) | val AUC 0.798 vs 0.748 hand-set | kept as `possession/models/control_params.json`; see `control-tuning.md` |
| — | `hmm_conf_ms` (`hmm_touch_ms` + tuned control features) | bench val 0.484 / 0.840 vs 0.473 / 0.838; **test 0.503 / 0.860 vs 0.498 / 0.861** (`hmm_touch_ms`, first test run) | small gain within noise; free to add if `hmm_final` is refitted |
| — | `hmm_conf_ms_hand` (the same, hand-set scales) | bench val 0.488 / 0.841 | tuning the scales did not help the HMM |
| — | `hmm_combo` (`hmm_touch_ms` + control confidence + cross-fitted P(alive) + cleaned tracking) | bench val 0.495 / 0.848 vs 0.473 / 0.838; ablations 0.487 (no conf), 0.490 (no alive); BAR-ATH ST agreement 0.832 | best on validation; needs refit + one test run; see `hmm-combo.md` |
| — | `hmm_combo_final` (`hmm_combo` refitted on train + val) | bench **test** 0.515 / 0.870 vs `hmm_final_ms` 0.497 / 0.860, ahead on all 6 match-seed pairs; BAR-ATH ST agreement 0.837 vs 0.814 | strongest candidate; see `hmm-combo.md` |
| — | `hmm_combo_final_islands` (`hmm_combo_final` + no play under 3 s between stoppages) | bench **test** 0.512 / 0.869 vs 0.515 / 0.870; BAR-ATH 327 possessions vs 353, ST agreement 0.836 vs 0.837 | a draw: the bench has few long breaks; see `hmm-combo.md` |
| — | `hmm_combo_islands` (`hmm_combo`, play under 3 s between two stoppages becomes dead ball) | bench val 0.496 / 0.848 vs 0.495 / 0.848; train 0.508 vs 0.511; BAR-ATH possessions 335 vs 362, ST agreement 0.832 unchanged | kept at 3 s by hand; see `dead-islands.md` |
