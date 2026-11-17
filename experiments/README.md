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
| — | `players_fill_keepers` (the same, keepers put on the right side) | bench: identical to `players_fill` (its sides are already right); BAR-ATH: Smart Tagging agreement 0.734 vs 0.727, possessions 537 vs 560, sub-second possessions 15.1% vs 14.6% | candidate; see `players-fill-keepers.md` |
| #26 | `hmm_viterbi` (3-state HMM, Viterbi per half) | bench val spellF1 0.335, macroF1 0.774 (v2) | superseded by `hmm_context` |
| #26 | `hmm_context` (+ windowed control evidence, gap context) | bench val 0.374 / 0.790 | superseded by `hmm_touch` |
| #26 | `hmm_touch` (+ side of the last and next control) | bench val 0.453 / 0.815 | the method; later variants within noise |
| #26 | `hmm_final` (`hmm_touch` refitted on train + val) | bench **test** 0.479 / 0.838 | candidate; BAR-ATH proxies pending |
| #24 | `stoppages` (dead balls from the players on every frame) | bench val 0.324 / 0.774; alive/dead F1 0.912, as `hmm_touch` | dropped as a method; see `stoppages.md` |
| — | `oracle_stoppages`, `oracle_team` (bench only) | on `hmm_touch`: true dead balls 0.518, true sides 0.847 | the gains left are in open-play changes of possession |
| — | `hmm_touch_ms` (ball speed from positions, m/s, on every match) | bench val 0.473 / 0.838; BAR-ATH 424 possessions, ST agreement 0.808 | fixes a unit mismatch; `hmm_final` should be refitted with it |
