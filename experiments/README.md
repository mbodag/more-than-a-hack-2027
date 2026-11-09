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

- These Markdown records, and any small metric files the experiment runner generates.
- **Never** experiment CSVs (they go in the gitignored `outputs/`). Only the final
  `<team_name>.csv` is committed.
- **Never** a score against the public 5-minute ground truth used to choose between experiments.
  That is a pipeline sanity check only (RULES.md).

## Summary

| Issue | Experiment | Key proxy metrics | Verdict |
|---|---|---|---|
| — | `baseline_smooth50` (organisers') | published full-match spellF1 0.2433 | the floor to beat |
