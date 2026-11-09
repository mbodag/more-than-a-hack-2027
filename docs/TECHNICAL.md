# Technical document

The third deliverable ([SUBMISSION.md](../SUBMISSION.md)). Kept up to date as we go: every PR that
changes the method updates the relevant section. At submission it is copied to `<team_name>.md`.
Markdown, not PDF: `*.pdf` is gitignored in this repo.

## Approach

_What we built, and why that rather than something else. To be filled once #5 decides the method._

## Assets used, and dropped

| Asset | Used? | How / why not |
|---|---|---|
| Tactical ATD (tracking) | | |
| Smart Tagging | | |
| Homography | | |
| Tactical video | | |
| Broadcast video and ATD | no (so far) | not synchronised to the scored timeline |

## Training data

_Every labelled dataset used: source, who labelled it, how much. None from the provided game.
Write "none" if the method uses no labels._

## Possession rules implemented

_Where we drew the line on loose balls, challenges, deflections, aerial duels and restarts. Use the
terms in [CONTEXT.md](../CONTEXT.md)._

## Limitations

_Where it fails, and what we would fix with more time._

## Reproduction

```bash
pip install -r requirements.txt                            # pinned versions
python scripts/check_data.py                               # the data matches the committed checksums (#2)
python -m possession run north_london_tactics              # regenerates "North London Tactics.csv"
python evaluation/validate_submission.py "North London Tactics.csv"
```

`python -m possession run` checks the data itself and stops if it fails, runs the chosen
experiment, fills each `X` inside a half from the labels around it (never across halftime), writes
the CSV and checks it as read back from disk with the official format check (ADR 0002). It then
prints a label-free proxy report for the graded halves (possessions, their lengths, label changes
per minute, dead-ball and home shares, agreement with Smart Tagging's in-possession phases), with
warnings for values outside provisional ranges, and saves it to
`experiments/metrics/<experiment>.json`. For the submission it also runs the organisers'
`evaluation/validate_submission.py` on the file. Only the format checks can fail a run. Add `--rebuild-cache` to rebuild every
prepared input item instead of loading it from `outputs/cache/`; the result must be byte-identical
either way.

## Pre-submission checklist

- [ ] A fresh clone plus the data bundle regenerates `<team_name>.csv` byte for byte.
- [ ] `validate_submission.py` passes on the exact file being uploaded.
- [ ] The public 5-minute score looks sane (pipeline check only; never used to choose).
- [ ] No data, video or ground truth is tracked in git.
- [ ] Every section above is filled in and matches the code.
- [ ] CSV and document named after the team.
