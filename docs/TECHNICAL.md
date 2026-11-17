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

- **IDSSE open Bundesliga data** (Bassek, Weber, Rein & Memmert, 2025, *Scientific Data*;
  CC BY 4.0; figshare): seven full matches of the 2022/23 Bundesliga and 2. Bundesliga, about
  1.0 million frames at 25 Hz. Labels are the DFL's own per-frame ball status and ball possession,
  produced by the provider's data collection, not by us. Split into four training, one validation
  and two test matches (ADR 0003). The tracking is degraded to look like the ATD before any
  method sees it: the ball is hidden in runs drawn from the ATD's own gap lengths, and the players
  are broken into ATD-like tracklets (degradation v2).
- Candidate `players_fill` (#22) fits two small logistic regressions on the training matches'
  hidden-ball frames (about 130,000 frames, every fifth): alive or dead, and which side has the
  ball, from side-level team-shape features. Their parameters are committed in
  `possession/models/players_fill.json`.

## Tracking that sees nobody, removed

Two faults of the ATD, both almost only in the keeper slots (Track 24 home, Track 46 away), each
confirmed by eye on candidate frames of BAR-ATH (`possession/clean.py`, `match.track_xy_clean`):

- **Held off camera.** The keeper slots go on being tracked when the camera cannot see the keeper:
  projected through the homography, their positions fall outside the picture on about half their
  graded frames (58% home, 47% away), against 0.01% for outfield slots. There they barely move
  (median 0.05 m a second, against 0.5-0.7 m on camera): the tracker holds a last guess. Every
  position outside the picture is removed.
- **Stuck.** A track can freeze where nobody is, also on camera: both keeper slots for the last
  minutes of the match, and the home one for minutes in the second half. Every stretch over which
  a slot moves less than 0.3 m (along x and along y) for 30 s is removed. On BAR-ATH no outfield
  slot is caught, even in stoppages where everyone stands still (they move 0.5-1.5 m).

On BAR-ATH's graded frames this removes 134,055 slot-frames off camera (140 of them outfield) and
27,821 stuck (all keeper slots): 5% of all tracked slot-frames.

## Keepers on the right side

Its metadata gives each side one keeper slot (Track 24 home, Track 46 away), and the two keepers'
tracklets land in either: a keeper standing in his own goal is often in the other side's slot,
sometimes while his own slot tracks him too. A slot's side is fixed, so the keeper counts for the
wrong team exactly when he has the ball (saves, goal kicks, build-up).

The rule (`possession/keepers.py`, `match.track_team_fixed`), on the cleaned tracking: each
tracklet of a keeper slot
belongs to the side defending the end it stands in (its median x, cut at the halves). Which side
defends which end is read at each half's kick-off, tagged by Smart Tagging, where every player is
in their own half: the lower median x of the outfield slots defends x = 0. Outfield slots are left
alone: only 3 outfield tracklets of 10 s or more live in the other side's keeper area.

Of the keeper slots' 132,798 graded slot-frames left after cleaning, 9,137 (7%) are given to the
other side (1,332 home slot, 7,805 away slot). With the cleaning, the nearest player's side changes
on 1,346 of the 94,214 frames with a ball. The team-shape features still use the raw tracking and
the metadata's sides, as their models were fitted on them.

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
