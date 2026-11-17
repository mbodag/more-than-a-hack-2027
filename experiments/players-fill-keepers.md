# `players_fill_keepers`: cleaned tracking, keepers put on the right side

- **Issue**: none yet (built on the `22-players-fill` branch)
- **Method**: `players_fill`, with the nearest player looked for in `match.track_xy_clean`
  (positions held off camera or stuck in place removed, `possession/clean.py`) and his side read
  from `match.track_team_fixed` instead of `match.track_team`. Each tracklet of a keeper slot (Track 24 home, Track 46 away in
  the ATD's metadata) is given to the side defending the end it stands in; ends come from each
  half's kick-off in Smart Tagging (`possession/keepers.py`). Same models, same settings.
- **Command**: `python -m possession run players_fill_keepers` (BAR-ATH)
- **Metrics**: `experiments/metrics/players_fill_keepers.json`
- **Validation**: the runner's official format check passed.

## What the rules change on BAR-ATH (graded frames)

| | First version (keepers only) | With cleaning |
|---|---|---|
| Slot-frames removed off camera | — | 134,055 (140 outfield) |
| Slot-frames removed as stuck | — | 27,821 (keeper slots only) |
| Keeper-slot frames given the other side | 17,405 | 9,137 of the 132,798 left |
| Frames whose nearest player changes side | 1,306 | 1,346 of 94,214 with a ball |
| Outfield tracklets of 10 s+ in the other side's keeper area | 3 (left alone) | 3 (left alone) |

The cleaning's candidates (frozen keeper tracks at the end of the match, keepers held off camera,
and an outfield stoppage as a control) were checked by eye in the viewer before the thresholds
were set.

## Proxy metrics

| | `players_fill` | keepers only | with cleaning |
|---|---|---|---|
| possessions | 560 | 537 | 535 |
| median possession | 5.24 s | 5.28 s | 5.36 s |
| sub-second possessions | 14.6% | 15.1% | 15.1% |
| home share | 0.568 | 0.575 | 0.574 |
| dead-ball share | 25.9% | 25.9% | 25.9% |
| agreement with Smart Tagging | 0.727 | 0.734 | 0.734 |

## What the bench can and cannot say

The bench's matches have the provider's own sides, keepers included, and no keeper slots, so
`track_team_fixed` is `track_team` there; its tracking has nothing held or stuck, so the cleaned
tracking is the raw one too. The two experiments score the same there. The rule's effect
is measured on BAR-ATH only, by label-free proxies.

## Verdict

Candidate. 25 fewer possessions and more agreement with Smart Tagging, with the dead-ball share
unchanged; sub-second possessions are slightly up. The cleaning itself changes little on the
labels (2 possessions): its main value is cleaner keeper tracks for whatever reads them next.
