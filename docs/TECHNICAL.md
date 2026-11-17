# Technical document

The third deliverable ([SUBMISSION.md](../SUBMISSION.md)). Kept up to date as we go: every PR that
changes the method updates the relevant section. At submission it is copied to `<team_name>.md`.
Markdown, not PDF: `*.pdf` is gitignored in this repo.

## Approach

**Submitted method: `hmm_final_ms`** (`possession/experiments/north_london_tactics.py` runs it).
It is `hmm_final` (#26) with the ball's speed recomputed in m/s from its positions on every match
(see "Ball speed across providers"). A classifier per frame, decoded as a sequence. For every frame, two logistic regressions give the probability that play is alive and,
if so, that the home side has the ball; one pair for frames where the ATD tracks the ball and one
for frames where it does not. A 3-state hidden Markov model (H, A, D) turns these into emission
scores, and Viterbi finds the most likely sequence over each half, paying a penalty for every
change of state. So possession only changes when the evidence for it holds up over time: that is
what spellF1 rewards, where a majority vote leaves flicker and cannot say `D` while the ball is
seen.

The evidence comes from tracking only: each side's nearest player to the ball and the margin
between them, ball speed, the ball off the pitch, the team shape (#22), control evidence averaged
over ±0.5 s and ±2 s, how long the ball has been missing and which side was nearest either side of
the gap, and most of all **the side of the last control before the frame and of the next one
after it**, because possession stays with the side that last controlled the ball through passes
and tracking gaps. The whole match is known when it is labelled, so looking ahead is fair.

Models and settings are fitted on other matches with a full ground truth (IDSSE, ADR 0003), never
on BAR-ATH. On the bench's held-out test matches, run once: spellF1 0.497, macroF1 0.860
(`hmm_final` 0.479 / 0.838; organisers' baseline 0.176 / 0.444 on the same bench). What we tried
and dropped: `experiments/26-hmm.md`, `experiments/stoppages.md`,
`experiments/control-confidence-probe.md`.

## Assets used, and dropped

| Asset | Used? | How / why not |
|---|---|---|
| Tactical ATD (tracking) | yes | every feature of the submitted method: ball and player positions, the metadata's sides, the team shape |
| Smart Tagging | not in the method | used to find which side defends which end in the keeper rule, and as a label-free check (agreement with its in-possession phases: 0.814); never as labels |
| Homography | not in the method | used by the tracking cleaning (positions outside the video picture), which the submitted model does not read yet |
| Tactical video | no | watched to debug only; nothing written down from it |
| Broadcast video and ATD | no | not synchronised to the scored timeline |

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
- Experiment `stoppages` fits one more logistic regression on every fifth frame of the training
  matches, ball or not (about 340,000 frames): alive or dead. Its parameters are committed in
  `possession/models/stoppages.json`.

## Ball speed across providers

`match.ball_speed` is the speed each provider's file gives, and the units differ: the DFL's is
km/h (3.6 times the speed from its positions), the ATD's about 0.19 times it. A model fitted on the
bench therefore sees BAR-ATH's ball as all but still on every frame (BAR-ATH's 90th percentile,
3.4, is below the bench's in-play 10th, 6.9). `hmm_touch_ms` recomputes ball speed from the
ball's positions in m/s on every match; on validation it scores spellF1 0.473 against
`hmm_touch`'s 0.453 (within noise), and on BAR-ATH it gives 424 possessions and a 35% dead-ball
share against 399 and 39%.
Refitted the same way as `hmm_final` (train + validation), `hmm_final_ms` scores 0.497 / 0.860 on
the bench's test matches against `hmm_final`'s 0.479 / 0.838, and agrees more with Smart Tagging
on BAR-ATH (0.814 against 0.803).

The DFL's tracking also freezes the ball while it is out of play (speed exactly 0 on 99% of dead
frames with a ball, against 1% on BAR-ATH), so on the bench a still ball means a dead ball, which
BAR-ATH never shows. Dropping ball speed from `hmm_touch` costs 0.065 spellF1 on validation, but
alive/dead F1 hardly moves (0.914 to 0.912): the speed's value is in play (passes), not in
stoppages. With m/s speed the bench's alive/dead F1 rises to 0.937, part of which may be that
frozen ball; the bench cannot tell.

## Stoppages, and where the rest is

`stoppages` (`possession/stoppage.py`) calls a stoppage on every frame from the players alone:
how many are standing, how bunched they are, how many are tracked, the team shape, and whether
the ball is tracked and beyond the lines, each also averaged over 1 to 10 s; no ball speed. It
reaches validation spellF1 0.324 against `players_fill`'s 0.253, with the same alive/dead F1 as
`hmm_touch` (0.912), so the HMM already finds stoppages as well as a dedicated model does.

Two bench-only oracles bound what is left for `hmm_touch`: true dead balls with its sides score
0.518 (+0.065), its dead balls with true sides 0.847 (+0.39). The remaining gains are in changes
of possession in open play, not in dead balls (`experiments/stoppages.md`).

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

We do not hand-write the rules: the models learn the DFL's own per-frame definitions from the
bench (ball status and ball possession), so where the line falls is the DFL's. What our decoding
adds on top:

- **Possession sticks with the side that last controlled the ball**, through passes in flight,
  loose balls and tracking gaps, until the evidence for the other side holds up; a deflection or
  a challenge that does not lead to a control does not change possession.
- **Every change of state costs** (the HMM's switch penalty), so brief touches and contested balls
  do not split a possession into flickers.
- **A stoppage lasts at least 4 s** (`dead_min` 100 frames): shorter dead stretches are absorbed
  into the possession around them. Restarts start a new possession for whichever side the evidence
  then favours.
- **We never abstain**: every graded frame gets `H`, `A` or `D`; any `X` inside a half is filled
  from the labels around it, never across halftime.

## Limitations

- **Open-play changes of possession are the main loss.** On the bench, giving the model the true
  sides would lift spellF1 from 0.453 to 0.847; giving it the true dead balls only to 0.518. Short
  possessions (most are under 10 s) are often missed or merged.
- **The labels are the DFL's, on Bundesliga matches.** If the organisers' ground truth draws the
  line differently (loose balls, restarts), we inherit the difference.
- **The bench only imitates the ATD.** The DFL's tracking is degraded to look like the ATD (ball
  gaps, broken tracklets), but its other habits differ: the DFL freezes the ball out of play, and
  the providers report ball speed in different units, which we found and fixed late.
- **The submitted model reads the raw tracking.** The cleaning (keepers held off camera, stuck
  tracks) and the keeper side fix are built but not yet fed to the HMM, so a keeper in the wrong
  slot can still count for the wrong side.
- With more time: control confidence and the stoppage model as HMM evidence on the cleaned
  tracking (`hmm_combo`), and evidence aimed at open-play turnovers.

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
