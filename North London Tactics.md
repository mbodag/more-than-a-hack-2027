# North London Tactics: who has the ball, frame by frame

Technical document for More than a Hack 2027, Phase 1 (Barcelona v Athletic Club). The submitted
file is made by `python -m possession run north_london_tactics`, which runs the experiment
`hmm_setpiece_islands`.

## The approach

We treat possession as a sequence problem rather than a per-frame one. The ranking metric scores
whole possessions, so a method that is right on most frames but flickers between teams for a frame
or two still loses badly. Early on we saw that the organisers' baseline (nearest player, then a
majority vote) gets reasonable frame accuracy and very few possessions right, and that told us
where to put the effort: in deciding *when* possession changes, not just who is closest to the ball.

What we built has four layers.

**1. Evidence per frame.** For every frame, two logistic regressions give the probability that the
ball is in play and, if it is, the probability that Barcelona (home) has it. There is one pair for
frames where the tracking has the ball and one for frames where it doesn't, because those are
really two different problems: with the ball we can look at who is near it; without it (41% of
graded frames) we have to read the players. The features come from the tracking only:

- each team's nearest player to the ball, and the margin between them;
- the ball's speed, and whether it is beyond a line;
- the team shape: where each side's players are and how they are spread;
- control evidence (a player within 2 m of the ball and clearly closer than any opponent),
  averaged over half a second and two seconds;
- how long the ball has been missing, and which side was nearest at each end of the gap;
- most importantly, **which side last controlled the ball before this frame and which side next
  controls it after**, and how long ago and how soon. Possession stays with the side that last
  had the ball through passes, loose balls and tracking gaps, so this turned out to be the
  strongest single cue. The whole match is available when we label it, so looking ahead is fair.

**2. A hidden Markov model over three states, H, A and D.** The probabilities above become emission
scores, and Viterbi decoding finds the most likely sequence for each half on its own. Every change of
state has a cost, learnt from how often possession changes and how often play stops on the
training matches. So a single frame where an Athletic player is nearest the ball doesn't hand them
possession; the evidence has to hold up for a while. We also make every stoppage last at least
4 seconds (100 frames), because short false stoppages were splitting real possessions in two.

**3. Set pieces from Smart Tagging.** We added this after watching our output in the match viewer.
The pattern we saw: while a set piece is being set up, the ball is often not tracked, so the model
has little to go on and can fill the stoppage with short, made-up possessions. And when a corner or
free kick is delivered into the box, the nearest players to the ball are often the defenders, so the
restart can go to the wrong side. Smart Tagging already marks every set piece, its type and, apart
from throw-ins, the team taking it. So, for each `SET PIECES` tag:

- we place the restart where the ball reappears (the end of the nearest ball gap of at least
  2 s, within 10 s of the tag start), or at the tag start if there is no such gap;
- from the start of that gap up to the restart (at most 30 s), the `D` state gets a bonus;
- for 5 s after the restart, the team named in the tag gets a bonus. Throw-ins get no team bonus,
  as they're all tagged `N/A`.

These are bonuses added to the scores before decoding, not overrides. If the tracking clearly
says otherwise, it still wins. We did it this way on purpose: Smart Tagging is automatic and has
its own mistakes, and its intervals are phases of play, not possessions, so copying them across
as labels would bring their errors with them.

**4. No possession shorter than 3 seconds between two stoppages.** In long breaks (drinks breaks,
injuries, VAR checks) players wander about while the ball isn't tracked, and the decoder sometimes
hands one side the ball for a second or two. On the ranking metric each of those is a whole false
possession. So after decoding, any stretch of play under 3 s with a stoppage directly before and
after it becomes dead ball.

**Why this rather than something else.** We tried several richer versions of the per-frame layer: a
small neural network instead of logistic regression, separate weights for hidden-ball frames, a
rule for who restarts after a stoppage, windowed team movement, hand-built feature combinations,
and a dedicated stoppage model. None beat this one by more than the noise on our validation match,
so we kept the simplest. The HMM itself was the big step: on our bench it took spellF1 from 0.18
for the smoothed baseline to 0.34, and the last-touch features took it to 0.45.

## Assets: what we used and what we dropped

| Asset | Used? | How |
|---|---|---|
| Tactical ATD (tracking) | yes | every per-frame feature: ball and player positions, the sides from the metadata, the team shape |
| Smart Tagging | yes | the `SET PIECES` tags (type and team) as evidence in the decoder; its in-possession phases only as a label-free check on our output, never as labels |
| Homography | no | used to build a cleaned version of the tracking (removing positions held off camera, mostly the keepers), which the submitted model doesn't use |
| Tactical video | no | watched in our viewer to debug and to look for failure patterns; nothing was written down from it as a label, and no setting is tied to any frame of the match |
| Broadcast video and ATD | no | not synchronised to the scored timeline |

We also built, and didn't use in the end: a rule putting the keepers on the right side (their
tracklets often land in the other team's keeper slot), a players-only stoppage detector, a
"control confidence" feature set, and ball speed recomputed in m/s (see the limitations). Each was
within noise on our bench on its own.

## Training data

We used one labelled dataset, and nothing from the provided match.

**IDSSE open Bundesliga data** (Bassek, Weber, Rein & Memmert, 2025, *Scientific Data*, CC BY 4.0,
on figshare). Seven full matches from the 2022/23 Bundesliga and 2. Bundesliga, about a million
frames at 25 Hz. The labels are the DFL's own per-frame ball status (in or out of play) and ball
possession (home or away), made by the league's data provider, not by us.

We fixed the split before looking at any scores: four matches for training, one for validation
and two for testing. Settings were chosen on training, methods compared on validation, and the test
matches scored once per candidate. The final models are refitted on the training and validation
matches together.

The DFL tracking is much cleaner than the ATD, so before any method sees it we degrade it to look
like the ATD. We hide the ball in runs drawn from the ATD's own gap lengths, and break the players
into short tracklets landing in random slots, as the ATD does. Three random seeds of this.

Layers 3 and 4 have **no training data**. The bench matches have no Smart Tagging, so the
set-piece windows and bonus sizes (2 s, 10 s, 30 s, 5 s, and a bonus of 1.0 per frame) were set by
hand, once, before the first run, and not adjusted afterwards. The 3-second threshold was also set
by hand: on the training matches every setting from off to 6 s scored within 0.01 of the others,
and they have far fewer long breaks than this match.

## The possession rules we implemented

We didn't hand-write the definitions of possession and dead ball. The models learn the DFL's
per-frame definitions from the training matches, so where the line falls on a loose ball or a
challenge is, in the first place, the DFL's. On top of that, the decoder adds:

- **Loose balls and challenges:** possession stays with the side that last controlled the ball
  until the other side's evidence holds up. A deflection, a block or a 50/50 that doesn't end in a
  control by the other team doesn't change possession.
- **Flicker:** every change of state costs something, so brief touches and contested balls don't
  break a possession into pieces.
- **Stoppages:** at least 4 seconds. Anything shorter is absorbed into the possession around it.
- **Restarts:** a set piece is a stoppage up to the moment the ball is played again, and the team
  taking it is favoured for the first 5 seconds after. Throw-ins have no team in Smart Tagging, so
  there the tracking alone decides who has the ball after the restart.
- **Breaks in play:** a possession can't be shorter than 3 seconds if there's a stoppage on both
  sides of it.
- **Abstaining:** we never do. Every graded frame gets `H`, `A` or `D`.

## Results

| On our bench's two test matches | spellF1 | macroF1 |
|---|---|---|
| Organisers' baseline | 0.176 | 0.444 |
| Layers 1–2 alone | 0.479 | 0.838 |
| The submitted method (scored once) | 0.479 | 0.837 |

The bench has no Smart Tagging, so there the set-piece layer does nothing, and the 3-second rule
makes no measurable difference either: the bench matches have few long breaks. So the bench says
the extra layers do no harm, and nothing about whether they help on this match. On Barcelona v Athletic we can only check things that
don't need labels:

| | layers 1–2 | + set pieces | submitted (+ 3 s rule) |
|---|---|---|---|
| possessions | 384 | 377 | 349 |
| share of frames dead | 40.3% | 41.7% | 42.4% |
| possessions under 1 s | 8.1% | 8.5% | 8.0% |
| agreement with Smart Tagging's in-possession phases | 0.803 | 0.815 | 0.815 |

The 3-second rule removes 28 possessions, all of them short stretches of play between two
stoppages.

The agreement check uses Smart Tagging's build-up and attacking phases, not the set pieces, so the
set-piece layer isn't marking its own homework there.

## Limitations

- **The set-piece layer is unmeasured.** We have no labelled match with Smart Tagging, so we can't
  say how much it helps, only that it moves the label-free checks in the right direction. A stand-in
  version of Smart Tagging built from the bench matches' own set-piece events would let us fit and
  test it properly. We got as far as building one, but not as far as running this method on it.
- **Restart timing is rough.** Smart Tagging's set-piece windows are long (around 35 s) and start a
  few seconds before the ball is played. When we can't see the ball reappear, we fall back to the
  tag start, which puts the restart a few seconds early.
- **Throw-ins carry no team**, and they are the most common restart.
- **The 3-second rule has a cost.** Real possessions that short, between two stoppages, do happen
  (36 of 405 such stretches on our training matches were under 4 s), and the rule removes them.
- **Ball speed is in the wrong units.** The two tracking providers report ball speed differently:
  the Bundesliga data we train on gives km/h, and this match's ATD gives roughly 0.19 × m/s. So our
  model sees this match's ball as moving much more slowly than it really is. We found this late. A
  version that recomputes speed from the ball's positions scored 0.497 on our bench test against
  0.479, which is within the bench's noise. We submitted the version we had examined most closely.
- **Open-play turnovers are where most of the remaining error is.** On the bench, giving our model
  the true dead balls would lift spellF1 by about 0.07; giving it the true sides would lift it by
  about 0.39. Short possessions, and most are under 10 s, are often missed or merged.
- **The labels are the DFL's.** If the organisers draw the line differently on loose balls or
  restarts, we inherit the difference.
- **The bench only imitates the ATD.** We copy its ball gaps and broken tracks, but not every habit.
- **The model reads the raw tracking.** The cleaned tracking and keeper fix exist but aren't fed in,
  so a keeper sitting in the wrong team's slot can still count for the wrong side.

With more time, we'd fix the ball-speed units in the submitted model, fit the set-piece layer on the
stand-in Smart Tagging rather than guess its settings, add the stoppage model and control-confidence
features (which together scored best on validation, 0.495), and go after open-play turnovers
directly.

## Reproduction

```bash
pip install -r requirements.txt
python scripts/check_data.py
python -m possession run north_london_tactics
python evaluation/validate_submission.py "North London Tactics.csv"
```

`check_data.py` confirms the match data matches the committed checksums. The run then checks the
data again, builds the features, decodes, writes `North London Tactics.csv` and validates it with
the organisers' own format check. The fitted models are committed in `possession/models/`, so
nothing is retrained. To refit them from the open data, run `python scripts/fetch_idsse.py`, then
`python -m possession.experiments.hmm_final`.
