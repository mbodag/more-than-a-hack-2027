# CONTEXT.md: glossary

The words we use for this challenge, defined once. When the organiser docs and this file disagree,
the organiser docs win ([CHALLENGE.md](CHALLENGE.md), [EVALUATION.md](EVALUATION.md)). When they
and the public ground truth disagree, the ground truth wins. Add a term here in the same PR that
starts using it.

## The task

- **Frame**: one frame of the *tactical* video, 25 fps. The submission has one row per frame,
  `0…199318`. Every timeline we use must be converted to this one.
- **Graded frame**: a frame the ground truth covers (about 158,857). Before kickoff, halftime and
  after the final whistle are not graded.
- **Home / away**: Barça (home, `H`) and Athletic (away, `A`) in `BAR-ATH`.
- **Label**: `H`, `A`, `D` or `X` for one frame.

## Possession

- **Ball in play / alive**: the referee allows play. While alive, exactly one team is in
  possession.
- **Dead ball (`D`)**: the ball is not in play. It is a stoppage, never a possession, and we never
  say who "owns" it.
- **Possession**: the team in control while the ball is alive. In contested or loose phases it
  stays with the last team that had it.
- **Controlled possession**: the opposing team truly gains the ball. Only this changes possession.
  A touch without control, a deflection that doesn't intercept, or an aerial duel won without
  follow-up control does **not**.
- **Change of possession / hand-over**: the frame where possession passes to the other team.
- **Spell / possession spell**: a maximal run of one team (`H` or `A`) with the ball alive.
  `H H H A A` is two spells; `H H D D H H` is two spells with a stoppage between.
- **Restart**: play resuming after a dead ball. Always has a clear possessor.
- **Abstain (`X`)**: we decline to guess. On ungraded frames it's free. Under `spellF1` it is
  filled with our last label in that half, so it's free inside a spell but costly across a
  hand-over. Under `macroF1` it always counts as a miss.

## Scoring

- **spellF1**: the ranking metric. Predicted and true spells are matched one-to-one by temporal
  IoU (intersection over union in time); F1 is averaged over IoU thresholds 0.3 to 0.7.
  Differences of about 0.04 are noise.
- **macroF1**: per-frame F1 averaged over `H`, `A`, `D`. The first tie-break.
- **Public ground truth**: the first 5 minutes (17 spells) in `evaluation/gt/`. For checking the
  pipeline and understanding the labelling, **never for tuning or training**.
- **Private ground truth**: the full-match truth we are scored on. Nobody has it.
- **Proxy metric**: a label-free check on a prediction, e.g. number of spells, spell-length
  distribution, flips per minute, dead-ball share, agreement with Smart Tagging. How we judge
  changes.

## Inputs

- **ATD**: the automatic tracking data (`*_atd.txt` plus `*_atd.xml` metadata): per frame, x/y and
  speed for tracked players and the ball. Uncorrected: gaps, fragmented tracks, identity switches,
  no player IDs, ball missing on 41% of graded frames.
- **Track**: one tracked object's sequence of positions. Not a player identity: tracks break and
  swap.
- **Smart Tagging**: automatically inferred tactical phases and set pieces (638 events). Inferred,
  so noisy.
- **Homography**: one 3×3 matrix per frame mapping camera pixels to pitch coordinates.
- **Tactical / broadcast**: the two recordings. Only tactical is aligned to the submission; we
  ignore broadcast for now.
