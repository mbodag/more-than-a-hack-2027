# Stoppages, oracles, and the ball's speed across providers

- **Issue**: relates to #24 (dead-ball detection); does not close it
- **Experiments**: `stoppages`, `oracle_stoppages`, `oracle_team`, `hmm_touch_nospeed`,
  `hmm_touch_noball`, `hmm_touch_ms`
- **Commands**: `python -m possession.bench train <experiment>` (the oracles: `val`),
  `python -m possession run <experiment>` for BAR-ATH
- **Metrics**: `experiments/metrics/bench/<experiment>.json`, `experiments/metrics/<experiment>.json`
- **Validation**: the runner's official format check passed for every BAR-ATH run.

All bench numbers: validation match J03WOY, degradation v2, seeds 0 1 2. Noise on spellF1 is about
±0.04.

## 1. `stoppages`: dead balls from the players, on every frame

One logistic model (`possession/stoppage.py`, every 5th frame of the training matches) gives
P(play is on) from the team shape plus how many players are standing (< 1 m/s), how bunched they
are (most players within 5 m of one; nearest-neighbour distance), how many are tracked, whether
the ball is tracked and beyond the lines, each also averaged over 1, 3 and 10 s and the 3 s before
and after. Stoppages: below 0.5, at least 3 s. The side (nearest player, players_fill's model in
ball gaps) is voted within each live stretch on its own. The grid scored 0.306–0.336 on train
whatever the settings.

| | spellF1 | macroF1 | alive/dead F1 | possessions (truth 456) |
|---|---|---|---|---|
| `players_fill` | 0.253 | 0.521 | 0.746 | 389–420 |
| `stoppages` | 0.324 | 0.774 | 0.912 | 352–379 |
| `hmm_touch` | 0.453 | 0.815 | 0.914 | 341–355 |

`stoppages` matches #24's `players_dead_ball` (0.327 on v2, on the `22-players-fill` branch) and
`hmm_touch`'s alive/dead F1: the HMM already finds stoppages as well as a dedicated model.

BAR-ATH: 408 possessions, dead-ball share 46% (bench matches 41–52%), Smart Tagging agreement
0.769. 55 of the 57 Smart Tagging set pieces overlap a detected stoppage; their windows are fixed
35 s and our stoppages end 3 to 18 s into them, so they cannot time restarts more finely.

## 2. Oracles: where the rest is

Bench only (they read the truth). `oracle_stoppages` puts the true dead balls on a method's
sides; `oracle_team` keeps its dead balls and puts the true side in play.

| base | as is | true dead balls | true sides |
|---|---|---|---|
| `stoppages` | 0.324 | 0.398 | 0.836 |
| `hmm_touch` | 0.453 | 0.518 | 0.847 |

Perfect dead balls are worth +0.07; perfect sides in play +0.39 to +0.51. Every method here finds
too few possessions (about 340–380 against 456): changes of possession in open play are missed.

## 3. The ball's speed is in different units on the bench and on BAR-ATH

`match.ball_speed` is the provider's own figure: DFL km/h (3.6 × the speed from positions), ATD
about 0.19 × it. BAR-ATH's 90th percentile (3.4) is below the bench's in-play 10th (6.9), so
`hmm_touch` sees BAR-ATH's ball as still throughout. Separately, the DFL freezes the ball while
it is out of play (speed 0 on 99% of dead frames with a ball, 1% on BAR-ATH).

| | spellF1 | macroF1 | alive/dead F1 | team acc. | BAR-ATH possessions | BAR-ATH dead share | BAR-ATH ST agreement |
|---|---|---|---|---|---|---|---|
| `hmm_touch` | 0.453 | 0.815 | 0.914 | 0.850 | 399 | 39.2% | 0.805 |
| `hmm_touch_nospeed` (no ball speed, no ball_dx) | 0.388 | 0.803 | 0.912 | 0.835 | 341 | 37.0% | 0.803 |
| `hmm_touch_noball` (also no ball position) | 0.371 | 0.786 | 0.897 | 0.832 | 359 | 37.1% | 0.798 |
| `hmm_touch_ms` (ball speed from positions, m/s) | 0.473 | 0.838 | 0.937 | 0.849 | 424 | 35.0% | 0.808 |

Ball speed matters in play, not for stoppages: removing it loses possessions and team accuracy,
while alive/dead F1 stays put. `hmm_touch_ms` puts it in the same unit everywhere; on the bench
it is within noise of `hmm_touch` (its alive/dead gain may partly be the frozen ball, which the
bench cannot tell apart), and on BAR-ATH it is the first version whose ball speed means what the
model learnt.

## Verdict

- `stoppages`: dropped as a method (the HMM does as well on stoppages and better elsewhere); its
  features stay in `possession/stoppage.py` for reuse.
- `hmm_touch_ms`: recommended fix for the candidate. `hmm_final` should be refitted with m/s ball
  speed before it is submitted; its test score has not been rerun.
- Next effort: changes of possession in open play.
