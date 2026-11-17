# Control confidence probe: how good are our ball controls?

- **Issue**: none yet (follows #26: `hmm_touch`'s gain came from "the side of the last control")
- **Method**: `possession/controls.py` finds control *episodes* (a run of frames where the same
  player is nearest a tracked ball, within 2.5 m) and scores each with hand-set rules:
  `confidence = prox * sep * max(dwell * together, impulse)` (closeness, margin over the nearest
  opponent, time in reach, ball moving with the player, sharp change of the ball's velocity).
  No fitting. Compared with `hmm_touch`'s two yes/no rules.
- **References** (training matches only, ADR 0003):
  - **precision**: share of controls on live play after which the DFL's possession, over the next
    0.5 s, is the controller's side;
  - **recall**: share of the DFL's contact events (passes, shots, ball claims, other ball actions;
    deleted events dropped) with a control of the same side, or the same player, within ±1 s.
    Event clocks differ from the tracking by -43 to +17 frames per match; the lag is estimated
    as where the named player is nearest the ball (median distance 2.3-3.3 m at that lag).
- **Command**: `python scripts/control_probe.py` (8 s once the IDSSE cache exists)
- **Metrics**: `experiments/metrics/control_probe.json`

## Clean tracking (mean of the 4 training matches)

| rule | controls | precision | recall (side) | recall (player) |
|---|---|---|---|---|
| `hmm_touch` 2 m, 1 m margin | 1702 | 0.717 | 0.874 | 0.718 |
| `hmm_touch` 3 m, 0.5 m margin | 2069 | 0.664 | 0.904 | 0.754 |
| confidence ≥ 0.2 | 1770 | 0.713 | 0.897 | 0.767 |
| **confidence ≥ 0.4** | 1543 | **0.763** | 0.878 | 0.744 |
| confidence ≥ 0.6 | 1306 | 0.822 | 0.850 | 0.714 |
| confidence ≥ 0.9 | 954 | 0.881 | 0.752 | 0.608 |

At the recall of the 2 m rule, confidence ≥ 0.4 is 4.6 points more precise with 160 fewer
controls; the 0.9 cut is 16 points more precise for 12 points less recall.

**Degradation v2 (seed 0)**: precision is the same at every cut (0.724 for the 2 m rule, 0.769 at
≥ 0.4, 0.886 at ≥ 0.9). Recall falls to about 0.53 for every rule because the ball is hidden on
41% of frames; no control rule can see those touches.

## What the rules say

| score | precision when low (< 0.5) | when high (≥ 0.5) |
|---|---|---|
| prox | 0.31 | 0.71 |
| sep | 0.41 | 0.66 |
| dwell | 0.40 | 0.74 |
| **together** | 0.48 | **0.87** |
| impulse | 0.45 | 0.69 |

| confidence bin | < 0.1 | 0.1-0.3 | 0.3-0.5 | 0.5-0.7 | 0.7-0.9 | ≥ 0.9 |
|---|---|---|---|---|---|---|
| episodes per match | 384 | 216 | 207 | 188 | 226 | 906 |
| precision | 0.40 | 0.35 | 0.38 | 0.49 | 0.70 | 0.88 |

- **The score ranks well at the top, and calibration is not monotone below 0.5.** Below 0.5 a
  "control" is *worse than a coin*: most are the ball passing a player of the side that does not
  have it (a pass flying past a defender). They are evidence for the *other* side, which a yes/no
  rule cannot say.
- **The ball moving with the player is the strongest single cue**, stronger than distance.

## Caveats

- Precision is against the DFL's possession, a close cousin of ours; and the DFL keeps
  possession through a deflection, so a real touch by a defender can count as wrong.
- The event clock offset is estimated per match with a 2-frame step; recall uses ±1 s.
- The rule scales are hand-set and were looked at only on training matches.

## Verdict (probe)

Worth feeding to the HMM.

## `hmm_conf`: the confidence in the HMM

- **Method**: `hmm_touch`'s features plus, from `possession/controls.py`: the signed confidence of
  the control in progress; for controls with confidence >= 0.5, the signed confidence of the last
  and next one, how long ago / until and decayed versions; for the rest ("the ball passed a
  player"), the decayed side of the last and next one. Same HMM. `switch`, `dead` and `dead_min`
  fixed at `hmm_touch`'s values (its grid was flat there); only `w` searched.
- **Command**: `python -m possession.bench train hmm_conf` (degradation v2, seeds 0-2; 107 s)
- **Metrics**: `experiments/metrics/bench/hmm_conf.json`; models `possession/models/hmm_conf.json`

| | train (best) | val seed 0 | seed 1 | seed 2 | **val spellF1** | val macroF1 |
|---|---|---|---|---|---|---|
| `hmm_touch` | 0.464 | 0.464 | 0.464 | 0.431 | 0.453 | 0.815 |
| `hmm_conf` (`w` 0.4) | 0.478 | 0.454 | 0.477 | 0.450 | **0.460** | 0.817 |

**Not kept for now: within noise.** +0.015 on train and +0.007 on validation, but seed 0 is worse
(-0.010), and team accuracy on live frames is unchanged (0.851-0.852). `hmm_split_w` (+0.003) and
`hmm_mlp` (+0.016) were judged the same way in #26. The confidence probably adds little because
`hmm_touch`'s logistic model already weighs distance and margin through its other features; the
probe's gain is in precision at a fixed threshold, which the HMM never applies.

Test not run. Ideas left: the score's rules as separate features (`together` was the strongest
cue), and calibrating confidence on train instead of hand-set scales.

## The viewer

`python -m possession.viz` (and `--external`) now writes the control episodes too: a yellow ring
on the controlling player (brighter and thicker the more confident), a "Ball controls" timeline row
(home top, away bottom, opacity = confidence), and a "Ball control" panel with the slot, side and
confidence, or the last control and how long ago. A "Controls" checkbox turns them off.
