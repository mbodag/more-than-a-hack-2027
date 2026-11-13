# 13 · nearest_smooth50

- **Issue and commit:** #13, the commit that adds this file.
- **Method:** the organisers' baseline (`starter/baseline.py`) through our pipeline: the side of the
  track nearest the ball, then a centred majority vote over ±50 frames; `D` where the ball is not
  tracked. Distances come from Match items (`ball_xy`, `track_xy`, `track_team`); the vote is
  `starter.baseline.smooth`, imported.
- **Command:** `python -m possession run nearest_smooth50` (and `north_london_tactics`, which points
  at it until #5).
- **Proxy metrics:** _not run yet. The cloud session that wrote this has no match data; a local
  session fills this from `experiments/metrics/nearest_smooth50.json`._
- **Validation:** _pending the local run._
- **Verdict:** the control, published full-match spellF1 0.2433.

## Checked without data

`tests/test_nearest_smooth50.py` writes a synthetic ATD file, runs the organisers'
`nearest_player_labels` and `smooth` on it, and checks that our version gives the same codes and the
same labels on every frame.

## Still to check locally (#13's Verify)

1. Labels agree with `starter/baseline.py`'s `baseline_smooth50.csv` on every frame. Our positions
   are stored as float32 metres; the baseline uses float64. A near-tie between two tracks could in
   principle flip the nearest one. If any frames disagree, check whether they are such ties before
   suspecting anything else.
2. A second run with `--rebuild-cache` gives a byte-identical CSV.
