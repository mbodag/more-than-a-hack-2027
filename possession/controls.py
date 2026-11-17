"""Ball controls from the tracking, each with a rule-based confidence.

hmm_touch reads "a control" as a yes/no rule on one frame (ball within 2 m of a player, 1 m nearer
than any opponent). That counts a ball flying past a player and misses a touch the tracking
smeared. Here a control is an *episode*: a run of frames where the same player is the nearest to a
tracked ball and within REACH. Each episode gets a few rule scores in [0, 1] and one confidence:

  prox      how close the ball came (2.5 m -> 0, 0.5 m -> 1)
  sep       how much nearer than the nearest opponent (-0.5 m -> 0, +1.5 m -> 1)
  dwell     how long the player kept it in reach (1 frame -> 0, 0.4 s -> 1)
  together  ball moving with the player (relative speed 4 m/s -> 0, 1 m/s -> 1)
  impulse   the ball's velocity changed sharply within reach (2 m/s -> 0, 8 m/s -> 1): a kick or trap

  confidence = prox * sep * max(dwell * together, impulse)

Everything is from tracking only, so the same code runs on BAR-ATH and on the bench. The scales
are HAND by default; scripts/tune_controls.py tunes them on the bench's training matches
(possession/models/control_params.json), and scripts/control_probe.py measures how well they rank.
"""

import numpy as np

FPS = 25
REACH = 2.5        # metres: a looser net than hmm_touch's; the scores do the sorting
MERGE = 2          # frames: a break this short in the same player's run does not end the episode
VEL_LAG = 2        # frames either side for a velocity
IMPULSE_LAG = 3    # frames either side for a change of velocity

# The hand-set scales: each score is 0 at `_zero` and 1 at `_one`, linear between.
HAND = {"prox_zero": REACH, "prox_one": 0.5, "sep_zero": -0.5, "sep_one": 1.5,
        "dwell_zero": 1.0, "dwell_one": 0.4 * FPS, "together_zero": 4.0, "together_one": 1.0,
        "impulse_zero": 2.0, "impulse_one": 8.0}

FIELDS = ("start", "end", "track", "side", "dmin", "margin", "frames", "rel_speed", "impulse",
          "prox", "sep", "dwell", "together", "impulse_score", "confidence")


def _ramp(x, zero, one):
    """0 at `zero`, 1 at `one`, linear and clipped between (either direction)."""
    return np.clip((np.asarray(x, dtype=float) - zero) / (one - zero), 0.0, 1.0)


def _velocity(xy, blocks, lag=VEL_LAG):
    """Central-difference velocity (m/s) along axis 0, NaN at block edges and where xy is NaN."""
    v = np.full(xy.shape, np.nan, dtype=np.float32)
    for a, b in blocks:
        if b - a + 1 <= 2 * lag:
            continue
        v[a + lag:b - lag + 1] = (xy[a + 2 * lag:b + 1] - xy[a:b - 2 * lag + 1]) * (FPS / (2 * lag))
    return v


def nearest(match):
    """Per frame: (nearest track, its distance, distance of the nearest track of the other side),
    -1 / NaN where the ball is not tracked. Sides from match.track_team_fixed."""
    ball = np.asarray(match.ball_xy, dtype=np.float32)
    tracks = np.asarray(match.track_xy, dtype=np.float32)
    side = np.asarray(match.track_team_fixed)
    d = np.linalg.norm(tracks - ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    j = np.argmin(d, axis=1)
    rows = np.arange(len(j))
    dmin = d[rows, j]
    mine = side[rows, j]
    other = np.where(side != mine[:, None], d, np.inf).min(axis=1)
    seen = np.isfinite(dmin)
    return np.where(seen, j, -1), np.where(seen, dmin, np.nan), np.where(seen, other, np.nan)


def episodes(match, params=None):
    """The control episodes of a match as a dict of arrays (keys FIELDS), sorted by start."""
    from possession.team_shape import blocks_of
    blocks = blocks_of(match)
    ball = np.asarray(match.ball_xy, dtype=np.float32)
    tracks = np.asarray(match.track_xy, dtype=np.float32)
    side_of = np.asarray(match.track_team_fixed)
    j, d, d_other = nearest(match)
    vb = _velocity(ball, blocks)
    vp = _velocity(tracks, blocks)
    n = len(j)
    rows = np.arange(n)
    rel = np.linalg.norm(vb - vp[rows, np.maximum(j, 0)], axis=1)
    jump = np.full(n, np.nan, dtype=np.float32)
    k = IMPULSE_LAG
    jump[k:n - k] = np.linalg.norm(vb[2 * k:] - vb[:n - 2 * k], axis=1)

    candidate = (j >= 0) & (d <= REACH)
    block_of = np.full(n, -1)
    for i, (a, b) in enumerate(blocks):
        block_of[a:b + 1] = i
    candidate &= block_of >= 0          # nothing outside the halves (BAR-ATH's pre-match, half-time)

    out = {f: [] for f in FIELDS}
    idx = np.flatnonzero(candidate)
    if len(idx) == 0:
        return {f: np.array(v) for f, v in out.items()}
    # Split into runs: a new episode when the player changes, the half changes, or the gap > MERGE.
    brk = np.r_[True, (j[idx[1:]] != j[idx[:-1]]) | (idx[1:] - idx[:-1] > MERGE + 1)
                | (block_of[idx[1:]] != block_of[idx[:-1]])]
    starts = np.flatnonzero(brk)
    ends = np.r_[starts[1:], len(idx)] - 1
    for s, e in zip(starts, ends):
        fr = idx[s:e + 1]
        a, b = int(fr[0]), int(fr[-1])
        t = int(j[a])
        out["start"].append(a)
        out["end"].append(b)
        out["track"].append(t)
        out["side"].append(str(side_of[a, t]))
        out["dmin"].append(float(np.nanmin(d[fr])))
        out["margin"].append(float(np.nanmax(d_other[fr] - d[fr])))
        out["frames"].append(len(fr))
        r = rel[fr]
        out["rel_speed"].append(float(np.nanmin(r)) if np.isfinite(r).any() else np.nan)
        w = jump[max(a - k, 0):b + k + 1]
        out["impulse"].append(float(np.nanmax(w)) if np.isfinite(w).any() else 0.0)
    ep = {f: np.array(v) for f, v in out.items() if v}
    return score(ep, params)


def score(ep, params=None):
    """Fill in the rule scores and the confidence of episodes, with `params` (default HAND)."""
    p = dict(HAND)
    p.update(params or {})
    if len(ep.get("start", ())) == 0:
        return ep
    ep["prox"] = _ramp(ep["dmin"], p["prox_zero"], p["prox_one"])
    ep["sep"] = _ramp(ep["margin"], p["sep_zero"], p["sep_one"])
    ep["dwell"] = _ramp(ep["frames"], p["dwell_zero"], p["dwell_one"])
    ep["together"] = np.nan_to_num(_ramp(ep["rel_speed"], p["together_zero"], p["together_one"]))
    ep["impulse_score"] = _ramp(ep["impulse"], p["impulse_zero"], p["impulse_one"])
    ep["confidence"] = ep["prox"] * ep["sep"] * np.maximum(ep["dwell"] * ep["together"],
                                                           ep["impulse_score"])
    return ep


def side_outcome(ep, labels, after=12):
    """Per episode, against a full ground truth: 1 if the side in possession over the `after` frames
    following its end is the controller's, 0 if the other side, -1 if those frames are all dead."""
    n = len(labels)
    out = np.empty(len(ep.get("start", ())), dtype=int)
    for i, (e, s) in enumerate(zip(ep["end"], ep["side"])):
        lab = labels[e:min(e + after, n)]
        lab = lab[lab != "D"]
        out[i] = -1 if len(lab) == 0 else int((lab == s).mean() >= 0.5)
    return out

def hmm_touch_rule(ep, reach, margin):
    """Which episodes hmm_touch's yes/no rule would count as a control (it is per frame; an episode
    qualifies if it comes within `reach` with at least `margin` over the nearest opponent)."""
    return (ep["dmin"] <= reach) & (ep["margin"] >= margin)
