"""Is play stopped? Per-frame features for every frame, and stoppages cut from a probability.

players_fill decides alive or dead only where the tracking has lost the ball, so where the ball is
tracked a stoppage can never be called. These features describe every frame, ball or not, and
what the players do over the seconds around it, because a stoppage is a stretch, not a frame:

    X, names = features(match)                 # (n, k), NaN replaced by 0
    dead = cut(p_alive, enter=0.4, leave=0.6, min_dead=50, min_live=25)

Beyond the team shape (possession/team_shape.py), in metres, m/s, counts, shares:
    n_tracked                    players tracked
    median_speed, share_slow     all players' median speed, and the share slower than 1 m/s
    crowd                        the most players within 5 m of any one player (celebrations,
                                 a wall, a group round the referee)
    nn_distance                  the median distance from a player to his nearest neighbour
    ball_tracked                 1 where the tracking has the ball
    ball_out                     metres the ball is beyond the lines (0 inside, or untracked)
and centred means of the play-telling ones over 1, 3 and 10 s, plus their means over the 3 s
before and the 3 s after (a restart is still before and moving after).

No ball speed, and nothing about where a dead ball lies: the DFL's tracking, which the model is
fitted on, freezes the ball while it is out of play (its speed is exactly 0 on 99% of dead frames
with a ball, against 1% on BAR-ATH's ATD), so the model would learn the provider's habit, not
football.
"""

import numpy as np

from possession import team_shape

FPS = 25
SLOW = 1.0      # m/s: a player slower than this is standing
CROWD = 5.0     # metres
CHUNK = 4000    # frames per block of the pairwise distances

ROLLED = ("mean_speed", "median_speed", "share_slow", "share_running", "crowd", "nn_distance",
          "n_tracked", "ball_tracked", "all_spread_x", "all_spread_y")
WINDOWS = {"1s": (12, 12), "3s": (37, 37), "10s": (125, 125), "past3s": (75, 0),
           "next3s": (0, 75)}
BASE = ("n_tracked", "median_speed", "share_slow", "crowd", "nn_distance", "ball_tracked",
        "ball_out")
NAMES = (tuple(team_shape.NAMES) + BASE
         + tuple("{}_{}".format(name, w) for name in ROLLED for w in WINDOWS))


def rolling_mean(values, before, after):
    """(n,) mean of the non-NaN values in [t - before, t + after]; NaN where there are none."""
    values = np.asarray(values, dtype=np.float64)
    ok = ~np.isnan(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(ok, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(ok)))
    index = np.arange(len(values))
    low = np.maximum(0, index - before)
    high = np.minimum(len(values), index + after + 1)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[high] - total[low]) / np.maximum(n, 1), np.nan)


def player_speeds(track_xy):
    """(n, slots) m/s over one second, NaN unless the slot was tracked throughout."""
    xy = np.asarray(track_xy, dtype=np.float64)
    lag = team_shape.LAG
    tracked = ~np.isnan(xy[..., 0])
    ok = team_shape.steady(tracked, lag)
    past = np.vstack([np.full((lag,) + xy.shape[1:], np.nan), xy[:-lag]])
    speed = np.linalg.norm(xy - past, axis=2) / (lag / FPS)
    return np.where(ok, speed, np.nan)


def crowding(track_xy, radius=CROWD, chunk=CHUNK):
    """((n,) the most other players within `radius` of any one player,
        (n,) the median distance from a player to his nearest neighbour), NaN without players."""
    xy = np.asarray(track_xy, dtype=np.float64)
    n = len(xy)
    crowd = np.full(n, np.nan)
    nearest = np.full(n, np.nan)
    for start in range(0, n, chunk):
        block = xy[start:start + chunk]
        d = np.linalg.norm(block[:, :, None, :] - block[:, None, :, :], axis=3)
        slots = d.shape[1]
        d[:, np.arange(slots), np.arange(slots)] = np.nan
        with team_shape._quiet():
            close = np.nansum(d < radius, axis=2).astype(np.float64)        # NaN compares False
            close[np.isnan(block[..., 0])] = np.nan
            crowd[start:start + chunk] = np.nanmax(close, axis=1)
            nearest[start:start + chunk] = np.nanmedian(np.nanmin(d, axis=2), axis=1)
    return crowd, nearest


def ball_columns(ball_xy, pitch):
    """{ball_tracked, ball_out}, each (n,), ball_out 0 where the ball is untracked."""
    ball = np.asarray(ball_xy, dtype=np.float64)
    length, width = pitch
    seen = ~np.isnan(ball[:, 0])
    x, y = ball[:, 0], ball[:, 1]
    beyond = np.maximum.reduce([-x, x - length, -y, y - width, np.zeros(len(x))])
    return {"ball_tracked": seen.astype(np.float64), "ball_out": np.where(seen, beyond, 0.0)}


def table(match):
    """{name: (n,) float64} for every name in NAMES, NaN where it cannot be told."""
    out = {name: np.asarray(v, dtype=np.float64) for name, v in team_shape.of(match).items()}
    xy = np.asarray(match.track_xy, dtype=np.float64)
    tracked = ~np.isnan(xy[..., 0])
    speed = player_speeds(xy)
    with team_shape._quiet():
        out["median_speed"] = np.nanmedian(speed, axis=1)
        out["share_slow"] = np.nanmean(np.where(np.isnan(speed), np.nan, speed < SLOW), axis=1)
    out["n_tracked"] = tracked.sum(axis=1).astype(np.float64)
    out["crowd"], out["nn_distance"] = crowding(xy)
    out.update(ball_columns(match.ball_xy, team_shape.pitch_of(match)))
    for name in ROLLED:
        for window, (before, after) in WINDOWS.items():
            out["{}_{}".format(name, window)] = rolling_mean(out[name], before, after)
    return out


def features(match, names=NAMES):
    """(X, names): the named columns as one (n, k) matrix for a model, NaN replaced by 0."""
    columns = table(match)
    return np.nan_to_num(np.column_stack([columns[name] for name in names])), tuple(names)


def runs(flags):
    """[(start, end, value)], end exclusive, of the runs of a 1-D bool array."""
    flags = np.asarray(flags, dtype=bool)
    if len(flags) == 0:
        return []
    edges = np.flatnonzero(np.diff(flags.astype(np.int8))) + 1
    bounds = np.concatenate(([0], edges, [len(flags)]))
    return [(int(s), int(e), bool(flags[s])) for s, e in zip(bounds[:-1], bounds[1:])]


def cut(p_alive, enter, leave, min_dead, min_live):
    """(n,) bool, True where play is stopped.

    Hysteresis: a stoppage starts where p_alive falls below `enter` and lasts until it rises above
    `leave` (leave >= enter), so a probability hovering near one threshold does not chop play into
    pieces. Then stoppages shorter than `min_dead` frames are dropped (each would split a
    possession for nothing), and live stretches shorter than `min_live` between two stoppages are
    absorbed into them.
    """
    if leave < enter:
        raise ValueError("leave ({}) must be at least enter ({})".format(leave, enter))
    p = np.asarray(p_alive, dtype=np.float64)
    below, above = p < enter, p > leave
    dead = np.zeros(len(p), dtype=bool)
    state = False
    # Only the frames where one threshold is crossed can change the state: walk those.
    for t in np.flatnonzero(below | above):
        if not state and below[t]:
            state, start = True, t
        elif state and above[t]:
            dead[start:t] = True
            state = False
    if state:
        dead[start:] = True
    for s, e, value in runs(dead):
        if value and e - s < min_dead:
            dead[s:e] = False
    pieces = runs(dead)
    for i, (s, e, value) in enumerate(pieces):
        if not value and 0 < i < len(pieces) - 1 and e - s < min_live:
            dead[s:e] = True
    return dead
