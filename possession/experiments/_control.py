"""Rule-based possession from tracking: control, hysteresis and dead balls. A helper, not an
experiment; experiments call `label` with their settings.

Candidate for option A of #5. Every setting is a round number from football or from how the
tracking behaves, chosen before any run, never fitted to the public ground truth (RULES.md).

1. Control evidence. On a frame where the ball is tracked, a side has control evidence when its
   nearest player is within `reach` metres of the ball and the other side's nearest player is at
   least `margin` metres further away. Close to both sides is a challenge, which is no evidence.
2. Dead balls. The ball is dead (D) while it is tracked outside the pitch (beyond `out_margin`
   metres of a line), and through any stretch where it is not tracked for longer than
   `max_bridge` frames. Out-of-pitch blips shorter than `min_dead` frames are tracking noise and
   stay alive. Shorter tracking gaps are bridged: possession carries on through them.
3. Hysteresis inside each stretch of live ball. Possession changes only when the other side shows
   `confirm` evidence frames in a row (gaps of up to `tolerance` frames without evidence allowed;
   any evidence for the side in possession breaks the run). A deflection or a duel won without
   follow-up control does not reach `confirm`, so it does not change possession (CONTEXT.md:
   controlled possession). The hand-over is dated to the first frame of that run, not the frame
   where it was confirmed.
4. Restarts. The first side to show `confirm_restart` evidence frames after a dead ball has
   possession from the restart, the first live frame. A live stretch with no evidence at all is
   X, and the runner's fill decides it.
"""

import numpy as np

from possession.experiments import _geometry

NONE, HOME_EV, AWAY_EV = 0, 1, 2
LETTER = {HOME_EV: "H", AWAY_EV: "A"}

DEFAULTS = dict(
    reach=2.0,            # m: a player this close to the ball can be playing it
    margin=1.0,           # m: the other side must be this much further away
    confirm=6,            # frames (0.24 s) of evidence in a row to change possession
    confirm_restart=3,    # frames of evidence to give the restart to a side
    tolerance=10,         # frames without evidence allowed inside a run
    max_bridge=75,        # frames (3 s): longer tracking gaps are treated as dead ball
    out_margin=1.0,       # m beyond a line before the ball counts as out
    min_dead=12,          # frames (~0.5 s): shorter out-of-pitch blips are noise
)


def evidence(home_dist, away_dist, reach, margin):
    """(n,) int8: HOME_EV, AWAY_EV or NONE per frame, from each side's nearest distance."""
    home = np.where(np.isnan(home_dist), np.inf, home_dist)
    away = np.where(np.isnan(away_dist), np.inf, away_dist)
    out = np.full(len(home), NONE, dtype=np.int8)
    with np.errstate(invalid="ignore"):  # inf - inf where neither side is known: no evidence
        out[(home <= reach) & (away - home >= margin)] = HOME_EV
        out[(away <= reach) & (home - away >= margin)] = AWAY_EV
    return out


def runs(mask):
    """[(start, end)] half-open runs of True in a 1-D bool array."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def dead_mask(ball_xy, pitch, max_bridge, out_margin, min_dead):
    """(n,) bool: frames where the ball is taken to be dead.

    Each run of frames where the ball is untracked or off the pitch is judged as a whole, so the
    ball going out and then being lost from view is one dead stretch. A run that has the ball off
    the pitch is dead if it lasts at least `min_dead` frames; a run of untracked frames only is
    dead if it lasts more than `max_bridge` frames.
    """
    ball = np.asarray(ball_xy, dtype=np.float64)
    missing = np.isnan(ball[:, 0])
    length, width = pitch
    with np.errstate(invalid="ignore"):
        out = ~missing & ((ball[:, 0] < -out_margin) | (ball[:, 0] > length + out_margin) |
                          (ball[:, 1] < -out_margin) | (ball[:, 1] > width + out_margin))
    dead = np.zeros(len(ball), dtype=bool)
    for start, end in runs(out | missing):
        limit = min_dead if out[start:end].any() else max_bridge + 1
        if end - start >= limit:
            dead[start:end] = True
    return dead


def hysteresis(ev, confirm, confirm_restart, tolerance):
    """(n,) labels "H"/"A"/"X" for one live stretch, from its evidence."""
    labels = np.full(len(ev), "X", dtype="<U1")
    current, run_side, run_count, run_start, last = NONE, NONE, 0, 0, -10 ** 9
    changes = []
    for i in np.flatnonzero(ev != NONE):
        side = ev[i]
        if side == run_side and i - last <= tolerance + 1:
            run_count += 1
        else:
            run_side, run_count, run_start = side, 1, i
        last = i
        need = confirm_restart if current == NONE else confirm
        if side != current and run_count >= need:
            changes.append((0 if current == NONE else run_start, side))
            current = side
    for k, (start, side) in enumerate(changes):
        end = changes[k + 1][0] if k + 1 < len(changes) else len(ev)
        labels[start:end] = LETTER[side]
    return labels


def label(ball_xy, track_xy, track_team, pitch, **settings):
    """(n,) labels "H", "A", "D" or "X" for the whole match."""
    s = dict(DEFAULTS, **settings)
    distances = _geometry.ball_distances(ball_xy, track_xy)
    home, away = _geometry.side_distances(distances, track_team)
    ev = evidence(home, away, s["reach"], s["margin"])
    dead = dead_mask(ball_xy, pitch, s["max_bridge"], s["out_margin"], s["min_dead"])
    out = np.full(len(ev), "D", dtype="<U1")
    for start, end in runs(~dead):
        out[start:end] = hysteresis(ev[start:end], s["confirm"], s["confirm_restart"],
                                    s["tolerance"])
    return out
