"""Ball-to-player distances, shared by the experiments. A helper, not an experiment.

Everything is vectorised over frames: (n_frames, n_tracks) arrays of distances in metres, NaN where
the ball or the track is missing.
"""

import numpy as np

NOBODY = -1
HOME, AWAY = 0, 1


def ball_distances(ball_xy, track_xy):
    """(n_frames, n_tracks) float64: distance from the ball to each track slot, NaN if either is
    missing."""
    ball = np.asarray(ball_xy, dtype=np.float64)[:, None, :]
    tracks = np.asarray(track_xy, dtype=np.float64)
    return np.sqrt(((tracks - ball) ** 2).sum(axis=2))


def nearest_side(distances, track_team):
    """(n_frames,) int8: HOME or AWAY for the side of the nearest track, NOBODY where no distance
    is known. Ties go to the lower slot, as in starter/baseline.py."""
    known = ~np.isnan(distances)
    seen = known.any(axis=1)
    filled = np.where(known, distances, np.inf)
    slot = filled.argmin(axis=1)
    side = np.where(np.asarray(track_team) == "H", HOME, AWAY).astype(np.int8)
    return np.where(seen, side[slot], NOBODY).astype(np.int8)


def side_distances(distances, track_team):
    """(home, away): per frame, the distance from the ball to the nearest track of each side,
    NaN where that side has no known distance."""
    team = np.asarray(track_team)
    out = []
    for letter in ("H", "A"):
        block = distances[:, team == letter]
        known = ~np.isnan(block)
        nearest = np.where(known, block, np.inf).min(axis=1)
        out.append(np.where(known.any(axis=1), nearest, np.nan))
    return tuple(out)
