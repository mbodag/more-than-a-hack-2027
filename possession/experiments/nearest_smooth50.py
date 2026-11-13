"""The organisers' baseline through our pipeline (#13): the nearest player's side, then a
centred majority vote over +/-50 frames. D where the ball is not tracked.

Published full-match spellF1 0.2433: the floor every other experiment has to beat.
"""

import numpy as np

from possession.experiments import _geometry
from starter import baseline  # organisers' code: imported, never edited

WINDOW = 50


def predict(match):
    distances = _geometry.ball_distances(match.ball_xy, match.track_xy)
    codes = _geometry.nearest_side(distances, match.track_team)
    # _geometry's codes are the baseline's: HOME 0, AWAY 1, NOBODY -1.
    return np.asarray(baseline.smooth(codes, WINDOW), dtype="<U1")
