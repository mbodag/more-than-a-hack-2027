"""players_fill, on cleaned tracking with the keepers put on the right side.

The same method and the same models as players_fill. The only difference is where the ball is
tracked: the nearest player is looked for among match.track_xy_clean (positions held off camera or
stuck in place removed, possession/clean.py), and his side is read from match.track_team_fixed, in
which each tracklet of a keeper slot belongs to the side defending the end it stands in
(possession/keepers.py), instead of the side the ATD's metadata gives the slot. The team-shape
models still read the raw tracking they were fitted on.

On the bench's matches the sides are already right, so this scores exactly as players_fill there;
its effect can only be seen on BAR-ATH, through the proxy report.
"""

import numpy as np

from possession.experiments.nearest_smooth import nearest_team_codes
from possession.experiments.players_fill import DEFAULT, GRID, gap_codes  # noqa: F401
from starter.baseline import smooth  # organisers': imported, never edited


def predict_with(match, window, alive_at):
    codes = nearest_team_codes(match.ball_xy, match.track_xy_clean, match.track_team_fixed)
    hidden = np.isnan(np.asarray(match.ball_xy)[:, 0])
    codes = np.where(hidden, gap_codes(match, alive_at), codes).astype(np.int8)
    return smooth(codes, window)


def predict(match):
    return predict_with(match, **DEFAULT)
