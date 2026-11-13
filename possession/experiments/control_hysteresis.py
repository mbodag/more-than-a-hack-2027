"""Option A of #5 with its default settings: control evidence, hysteresis, dead balls from the
ball leaving the pitch or the tracking losing it for more than 3 s. See _control.py."""

import games
from possession.experiments import _control


def predict(match):
    return _control.label(match.ball_xy, match.track_xy, match.track_team,
                          games.GAMES[games.GAME]["pitch"])
