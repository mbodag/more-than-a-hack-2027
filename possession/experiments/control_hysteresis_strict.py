"""control_hysteresis needing twice the evidence (12 frames, ~0.5 s) to change possession: fewer,
longer spells. A variant to compare on the proxy report, not to tune against labels."""

import games
from possession.experiments import _control


def predict(match):
    return _control.label(match.ball_xy, match.track_xy, match.track_team,
                          games.GAMES[games.GAME]["pitch"], confirm=12, tolerance=15)
