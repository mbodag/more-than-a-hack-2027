"""control_hysteresis with a wider reach (3 m) and 5 s tracking gaps bridged, for noisier ball
tracking than the defaults assume. A variant to compare on the proxy report, not to tune against
labels."""

import games
from possession.experiments import _control


def predict(match):
    return _control.label(match.ball_xy, match.track_xy, match.track_team,
                          games.GAMES[games.GAME]["pitch"], reach=3.0, max_bridge=125)
