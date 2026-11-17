"""hmm_touch without the ball's movement: is its bench score the DFL's frozen ball?

The DFL's tracking, which the bench is built on, freezes the ball while it is out of play: its
speed is exactly 0 on 99% of dead frames with a ball, against 1% on BAR-ATH's ATD. A model fitted
there can call a stoppage from "the ball is not moving", which BAR-ATH will never show. This is
hmm_touch with every column that measures the ball's movement set to 0 (ball_speed, its windowed
means, ball_dx), refitted, at hmm_touch's own settings. The drop from hmm_touch on the bench is
how much of its score rests on the frozen ball.

    python -m possession.bench train hmm_touch_nospeed
"""

import numpy as np

from possession.experiments import hmm_touch
from possession.hmm_models import HmmModels

DROPPED = ("ball_speed", "ball_dx")
GRID = {name: [value] for name, value in hmm_touch.DEFAULT.items()}   # hmm_touch's, not re-chosen
DEFAULT = dict(hmm_touch.DEFAULT)


def without(dropped):
    """features(match) of hmm_touch with the named columns (and "speed_w*" windows) set to 0."""
    def features(match):
        X, names = hmm_touch.features(match)
        X = X.copy()
        for i, name in enumerate(names):
            if name in dropped or name.startswith("speed_w"):
                X[:, i] = 0.0
        return X, names
    return features


MODEL = HmmModels("hmm_touch_nospeed", without(DROPPED))
fit = MODEL.fit
predict_with = MODEL.predict_with


def predict(match):
    return predict_with(match, **DEFAULT)
