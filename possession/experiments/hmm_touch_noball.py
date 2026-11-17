"""hmm_touch without the ball's movement or where a ball lies: hmm_touch_nospeed, and also
ball_out, ball_x and ball_y_centre set to 0, since a frozen dead ball also sits where it went out.
Who is nearest the ball is kept: that is the possession signal itself.

    python -m possession.bench train hmm_touch_noball
"""

from possession.experiments import hmm_touch
from possession.experiments.hmm_touch_nospeed import DROPPED, without
from possession.hmm_models import HmmModels

GRID = {name: [value] for name, value in hmm_touch.DEFAULT.items()}   # hmm_touch's, not re-chosen
DEFAULT = dict(hmm_touch.DEFAULT)

MODEL = HmmModels("hmm_touch_noball", without(DROPPED + ("ball_out", "ball_x", "ball_y_centre")))
fit = MODEL.fit
predict_with = MODEL.predict_with


def predict(match):
    return predict_with(match, **DEFAULT)
