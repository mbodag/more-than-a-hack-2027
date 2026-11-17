"""hmm_touch with the ball's speed in the same unit on every match: m/s, from its positions.

hmm_touch reads match.ball_speed, the speed the provider's file gives. The DFL's is in km/h (3.6
times the speed from its positions) and the ATD's in some other unit (about 0.19 times it), so a
model fitted on the bench sees BAR-ATH's ball as all but still on every frame: BAR-ATH's 90th
percentile (3.4) is below the bench's in-play 10th (6.9). Here ball_speed and its windowed means
(speed_w12, speed_w50) are recomputed from the ball's positions over 0.2 s, in m/s, capped at
40, 0 where the ball is not tracked; everything else is hmm_touch's, at hmm_touch's settings.

    python -m possession.bench train hmm_touch_ms
"""

import numpy as np

from possession.experiments import hmm_context, hmm_touch
from possession.hmm_models import HmmModels

LAG = 5          # frames: 0.2 s
CAP = 40.0       # m/s: faster is a jump between tracks, not a kick
GRID = {name: [value] for name, value in hmm_touch.DEFAULT.items()}   # hmm_touch's, not re-chosen
DEFAULT = dict(hmm_touch.DEFAULT)


def ball_speed_ms(ball_xy):
    """(n,) m/s over the last LAG frames, capped at CAP; 0 where it cannot be told."""
    ball = np.asarray(ball_xy, dtype=np.float64)
    past = np.vstack([np.full((LAG, 2), np.nan), ball[:-LAG]])
    speed = np.linalg.norm(ball - past, axis=1) * 25.0 / LAG
    return np.nan_to_num(np.minimum(speed, CAP))


def features(match):
    X, names = hmm_touch.features(match)
    X = X.copy()
    speed = ball_speed_ms(match.ball_xy)
    X[:, names.index("ball_speed")] = speed
    for half in hmm_context.WINDOWS:
        X[:, names.index("speed_w{}".format(half))] = hmm_context.centred_mean(speed, half)
    return X, names


MODEL = HmmModels("hmm_touch_ms", features)
fit = MODEL.fit
predict_with = MODEL.predict_with


def predict(match):
    return predict_with(match, **DEFAULT)
