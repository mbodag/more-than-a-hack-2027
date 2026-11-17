"""hmm_conf_ms with the hand-set confidence scales (controls.HAND): the control for the tuning.

    python -m possession.bench train hmm_conf_ms_hand
"""

from possession.experiments.hmm_conf_ms import GRID, make_features  # noqa: F401
from possession.hmm_models import HmmModels

DEFAULT = {"w": 0.4, "switch": 0.5, "dead": 1.0, "dead_min": 100}  # chosen by `bench train`

MODEL = HmmModels("hmm_conf_ms_hand", make_features(None))
fit = MODEL.fit
predict_with = MODEL.predict_with


def predict(match):
    return predict_with(match, **DEFAULT)
