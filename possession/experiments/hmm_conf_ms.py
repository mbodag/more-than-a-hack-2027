"""hmm_touch_ms plus graded controls, with the control confidence's scales tuned on train.

The features of hmm_conf (the signed confidence of the control in progress, the last and next
confident control, the last and next ball passing a player), added to hmm_touch_ms (ball speed in
m/s, the best on validation, 0.473). The confidence uses the scales scripts/tune_controls.py chose
on the training matches (possession/models/control_params.json; validation AUC 0.798 against 0.748
hand-set). hmm_conf_ms_hand is the same with the hand-set scales, to tell the two apart.

    python scripts/tune_controls.py                 # once: the scales
    python -m possession.bench train hmm_conf_ms
"""

import json
import os

import numpy as np

from possession.experiments import hmm_conf, hmm_touch_ms
from possession.hmm_models import MODELS_DIR, HmmModels

GRID = {"w": [0.4, 0.7], "switch": [0.5], "dead": [1.0], "dead_min": [100]}
DEFAULT = {"w": 0.4, "switch": 0.5, "dead": 1.0, "dead_min": 100}  # chosen by `bench train`


def tuned_params():
    with open(os.path.join(MODELS_DIR, "control_params.json")) as handle:
        return json.load(handle)["params"]


def make_features(params):
    def features(match):
        X, names = hmm_touch_ms.features(match)
        C, cnames = hmm_conf.conf_features(match, params)
        return np.nan_to_num(np.column_stack([X, C])), names + cnames
    return features


MODEL = HmmModels("hmm_conf_ms", make_features(tuned_params()))
fit = MODEL.fit
predict_with = MODEL.predict_with


def predict(match):
    return predict_with(match, **DEFAULT)
