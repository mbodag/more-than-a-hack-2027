"""Ablation of hmm_combo: the same, with alive=False.

    python -m possession.bench train hmm_combo_noalive
"""

from possession.experiments import hmm_combo
from possession.hmm_models import HmmModels

GRID = dict(hmm_combo.GRID)
DEFAULT = dict(hmm_combo.DEFAULT)

MODEL = HmmModels("hmm_combo_noalive", hmm_combo.make_features(alive=False))
predict_with = MODEL.predict_with


def fit(matches, say=print):
    hmm_combo.fit_alive(matches, say=say)
    return MODEL.fit(matches, say=say)


def predict(match):
    return predict_with(match, **DEFAULT)
