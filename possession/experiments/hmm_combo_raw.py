"""Ablation of hmm_combo for BAR-ATH only: hmm_combo's models, features from the raw tracking.

On the bench the cleaned tracking is the raw tracking, so this is hmm_combo there; on BAR-ATH it
shows what the cleaning and the keeper fix change. Uses possession/models/hmm_combo.json.

    python -m possession run hmm_combo_raw
"""

from possession.experiments import hmm_combo
from possession.hmm_models import HmmModels

DEFAULT = dict(hmm_combo.DEFAULT)

MODEL = HmmModels("hmm_combo", hmm_combo.make_features(clean=False))
predict_with = MODEL.predict_with


def predict(match):
    return predict_with(match, **DEFAULT)
