"""Bench only: a method's dead balls with the true sides. How much is left in team work?

Takes another experiment's labels (`base`) and keeps its dead balls; wherever it says play is on,
the true side instead (carried forward over the truth's own dead balls that `base` missed). The
gap between this and `base` on the bench is what perfect team possession would add to that
method. It reads match.labels, so it runs on the bench's matches only, never on BAR-ATH.

    python -m possession.bench val oracle_team
    python -m possession.bench val oracle_team --set 'base="stoppages"'
"""

import numpy as np

from possession.experiments.oracle_stoppages import base_labels
from possession.fill import fill

DEFAULT = {"base": "hmm_touch"}   # the method on validation (hmm_final has seen val)


def predict_with(match, base):
    labels = base_labels(match, base)
    truth = np.asarray(match.labels)
    side = fill(np.where(truth == "D", "X", truth), blocks=match.blocks)
    return np.where(labels == "D", "D", side)


def predict(match):
    return predict_with(match, **DEFAULT)
