"""Bench only: a method's sides with the true dead balls. How much is left in dead-ball work?

Takes another experiment's labels (`base`) and replaces its dead balls with the ground truth's:
D wherever the truth has D; where the truth is in play but `base` said D, the side `base` gives
next to it (carried over the gap, as the scorer fills X). The gap between this and `base` on the
bench is what perfect dead-ball detection would add to that method. It reads match.labels, so it
runs on the bench's matches only, never on BAR-ATH.

    python -m possession.bench val oracle_stoppages
    python -m possession.bench val oracle_stoppages --set 'base="stoppages"'
"""

import importlib

import numpy as np

from possession.fill import fill

DEFAULT = {"base": "hmm_touch"}   # the method on validation (hmm_final has seen val)


def base_labels(match, base):
    return np.asarray(importlib.import_module("possession.experiments." + base).predict(match))


def predict_with(match, base):
    labels = base_labels(match, base)
    dead = np.asarray(match.labels) == "D"
    side = fill(np.where(labels == "D", "X", labels), blocks=match.blocks)
    return np.where(dead, "D", side)


def predict(match):
    return predict_with(match, **DEFAULT)
