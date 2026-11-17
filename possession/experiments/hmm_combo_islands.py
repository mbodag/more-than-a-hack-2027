"""hmm_combo, then any play stretch shorter than `island_min` frames that sits between two
stoppages becomes dead ball (hmm.drop_dead_islands).

Aimed at long breaks (cooling breaks, injuries, VAR) where the decoder invents a moment of
possession: each such moment is a whole false possession on spellF1. The cost is the real short
ones: on the training matches' ground truth, 36 of 405 play stretches between two stoppages are
under 4 s.

By hand at 3 s (75 frames): bench train prefers it off, by less than the noise, and BAR-ATH's
long breaks are where it is aimed. Uses hmm_combo's fitted models (possession/models/hmm_combo.json); nothing is refitted.

    python -m possession.bench train hmm_combo_islands   # picks island_min on train
    python -m possession.bench val hmm_combo_islands
"""

from possession import hmm, team_shape
from possession.experiments import hmm_combo

GRID = dict(hmm_combo.GRID, island_min=[1, 50, 75, 100, 150])
DEFAULT = dict(hmm_combo.DEFAULT, island_min=75)   # 3 s, set by hand: bench train picks 1 (off),
#   but the bench has few long breaks; see experiments/dead-islands.md


def predict_with(match, island_min, **settings):
    labels = hmm_combo.predict_with(match, **dict(hmm_combo.DEFAULT, **settings))
    return hmm.drop_dead_islands(labels, team_shape.blocks_of(match), island_min)


def predict(match):
    return predict_with(match, **DEFAULT)
