"""hmm_setpiece, then any play stretch shorter than 3 s between two stoppages becomes dead ball.

The set-piece evidence is hmm_setpiece's (Smart Tagging's SET PIECES tags as bonuses on
hmm_final's emissions before Viterbi). The dead-island rule is hmm.drop_dead_islands at
`island_min` 75 frames (3 s), as in hmm_combo_islands (#35): in long breaks the decoder can invent
a moment of possession, and each one is a whole false possession on spellF1. Both are set by
hand, not fitted: the bench has no Smart Tagging and few long breaks. Nothing is refitted.

    python -m possession run hmm_setpiece_islands
"""

from possession import hmm, team_shape
from possession.experiments import hmm_final, hmm_setpiece

DEFAULT = dict(hmm_final.DEFAULT, island_min=75)   # 3 s, set by hand as in hmm_combo_islands


def predict_with(match, island_min, **settings):
    labels = hmm_setpiece.predict_on(hmm_final, match, **settings)
    return hmm.drop_dead_islands(labels, team_shape.blocks_of(match), island_min)


def predict(match):
    return predict_with(match, **DEFAULT)
