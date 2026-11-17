"""hmm_combo_final, then no possession under 3 s between two stoppages (hmm_combo_islands' rule).

hmm_combo_final's models and settings (refitted on train + validation), then hmm.drop_dead_islands
with island_min 75 frames, set by hand on hmm_combo_islands (experiments/dead-islands.md): the
bench calls it a draw, and BAR-ATH's long breaks are where it is aimed. Nothing is refitted.

    python -m possession.bench test hmm_combo_final_islands   # once
    python -m possession run hmm_combo_final_islands           # BAR-ATH
"""

from possession import hmm, team_shape
from possession.experiments import hmm_combo_final, hmm_combo_islands

DEFAULT = dict(hmm_combo_final.DEFAULT, island_min=hmm_combo_islands.DEFAULT["island_min"])


def predict_with(match, island_min, **settings):
    labels = hmm_combo_final.predict_with(match, **dict(hmm_combo_final.DEFAULT, **settings))
    return hmm.drop_dead_islands(labels, team_shape.blocks_of(match), island_min)


def predict(match):
    return predict_with(match, **DEFAULT)
