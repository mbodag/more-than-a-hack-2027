"""The submission: hmm_setpiece_islands, written as "North London Tactics.csv".

hmm_setpiece_islands is hmm_final (a 3-state HMM over per-frame logistic evidence, refitted on the
bench's train + validation matches) with Smart Tagging's set pieces as extra evidence before
decoding (hmm_setpiece), then no possession under 3 s between two stoppages
(hmm.drop_dead_islands). See experiments/setpiece-trial.md and docs/TECHNICAL.md.

    python -m possession run north_london_tactics
    python evaluation/validate_submission.py "North London Tactics.csv"
"""

from possession.experiments import hmm_setpiece_islands

DEFAULT = dict(hmm_setpiece_islands.DEFAULT)


def predict(match):
    return hmm_setpiece_islands.predict(match)
