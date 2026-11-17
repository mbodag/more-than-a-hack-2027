"""The submission: hmm_final_ms, written as "North London Tactics.csv".

hmm_final_ms is hmm_final (a 3-state HMM over per-frame logistic evidence, refitted on the bench's
train + validation matches) with the ball's speed in m/s from its positions on every match. Bench
test (run once): spellF1 0.497, macroF1 0.860. See experiments/stoppages.md and docs/TECHNICAL.md.

    python -m possession run north_london_tactics
    python evaluation/validate_submission.py "North London Tactics.csv"
"""

from possession.experiments import hmm_final_ms

DEFAULT = dict(hmm_final_ms.DEFAULT)


def predict(match):
    return hmm_final_ms.predict(match)
