"""TRIAL (not benched): the submission's model, hmm_final_ms, with hmm_setpiece's Smart Tagging
set-piece evidence. Everything else (models, settings, Viterbi) is the submission's, unchanged.

    python -m possession run hmm_setpiece_ms        # outputs/hmm_setpiece_ms.csv
"""

from possession.experiments import hmm_final_ms, hmm_setpiece

DEFAULT = dict(hmm_final_ms.DEFAULT)


def predict(match):
    return hmm_setpiece.predict_on(hmm_final_ms, match)
