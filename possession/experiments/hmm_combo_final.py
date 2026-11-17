"""hmm_combo refitted on every non-test match: the candidate that could replace hmm_final_ms.

hmm_combo's method and settings (chosen on train, judged on validation: spellF1 0.495, see
experiments/hmm-combo.md), refitted on train + validation (degradation v2, seeds 0-2), as
hmm_final_ms was. The alive models are refitted too, cross-fitted over the same five matches, in
their own file. Test is untouched until `bench test`, run once.

    python -m possession.experiments.hmm_combo_final   # refit possession/models/hmm_combo_final*.json
    python -m possession.bench test hmm_combo_final     # once
    python -m possession run hmm_combo_final            # BAR-ATH
"""

import os

from possession.experiments import hmm_combo, hmm_final_ms
from possession.hmm_models import MODELS_DIR, HmmModels

ALIVE_PATH = os.path.join(MODELS_DIR, "hmm_combo_final_alive.json")
SEEDS = hmm_final_ms.SEEDS
DEFAULT = dict(hmm_combo.DEFAULT)

MODEL = HmmModels("hmm_combo_final", hmm_combo.make_features(alive_path=ALIVE_PATH))
predict_with = MODEL.predict_with


def predict(match):
    return predict_with(match, **DEFAULT)


def refit(say=print):
    """Fit on train + validation, every degradation seed; checks the data first."""
    import sys
    from possession import external
    sys.path.insert(0, external.REPO + "/scripts")
    import check_data
    import fetch_idsse
    problems = fetch_idsse.check()
    if problems:
        raise SystemExit("\n".join(problems))
    check_data.check_data()
    matches = [m for seed in SEEDS for split in ("train", "val")
               for m in external.split(split, degrade=seed)]
    say("refitting hmm_combo_final on {} ({} match copies)".format(
        ", ".join(external.SPLITS["train"] + external.SPLITS["val"]), len(matches)))
    hmm_combo.fit_alive(matches, path=ALIVE_PATH, say=say)
    return MODEL.fit(matches, say=say)


if __name__ == "__main__":
    refit()
