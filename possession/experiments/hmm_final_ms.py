"""hmm_final with the ball's speed in m/s from its positions on every match (hmm_touch_ms).

hmm_final reads the provider's own ball speed, which is km/h on the bench and about 0.19 x m/s
on BAR-ATH's ATD, so on BAR-ATH its model sees the ball as all but still. This is hmm_final's
method, settings and refit (train + validation, degradation v2, seeds 0-2) with hmm_touch_ms's
features instead. Test is run once.

    python -m possession.experiments.hmm_final_ms   # refit possession/models/hmm_final_ms.json
    python -m possession.bench test hmm_final_ms     # once
    python -m possession run hmm_final_ms            # BAR-ATH
"""

import sys

from possession import external
from possession.experiments import hmm_final, hmm_touch_ms
from possession.hmm_models import HmmModels

SEEDS = hmm_final.SEEDS
DEFAULT = dict(hmm_final.DEFAULT)

MODEL = HmmModels("hmm_final_ms", hmm_touch_ms.features)
predict_with = MODEL.predict_with


def predict(match):
    return predict_with(match, **DEFAULT)


def refit(say=print):
    """Fit on train + validation, every degradation seed; checks the data first."""
    sys.path.insert(0, external.REPO + "/scripts")
    import check_data
    import fetch_idsse
    problems = fetch_idsse.check()
    if problems:
        raise SystemExit("\n".join(problems))
    check_data.check_data()
    matches = [m for seed in SEEDS for split in ("train", "val")
               for m in external.split(split, degrade=seed)]
    say("refitting hmm_final_ms on {} ({} match copies)".format(
        ", ".join(external.SPLITS["train"] + external.SPLITS["val"]), len(matches)))
    return MODEL.fit(matches, say=say)


if __name__ == "__main__":
    refit()
