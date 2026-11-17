"""hmm_touch plus graded controls: the side of the last and next control *weighted by confidence*.

hmm_touch reads a control as a yes/no rule per frame. The control probe
(experiments/control-confidence-probe.md) found that possession/controls.py's confidence ranks
controls well (precision 0.88 at >= 0.9 against 0.72 for the 2 m rule) and that a low-confidence
"control" is evidence for the *other* side (a pass flying past a defender). So, on top of
hmm_touch's features:
  - the signed confidence of the control in progress (+ home, - away; 0 outside one);
  - for controls with confidence >= STRONG: the signed confidence of the last one before the frame
    and of the next one after it, how long ago / until, and decayed versions;
  - for the rest ("the ball passed a player"): the side of the last and next one, and how long ago.

    python -m possession.bench train hmm_conf
"""

import numpy as np

from possession import controls
from possession.experiments import hmm_touch
from possession.hmm_models import HmmModels

STRONG = 0.5
GRID = {"w": [0.4, 0.7], "switch": [0.5], "dead": [1.0], "dead_min": [100]}
DEFAULT = {"w": 0.4, "switch": 0.5, "dead": 1.0, "dead_min": 100}  # chosen by `bench train`


def signed_frames(match, ep, keep, values):
    """(n,) per frame: +value for a home episode in `keep`, -value for an away one, else 0."""
    out = np.zeros(match.n_frames)
    for a, b, side, v in zip(ep["start"][keep], ep["end"][keep], ep["side"][keep], values[keep]):
        out[a:b + 1] = v if side == "H" else -v
    return out


def conf_features(match):
    ep = controls.episodes(match)
    if len(ep.get("start", ())) == 0:
        ep = {k: np.zeros(0) for k in controls.FIELDS}
    conf = ep["confidence"]
    strong = conf >= STRONG
    now = signed_frames(match, ep, np.ones(len(conf), dtype=bool), conf)
    columns, names = [now], ["conf_now"]

    graded = signed_frames(match, ep, strong, conf)
    last, since, nxt, until = hmm_touch.carry(graded)
    for name, values in (("conf_last", last), ("conf_next", nxt), ("conf_agree", last * nxt),
                         ("conf_log_since", np.log1p(since)), ("conf_log_until", np.log1p(until)),
                         ("conf_last_fresh", last / (1 + since / 25.0)),
                         ("conf_next_fresh", nxt / (1 + until / 25.0))):
        columns.append(values)
        names.append(name)

    passby = signed_frames(match, ep, ~strong, np.ones(len(conf)))
    last, since, nxt, until = hmm_touch.carry(passby)
    for name, values in (("passby_last_fresh", last / (1 + since / 25.0)),
                         ("passby_next_fresh", nxt / (1 + until / 25.0))):
        columns.append(values)
        names.append(name)
    return np.column_stack(columns), tuple(names)


def features(match):
    X, names = hmm_touch.features(match)
    C, cnames = conf_features(match)
    return np.nan_to_num(np.column_stack([X, C])), names + cnames


MODEL = HmmModels("hmm_conf", features)
fit = MODEL.fit
predict_with = MODEL.predict_with


def predict(match):
    return predict_with(match, **DEFAULT)
