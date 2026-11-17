"""hmm_touch_ms plus the three features added on 17 November: control confidence, P(alive) from the
stoppage model, and the cleaned tracking with the keepers on the right side.

On top of hmm_touch_ms's features:
  - conf: hmm_conf's control features (possession/controls.py): the signed confidence of the
    control in progress, of the last and next strong one, and the pass-bys;
  - alive: P(alive) from possession/stoppage.py's features (the players alone), its logit, and its
    means over ±1 s and ±4 s. The alive model is **cross-fitted**: on a match it was fitted on, the
    value comes from the model fitted on the other matches, so the HMM does not learn to trust an
    in-sample P(alive) more than it deserves. Unseen matches (validation, test, BAR-ATH) use the
    model fitted on every match given to `fit`;
  - clean: hmm_touch_ms's features and the controls read the cleaned tracking
    (`match.track_xy_clean`) with each slot's per-frame side (`match.track_team_fixed`, keepers
    fixed). On the bench these are the raw tracking and the DFL's sides, so this part changes
    nothing there and can only be judged on BAR-ATH's label-free checks. The stoppage features stay
    on the raw tracking: their model counts tracked players, and the bench never shows removals.

Ablations share this code: hmm_combo_noconf, hmm_combo_noalive, hmm_combo_raw (same models,
features from the raw tracking: BAR-ATH only).

    python -m possession.bench train hmm_combo
    python -m possession.bench val hmm_combo
"""

import json
import os
import weakref

import numpy as np

from possession import learn, stoppage
from possession.experiments import hmm_conf, hmm_context, hmm_touch_ms
from possession.hmm_models import MODELS_DIR, HmmModels

ALIVE_PATH = os.path.join(MODELS_DIR, "hmm_combo_alive.json")
EVERY = 5
GRID = dict(hmm_touch_ms.GRID)          # hmm_touch's settings, not re-chosen
DEFAULT = dict(hmm_touch_ms.DEFAULT)
ALIVE_WINDOWS = (25, 100)               # half-widths in frames: ±1 s, ±4 s


# Cleaned tracking --------------------------------------------------------------------------------

class CleanView:
    """The match, with the cleaned tracking and per-frame sides in place of the raw ones."""

    def __init__(self, match):
        self._match = match

    def __getattr__(self, name):
        return getattr(self._match, name)

    track_xy = property(lambda self: self._match.track_xy_clean)
    track_speed = property(lambda self: self._match.track_speed_clean)
    track_team = property(lambda self: self._match.track_team_fixed)


_views = weakref.WeakKeyDictionary()


def clean_view(match):
    try:
        return _views[match]
    except KeyError:
        view = _views[match] = CleanView(match)
        return view


# Cross-fitted P(alive) ---------------------------------------------------------------------------

def _fit_alive(matches):
    rows, alive = [], []
    for match in matches:
        X, names = stoppage.features(match)
        rows.append(X[::EVERY])
        alive.append(np.asarray(match.labels)[::EVERY] != "D")
    return learn.fit_logistic(np.concatenate(rows), np.concatenate(alive), names=names)


def fit_alive(matches, path=ALIVE_PATH, say=print):
    ids = sorted({m.match_id for m in matches})
    model = {"about": "hmm_combo's alive models: 'all' on every match given to fit, and one per "
                      "match fitted without it (cross-fitting).",
             "matches": ids,
             "all": _fit_alive(matches),
             "folds": {i: _fit_alive([m for m in matches if m.match_id != i]) for i in ids}}
    with open(path, "w") as handle:
        json.dump(model, handle, indent=1)
        handle.write("\n")
    say("  alive: fitted on {} matches, {} folds; saved {}".format(
        len(ids), len(ids), os.path.relpath(path)))
    _alive_models.pop(path, None)
    _p_alive.pop(path, None)
    return model


_alive_models = {}                      # path -> the saved alive models
_p_alive = {}                           # path -> {match: P(alive)}


def _load_alive(path=ALIVE_PATH):
    if path not in _alive_models:
        with open(path) as handle:
            _alive_models[path] = json.load(handle)
    return _alive_models[path]


def p_alive(match, path=ALIVE_PATH):
    """(n,) P(play is on), out of fold on any match the alive model was fitted on."""
    memo = _p_alive.setdefault(path, weakref.WeakKeyDictionary())
    if match in memo:
        return memo[match]
    models = _load_alive(path)
    model = models["folds"].get(getattr(match, "match_id", None), models["all"])
    p = learn.predict_proba(model, stoppage.features(match)[0])
    memo[match] = p
    return p


def alive_features(match, path=ALIVE_PATH):
    p = np.clip(p_alive(match, path), 1e-4, 1 - 1e-4)
    columns, names = [p, np.log(p / (1 - p))], ["p_alive", "alive_logit"]
    for half in ALIVE_WINDOWS:
        columns.append(hmm_context.centred_mean(p, half))
        names.append("p_alive_w{}".format(half))
    return np.column_stack(columns), tuple(names)


# Features and models -----------------------------------------------------------------------------

def make_features(conf=True, alive=True, clean=True, alive_path=ALIVE_PATH, extra=()):
    """hmm_combo's feature function. Each group can be switched off; `extra` is a tuple of more
    sources, each match -> (X, names), appended last (e.g. dead-ball evidence from elsewhere, in
    place of or next to P(alive))."""
    def features(match):
        tracked = clean_view(match) if clean else match
        X, names = hmm_touch_ms.features(tracked)
        parts, names = [X], tuple(names)
        if conf:
            C, cnames = hmm_conf.conf_features(tracked)
            parts.append(C)
            names += cnames
        if alive:
            A, anames = alive_features(match, alive_path)
            parts.append(A)
            names += anames
        for source in extra:
            E, enames = source(match)
            parts.append(E)
            names += tuple(enames)
        return np.nan_to_num(np.column_stack(parts)), names
    return features


features = make_features()
MODEL = HmmModels("hmm_combo", features)
predict_with = MODEL.predict_with


def fit(matches, say=print):
    fit_alive(matches, say=say)
    return MODEL.fit(matches, say=say)


def predict(match):
    return predict_with(match, **DEFAULT)
