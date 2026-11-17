"""players_fill_keepers, with stoppages found on every frame instead of only in the ball's gaps.

players_fill calls a frame dead only where the tracking has lost the ball; where it has the ball,
the nearest player's side always wins, so throw-ins, goal kicks and fouls with the ball in view go
unseen, and a possession runs straight through them. Here:

  1. alive: one model reads every frame (possession/stoppage.py: the team shape, how many players
     are standing, how bunched they are, how many are tracked, where the ball is and how fast it
     moves, each also averaged over the seconds around the frame) and gives P(play is on);
  2. stoppages are cut from it with hysteresis and minimum lengths (stoppage.cut): D there;
  3. side: in the live stretches, the nearest player's side where the ball is tracked (keepers
     on the right side, as players_fill_keepers), players_fill's home model
     where it is not;
  4. the centred majority vote, run inside each live stretch on its own, so votes never cross a
     stoppage and the possession after a restart does not inherit the one before it.

The alive model is fitted on the bench's training matches (all their frames, ball or not) and
saved in possession/models/stoppages.json; the home model is players_fill's. Refit and choose the
settings with:

    python -m possession.bench train stoppages
"""

import json
import os
import weakref

import numpy as np

from possession import learn, stoppage, team_shape
from possession.experiments import players_fill
from possession.experiments.nearest_smooth import nearest_team_codes
from starter.baseline import AWAY, HOME, NOBODY, smooth  # organisers': imported, never edited

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "stoppages.json")
EVERY = 5   # fit on every 5th frame: neighbouring frames are near copies

GRID = {"window": [25, 50, 75], "enter": [0.35, 0.5], "leave": [0.5, 0.65],
        "min_dead": [25, 75]}
DEFAULT = {"window": 50, "enter": 0.5, "leave": 0.5, "min_dead": 75}  # chosen by `bench train`
MIN_LIVE = 25

# match -> P(alive) and match -> side codes, so a GRID search computes each once per match.
_p_alive = weakref.WeakKeyDictionary()
_sides = weakref.WeakKeyDictionary()


def _memo(store, match, compute):
    try:
        return store[match]
    except (KeyError, TypeError):
        pass
    value = compute(match)
    try:
        store[match] = value
    except TypeError:
        pass
    return value


def fit(matches, path=MODEL_PATH, say=print):
    """Fit the alive model on every EVERY-th frame of `matches` (they carry labels) and save it."""
    rows, alive = [], []
    for match in matches:
        X, names = stoppage.features(match)
        rows.append(X[::EVERY])
        alive.append(np.asarray(match.labels)[::EVERY] != "D")
    X, alive = np.concatenate(rows), np.concatenate(alive)
    model = {
        "about": "stoppages alive model, fitted by `python -m possession.bench train stoppages` "
                 "on the bench's training matches (ADR 0003). Every 5th frame, ball or not.",
        "matches": sorted({repr(m)[14:-1] for m in matches}),
        "alive": learn.fit_logistic(X, alive, names=names),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(model, handle, indent=1)
        handle.write("\n")
    say("  fitted on {} frames ({} alive); saved {}".format(
        len(X), int(alive.sum()), os.path.relpath(path)))
    _load.cache = None
    _p_alive.clear()
    return model


def _load(path=MODEL_PATH):
    if getattr(_load, "cache", None) is None:
        with open(path) as handle:
            _load.cache = json.load(handle)
    return _load.cache


def p_alive(match):
    """(n,) P(play is on) for every frame, computed once per match object."""
    return _memo(_p_alive, match,
                 lambda m: learn.predict_proba(_load()["alive"], stoppage.features(m)[0]))


def side_codes(match):
    """(n,) int8 HOME / AWAY on every frame: the nearest player where the ball is tracked, the
    players_fill home model where it is not (or where nobody is tracked near it)."""
    return _memo(_sides, match, _side_codes)


def _side_codes(match):
    near = nearest_team_codes(match.ball_xy, match.track_xy, match.track_team_fixed)
    X, _ = team_shape.features(match)
    home = learn.predict_proba(players_fill._load()["home"], X) >= 0.5
    shape = np.where(home, HOME, AWAY).astype(np.int8)
    return np.where(near == NOBODY, shape, near).astype(np.int8)


def label(codes, dead, window):
    """(n,) "H" / "A" / "D": D on `dead`, the majority vote of `codes` within each live stretch."""
    labels = np.full(len(codes), "D", dtype="<U1")
    for start, end, is_dead in stoppage.runs(dead):
        if not is_dead:
            labels[start:end] = smooth(codes[start:end], window)
    return labels


def predict_with(match, window, enter, leave, min_dead):
    dead = stoppage.cut(p_alive(match), enter, leave, min_dead, MIN_LIVE)
    return label(side_codes(match), dead, window)


def predict(match):
    return predict_with(match, **DEFAULT)
