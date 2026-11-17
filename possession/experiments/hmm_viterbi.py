"""A 3-state hidden Markov model (H, A, D) decoded by Viterbi (possession/hmm.py).

Instead of a hard guess per frame and a majority vote, every frame gets a probability for each
state, and Viterbi picks the best sequence over each half, paying a penalty for every change:
  - ball tracked: two logistic models on ball features (each side's nearest player, the margin
    between them, ball speed, how far outside the pitch, where along it) plus the team shape;
  - ball hidden: players_fill's two models on the team shape, as they are.
Each pair gives p(alive) and p(home | alive), so p(D) = 1 - p_alive, p(H) = p_alive * p_home and
p(A) = p_alive * (1 - p_home). The emission of a state is w * (log p(state) - log prior(state)).
Neighbouring frames are near copies, so their evidence is not independent: `w` (well below 1)
says how much one frame counts. Transitions start from the rates counted in the training labels,
and `switch` / `dead` scale the cost of a hand-over and of entering or leaving D.

The models and rates are fitted on the bench's training matches only (ADR 0003), and the GRID is
chosen there, by:

    python -m possession.bench train hmm_viterbi
"""

import json
import os
import weakref

import numpy as np

from possession import hmm, learn, team_shape
from possession.experiments import players_fill

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "hmm_viterbi.json")
EVERY = 5     # fit on every 5th frame: neighbouring frames are near copies
CAP = 30.0    # m: farther than this, a player is simply far from the ball
EPS = 1e-4    # probabilities are kept within [EPS, 1 - EPS]

GRID = {"w": [0.4, 0.7, 1.0, 1.5], "switch": [0.5, 0.75, 1.0], "dead": [0.5, 0.75, 1.0]}
DEFAULT = {"w": 0.4, "switch": 0.5, "dead": 0.75}  # chosen by `bench train` (experiments/metrics/bench)

BALL_NAMES = ("near_home", "near_away", "near_margin", "near_any", "ball_speed", "ball_out",
              "ball_x", "ball_y_centre")


# Features ----------------------------------------------------------------------------------------

def ball_features(match):
    """(X, names): ball features per frame, 0 where the ball is not tracked. x is turned so that the
    home side attacks +x, like the team shape."""
    ball = np.asarray(match.ball_xy, dtype=np.float64)
    xy = np.asarray(match.track_xy, dtype=np.float64)
    team = np.asarray(match.track_team)
    length, width = team_shape.pitch_of(match)
    n = len(ball)

    with np.errstate(invalid="ignore"):
        dist = np.linalg.norm(xy - ball[:, None, :], axis=2)
    dist = np.where(np.isnan(dist), np.inf, dist)
    if team.ndim == 2:   # one side per slot and frame (match.track_team_fixed)
        near_home = np.minimum(np.where(team == "H", dist, np.inf).min(axis=1), CAP)
        near_away = np.minimum(np.where(team == "A", dist, np.inf).min(axis=1), CAP)
        slot_home = (team == "H").mean(axis=0) > 0.5   # the attack direction reads each slot's usual side
    else:
        near_home = np.minimum(dist[:, team == "H"].min(axis=1), CAP)
        near_away = np.minimum(dist[:, team == "A"].min(axis=1), CAP)
        slot_home = team == "H"

    sign = team_shape.home_attacks_plus_x(xy, slot_home, ~slot_home if team.ndim == 2 else team == "A",
                                          team_shape.blocks_of(match), n)
    x, y = ball[:, 0], ball[:, 1]
    out = np.maximum.reduce([np.zeros(n), -x, x - length, -y, y - width])
    speed = np.minimum(np.asarray(match.ball_speed, dtype=np.float64), 40.0)
    columns = (near_home, near_away, near_home - near_away, np.minimum(near_home, near_away),
               speed, np.minimum(out, 20.0), np.where(sign > 0, x, length - x),
               np.abs(y - width / 2))
    X = np.column_stack(columns)
    return np.nan_to_num(np.where(np.isnan(x)[:, None], 0.0, X)), BALL_NAMES


def visible_features(match):
    ball, bnames = ball_features(match)
    shape, snames = team_shape.features(match)
    return np.column_stack([ball, shape]), bnames + snames


# Fitting -----------------------------------------------------------------------------------------

def fit(matches, path=MODEL_PATH, say=print):
    """Fit the ball-tracked models and the transition rates on `matches` (they carry labels)."""
    rows, alive, home = [], [], []
    for match in matches:
        X, names = visible_features(match)
        seen = np.flatnonzero(~np.isnan(np.asarray(match.ball_xy)[:, 0]))[::EVERY]
        rows.append(X[seen])
        alive.append(match.labels[seen] != "D")
        home.append(match.labels[seen] == "H")
    X, alive, home = np.concatenate(rows), np.concatenate(alive), np.concatenate(home)

    halves, seen_ids = [], set()
    for match in matches:              # the same match under several seeds: count its labels once
        if match.match_id in seen_ids:
            continue
        seen_ids.add(match.match_id)
        halves += [match.labels[first:last + 1] for first, last in match.blocks]
    labels = np.concatenate(halves)
    prior = {s: float(np.mean(labels == s)) for s in hmm.STATES}

    models = {
        "about": "hmm_viterbi models, fitted by `python -m possession.bench train hmm_viterbi` on "
                 "the bench's training matches (ADR 0003). Ball-tracked frames only; hidden-ball "
                 "frames use players_fill's models.",
        "matches": sorted(seen_ids),
        "alive": learn.fit_logistic(X, alive, names=names),
        "home": learn.fit_logistic(X[alive], home[alive], names=names),
        "rates": hmm.fit_rates(halves),
        "prior": prior,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(models, handle, indent=1)
        handle.write("\n")
    say("  fitted on {} ball-tracked frames ({} alive); rates {}; saved {}".format(
        len(X), int(alive.sum()), json.dumps({k: round(v, 5) for k, v in models["rates"].items()}),
        os.path.relpath(path)))
    _load.cache = None
    _STATE_LOGP.clear()
    return models


def _load(path=None):
    if getattr(_load, "cache", None) is None:
        with open(path or MODEL_PATH) as handle:
            _load.cache = json.load(handle)
    return _load.cache


# Prediction --------------------------------------------------------------------------------------

_STATE_LOGP = weakref.WeakKeyDictionary()  # match -> (n, 3) log p(state); the grid reuses it


def state_logp(match, models=None):
    """(n, 3) log p(H), log p(A), log p(D) per frame, from whichever model pair fits the frame."""
    if models is None and match in _STATE_LOGP:
        return _STATE_LOGP[match]
    models = models or _load()
    hidden = np.isnan(np.asarray(match.ball_xy)[:, 0])

    shape, _ = team_shape.features(match)
    gap = players_fill._load()
    p_alive = learn.predict_proba(gap["alive"], shape)
    p_home = learn.predict_proba(gap["home"], shape)
    if (~hidden).any():
        X, _ = visible_features(match)
        p_alive = np.where(hidden, p_alive, learn.predict_proba(models["alive"], X))
        p_home = np.where(hidden, p_home, learn.predict_proba(models["home"], X))

    p_alive, p_home = np.clip(p_alive, EPS, 1 - EPS), np.clip(p_home, EPS, 1 - EPS)
    logp = np.log(np.column_stack([p_alive * p_home, p_alive * (1 - p_home), 1 - p_alive]))
    if models is _load.cache:
        _STATE_LOGP[match] = logp
    return logp


def predict_with(match, w, switch, dead):
    models = _load()
    log_prior = np.log([models["prior"][s] for s in hmm.STATES])
    log_emit = w * (state_logp(match) - log_prior)
    T = hmm.transitions(models["rates"], switch=switch, dead=dead)
    return hmm.decode(log_emit, T, team_shape.blocks_of(match))


def predict(match):
    return predict_with(match, **DEFAULT)
