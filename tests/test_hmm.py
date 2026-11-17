"""The HMM decoder (possession/hmm.py) and hmm_viterbi, on synthetic data."""

import itertools

import numpy as np

from possession import hmm
from possession.experiments import hmm_viterbi

from tests.test_players_fill import synthetic_match

RATES = {"switch": 0.005, "to_dead": 0.004, "from_dead": 0.01}


def emissions(states, strength=1.0):
    """(n, 3) log scores that favour `states` (a string of H/A/D) by `strength` per frame."""
    out = np.zeros((len(states), 3))
    for t, s in enumerate(states):
        out[t, hmm.STATES.index(s)] = strength
    return out


def letters(path):
    return "".join(hmm.STATES[i] for i in path)


def test_a_short_flicker_is_ignored_but_a_long_spell_is_kept():
    truth = "H" * 100 + "A" * 3 + "H" * 100 + "A" * 100
    path = hmm.viterbi(emissions(truth), hmm.transitions(RATES))
    assert letters(path) == "H" * 203 + "A" * 100


def test_a_run_of_dead_ball_evidence_becomes_one_stoppage():
    truth = "H" * 80 + "D" * 60 + "A" * 80
    noisy = emissions(truth)
    noisy[100:103] = emissions("HHH")          # three stray frames inside the stoppage
    path = letters(hmm.viterbi(noisy, hmm.transitions(RATES)))
    assert path == truth


def test_viterbi_matches_brute_force_on_tiny_inputs():
    rng = np.random.default_rng(0)
    T = np.log(rng.dirichlet(np.ones(3), size=3))
    for _ in range(20):
        E = rng.normal(size=(5, 3))
        best = max(itertools.product(range(3), repeat=5),
                   key=lambda p: E[0, p[0]] + sum(T[p[t - 1], p[t]] + E[t, p[t]]
                                                  for t in range(1, 5)))
        assert tuple(hmm.viterbi(E, T)) == best


def test_decode_never_crosses_a_block_and_fills_outside():
    truth = "H" * 50 + "A" * 50
    labels = hmm.decode(emissions(truth), hmm.transitions(RATES), [(0, 49), (60, 99)])
    assert "".join(labels[:50]) == "H" * 50
    assert "".join(labels[50:60]) == "D" * 10
    assert "".join(labels[60:]) == "A" * 40


def test_stickier_switch_needs_longer_evidence():
    truth = "H" * 100 + "A" * 4 + "H" * 100
    loose = hmm.transitions(RATES, switch=0.3)
    sticky = hmm.transitions(RATES, switch=3.0)
    assert "A" in letters(hmm.viterbi(emissions(truth, 2.0), loose))
    assert "A" not in letters(hmm.viterbi(emissions(truth, 2.0), sticky))


def test_fit_rates_counts_transitions():
    rates = hmm.fit_rates(["HHHHAAAADDDD", "HHHHHHHHHHHH"])
    assert all(0 < v < 1 for v in rates.values())
    assert rates["from_dead"] > rates["switch"]   # 3 of 4 D frames stay D; 1 of 18 alive switches


def test_hmm_viterbi_fits_and_predicts_on_a_synthetic_match(tmp_path, monkeypatch):
    match = synthetic_match(n=600)
    path = tmp_path / "hmm.json"
    hmm_viterbi.fit([match], path=str(path), say=lambda *a: None)
    monkeypatch.setattr(hmm_viterbi, "MODEL_PATH", str(path))
    labels = hmm_viterbi.predict_with(match, w=0.2, switch=1.0, dead=1.0)
    again = hmm_viterbi.predict_with(match, w=0.2, switch=1.0, dead=1.0)
    assert labels.shape == (600,) and set(labels) <= {"H", "A", "D"}
    np.testing.assert_array_equal(labels, again)
    hmm_viterbi._load.cache = None
    hmm_viterbi._STATE_LOGP.clear()


def test_hmm_context_windows_and_gaps():
    from possession.experiments import hmm_context
    mask = np.array([0, 1, 0, 0, 1, 0], dtype=bool)
    assert list(hmm_context.last_index(mask)) == [-1, 1, 1, 1, 4, 4]
    assert list(hmm_context.next_index(mask)) == [1, 1, 4, 4, 4, 6]
    assert np.allclose(hmm_context.centred_mean([0, 0, 3, 0, 0], 1), [0, 1, 1, 1, 0])


def test_hmm_context_fits_and_predicts_with_hidden_ball(tmp_path, monkeypatch):
    from possession.experiments import hmm_context
    match = synthetic_match(n=600)
    ball = match.ball_xy.copy()
    ball[100:180] = np.nan                       # a gap, so both model pairs have frames
    match = match.with_arrays("gap", ball_xy=ball)
    path = tmp_path / "ctx.json"
    hmm_context.fit([match], path=str(path), say=lambda *a: None)
    monkeypatch.setattr(hmm_context, "MODEL_PATH", str(path))
    labels = hmm_context.predict(match)
    assert labels.shape == (600,) and set(labels) <= {"H", "A", "D"}
    np.testing.assert_array_equal(labels, hmm_context.predict(match))
    hmm_context._load.cache = None
    hmm_context._STATE_LOGP.clear()


def _chain_score(path, E, T, k):
    """Score of a 3-state path under viterbi_dead_min's model, or None if a stoppage that ends
    inside the sequence is shorter than k. The first k - 1 frames of a stoppage cost nothing to
    stay in; after that staying costs T[D, D]."""
    score, run = E[0, path[0]], 1 if path[0] == hmm.D else 0
    for t in range(1, len(path)):
        a, b = path[t - 1], path[t]
        if a == hmm.D and b == hmm.D:
            score += 0.0 if run < k else T[hmm.D, hmm.D]
            run += 1
        else:
            if a == hmm.D and run < k:
                return None
            score += T[a, b]
            run = 1 if b == hmm.D else 0
        score += E[t, b]
    return score


def test_viterbi_dead_min_matches_brute_force():
    rng = np.random.default_rng(1)
    T = np.log(rng.dirichlet(np.ones(3), size=3))
    for k in (2, 3):
        for _ in range(15):
            E = rng.normal(size=(6, 3))
            E[0, hmm.D] = -1e9                      # no stoppage at the very start
            scored = [(s, p) for p in itertools.product(range(3), repeat=6)
                      if (s := _chain_score(p, E, T, k)) is not None]
            best = max(scored, key=lambda sp: sp[0])
            got = tuple(hmm.viterbi_dead_min(E, T, k))
            assert abs(_chain_score(got, E, T, k) - best[0]) < 1e-9


def test_dead_min_removes_a_short_stoppage_and_keeps_a_long_one():
    truth = "H" * 80 + "D" * 5 + "H" * 80 + "D" * 60 + "A" * 80
    T = hmm.transitions(RATES, switch=0.3, dead=0.3)
    plain = letters(hmm.viterbi(emissions(truth, 2.0), T))
    chain = letters(hmm.viterbi_dead_min(emissions(truth, 2.0), T, 25))
    assert plain == truth
    assert chain == "H" * 165 + "D" * 60 + "A" * 80
    assert letters(hmm.viterbi_dead_min(emissions(truth, 2.0), T, 1)) == plain


def test_mlp_learns_an_interaction_and_is_deterministic():
    from possession import learn, mlp
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 2))
    y = (X[:, 0] * X[:, 1]) > 0                          # XOR-like: no straight line separates it
    model = mlp.fit_mlp(X, y, hidden=16, steps=300, rate=0.05)
    assert model == mlp.fit_mlp(X, y, hidden=16, steps=300, rate=0.05)
    accuracy = np.mean((mlp.predict_proba(model, X) > 0.5) == y)
    linear = np.mean((learn.predict_proba(learn.fit_logistic(X, y), X) > 0.5) == y)
    assert accuracy > 0.9 and accuracy > linear + 0.2


def test_restart_hint_points_to_the_other_side_and_fades():
    from possession.experiments import hmm_restart
    labels = np.array(list("HHHHDDDAAAAAA"))
    last_control = np.array([1.0] * 4 + [1.0] * 3 + [-1.0] * 6)   # home touched it last
    ball_out = np.zeros(13, dtype=bool)
    ball_out[5] = True
    hint, hint_out, log_since = hmm_restart.restart_hints(labels, last_control, ball_out)
    assert (hint[:7] == 0).all()                       # nothing before the restart
    assert hint[7] == -1.0                             # away restarts, at full strength
    assert 0 > hint[12] > hint[7]                      # and it fades
    np.testing.assert_array_equal(hint_out, hint)      # the ball was seen out
    assert log_since[7] == 0.0


def test_rich_run_lengths_and_spots():
    from possession.experiments import hmm_rich
    before, after = hmm_rich.run_lengths(np.array([0, 1, 1, 1, 0, 1], dtype=bool))
    assert list(before) == [0, 1, 2, 3, 0, 1]
    assert list(after) == [0, 3, 2, 1, 0, 1]
    d = hmm_rich.spot_distance(np.array([0.5, 52.5, 30.0]), np.array([0.0, 34.0, 20.0]), 105, 68)
    assert d[0] == 0.5 and d[1] == 0.0 and d[2] > 10


def drop(s, blocks=None, k=4):
    labels = np.array(list(s))
    return "".join(hmm.drop_dead_islands(labels, blocks or [(0, len(s) - 1)], k))


def test_a_short_possession_between_two_stoppages_becomes_dead_ball():
    assert drop("DDDHHHDDD") == "DDDDDDDDD"
    assert drop("DDDHAHDDD") == "DDDDDDDDD"        # a flicker of both sides is one stretch


def test_a_possession_long_enough_or_touching_play_is_kept():
    assert drop("DDHHHHDD") == "DDHHHHDD"          # 4 frames: not shorter than the minimum
    assert drop("HHHDDDHH") == "HHHDDDHH"          # runs to the block edge on one side
    assert drop("DDHHDD", k=1) == "DDHHDD"         # off


def test_stretches_are_judged_inside_each_block():
    # D H | H D across a halftime boundary: each H touches its block's edge, so both stay.
    assert drop("DHHDDHHD", blocks=[(0, 2), (3, 7)]) == "DHHDDDDD"
