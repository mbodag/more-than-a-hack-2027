"""hmm_viterbi.ball_features with per-frame sides, as hmm_combo's cleaned view gives them."""

import numpy as np

from possession.experiments import hmm_viterbi


class Fake:
    def __init__(self, track_team):
        n = 50
        t = np.arange(n) / 25.0
        home = np.stack([20 + 5 * t, np.full(n, 30.0)], axis=1)
        away = np.stack([60 - 5 * t, np.full(n, 40.0)], axis=1)
        keeper = home + [1.0, 0.0]          # the keeper slot's tracklet, by the ball
        self.track_xy = np.stack([home, away, keeper], axis=1).astype(np.float32)
        self.ball_xy = (home + [0.5, 0.0]).astype(np.float32)
        self.ball_speed = np.full(n, 5.0, dtype=np.float32)
        self.track_team = track_team
        self.n_frames = n
        self.blocks = [(0, n - 1)]
        self.pitch = (105.0, 68.0)


def test_per_frame_sides_that_never_change_give_the_slot_sides_features():
    sides = np.array(["H", "A", "A"])
    flat, names = hmm_viterbi.ball_features(Fake(sides))
    per_frame, _ = hmm_viterbi.ball_features(Fake(np.tile(sides, (50, 1))))
    np.testing.assert_allclose(flat, per_frame)


def test_a_slot_moved_to_the_other_side_counts_for_that_side_on_those_frames():
    sides = np.tile(np.array(["H", "A", "A"]), (50, 1))
    sides[25:, 2] = "H"            # the keeper slot is given to home from frame 25
    X, names = hmm_viterbi.ball_features(Fake(sides))
    near_away = X[:, names.index("near_away")]
    assert near_away[10] < 1.0      # the slot counts for away: away is by the ball
    assert near_away[40] > 5.0      # given to home: the nearest away player is the runner
    flat, _ = hmm_viterbi.ball_features(Fake(np.array(["H", "A", "A"])))
    np.testing.assert_allclose(X[:25], flat[:25])
