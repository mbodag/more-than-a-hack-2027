"""Stoppage features, the hysteresis cut and the stoppages experiment, on synthetic data."""

import numpy as np
import pytest

from possession import stoppage
from possession.experiments import stoppages
from starter.baseline import AWAY, HOME

from tests.test_players_fill import synthetic_match


def test_rolling_mean_skips_nan_and_clips_at_the_edges():
    values = np.array([1.0, np.nan, 3.0, 5.0])
    out = stoppage.rolling_mean(values, 1, 1)
    np.testing.assert_allclose(out, [1.0, 2.0, 4.0, 4.0])
    assert np.isnan(stoppage.rolling_mean(np.array([np.nan, np.nan]), 1, 0)).all()
    np.testing.assert_allclose(stoppage.rolling_mean(np.arange(4.0), 2, 0), [0, 0.5, 1, 2])


def test_crowding_counts_neighbours_and_ignores_missing_players():
    xy = np.full((2, 4, 2), np.nan)
    xy[0] = [[0, 0], [1, 0], [2, 0], [50, 0]]   # three within 5 m of the middle one
    xy[1, :2] = [[0, 0], [30, 0]]               # two far apart, two missing
    crowd, nearest = stoppage.crowding(xy, radius=5.0, chunk=1)
    np.testing.assert_allclose(crowd, [2, 0])
    np.testing.assert_allclose(nearest, [1.0, 30.0])


def test_ball_out_is_the_distance_beyond_the_lines():
    ball = np.array([[50, 30], [-2, 30], [50, 71], [np.nan, np.nan]])
    cols = stoppage.ball_columns(ball, (105, 68))
    np.testing.assert_allclose(cols["ball_out"], [0, 2, 3, 0])
    np.testing.assert_allclose(cols["ball_tracked"], [1, 1, 1, 0])


def test_cut_uses_hysteresis():
    p = np.array([0.9, 0.45, 0.55, 0.45, 0.55, 0.7, 0.9])
    # Below 0.5 at frame 1; 0.55 does not reach leave=0.6, so it stays stopped until frame 5.
    dead = stoppage.cut(p, enter=0.5, leave=0.6, min_dead=0, min_live=0)
    np.testing.assert_array_equal(dead, [0, 1, 1, 1, 1, 0, 0])
    # With no gap between the thresholds the same probability flickers.
    dead = stoppage.cut(p, enter=0.5, leave=0.5, min_dead=0, min_live=0)
    np.testing.assert_array_equal(dead, [0, 1, 0, 1, 0, 0, 0])


def test_cut_drops_short_stoppages_and_fills_short_live_stretches():
    p = np.r_[np.ones(10), np.zeros(3), np.ones(10), np.zeros(20), np.ones(2), np.zeros(20),
              np.ones(10)]
    dead = stoppage.cut(p, enter=0.5, leave=0.5, min_dead=5, min_live=5)
    assert not dead[10:13].any()            # 3 frames: too short to be a stoppage
    assert dead[23:65].all()                # the 2 live frames between two stoppages are absorbed
    assert not dead[:23].any() and not dead[65:].any()


def test_cut_refuses_leave_below_enter():
    with pytest.raises(ValueError):
        stoppage.cut(np.ones(3), enter=0.6, leave=0.4, min_dead=0, min_live=0)


def test_cut_keeps_a_stoppage_running_to_the_end():
    dead = stoppage.cut(np.r_[np.ones(3), np.zeros(4)], 0.5, 0.5, 0, 0)
    np.testing.assert_array_equal(dead, [0, 0, 0, 1, 1, 1, 1])


def test_label_never_votes_across_a_stoppage():
    codes = np.r_[np.full(10, HOME), np.full(4, HOME), np.full(10, AWAY)].astype(np.int8)
    dead = np.zeros(24, dtype=bool)
    dead[10:14] = True
    labels = stoppages.label(codes, dead, window=50)
    assert "".join(labels) == "H" * 10 + "D" * 4 + "A" * 10


def test_features_have_one_column_per_name_and_no_nan():
    match = synthetic_match()
    X, names = stoppage.features(match)
    assert X.shape == (match.n_frames, len(stoppage.NAMES))
    assert names == stoppage.NAMES
    assert not np.isnan(X).any()


def test_fit_and_predict_on_a_synthetic_match(tmp_path, monkeypatch):
    match = synthetic_match()
    path = str(tmp_path / "stoppages.json")
    monkeypatch.setattr(stoppages, "MODEL_PATH", path)
    stoppages._load.cache = None
    model = stoppages.fit([match], path=path, say=lambda *a: None)
    assert model["alive"]["names"] == list(stoppage.NAMES)
    monkeypatch.setattr(stoppages._load, "cache", model)
    labels = stoppages.predict(match)
    assert labels.shape == (match.n_frames,)
    assert set(labels) <= {"H", "A", "D"}
    stoppages._load.cache = None


class _Truth:
    labels = np.array(list("HHDDAAAA"))
    blocks = [(0, 7)]


def test_oracle_stoppages_puts_the_true_dead_balls_on_the_base(monkeypatch):
    from possession.experiments import oracle_stoppages
    monkeypatch.setattr(oracle_stoppages, "base_labels",
                        lambda match, base: np.array(list("HHHHDAAH")))
    out = oracle_stoppages.predict_with(_Truth(), base="any")
    assert "".join(out) == "HHDDHAAH"   # base's D at frame 4 takes the side before it


def test_oracle_team_keeps_the_base_dead_balls_and_takes_the_true_sides(monkeypatch):
    from possession.experiments import oracle_team
    monkeypatch.setattr(oracle_team, "base_labels",
                        lambda match, base: np.array(list("DAAHHDHH")))
    out = oracle_team.predict_with(_Truth(), base="any")
    assert "".join(out) == "DHHHADAA"   # truth's D at 2-3 missed by base: the side carried on
