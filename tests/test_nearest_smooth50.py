"""#13: the baseline through our Match items agrees with starter/baseline.py, on a synthetic ATD."""

import types

import numpy as np

from possession.experiments import _geometry, nearest_smooth50
from possession.match import parse_atd
from starter import baseline

PITCH = (105.0, 68.0)
N_SLOTS = 6
TEAM = np.array(["H", "H", "H", "A", "A", "A"])


def synthetic_atd(path, n_frames=400, seed=0):
    rng = np.random.default_rng(seed)
    lines = []
    for frame in range(1, n_frames + 1):
        groups = []
        for _ in range(N_SLOTS):
            if rng.random() < 0.15:
                groups.append("NaN,NaN,NaN")
            else:
                groups.append("{:.5f},{:.5f},{:.2f}".format(rng.random(), rng.random(),
                                                            rng.random() * 8))
        ball = ("NaN,NaN,NaN" if rng.random() < 0.4 else
                "{:.5f},{:.5f},{:.2f}".format(rng.random(), rng.random(), rng.random() * 20))
        lines.append("{}:{}:{}".format(frame, ";".join(groups), ball))
    path.write_text("\n".join(lines) + "\n")
    return n_frames


def test_nearest_side_matches_the_baseline(tmp_path):
    raw = tmp_path / "atd.txt"
    n = synthetic_atd(raw)
    team_codes = np.where(TEAM == "H", baseline.HOME, baseline.AWAY).astype(np.int8)
    expected = baseline.nearest_player_labels(str(raw), team_codes, n, PITCH)

    items = parse_atd(str(raw), N_SLOTS, n, PITCH)
    ours = _geometry.nearest_side(_geometry.ball_distances(items["ball_xy"], items["track_xy"]),
                                  TEAM)
    assert np.array_equal(ours, expected)

    match = types.SimpleNamespace(ball_xy=items["ball_xy"], track_xy=items["track_xy"],
                                  track_team=TEAM)
    assert np.array_equal(nearest_smooth50.predict(match), baseline.smooth(expected, 50))


def test_side_distances():
    d = np.array([[1.0, np.nan, 3.0, 0.5], [np.nan, np.nan, 2.0, np.nan]])
    home, away = _geometry.side_distances(d, np.array(["H", "H", "A", "A"]))
    assert home[0] == 1.0 and np.isnan(home[1])
    assert away.tolist() == [0.5, 2.0]
