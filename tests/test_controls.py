"""possession/controls.py on small made-up matches."""

import numpy as np

from possession import controls, viz


class Fake:
    def __init__(self, ball_xy, track_xy, sides):
        self.ball_xy = np.asarray(ball_xy, dtype=np.float32)
        self.track_xy = np.asarray(track_xy, dtype=np.float32)
        self.n_frames = len(self.ball_xy)
        self.blocks = [(0, self.n_frames - 1)]
        self.track_team_fixed = np.broadcast_to(np.array(sides)[None, :],
                                                (self.n_frames, len(sides)))


def dribble(n=40):
    """A home player runs at 5 m/s with the ball at their feet; an away player stands 10 m off."""
    t = np.arange(n) / 25.0
    player = np.stack([10 + 5 * t, np.full(n, 30.0)], axis=1)
    ball = player + [0.4, 0.0]
    away = np.tile([[30.0, 40.0]], (n, 1))
    return Fake(ball, np.stack([player, away], axis=1), ["H", "A"])


def test_a_dribble_is_one_confident_home_control():
    ep = controls.episodes(dribble())
    assert len(ep["start"]) == 1
    assert ep["side"][0] == "H" and ep["track"][0] == 0
    assert ep["confidence"][0] > 0.9


def test_a_ball_flying_past_is_not_confident():
    n = 40
    ball = np.stack([np.linspace(0, 60, n), np.full(n, 31.5)], axis=1)   # 37 m/s, passes 1.5 m away
    match = Fake(ball, np.stack([np.tile([[30.0, 30.0]], (n, 1)),
                                 np.tile([[30.0, 50.0]], (n, 1))], axis=1), ["A", "H"])
    ep = controls.episodes(match)
    assert len(ep["start"]) == 1 and ep["side"][0] == "A"
    assert ep["confidence"][0] < 0.5


def test_no_control_where_the_ball_is_not_tracked():
    match = dribble()
    match.ball_xy[:] = np.nan
    assert len(controls.episodes(match).get("start", ())) == 0
    assert viz.control_rows(match) == []


def test_control_rows_use_page_slots():
    rows = viz.control_rows(dribble())
    assert rows[0][2] == 1 and rows[0][3] == "H"   # slot 0 is the ball, so track 0 is slot 1
