"""The rule-based candidate (_control), on synthetic play with one player per side."""

import numpy as np

from possession.experiments import _control

PITCH = (105.0, 68.0)
TEAM = np.array(["H", "A"])
FAR = np.array([50.0, 10.0])  # where the player without the ball stands


class Play:
    """Builds ball and track positions frame by frame."""

    def __init__(self):
        self.ball, self.tracks = [], []

    def hold(self, side, frames, at=(50.0, 34.0)):
        """`side` ("H" or "A") has the ball at its feet; the other player is far away."""
        for _ in range(frames):
            near = np.array(at) + [0.5, 0.0]
            pair = [near, FAR] if side == "H" else [FAR, near]
            self.ball.append(at)
            self.tracks.append(pair)
        return self

    def duel(self, frames, at=(50.0, 34.0)):
        """Both players next to the ball."""
        for _ in range(frames):
            self.ball.append(at)
            self.tracks.append([np.array(at) + [0.5, 0], np.array(at) - [0.5, 0]])
        return self

    def lost(self, frames):
        for _ in range(frames):
            self.ball.append((np.nan, np.nan))
            self.tracks.append([FAR, FAR + 5])
        return self

    def out(self, frames):
        for _ in range(frames):
            self.ball.append((-5.0, 34.0))
            self.tracks.append([FAR, FAR + 5])
        return self

    def label(self, **settings):
        return _control.label(np.array(self.ball, dtype=np.float32),
                              np.array(self.tracks, dtype=np.float32), TEAM, PITCH, **settings)


def spells(labels):
    """[(label, length)] runs."""
    out = []
    for value in labels:
        if out and out[-1][0] == value:
            out[-1][1] += 1
        else:
            out.append([value, 1])
    return [tuple(run) for run in out]


def test_interception_changes_possession_at_its_first_frame():
    labels = Play().hold("H", 50).hold("A", 50).label()
    assert spells(labels) == [("H", 50), ("A", 50)]


def test_a_deflection_does_not_change_possession():
    labels = Play().hold("H", 50).hold("A", 3).hold("H", 50).label()
    assert spells(labels) == [("H", 103)]


def test_a_duel_is_not_evidence():
    labels = Play().hold("H", 50).duel(40).hold("H", 10).label()
    assert spells(labels) == [("H", 100)]


def test_short_tracking_gaps_are_bridged():
    labels = Play().hold("H", 50).lost(40).hold("H", 50).label()
    assert spells(labels) == [("H", 140)]


def test_long_tracking_gaps_are_dead_ball_and_the_restart_goes_to_the_first_side_in_control():
    labels = Play().hold("H", 50).lost(200).duel(5).hold("A", 50).label()
    assert spells(labels) == [("H", 50), ("D", 200), ("A", 55)]


def test_ball_off_the_pitch_is_dead_but_a_blip_is_not():
    assert spells(Play().hold("H", 50).out(30).lost(20).hold("A", 50).label()) == [
        ("H", 50), ("D", 50), ("A", 50)]
    assert spells(Play().hold("H", 50).out(3).hold("H", 50).label()) == [("H", 103)]


def test_live_ball_with_no_evidence_abstains():
    labels = Play().duel(30).label()
    assert spells(labels) == [("X", 30)]


def test_settings_change_the_outcome():
    play = Play().hold("H", 50).hold("A", 8).hold("H", 50)
    assert spells(play.label()) == [("H", 50), ("A", 8), ("H", 50)]
    assert spells(play.label(confirm=10)) == [("H", 108)]
