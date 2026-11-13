"""scripts/inspect_tracking.py on synthetic arrays."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
import inspect_tracking  # noqa: E402


def test_summarise_counts_gaps_and_off_pitch_frames():
    n = 100
    ball = np.tile([50.0, 30.0], (n, 1))
    ball[10:20] = np.nan        # a 0.4 s gap
    ball[50:55] = [-3.0, 30.0]  # off the pitch
    tracks = np.zeros((n, 2, 2))
    tracks[:, 0] = [50.5, 30.0]
    tracks[:, 1] = [10.0, 10.0]
    graded = inspect_tracking.graded_mask(n, [(0, 99)])
    facts = inspect_tracking.summarise(ball, tracks, np.array(["H", "A"]), (105.0, 68.0), graded)
    assert facts["graded_frames"] == 100
    assert facts["ball_missing_share"] == 0.1
    assert facts["gap_count"] == 1
    assert facts["ball_off_pitch_share_of_seen"] == round(5 / 90, 3)
    assert facts["evidence_share_reach_2.0m_H_A"] == [0.85, 0.0]
