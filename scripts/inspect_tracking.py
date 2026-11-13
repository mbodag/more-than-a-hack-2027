"""Label-free facts about the tracking that the rule-based settings rest on (#5, option A).

    python scripts/inspect_tracking.py          # needs the match data

Prints, for the graded frames: the range of the ball's coordinates (does it span the pitch from 0?),
how often the ball is untracked and for how long at a stretch, how often it is off the pitch, and
how often each side has control evidence at a few reaches. Reads no ground truth.
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import games  # noqa: E402
from possession.experiments import _control, _geometry  # noqa: E402
from possession.fill import half_blocks  # noqa: E402

FPS = games.GAMES[games.GAME]["fps"]


def graded_mask(n_frames, blocks):
    mask = np.zeros(n_frames, dtype=bool)
    for first, last in blocks:
        mask[first:last + 1] = True
    return mask


def summarise(ball_xy, track_xy, track_team, pitch, graded, reaches=(1.0, 2.0, 3.0)):
    """A dict of the facts above, over the graded frames only."""
    ball = np.asarray(ball_xy, dtype=np.float64)[graded]
    missing = np.isnan(ball[:, 0])
    seen = ball[~missing]
    gaps = np.array([end - start for start, end in _control.runs(missing)]) / FPS
    length, width = pitch
    with np.errstate(invalid="ignore"):
        off = ((seen[:, 0] < 0) | (seen[:, 0] > length) | (seen[:, 1] < 0) | (seen[:, 1] > width))
    distances = _geometry.ball_distances(ball_xy, track_xy)[graded]
    home, away = _geometry.side_distances(distances, track_team)
    team = np.asarray(track_team)
    tracked = ~np.isnan(np.asarray(track_xy)[graded][:, :, 0])
    out = {
        "graded_frames": int(graded.sum()),
        "ball_x_percentiles_0_1_50_99_100": np.percentile(seen[:, 0], [0, 1, 50, 99, 100]).round(2)
        .tolist() if len(seen) else [],
        "ball_y_percentiles_0_1_50_99_100": np.percentile(seen[:, 1], [0, 1, 50, 99, 100]).round(2)
        .tolist() if len(seen) else [],
        "pitch": list(pitch),
        "ball_missing_share": round(float(missing.mean()), 3),
        "ball_off_pitch_share_of_seen": round(float(off.mean()), 3) if len(seen) else None,
        "gap_count": int(len(gaps)),
        "gap_seconds_percentiles_50_90_99": np.percentile(gaps, [50, 90, 99]).round(2).tolist()
        if len(gaps) else [],
        "missing_frames_in_gaps_over_s": {
            s: round(float(gaps[gaps > s].sum() * FPS / max(1, missing.sum())), 3)
            for s in (1, 3, 5, 10)},
        "tracks_per_frame_home_away_median": [int(np.median(tracked[:, team == t].sum(axis=1)))
                                              for t in ("H", "A")],
    }
    for reach in reaches:
        ev = _control.evidence(home, away, reach, _control.DEFAULTS["margin"])
        out["evidence_share_reach_{}m_H_A".format(reach)] = [
            round(float((ev == side).mean()), 3) for side in (_control.HOME_EV, _control.AWAY_EV)]
    return out


def main():
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import check_data
    from possession.match import Match

    check_data.check_data()
    match = Match()
    graded = graded_mask(match.n_frames, half_blocks())
    facts = summarise(match.ball_xy, match.track_xy, match.track_team,
                      games.GAMES[games.GAME]["pitch"], graded)
    for key, value in facts.items():
        print("{:<42} {}".format(key, value))


if __name__ == "__main__":
    main()
