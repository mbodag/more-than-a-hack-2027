"""TRIAL (not benched, not committed): hmm_final plus Smart Tagging set pieces as extra evidence.

Every set piece is a restart, and every restart ends a stoppage. For each `SET PIECES` tag:
  - the restart frame r: the end of the ball gap (at least MIN_GAP frames) nearest the tag start,
    if one ends within SEARCH frames of it; otherwise the tag start itself;
  - before r, back to the start of that gap (at most MAX_DEAD frames), D gets +D_BONUS per frame;
    with no gap, the PRE frames before r do;
  - for POST frames from r, the tagged team gets +TEAM_BONUS per frame (not for throw-ins, which
    are all tagged N/A, nor for tags whose team is mixed).
The bonuses are added to hmm_final's emissions, then the same Viterbi decodes. Bonus sizes and
windows are guesses, not fitted: the bench matches have no Smart Tagging.

    python -m possession.experiments.hmm_setpiece   # aggregate timing statistics only
    python -m possession run hmm_setpiece           # outputs/hmm_setpiece.csv
"""

import numpy as np

from possession import hmm, team_shape
from possession.experiments import hmm_final

FPS = 25
MIN_GAP = 2 * FPS        # a ball gap shorter than this is tracking noise, not a stoppage
SEARCH = 10 * FPS        # how far from the tag start the restart may be
MAX_DEAD = 30 * FPS      # longest stretch before a restart that gets the D bonus
PRE = 5 * FPS            # D window when no ball gap is found
POST = 5 * FPS           # frames after the restart that favour the tagged team
D_BONUS = 1.0
TEAM_BONUS = 1.0
PREFIX = "SET PIECES"


def _runs(flags):
    """[(start, end)] half-open runs of True in a 1-D bool array."""
    edges = np.diff(np.concatenate(([0], flags.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def set_pieces(match):
    """[(start, end, type, side)] for every set-piece interval; side is "H", "A", "N" or "?"."""
    codes = list(np.asarray(match.smart_tagging_codes).tolist())
    team = np.asarray(match.smart_tagging_team)
    out = []
    for j, code in enumerate(codes):
        if not code.upper().startswith(PREFIX):
            continue
        kind = code.split(":", 1)[1].strip() if ":" in code else ""
        column = team[:, j]
        for s, e in _runs(column != ""):
            sides = set(column[s:e].tolist())
            out.append((int(s), int(e), kind, sides.pop() if len(sides) == 1 else "?"))
    return sorted(out)


def restart_windows(match):
    """[(type, side, restart, dead_from, gap_found)] for every set piece."""
    hidden = np.isnan(np.asarray(match.ball_xy)[:, 0])
    gaps = [(s, e) for s, e in _runs(hidden) if e - s >= MIN_GAP]
    ends = np.array([e for _, e in gaps]) if gaps else np.array([], dtype=int)
    out = []
    for s, _, kind, side in set_pieces(match):
        if len(ends):
            k = int(np.argmin(np.abs(ends - s)))
            if abs(ends[k] - s) <= SEARCH:
                r = int(ends[k])
                out.append((kind, side, r, max(int(gaps[k][0]), r - MAX_DEAD), True))
                continue
        out.append((kind, side, s, max(0, s - PRE), False))
    return out


def bonus(match):
    """(n, 3) additions to the emissions over H, A, D."""
    B = np.zeros((match.n_frames, 3))
    H, A, D = (hmm.STATES.index(s) for s in ("H", "A", "D"))
    for _, side, r, dead_from, _ in restart_windows(match):
        B[dead_from:r, D] += D_BONUS
        if side in ("H", "A"):
            B[r:r + POST, H if side == "H" else A] += TEAM_BONUS
    return B


def predict(match, **settings):
    return predict_on(hmm_final, match, **settings)


def predict_on(base, match, **settings):
    """`base` (an experiment module with MODEL and DEFAULT) decoded with the set-piece bonuses."""
    s = dict(base.DEFAULT, **settings)
    model = base.MODEL
    models = model.load()
    T = hmm.transitions(models["rates"], switch=s["switch"], dead=s["dead"])
    E = model.emissions(match, s["w"]) + bonus(match)
    return hmm.decode(E, T, team_shape.blocks_of(match), dead_min=s["dead_min"])


def timing_report(match):
    """Aggregates only: how set-piece tags line up with the ball gaps, per type."""
    hidden = np.isnan(np.asarray(match.ball_xy)[:, 0])
    pieces = set_pieces(match)
    windows = restart_windows(match)
    print("{:<24}{:>4}{:>10}{:>12}{:>10}{:>14}{:>12}".format(
        "type", "n", "len s", "missing", "gap near", "r - start s", "dead s"))
    for kind in sorted({p[2] for p in pieces}):
        rows = [(p, w) for p, w in zip(pieces, windows) if p[2] == kind]
        lens = [(p[1] - p[0]) / FPS for p, _ in rows]
        miss = [hidden[p[0]:p[1]].mean() for p, _ in rows]
        found = [w for _, w in rows if w[4]]
        offs = [(w[2] - p[0]) / FPS for p, w in rows if w[4]]
        dead = [(w[2] - w[3]) / FPS for _, w in rows]
        sides = {sd: sum(1 for p, _ in rows if p[3] == sd) for sd in "HAN?"}
        print("{:<24}{:>4}{:>10.1f}{:>12.2f}{:>10}{:>14}{:>12.1f}   sides {}".format(
            kind, len(rows), np.median(lens), np.median(miss), len(found),
            "{:.1f}".format(np.median(offs)) if offs else "-", np.median(dead),
            " ".join("{}={}".format(k, v) for k, v in sides.items() if v)))


if __name__ == "__main__":
    from possession.match import Match
    timing_report(Match(log=lambda *args: None))
