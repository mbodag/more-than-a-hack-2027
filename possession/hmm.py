"""A small hidden Markov model over possession states, decoded by Viterbi. numpy only.

The states are H, A and D (home in possession, away in possession, dead ball), in that order. Each
frame brings evidence about the state (an emission score per state); moving from one state to
another costs a transition penalty. Viterbi finds the single sequence of states with the best total
score over a whole half, so one odd frame cannot flip possession, but evidence that holds up over
time can. That is what spellF1 rewards: few, well-placed hand-overs (CONTEXT.md).

    T = transitions(rates, switch=2.0, dead=1.0)    # (3, 3) log scores
    states = viterbi(log_emit, T)                     # (n,) int, 0 = H, 1 = A, 2 = D
    labels = decode(log_emit, T, blocks)              # (n,) "H"/"A"/"D", each half on its own

Everything is deterministic: ties go to the lowest state index.
"""

import numpy as np

STATES = ("H", "A", "D")
H, A, D = 0, 1, 2


def fit_rates(label_arrays):
    """{"switch", "to_dead", "from_dead"}: per-frame transition rates counted in `label_arrays`
    (sequences of "H"/"A"/"D", one per half or match). Laplace-smoothed so none is zero."""
    counts = np.ones((3, 3))
    index = {s: i for i, s in enumerate(STATES)}
    for labels in label_arrays:
        codes = np.array([index[l] for l in labels], dtype=np.int64)
        np.add.at(counts, (codes[:-1], codes[1:]), 1)
    alive = counts[:2].sum()
    return {
        "switch": float((counts[H, A] + counts[A, H]) / alive),
        "to_dead": float((counts[H, D] + counts[A, D]) / alive),
        "from_dead": float((counts[D, H] + counts[D, A]) / counts[D].sum()),
    }


def transitions(rates, switch=1.0, dead=1.0):
    """(3, 3) log transition scores from per-frame `rates` (fit_rates). `switch` and `dead`
    multiply the log cost of a hand-over (H<->A) and of entering or leaving D: above 1 makes the
    decoder stickier than the counted rates, below 1 more willing to change."""
    s, td, fd = rates["switch"], rates["to_dead"], rates["from_dead"]
    T = np.empty((3, 3))
    T[H, H] = T[A, A] = np.log(1 - s - td)
    T[H, A] = T[A, H] = switch * np.log(s)
    T[H, D] = T[A, D] = dead * np.log(td)
    T[D, D] = np.log(1 - fd)
    T[D, H] = T[D, A] = dead * np.log(fd / 2)
    return T


def viterbi(log_emit, log_trans, log_start=None):
    """(n,) int: the state sequence with the highest total of emission and transition scores.

    log_emit is (n, k); log_trans is (k, k), from row to column; log_start (k,) defaults to 0.
    """
    log_emit = np.asarray(log_emit, dtype=np.float64)
    n, k = log_emit.shape
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    T = np.asarray(log_trans, dtype=np.float64)
    back = np.empty((n, k), dtype=np.int8)
    score = log_emit[0] + (np.zeros(k) if log_start is None else np.asarray(log_start))
    cols = np.arange(k)
    for t in range(1, n):
        cand = score[:, None] + T                   # cand[i, j]: be in i, then move to j
        best = cand.argmax(axis=0)                  # first maximum: ties to the lowest index
        back[t] = best
        score = cand[best, cols] + log_emit[t]
    path = np.empty(n, dtype=np.int64)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def viterbi_dead_min(log_emit, log_trans, k):
    """(n,) int over H, A, D like viterbi(), but every stoppage entered lasts at least `k` frames.

    D is a chain of k sub-states D1 -> D2 -> ... -> Dk; play goes into D1, only Dk may stay or
    leave, and all sub-states share D's emission. A half may start inside a stoppage, so the
    first frame may be in any sub-state. k = 1 is plain viterbi().
    """
    if k <= 1:
        return viterbi(log_emit, log_trans)
    E = np.asarray(log_emit, dtype=np.float64)
    T = np.asarray(log_trans, dtype=np.float64)
    n = len(E)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    alive = E[0, :2].copy()                 # scores of H, A
    dead = np.full(k, E[0, D])              # scores of D1..Dk
    back_alive = np.empty((n, 2), dtype=np.int8)   # 0 = H, 1 = A, 2 = Dk
    back_d1 = np.empty(n, dtype=np.int8)           # 0 = H, 1 = A
    back_dk = np.empty(n, dtype=np.int8)           # 0 = D(k-1), 1 = Dk
    cols = np.array([0, 1])
    for t in range(1, n):
        prev = np.array([alive[0], alive[1], dead[-1]])
        cand = prev[:, None] + T[:, :2]                       # from H, A, D into H, A
        b_alive = cand.argmax(axis=0)
        new_alive = cand[b_alive, cols] + E[t, :2]
        enter = alive + T[:2, D]
        b_d1 = int(enter[1] > enter[0])
        stay_prev, stay_self = dead[-2], dead[-1] + T[D, D]
        b_dk = int(stay_self > stay_prev)
        new_dead = np.empty(k)
        new_dead[0] = enter[b_d1]
        new_dead[1:-1] = dead[:-2]
        new_dead[-1] = stay_self if b_dk else stay_prev
        new_dead += E[t, D]
        back_alive[t], back_d1[t], back_dk[t] = b_alive, b_d1, b_dk
        alive, dead = new_alive, new_dead

    # Backtrack over H, A, or D with a sub-state index 0..k-1.
    path = np.empty(n, dtype=np.int64)
    best = int(np.argmax(np.r_[alive, dead]))
    state, sub = (best, None) if best < 2 else (D, best - 2)
    for t in range(n - 1, -1, -1):
        path[t] = state
        if t == 0:
            break
        if state != D:
            b = int(back_alive[t, state])
            state, sub = (b, None) if b < 2 else (D, k - 1)
        elif sub == 0:
            state, sub = int(back_d1[t]), None
        elif sub == k - 1:
            sub = k - 2 if back_dk[t] == 0 else k - 1
        else:
            sub -= 1
    return path


def drop_dead_islands(labels, blocks, min_frames):
    """Copy of `labels` where every play stretch shorter than `min_frames` with a stoppage both
    right before and right after it, inside one block, becomes "D".

    A play stretch is a maximal run of anything but "D", so H and A flickering together count as
    one stretch. Stretches touching a block's edge are kept: there is no stoppage on that side.
    The point is long breaks (cooling breaks, injuries) where a moment of possession is invented.
    """
    out = np.array(labels, copy=True)
    if min_frames <= 1:
        return out
    for first, last in blocks:
        alive = out[first:last + 1] != "D"
        edges = np.diff(np.concatenate(([0], alive.astype(np.int8), [0])))
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        for start, stop in zip(starts, stops):
            if start > 0 and stop < len(alive) and stop - start < min_frames:
                out[first + start:first + stop] = "D"
    return out


def decode(log_emit, log_trans, blocks, outside="D", dead_min=1):
    """(n,) "H"/"A"/"D" labels: Viterbi on each block (first, last inclusive) separately, never
    across halftime; frames outside every block get `outside`. `dead_min` > 1: every stoppage
    lasts at least that many frames (viterbi_dead_min)."""
    labels = np.full(len(log_emit), outside, dtype="<U1")
    letters = np.array(STATES)
    for first, last in blocks:
        labels[first:last + 1] = letters[viterbi_dead_min(log_emit[first:last + 1], log_trans,
                                                          dead_min)]
    return labels
