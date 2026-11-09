"""Step 5 of ADR 0002: fill each X inside a graded half from the labels around it.

Inside each half, an X takes the last real label before it; X's at the very start of a half take
the first real label after them. Nothing is carried across halftime, frames outside the halves are
left as they are, and a half with no real label at all stays X. This is what the official scorer
does before computing spellF1, so filling never changes spellF1; it only stops macroF1 counting
those frames as misses. It never changes a label the experiment gave.
"""

import numpy as np

import games


def half_blocks(game=games.GAME):
    """[(first_frame, last_frame)] of each graded half, inclusive, in video frames."""
    return [(first, last) for _, first, last in games.half_video_blocks(game)]


def fill(labels, blocks=None):
    """A filled copy of `labels` (a 1-D array of "H", "A", "D", "X"); the input is not changed.

    `blocks` is [(first, last)] inclusive; it defaults to this match's two halves.
    """
    out = np.array(labels, dtype="<U1", copy=True)
    for first, last in (half_blocks() if blocks is None else blocks):
        out[first:last + 1] = _fill_span(out[first:last + 1])
    return out


def _fill_span(span):
    real = span != "X"
    if not real.any():
        return span
    positions = np.arange(len(span))
    # Index of the last real label at or before each frame; -1 before the first one.
    last_real = np.maximum.accumulate(np.where(real, positions, -1))
    last_real[last_real < 0] = np.argmax(real)  # head of the half: the first real label after
    return span[last_real]
