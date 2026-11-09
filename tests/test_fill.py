"""Step 5, fill, on hand-made arrays."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from possession.fill import fill, half_blocks  # noqa: E402

# Two "halves": frames 2-7 and 10-13; 0-1, 8-9 and 14-15 are outside them.
BLOCKS = [(2, 7), (10, 13)]


def labels(text):
    return np.array(list(text), dtype="<U1")


def test_forward_fill_inside_a_half():
    assert "".join(fill(labels("XXHXXAXXXXHXXAXX"), BLOCKS)) == "XXHHHAAAXXHHHAXX"


def test_head_of_a_half_takes_the_first_label_after():
    assert "".join(fill(labels("XXXXDHXXXXXXAXXX"), BLOCKS)) == "XXDDDHHHXXAAAAXX"


def test_nothing_carried_across_halftime():
    # Half 1 ends on A; half 2 starts with X, which must take half 2's D, not half 1's A.
    assert "".join(fill(labels("XXHHHHHAXXXXDXXX"), BLOCKS)) == "XXHHHHHAXXDDDDXX"


def test_ungraded_frames_untouched():
    out = fill(labels("HXHXXXXXAXXXXXDX"), BLOCKS)
    assert "".join(out[[0, 1, 8, 9, 14, 15]]) == "HXAXDX"


def test_all_x_half_stays_x():
    assert "".join(fill(labels("XXXXXXXXXXAXXXXX"), BLOCKS)) == "XXXXXXXXXXAAAAXX"


def test_real_labels_never_change_and_input_is_not_modified():
    original = labels("XAHDXHXAXXDXHXAX")
    before = original.copy()
    out = fill(original, BLOCKS)
    assert np.array_equal(original, before)
    real = original != "X"
    assert np.array_equal(out[real], original[real])


def test_default_blocks_are_the_two_halves_of_the_match():
    assert half_blocks() == [(11185, 90651), (110808, 190197)]
