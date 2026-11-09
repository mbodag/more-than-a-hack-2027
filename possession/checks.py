"""Steps 4 and 7 of ADR 0002: check what an experiment returned, then the file on disk.

Each check raises RunError with a message naming the experiment; the runner prints it and stops.
"""

import numpy as np

import possession_metrics as pm  # the organisers' scorer, imported, never edited

from possession import LABELS, N_FRAMES


class RunError(Exception):
    """The run cannot go on; the message says why."""


def check_output(experiment, labels, n_frames=N_FRAMES):
    """Step 4: `labels` must be n_frames labels, each "H", "A", "D" or "X". Returns them as an
    array of one-character strings."""
    try:
        array = np.asarray(labels)
    except Exception as error:  # e.g. a ragged list
        raise RunError("{}: predict() returned something that is not an array: {}".format(
            experiment, error))
    if array.ndim != 1:
        raise RunError("{}: predict() returned an array of shape {}; expected ({},)".format(
            experiment, array.shape, n_frames))
    if len(array) != n_frames:
        raise RunError("{}: predict() returned {} labels; expected {}".format(
            experiment, len(array), n_frames))
    as_text = np.array([v if isinstance(v, str) else repr(v) for v in array.tolist()],
                       dtype=object)
    bad = ~np.isin(as_text, LABELS)
    if bad.any():
        first = int(np.argmax(bad))
        values = sorted(set(as_text[bad].tolist()))
        raise RunError("{}: predict() returned {} labels outside {} (first at frame {}): {}"
                       .format(experiment, int(bad.sum()), list(LABELS), first, values[:5]))
    return as_text.astype("<U1")


def check_file(experiment, path, n_frames=N_FRAMES):
    """Step 7, the hook issue #12 extends: check the CSV as read back from disk.

    For now, only the official format check (possession_metrics.validate_submission, the function
    the scorer calls). #12 adds the proxy report, the metrics record and, for the submission, the
    organisers' own validate_submission.py. Only the format check may fail a run.
    """
    try:
        table = pm.read_label_csv(path)
    except Exception as error:
        raise RunError("{}: cannot read back {}: {}".format(experiment, path, error))
    reason = pm.validate_submission(table, n_frames)
    if reason:
        raise RunError("{}: {} fails the official format check: {}".format(
            experiment, path, reason))
    return table
