"""Step 6 of ADR 0002: where an experiment's CSV goes, and writing it byte for byte."""

import os

from possession import REPO

SUBMISSION_EXPERIMENT = "north_london_tactics"
SUBMISSION_FILE = "North London Tactics.csv"  # the team name, exactly: the scorer labels our row with it


def output_path(experiment, repo=REPO):
    """outputs/<experiment>.csv, except the submission, which goes to the repo root."""
    if experiment == SUBMISSION_EXPERIMENT:
        return os.path.join(repo, SUBMISSION_FILE)
    return os.path.join(repo, "outputs", experiment + ".csv")


def csv_bytes(labels):
    """The header `frame,label`, then `i,label` for every frame, `\\n` line endings, ASCII.
    Nothing in it varies between runs."""
    rows = ["frame,label\n"]
    rows.extend("{},{}\n".format(i, label) for i, label in enumerate(labels))
    return "".join(rows).encode("ascii")


def write_csv(labels, path):
    """Write the CSV, replacing any earlier one only once the new one is complete."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        handle.write(csv_bytes(labels))
    os.replace(tmp, path)
    return path
