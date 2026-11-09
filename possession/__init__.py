"""Our possession pipeline. Run an experiment with `python -m possession run <experiment>`.

How an experiment runs, from data files to a checked CSV, is ADR 0002
(docs/adr/0002-experiment-runner.md).
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# games.py sits at the repo root and the organisers' scorer in evaluation/; neither is a package,
# so both folders go on the path once, here, and nothing else spells this out.
for _folder in (REPO, os.path.join(REPO, "evaluation")):
    if _folder not in sys.path:
        sys.path.insert(0, _folder)

import games  # noqa: E402

N_FRAMES = games.GAMES[games.GAME]["n_video_frames"]
LABELS = ("H", "A", "D", "X")
