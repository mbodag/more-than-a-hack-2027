"""Put the repo root on sys.path, so tests import `possession` and `games` however pytest is run."""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
