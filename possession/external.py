"""Other matches with a full ground truth: the IDSSE Bundesliga open data, as Match-like objects.

    from possession.external import load, SPLITS
    match = load("J03WR9")                 # clean DFL tracking, read once and cached
    match = load("J03WR9", degrade=0)      # the same match with the ball hidden like the ATD
    match.labels                           # (n_frames,) "H" / "A" / "D", the provider's own

The IDSSE dataset (Bassek, Weber, Rein & Memmert, 2025, Scientific Data; CC BY 4.0) has seven full
matches of the 2022/23 Bundesliga and 2. Bundesliga: player and ball positions at 25 Hz, the DFL's
per-frame ball status (in or out of play) and ball possession (home or away), and the official
event data. scripts/fetch_idsse.py downloads it into data/external/idsse/ (gitignored).

An ExternalMatch answers the attributes experiments use from the Match (ball_xy, ball_speed,
track_xy, track_speed, track_team, n_frames, the Smart Tagging ones), in the same units and the
same pitch frame: metres, origin at a corner, x along the length. Its frame axis is the two halves
back to back with nothing between, so `blocks` replaces fill.half_blocks() for it.

The DFL's possession is not exactly ours (CONTEXT.md): it is the provider's own definition, so
scores here measure agreement with a close cousin of the challenge's labelling, not the real thing.
Smart Tagging does not exist for these matches; its attributes are empty (no codes).
"""

import os
import re

import numpy as np

from possession import REPO
from possession.match import ST_NONE

DATA_DIR = os.path.join(REPO, "data", "external", "idsse")
CACHE_DIR = os.path.join(REPO, "outputs", "cache", "idsse")
PARSE_VERSION = 1

# Fixed before any score was looked at. Training picks settings, validation compares candidates,
# and test is run once per candidate we intend to submit. Two Düsseldorf home games go to test so
# that neither validation nor training sees all of one home side's matches.
SPLITS = {
    "train": ("J03WMX", "J03WN1", "J03WOH", "J03WQQ"),
    "val": ("J03WOY",),
    "test": ("J03WPY", "J03WR9"),
}
MATCHES = tuple(m for split in SPLITS.values() for m in split)

HALVES = ("firstHalf", "secondHalf")
FPS = 25
_FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)"[^>]*PersonId="([^"]+)"')
_FRAME = re.compile(r'<Frame N="(\d+)"[^>]* X="([-\d.]+)" Y="([-\d.]+)"(?: Z="[-\d.]+")?'
                    r' D="[-\d.]+" S="([-\d.]+)"(?:[^>]*BallPossession="(\d)" BallStatus="(\d)")?')
_ATTR = re.compile(r'(\w+)="([^"]*)"')


def match_dir(match_id, data_dir=DATA_DIR):
    return os.path.join(data_dir, match_id)


# Parsing -----------------------------------------------------------------------------------------

def read_meta(path):
    """(home team id, away team id, home name, away name, pitch (length, width))."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    general = dict(_ATTR.findall(re.search(r"<General [^>]*>", text).group(0)))
    env = dict(_ATTR.findall(re.search(r"<Environment [^>]*>", text).group(0)))
    return (general["HomeTeamId"], general["GuestTeamId"], general["HomeTeamName"],
            general["GuestTeamName"], (float(env["PitchX"]), float(env["PitchY"])))


def parse_tracking(path, home_id, away_id, pitch):
    """Arrays for the two halves back to back, from the DFL position file.

    The ball's frames define each half; a player's frame outside them is dropped. DFL positions are
    metres from the centre spot; they are shifted so the origin is a corner, as the Match has it.
    """
    sets = {}  # (half, team, person) -> list of (N, x, y, speed[, possession, status])
    current = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                m = _FRAME.match(line)
                current.append(m.groups())
            elif line.startswith("<FrameSet "):
                half, team, person = _FRAMESET.match(line).groups()
                current = sets.setdefault((half, team, person), [])

    length, width = pitch
    # Two matches also track the referees; they are nobody's player.
    sets = {key: v for key, v in sets.items() if key[1] != "referee"}
    people = sorted({(team, person) for (_, team, person) in sets if team != "BALL"},
                    key=lambda tp: (tp[0] != home_id, tp[1]))
    column = {tp: j for j, tp in enumerate(people)}
    side = np.array(["H" if team == home_id else "A" for team, _ in people], dtype="<U1")
    if set(t for t, _ in people) - {home_id, away_id}:
        raise ValueError("{}: a FrameSet belongs to neither team".format(path))

    halves = []
    for half in HALVES:
        ball = [v for (h, team, _), v in sets.items() if h == half and team == "BALL"]
        if len(ball) != 1:
            raise ValueError("{}: expected one ball FrameSet in {}".format(path, half))
        ball = np.array(ball[0], dtype=object)
        n = ball[:, 0].astype(np.int64)
        first = n[0]
        if not np.array_equal(n, np.arange(first, first + len(n))):
            raise ValueError("{}: ball frames of {} are not consecutive".format(path, half))
        count = len(n)
        track_xy = np.full((count, len(people), 2), np.nan, dtype=np.float32)
        track_speed = np.full((count, len(people)), np.nan, dtype=np.float32)
        for (h, team, person), values in sets.items():
            if h != half or team == "BALL":
                continue
            v = np.array([row[:4] for row in values], dtype=np.float64)
            idx = v[:, 0].astype(np.int64) - first
            keep = (idx >= 0) & (idx < count)
            j = column[(team, person)]
            track_xy[idx[keep], j, 0] = v[keep, 1] + length / 2
            track_xy[idx[keep], j, 1] = v[keep, 2] + width / 2
            track_speed[idx[keep], j] = v[keep, 3]
        possession = ball[:, 4].astype(str)
        status = ball[:, 5].astype(str)
        labels = np.where(status == "1", np.where(possession == "1", "H", "A"), "D")
        halves.append({
            "ball_xy": np.stack([ball[:, 1].astype(np.float64) + length / 2,
                                 ball[:, 2].astype(np.float64) + width / 2], axis=1)
                         .astype(np.float32),
            "ball_speed": ball[:, 3].astype(np.float32),
            "track_xy": track_xy,
            "track_speed": track_speed,
            "labels": labels.astype("<U1"),
        })

    out = {key: np.concatenate([h[key] for h in halves]) for key in halves[0]}
    out["track_team"] = side
    out["half_lengths"] = np.array([len(h["labels"]) for h in halves])
    return out


def _cached(match_id, data_dir, cache_dir):
    path = os.path.join(cache_dir, "{}.v{}.npz".format(match_id, PARSE_VERSION))
    if os.path.isfile(path):
        with np.load(path) as stored:
            return {key: stored[key] for key in stored.files}
    folder = match_dir(match_id, data_dir)
    home, away, home_name, away_name, pitch = read_meta(os.path.join(folder, "meta.xml"))
    arrays = parse_tracking(os.path.join(folder, "tracking.xml"), home, away, pitch)
    arrays["team_names"] = np.array([home_name, away_name])
    arrays["pitch"] = np.array(pitch)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return arrays


# The match ---------------------------------------------------------------------------------------

class ExternalMatch:
    """One IDSSE match, answering what an experiment reads from the Match, plus its labels."""

    def __init__(self, match_id, arrays, degraded=None):
        self.match_id = match_id
        self.degraded = degraded
        self._a = arrays
        self.n_frames = len(arrays["labels"])
        n1 = int(arrays["half_lengths"][0])
        self.blocks = [(0, n1 - 1), (n1, self.n_frames - 1)]
        self.team_names = tuple(str(t) for t in arrays["team_names"])
        self.pitch = tuple(float(p) for p in arrays["pitch"])
        self._team_shape = None

    def __repr__(self):
        return "ExternalMatch({}{})".format(
            self.match_id, "" if self.degraded is None else ", degraded={}".format(self.degraded))

    labels = property(lambda self: self._a["labels"])
    ball_xy = property(lambda self: self._a["ball_xy"])
    ball_speed = property(lambda self: self._a["ball_speed"])
    track_xy = property(lambda self: self._a["track_xy"])
    track_speed = property(lambda self: self._a["track_speed"])
    track_team = property(lambda self: self._a["track_team"])

    # The DFL's tracking has no held or stuck positions to remove: the cleaned tracking is the same.
    track_xy_clean = property(lambda self: self._a["track_xy"])
    track_speed_clean = property(lambda self: self._a["track_speed"])

    @property
    def track_removed(self):
        return np.zeros(np.asarray(self.track_xy).shape[:2], dtype=np.uint8)

    @property
    def track_team_fixed(self):
        """The Match's per-frame sides. The DFL's own sides are right, keepers included, and its
        slots are not the ATD's keeper slots, so this is track_team on every frame (read-only)."""
        team = np.asarray(self.track_team)
        return np.broadcast_to(team[None, :], (self.n_frames, len(team)))

    @property
    def team_shape(self):
        """The same as Match.team_shape, computed on first use and kept on this object."""
        if self._team_shape is None:
            from possession import team_shape
            self._team_shape = team_shape.compute(self.track_xy, self.track_team, self.blocks,
                                                  self.pitch)
        return self._team_shape

    smart_tagging_codes = property(lambda self: np.array([], dtype=str))
    smart_tagging_team = property(lambda self: np.full((self.n_frames, 0), ST_NONE, dtype="<U1"))
    smart_tagging_phase = property(lambda self: np.zeros((self.n_frames, 0), dtype=bool))

    def with_arrays(self, degraded, **changes):
        arrays = dict(self._a)
        arrays.update(changes)
        return ExternalMatch(self.match_id, arrays, degraded)


def load(match_id, degrade=None, data_dir=DATA_DIR, cache_dir=CACHE_DIR):
    """An ExternalMatch; with `degrade` (a seed), the ball is hidden like the ATD's (degrade.py)."""
    if match_id not in MATCHES:
        raise ValueError("unknown IDSSE match {!r}; known: {}".format(match_id, ", ".join(MATCHES)))
    match = ExternalMatch(match_id, _cached(match_id, data_dir, cache_dir))
    if degrade is None:
        return match
    from possession import degrade as _degrade
    return _degrade.degrade(match, seed=degrade)


def split(name, degrade=None, **kwargs):
    """Every match of a split ("train", "val", "test"), in order."""
    return [load(m, degrade=degrade, **kwargs) for m in SPLITS[name]]
