"""The Match: every input an experiment may use, prepared on first access (ADR 0002, step 2).

    match = Match()            # reads nothing
    match.ball_xy              # first access: built from the tracking file and cached
    match.ball_xy              # later accesses, and later runs: loaded from outputs/cache/

Every per-frame item is a numpy array whose first axis is the frame of the tactical video, length
199,319, so item[i] belongs to submission row i. The tracking file counts frames from 1; that is
converted here, once (ATD frame_id 1 is frame 0).

Items are only ever added. An item's name and format never change; a different format is a new
item with a new name. If you change how an existing item is built, bump its version in ITEMS.

The Match does not check the data against the committed checksums; the runner does that first
(scripts/check_data.py). Cache keys are made from those committed checksums.
"""

import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import games  # noqa: E402
from possession.cache import Cache, Item  # noqa: E402

SOURCE = "tactical"
CACHE_DIR = os.path.join(REPO, "outputs", "cache")
CHECKSUMS = os.path.join(REPO, "scripts", "data_checksums.json")

# Smart Tagging's Team tag on phases that belong to neither side (kick-offs, for instance).
NO_TEAM_TAG = "N/A"
# smart_tagging_team values: no phase of that code, home, away, tagged N/A, tags disagree.
ST_NONE, ST_HOME, ST_AWAY, ST_NO_TEAM, ST_MIXED = "", "H", "A", "N", "?"


# Builders ----------------------------------------------------------------------------------------

def parse_atd(raw_path, n_slots, n_frames, pitch):
    """The raw ATD file as arrays indexed by frame (frame_id - 1).

    Each line is "<frame_id>:<slot>;<slot>;...:<ball>", every group "x,y,speed", x and y as
    fractions of the pitch, NaN where nothing was tracked. Positions are returned in metres
    (x along the length, y along the width, origin where the file has it); speeds as in the file.
    """
    with open(raw_path) as handle:
        text = handle.read()
    lines = text.strip().split("\n")
    width = 1 + 3 * n_slots + 3
    # One flat parse of every number in the file: far faster than splitting line by line.
    flat = np.array(",".join(lines).replace(":", ",").replace(";", ",").split(","),
                    dtype=np.float64)
    if flat.size != len(lines) * width:
        raise ValueError("{}: expected {} values per line ({} slots and the ball), got {} values "
                         "over {} lines".format(raw_path, width, n_slots, flat.size, len(lines)))
    table = flat.reshape(len(lines), width)

    frame_ids = table[:, 0]
    if len(frame_ids) != n_frames or not np.array_equal(frame_ids, np.arange(1, n_frames + 1)):
        raise ValueError("{}: expected frame ids 1..{} in order, one per line".format(
            raw_path, n_frames))

    length, breadth = pitch
    scale = np.array([length, breadth])
    slots = table[:, 1:1 + 3 * n_slots].reshape(n_frames, n_slots, 3)
    ball = table[:, 1 + 3 * n_slots:]
    return {
        "ball_xy": (ball[:, :2] * scale).astype(np.float32),
        "ball_speed": ball[:, 2].astype(np.float32),
        "track_xy": (slots[:, :, :2] * scale).astype(np.float32),
        "track_speed": slots[:, :, 2].astype(np.float32),
    }


def read_teams(meta_path):
    """((home name, away name), side of each track slot as "H"/"A").

    The first team in the ATD metadata is the home side (the FIFA/EPTS convention the organisers'
    baseline follows); each Player entry, in file order, is one track slot.
    """
    root = ET.parse(meta_path).getroot()
    teams = [(team.get("id"), team.findtext("Name")) for team in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    side = {teams[0][0]: ST_HOME, teams[1][0]: ST_AWAY}
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots or any(team_id not in side for team_id in slots):
        raise ValueError("{}: every Player needs a teamId of one of the two teams".format(
            meta_path))
    return (teams[0][1], teams[1][1]), np.array([side[t] for t in slots], dtype="<U1")


def smart_tagging_code(event):
    """Column name for an event. Set pieces are split by type: a throw-in is not a corner."""
    code = event["code"]
    kind = event["tags"].get("Type")
    return "{}: {}".format(code, kind) if kind else code


def paint_smart_tagging(events, team_names, n_frames):
    """{codes: (k,), team: (n_frames, k)} from Smart Tagging events in video frames.

    team[i, j] says which side the phases of code j covering frame i are tagged with: "" (none),
    "H", "A", "N" (tagged N/A) or "?" (overlapping phases of that code disagree). Phases overlap,
    within one code and across codes, so one value per frame would throw information away.
    Intervals are half-open, [start_frame, end_frame), as the organisers' loader gives them.
    """
    home, away = team_names
    side_of = {home: 0, away: 1, NO_TEAM_TAG: 2}
    codes = sorted({smart_tagging_code(event) for event in events})
    column = {code: j for j, code in enumerate(codes)}

    flags = np.zeros((n_frames, len(codes), 3), dtype=bool)
    for event in events:
        team = event["tags"].get("Team")
        if team not in side_of:
            raise ValueError("Smart Tagging team {!r} is neither {!r}, {!r} nor {!r}".format(
                team, home, away, NO_TEAM_TAG))
        start = max(0, event["start_frame"])
        end = min(n_frames, event["end_frame"])
        flags[start:end, column[smart_tagging_code(event)], side_of[team]] = True

    team = np.full((n_frames, len(codes)), ST_NONE, dtype="<U1")
    count = flags.sum(axis=2)
    for k, value in enumerate((ST_HOME, ST_AWAY, ST_NO_TEAM)):
        team[(count == 1) & flags[:, :, k]] = value
    team[count > 1] = ST_MIXED
    return {"codes": np.array(codes, dtype=str), "team": team}


def _build_tracking(inputs):
    _, slots = read_teams(inputs.path(SOURCE, "atd_meta"))
    return parse_atd(inputs.path(SOURCE, "atd_raw"), len(slots),
                     games.source_frames(SOURCE), games.GAMES[games.GAME]["pitch"])


def _build_track_team(inputs):
    return read_teams(inputs.path(SOURCE, "atd_meta"))[1]


def _build_smart_tagging(inputs):
    from starter import load_smart_tagging  # organisers' loader: imported, never edited

    team_names, _ = read_teams(inputs.path(SOURCE, "atd_meta"))
    path = inputs.path(SOURCE, "smart_tagging")
    events = load_smart_tagging.load(data_dir=os.path.dirname(os.path.dirname(path)))
    return paint_smart_tagging(events, team_names, games.source_frames(SOURCE))


# The cache items. A Match attribute may expose one array of an item.
ITEMS = {item.name: item for item in (
    Item("tracking", version=1, build=_build_tracking,
         raw=((SOURCE, "atd_raw"), (SOURCE, "atd_meta")),
         doc="ball and track positions and speeds per frame, from the raw ATD"),
    Item("track_team", version=1, build=_build_track_team,
         raw=((SOURCE, "atd_meta"),),
         doc="side of each track slot"),
    Item("smart_tagging", version=1, build=_build_smart_tagging,
         raw=((SOURCE, "smart_tagging"), (SOURCE, "atd_meta")),
         doc="Smart Tagging phases and their team, per frame and code"),
)}


def load_checksums(path=CHECKSUMS):
    with open(path) as handle:
        return json.load(handle)


# The Match ---------------------------------------------------------------------------------------

class Match:
    """The inputs of BAR-ATH as items, each prepared the first time it is read.

    data_dir    where the match data is; games.DATA_DIR (MTAH_DATA_DIR or ./data) by default
    cache_dir   where prepared items are kept; outputs/cache/ by default
    rebuild     None, True (rebuild every item used) or a list of item names; see ITEMS
    """

    n_frames = games.source_frames(SOURCE)

    def __init__(self, data_dir=None, cache_dir=None, rebuild=None, log=print):
        self.data_dir = data_dir
        paths = games.asset_paths(data_dir=data_dir, source=SOURCE)
        self._cache = Cache(cache_dir or CACHE_DIR, ITEMS, load_checksums(),
                            {(SOURCE, asset): path for asset, path in paths.items()},
                            rebuild=rebuild, log=log)

    def item(self, name):
        """A cache item by name: an array, or a dict of arrays."""
        return self._cache.get(name)

    @property
    def video_path(self):
        """Path of the tactical video. Not checked for existence; decode it yourself."""
        return games.video_path(data_dir=self.data_dir, source=SOURCE)

    # Tracking: positions in metres, NaN where not tracked. Slots are tracklets, not players.

    @property
    def ball_xy(self):
        """(n_frames, 2) float32: ball x, y in metres; NaN where the ball was not tracked."""
        return self.item("tracking")["ball_xy"]

    @property
    def ball_speed(self):
        """(n_frames,) float32: ball speed as given in the ATD; NaN where not tracked."""
        return self.item("tracking")["ball_speed"]

    @property
    def track_xy(self):
        """(n_frames, n_tracks, 2) float32: each track slot's x, y in metres; NaN when absent."""
        return self.item("tracking")["track_xy"]

    @property
    def track_speed(self):
        """(n_frames, n_tracks) float32: each track slot's speed as given in the ATD."""
        return self.item("tracking")["track_speed"]

    @property
    def track_team(self):
        """(n_tracks,) str: "H" or "A" for each track slot."""
        return self.item("track_team")

    # Smart Tagging: inferred, noisy, and its phases overlap.

    @property
    def smart_tagging_codes(self):
        """(k,) str: Smart Tagging codes, sorted; set pieces split by type ("SET PIECES: Throw-in")."""
        return self.item("smart_tagging")["codes"]

    @property
    def smart_tagging_team(self):
        """(n_frames, k) str: per frame and code, "" (no phase), "H", "A", "N" (tagged N/A)
        or "?" (overlapping phases of that code disagree)."""
        return self.item("smart_tagging")["team"]

    @property
    def smart_tagging_phase(self):
        """(n_frames, k) bool: a phase of that code covers the frame."""
        return self.smart_tagging_team != ST_NONE
