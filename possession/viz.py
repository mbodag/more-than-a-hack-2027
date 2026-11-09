"""The match viewer (issue #3): the tactical video, the tracking and the labels on one frame clock.

    python -m possession.viz                 # writes outputs/viewer/; open index.html
    python -m possession.viz --out <folder>

The page is plain HTML and JavaScript (possession/viewer/), opened from disk with no server. A page
opened that way cannot fetch data files, so the data is written as scripts next to it:

    meta.js          everything small: halves, teams, Smart Tagging, per-frame ball and player
                     counts, the public ground truth, where the video is
    chunks/cNNNN.js  positions and homography for frames [NNNN * 1500, (NNNN + 1) * 1500), loaded
                     as the playhead reaches them; only chunks touching a half are written

The frame index is the tactical video's throughout: frame i of the page is submission row i.

Everything written here is derived from the match data. It stays in outputs/ (gitignored) and is
never committed or shared. The ground truth shown by the page is a pipeline sanity check only.
"""

import argparse
import base64
import csv
import json
import os
import pathlib
import shutil
import sys

import numpy as np

import games
from possession import REPO
from possession.fill import half_blocks
from possession.match import Match, read_teams

sys.path.insert(0, os.path.join(REPO, "scripts"))
import check_data as _check_data  # noqa: E402  (scripts/ is not a package)

OUT_DIR = os.path.join(REPO, "outputs", "viewer")
PAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer")
PAGE_FILES = ("index.html", "viewer.css", "core.js", "viewer.js")
GT_PATH = os.path.join(REPO, "evaluation", "gt",
                       "{}_possession_gt_public.csv".format(games.GAME))

CHUNK_FRAMES = 1500          # one minute at 25 fps
VIDEO_SIZE = (1920, 1080)    # the tactical video, DATA.md

# Positions are stored as unsigned 16-bit fractions of the pitch. The range is wider than 0-1
# because players and the ball do leave the pitch; one step is about 3 mm.
Q_LOW, Q_SPAN, Q_MAX, Q_MISSING = -0.5, 2.0, 65534, 65535


# Encoding ----------------------------------------------------------------------------------------

def quantise(xy):
    """(..., 2) pitch fractions -> uint16; a point with a NaN coordinate becomes Q_MISSING."""
    xy = np.asarray(xy, dtype=np.float64)
    missing = np.isnan(xy).any(axis=-1, keepdims=True)
    steps = np.clip(np.rint((np.nan_to_num(xy) - Q_LOW) / Q_SPAN * Q_MAX), 0, Q_MAX)
    return np.where(missing, Q_MISSING, steps).astype("<u2")


def dequantise(steps):
    """The inverse of quantise(), NaN where missing. The page does the same in core.js."""
    steps = np.asarray(steps)
    return np.where(steps == Q_MISSING, np.nan, steps / Q_MAX * Q_SPAN + Q_LOW)


def objects_xy(ball_xy, track_xy, pitch):
    """(n_frames, 1 + n_tracks, 2) pitch fractions: the ball first, then every track slot.

    The Match gives metres; the homography takes fractions of the pitch, as the ATD file has them.
    """
    both = np.concatenate([np.asarray(ball_xy)[:, None, :], np.asarray(track_xy)], axis=1)
    return both / np.asarray(pitch, dtype=np.float64)


def chunk_indices(blocks, chunk_frames=CHUNK_FRAMES):
    """Every chunk that holds at least one frame of a block; blocks are (first, last) inclusive."""
    wanted = set()
    for first, last in blocks:
        wanted.update(range(first // chunk_frames, last // chunk_frames + 1))
    return sorted(wanted)


def chunk_name(index):
    return "c{:04d}.js".format(index)


def _b64(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def chunk_js(index, steps, homography, carry, chunk_frames=CHUNK_FRAMES):
    """The script for one chunk: a call to the page's viewerChunk(index, payload).

    steps       (n, objects, 2) uint16 from quantise(), for frames index * chunk_frames onwards
    homography  (n, 3, 3): pitch fractions -> pixels, NaN where there is none
    carry       [frame, x step, y step] of the last tracked ball before this chunk, or None
    """
    payload = {
        "first": index * chunk_frames,
        "n": int(len(steps)),
        "xy": _b64(np.asarray(steps, dtype="<u2")),
        "h": _b64(np.asarray(homography, dtype="<f4")),
        "carry": carry,
    }
    return "viewerChunk({}, {});\n".format(index, json.dumps(payload))


def read_chunk_js(text):
    """(index, payload with xy and h as arrays): what the page decodes, for the tests."""
    head, _, rest = text.partition("(")
    if head != "viewerChunk":
        raise ValueError("not a viewer chunk")
    index, _, body = rest.partition(", ")
    payload = json.loads(body.rstrip().rstrip(";").rstrip(")"))
    n = payload["n"]
    payload["xy"] = np.frombuffer(base64.b64decode(payload["xy"]), dtype="<u2").reshape(n, -1, 2)
    payload["h"] = np.frombuffer(base64.b64decode(payload["h"]), dtype="<f4").reshape(n, 3, 3)
    return int(index), payload


def last_ball_frame(ball_present):
    """(n_frames,) int: the latest frame up to and including i with a tracked ball, -1 if none."""
    frames = np.where(ball_present, np.arange(len(ball_present)), -1)
    return np.maximum.accumulate(frames)


# Reading the inputs ------------------------------------------------------------------------------

def read_ground_truth(path=GT_PATH):
    """{first, labels} for the stretch of the public ground truth that is labelled, or None.

    The file has a row for every frame of the match, X outside its five minutes.
    """
    if not os.path.isfile(path):
        return None
    with open(path, newline="") as handle:
        labels = [row["label"] for row in csv.DictReader(handle)]
    graded = [i for i, label in enumerate(labels) if label != "X"]
    if not graded:
        return None
    return {"first": graded[0], "labels": "".join(labels[graded[0]:graded[-1] + 1])}


def tagging_events(events):
    """Smart Tagging events as the page wants them; frames are [s, e), as the loader gives."""
    return [{"s": event["start_frame"], "e": event["end_frame"], "code": event["code"],
             "tags": {group: text for group, text in event["tags"].items() if text is not None}}
            for event in events]


# Writing -----------------------------------------------------------------------------------------

def write_viewer(out_dir, ball_xy, track_xy, track_team, team_names, homography, events, blocks,
                 pitch, fps, video_path, ground_truth=None, chunk_frames=CHUNK_FRAMES,
                 say=print):
    """Write the page and its data into `out_dir`; returns the path of index.html.

    ball_xy (n, 2) and track_xy (n, tracks, 2) are in metres, NaN where not tracked, as the Match
    gives them; homography is (n, 3, 3); events come from tagging_events().
    """
    n_frames = len(ball_xy)
    track_team = np.asarray(track_team)
    xy = objects_xy(ball_xy, track_xy, pitch)
    steps = quantise(xy)
    ball_present = steps[:, 0, 0] != Q_MISSING
    tracked = steps[:, 1:, 0] != Q_MISSING
    last_ball = last_ball_frame(ball_present)

    chunk_dir = os.path.join(out_dir, "chunks")
    if os.path.isdir(chunk_dir):
        shutil.rmtree(chunk_dir)  # ours alone: chunks of an earlier run
    os.makedirs(chunk_dir)

    indices = chunk_indices(blocks, chunk_frames)
    for index in indices:
        first = index * chunk_frames
        last = min(first + chunk_frames, n_frames)
        before = int(last_ball[first - 1]) if first > 0 else -1
        carry = None if before < 0 else [before] + steps[before, 0].tolist()
        with open(os.path.join(chunk_dir, chunk_name(index)), "w") as handle:
            handle.write(chunk_js(index, steps[first:last], homography[first:last], carry,
                                  chunk_frames))
    say("    {} chunks of {} frames".format(len(indices), chunk_frames))

    meta = {
        "game": games.GAME,
        "n_frames": n_frames,
        "fps": fps,
        "pitch": list(pitch),
        "blocks": [list(block) for block in blocks],
        "chunk_frames": chunk_frames,
        "chunks": indices,
        "n_objects": int(steps.shape[1]),
        "q": {"low": Q_LOW, "span": Q_SPAN, "max": Q_MAX, "missing": Q_MISSING},
        "teams": list(team_names),
        "track_team": "".join(track_team.tolist()),
        "events": events,
        "ball_present": _b64(ball_present.astype(np.uint8)),
        "n_home": _b64((tracked & (track_team == "H")).sum(axis=1).astype(np.uint8)),
        "n_away": _b64((tracked & (track_team == "A")).sum(axis=1).astype(np.uint8)),
        "ground_truth": ground_truth,
        "video": {"uri": pathlib.Path(video_path).resolve().as_uri(), "path": video_path,
                  "width": VIDEO_SIZE[0], "height": VIDEO_SIZE[1]},
    }
    with open(os.path.join(out_dir, "meta.js"), "w") as handle:
        handle.write("window.VIEWER_META = {};\n".format(json.dumps(meta)))

    for name in PAGE_FILES:
        shutil.copyfile(os.path.join(PAGE_DIR, name), os.path.join(out_dir, name))
    return os.path.join(out_dir, "index.html")


def build(out_dir=OUT_DIR, say=print):
    """Check the data, read it through the Match and the organisers' loaders, write the viewer."""
    from starter import load_homography, load_smart_tagging  # organisers': imported, never edited

    say("checking the data")
    _check_data.check_data()
    match = Match(log=say)
    paths = games.asset_paths(data_dir=match.data_dir)
    team_names, _ = read_teams(paths["atd_meta"])
    homography, _, _ = load_homography.load(data_dir=match.data_dir)
    events = tagging_events(load_smart_tagging.load(data_dir=match.data_dir))

    say("writing {}".format(os.path.relpath(out_dir, REPO)))
    index = write_viewer(
        out_dir, match.ball_xy, match.track_xy, match.track_team, team_names, homography, events,
        blocks=half_blocks(), pitch=games.GAMES[games.GAME]["pitch"],
        fps=games.GAMES[games.GAME]["fps"], video_path=match.video_path,
        ground_truth=read_ground_truth(), say=say)
    if not os.path.isfile(match.video_path):
        say("    no video at {}: the page will show the pitch only until it is there".format(
            match.video_path))
    say("open {}".format(index))
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m possession.viz",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=OUT_DIR, help="folder to write (default outputs/viewer)")
    args = parser.parse_args(argv)
    try:
        build(os.path.abspath(args.out))
    except (_check_data.DataCheckError, IOError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
