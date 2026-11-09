"""Steps 4 and 7 of ADR 0002: check what an experiment returned, then the file on disk.

Each check raises RunError with a message naming the experiment; the runner prints it and stops.
Only the format checks can fail a run. The proxy report (step 7.2) measures the prediction without
any ground truth and only ever warns; it never reads evaluation/gt/.
"""

import json
import os
import subprocess
import sys

import numpy as np

import games
import possession_metrics as pm  # the organisers' scorer, imported, never edited

from possession import LABELS, N_FRAMES, REPO
from possession.fill import half_blocks

FPS = games.GAMES[games.GAME]["fps"]

# Proxy-report warning thresholds. Provisional: simple, round values from typical football (about
# 150-200 possessions per team per match, possessions of 10-15 s on average, the ball dead for
# roughly 35-45% of the clock) until #5 settles the measures and thresholds. Never tuned on the
# public ground truth. A value outside its range prints a warning; it never fails a run.
THRESHOLDS = {
    "spells_per_min": (1.5, 6.0),       # possessions per graded minute
    "mean_spell_s": (5.0, 30.0),        # mean possession length, seconds
    "short_spell_s": 1.0,               # a possession shorter than this is "very short"...
    "max_short_spell_share": 0.10,      # ...and more than this share of them is suspicious
    "dead_ball_share": (0.20, 0.50),    # share of graded frames labelled D
    "home_share": (0.25, 0.75),         # share of H among H and A frames
    "min_smart_tagging_agreement": 0.60,
}

# Smart Tagging codes whose Team tag is the side in possession. Defending codes are tagged with the
# defending side, and set pieces, long balls, goals and the box codes are ambiguous or N/A, so
# they are left out of the agreement measure.
IN_POSSESSION_CODES = ("BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD", "CREATING CHANCES",
                       "ATTACKING TRANSITION")


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
    """Step 7.1: the official format check (possession_metrics.validate_submission, the function
    the scorer calls) on the CSV as read back from disk. Returns the table read."""
    try:
        table = pm.read_label_csv(path)
    except Exception as error:
        raise RunError("{}: cannot read back {}: {}".format(experiment, path, error))
    reason = pm.validate_submission(table, n_frames)
    if reason:
        raise RunError("{}: {} fails the official format check: {}".format(
            experiment, path, reason))
    return table


def run_official_validator(experiment, path):
    """Step 7.4, submission only: the organisers' own `python evaluation/validate_submission.py
    <file>`, unmodified. Returns its output; raises RunError if it fails."""
    script = os.path.join(REPO, "evaluation", "validate_submission.py")
    result = subprocess.run([sys.executable, script, path], cwd=REPO,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True)
    if result.returncode != 0:
        raise RunError("{}: evaluation/validate_submission.py fails on {}:\n{}".format(
            experiment, path, result.stdout.strip()))
    return result.stdout


# -- step 7.2: the proxy report -------------------------------------------------------------------

def smart_tagging_side(codes, team, in_possession=IN_POSSESSION_CODES):
    """(n_frames,) "H", "A" or "": the side Smart Tagging's in-possession phases give each frame.

    `codes` and `team` are Match.smart_tagging_codes and Match.smart_tagging_team. A frame gets a
    side only when every in-possession phase covering it names that same side.
    """
    columns = [j for j, code in enumerate(np.asarray(codes).tolist()) if code in in_possession]
    tags = np.asarray(team)[:, columns]
    home = (tags == "H").any(axis=1)
    away = (tags == "A").any(axis=1)
    mixed = (tags == "?").any(axis=1)
    side = np.full(len(tags), "", dtype="<U1")
    side[home & ~away & ~mixed] = "H"
    side[away & ~home & ~mixed] = "A"
    return side


def proxy_report(labels, blocks=None, smart_tagging=None, fps=FPS):
    """Label-free measures of a prediction, inside the graded halves only, plus warnings.

    labels         one label per frame, as written to the CSV (after fill)
    blocks         [(first, last)] inclusive graded halves; this match's by default
    smart_tagging  optional, smart_tagging_side() per frame, for the agreement measure
    Returns a dict ready for JSON: floats rounded, no timestamps, so it is deterministic.
    """
    labels = np.asarray(labels).astype("<U1")
    blocks = half_blocks() if blocks is None else blocks
    runs = [(first, last + 1) for first, last in blocks]
    graded = np.concatenate([np.arange(a, b) for a, b in runs])
    in_halves = labels[graded]
    n_graded = len(graded)
    minutes = n_graded / float(fps) / 60.0

    codes = pm.label_codes(labels)
    spells = pm.possession_spells(codes, runs)  # the scorer's own definition of a possession
    lengths = np.array([b - a for a, b, _ in spells], dtype=float) / fps
    changes = sum(int((labels[a + 1:b] != labels[a:b - 1]).sum()) for a, b in runs)
    alive = int(np.isin(in_halves, ("H", "A")).sum())

    report = {
        "graded_frames": n_graded,
        "label_share": {k: _share((in_halves == k).sum(), n_graded) for k in LABELS},
        "halves": [{"half": i + 1, "first": a, "last": b - 1,
                    "n_spells": sum(1 for s, _, _ in spells if a <= s < b),
                    "all_x": bool((labels[a:b] == "X").all())}
                   for i, (a, b) in enumerate(runs)],
        "n_spells": len(spells),
        "spells_per_min": _round(len(spells) / minutes),
        "spell_length_s": _distribution(lengths),
        "short_spell_share": _share((lengths < THRESHOLDS["short_spell_s"]).sum(), len(lengths)),
        "label_changes_per_min": _round(changes / minutes),
        "dead_ball_share": _share((in_halves == "D").sum(), n_graded),
        "home_share": _share((in_halves == "H").sum(), alive),
        "smart_tagging": None,
    }
    if smart_tagging is not None:
        side = np.asarray(smart_tagging)[graded]
        compared = (side != "") & np.isin(in_halves, ("H", "A"))
        report["smart_tagging"] = {
            "coverage": _share((side != "").sum(), n_graded),
            "agreement": _share((side[compared] == in_halves[compared]).sum(), compared.sum()),
        }
    report["warnings"] = proxy_warnings(report)
    return report


def proxy_warnings(report):
    """Plain-English warnings for every measure outside THRESHOLDS. Never fails a run."""
    t = THRESHOLDS
    out = ["half {} has no real label: every frame is X".format(h["half"])
           for h in report["halves"] if h["all_x"]]

    def outside(name, value, bounds, what):
        if value is not None and not bounds[0] <= value <= bounds[1]:
            out.append("{} {} is outside {}-{} ({})".format(name, value, bounds[0], bounds[1], what))

    outside("spells_per_min", report["spells_per_min"], t["spells_per_min"],
            "possessions per minute")
    outside("mean_spell_s", report["spell_length_s"]["mean"], t["mean_spell_s"],
            "mean possession length in seconds")
    if (report["short_spell_share"] or 0) > t["max_short_spell_share"]:
        out.append("short_spell_share {} is above {} (possessions under {} s)".format(
            report["short_spell_share"], t["max_short_spell_share"], t["short_spell_s"]))
    outside("dead_ball_share", report["dead_ball_share"], t["dead_ball_share"],
            "share of graded frames labelled D")
    outside("home_share", report["home_share"], t["home_share"], "share of H among H and A")
    st = report["smart_tagging"]
    if st and st["agreement"] is not None and st["agreement"] < t["min_smart_tagging_agreement"]:
        out.append("smart_tagging agreement {} is below {}".format(
            st["agreement"], t["min_smart_tagging_agreement"]))
    return out


def format_report(report):
    """The report as a few printable lines."""
    lengths = report["spell_length_s"]
    st = report["smart_tagging"]
    lines = [
        "    graded frames {}, labels {}".format(report["graded_frames"], ", ".join(
            "{} {}".format(k, v) for k, v in report["label_share"].items())),
        "    possessions {} ({} per min; {})".format(report["n_spells"], report["spells_per_min"],
                                                     ", ".join("{} per half {}".format(
                                                         h["n_spells"], h["half"])
                                                         for h in report["halves"])),
        "    possession length s: min {min}, p10 {p10}, median {median}, mean {mean}, "
        "p90 {p90}, max {max}".format(**lengths),
        "    label changes per min {}, dead-ball share {}, home share {}".format(
            report["label_changes_per_min"], report["dead_ball_share"], report["home_share"]),
    ]
    if st:
        lines.append("    Smart Tagging: coverage {}, agreement {}".format(
            st["coverage"], st["agreement"]))
    lines.extend("    warning: " + w for w in report["warnings"])
    return "\n".join(lines)


def metrics_path(experiment, repo=REPO):
    return os.path.join(repo, "experiments", "metrics", experiment + ".json")


def write_metrics(experiment, report, repo=REPO):
    """Step 7.3: save the report to experiments/metrics/<experiment>.json (committed)."""
    path = metrics_path(experiment, repo)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    record = dict(experiment=experiment, thresholds=THRESHOLDS, **report)
    with open(path, "w", newline="\n") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _round(value):
    return round(float(value), 4)


def _share(part, whole):
    return _round(part / float(whole)) if whole else None


def _distribution(values):
    keys = ("min", "p10", "median", "mean", "p90", "max")
    if not len(values):
        return dict.fromkeys(keys)
    numbers = (values.min(), np.percentile(values, 10), np.median(values), values.mean(),
               np.percentile(values, 90), values.max())
    return {k: _round(v) for k, v in zip(keys, numbers)}
