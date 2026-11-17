"""Probe: how good are the controls of possession/controls.py, against the DFL's own data?

Training matches only (ADR 0003). Two references, both the provider's:
  1. Side: just after a control, the DFL's per-frame possession should be the controller's side.
     Precision = share of controls (on alive play) whose side matches the label over the next 0.5 s.
  2. Contacts: the DFL's event data names the player on every pass, shot, ball claim and other ball
     action. Recall = share of those events with a control of the same side (and, on clean
     tracking, the same player) within +-1 s. Events carry wall-clock times; the clock offset to
     the tracking is estimated per match (the lag where the named player is nearest the ball).

Run on clean tracking and on degradation v2 (seed 0). Compares the confidence score with
hmm_touch's two yes/no rules. Writes experiments/metrics/control_probe.json.

    python scripts/control_probe.py
"""

import datetime
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from possession import REPO, controls, external  # noqa: E402

FPS = 25
AFTER = 12                    # frames after a control over which its side is checked
TOL = 25                      # frames: an event matches a control this close
CONTACT_TAGS = ("Play", "ShotAtGoal", "OtherBallAction", "BallClaiming")
THRESHOLDS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_ATTR = re.compile(r'(\w+)="([^"]*)"')
CACHE = os.path.join(REPO, "outputs", "cache", "idsse")


def _time(text):
    return datetime.datetime.fromisoformat(text)


def tracking_index(match_id):
    """(person ids in the column order of external.parse_tracking, [start time of each half]).
    One pass over the position file, cached as JSON."""
    path = os.path.join(CACHE, "{}.persons.json".format(match_id))
    if os.path.isfile(path):
        with open(path) as handle:
            got = json.load(handle)
        return got["people"], [_time(t) for t in got["starts"]]
    folder = external.match_dir(match_id)
    home, away = external.read_meta(os.path.join(folder, "meta.xml"))[:2]
    people, starts, want = set(), {}, None
    with open(os.path.join(folder, "tracking.xml"), encoding="utf-8") as handle:
        for line in handle:
            if want is not None:
                starts[want] = dict(_ATTR.findall(line))["T"]
                want = None
            elif line.startswith("<FrameSet "):
                half, team, person = external._FRAMESET.match(line).groups()
                if team == "BALL":
                    want = half
                elif team != "referee":
                    people.add((team, person))
    order = sorted(people, key=lambda tp: (tp[0] != home, tp[1]))
    out = {"people": [p for _, p in order], "starts": [starts[h] for h in external.HALVES]}
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(out, handle)
    return out["people"], [_time(t) for t in out["starts"]]


def contact_events(match_id, starts, half_lengths):
    """DFL events naming a player in contact with the ball: [(frame, side, person, tag)]."""
    folder = external.match_dir(match_id)
    home = external.read_meta(os.path.join(folder, "meta.xml"))[0]
    with open(os.path.join(folder, "event.xml"), encoding="utf-8") as handle:
        blocks = handle.read().split("<Event ")[1:]
    deleted = {dict(_ATTR.findall(b[:b.find(">")])).get("EventId") for b in blocks if "<Delete" in b}
    offsets = [0, int(half_lengths[0])]
    out = []
    for block in blocks:
        head = dict(_ATTR.findall(block[:block.find(">")]))
        if head.get("EventId") in deleted or "EventTime" not in head:
            continue
        # The innermost contact element names the player (set pieces wrap a Play).
        found = None
        for tag, attrs in re.findall(r"<(\w+)((?: \w+=\"[^\"]*\")*)\s*/?>", block):
            if tag in CONTACT_TAGS:
                a = dict(_ATTR.findall(attrs))
                if "Player" in a and "Team" in a:
                    found = (tag, a)
        if found is None:
            continue
        tag, a = found
        when = _time(head["EventTime"])
        half = 1 if when >= starts[1] else 0
        frame = offsets[half] + int(round((when - starts[half]).total_seconds() * FPS))
        if not offsets[half] <= frame < offsets[half] + int(half_lengths[half]):
            continue
        out.append((frame, "H" if a["Team"] == home else "A", a["Player"], tag))
    return out


def clock_offset(match, events, column):
    """Lag (frames) to add to event frames: where the named player is nearest the ball, median."""
    ball = np.asarray(match.ball_xy)
    tracks = np.asarray(match.track_xy)
    best = None
    for lag in range(-75, 76, 2):
        ds = []
        for frame, _, person, _ in events:
            f = frame + lag
            if 0 <= f < match.n_frames and person in column:
                dist = np.linalg.norm(tracks[f, column[person]] - ball[f])
                if np.isfinite(dist):
                    ds.append(dist)
        med = float(np.median(ds))
        if best is None or med < best[1]:
            best = (lag, med)
    return best


def side_check(ep, labels):
    """Per episode: 1 if the DFL side over the AFTER frames following its end matches, 0 if the
    other side, -1 if those frames are all dead (D)."""
    n = len(labels)
    out = np.empty(len(ep["start"]), dtype=int)
    for i, (e, s) in enumerate(zip(ep["end"], ep["side"])):
        lab = labels[e:min(e + AFTER, n)]
        lab = lab[lab != "D"]
        if len(lab) == 0:
            out[i] = -1
        else:
            out[i] = int((lab == s).mean() >= 0.5)
    return out


def event_recall(ep, keep, events, lag, match_column=None):
    """Share of events with a kept episode of the same side (and player) within TOL frames."""
    starts, ends = ep["start"][keep], ep["end"][keep]
    sides, tracks = ep["side"][keep], ep["track"][keep]
    hits = 0
    for frame, side, person, _ in events:
        f = frame + lag
        near = (starts <= f + TOL) & (ends >= f - TOL) & (sides == side)
        if match_column is not None:
            near &= tracks == match_column.get(person, -2)
        hits += bool(near.any())
    return hits / max(len(events), 1)


def probe(match, events, lag, column=None):
    ep = controls.episodes(match)
    check = side_check(ep, np.asarray(match.labels))
    alive = check >= 0
    rows = {"n_episodes": int(len(check)), "dead_share": float((~alive).mean()),
            "n_events": len(events)}

    def row(keep):
        k = keep & alive
        return {"controls": int(keep.sum()),
                "precision": float(check[k].mean()) if k.any() else None,
                "recall_side": event_recall(ep, keep, events, lag),
                "recall_player": None if column is None else
                event_recall(ep, keep, events, lag, column)}

    rows["rules"] = {"hmm_touch r2 (2 m, 1 m)": row(controls.hmm_touch_rule(ep, 2.0, 1.0)),
                     "hmm_touch r3 (3 m, 0.5 m)": row(controls.hmm_touch_rule(ep, 3.0, 0.5))}
    rows["confidence"] = {"{:.1f}".format(t): row(ep["confidence"] >= t) for t in THRESHOLDS}
    # Calibration: precision within confidence bins, and what each rule score says on its own.
    bins = np.digitize(ep["confidence"], [0.1, 0.3, 0.5, 0.7, 0.9])
    rows["by_bin"] = {}
    for b, name in enumerate(("<0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", ">=0.9")):
        k = (bins == b) & alive
        rows["by_bin"][name] = {"n": int(k.sum()),
                                "precision": float(check[k].mean()) if k.any() else None}
    rows["by_score"] = {}
    for score in ("prox", "sep", "dwell", "together", "impulse_score"):
        lo, hi = (ep[score] < 0.5) & alive, (ep[score] >= 0.5) & alive
        rows["by_score"][score] = {
            "low": float(check[lo].mean()) if lo.any() else None,
            "high": float(check[hi].mean()) if hi.any() else None,
            "high_share": float(hi.sum() / max(alive.sum(), 1))}
    return rows


def pooled(per_match, path):
    """Average a nested metric over matches, weighted equally."""
    vals = []
    for r in per_match:
        v = r
        for key in path:
            v = v[key]
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else None


def main():
    report = {"split": "train", "after_frames": AFTER, "tolerance_frames": TOL, "regimes": {}}
    for regime, seed in (("clean", None), ("degraded v2 seed 0", 0)):
        per_match = []
        for match_id in external.SPLITS["train"]:
            match = external.load(match_id, degrade=seed)
            people, starts = tracking_index(match_id)
            half_lengths = (match.blocks[0][1] + 1, match.n_frames - match.blocks[0][1] - 1)
            events = contact_events(match_id, starts, half_lengths)
            column = {p: i for i, p in enumerate(people)}
            clean = external.load(match_id)
            lag, med = clock_offset(clean, events, column)
            r = probe(match, events, lag, column if seed is None else None)
            r.update(match=match_id, clock_lag_frames=lag, clock_median_dist=med)
            per_match.append(r)
            print("{} {}: {} episodes, {} events, lag {} frames (median {:.2f} m)".format(
                regime, match_id, r["n_episodes"], len(events), lag, med), flush=True)
        summary = {"rules": {}, "confidence": {}, "by_bin": {}, "by_score": {}}
        for group in ("rules", "confidence", "by_bin", "by_score"):
            for key in per_match[0][group]:
                summary[group][key] = {m: pooled(per_match, (group, key, m))
                                       for m in per_match[0][group][key]}
        report["regimes"][regime] = {"summary": summary, "per_match": per_match}

        print("\n== {} (mean of the 4 training matches)".format(regime))
        print("{:<28}{:>9}{:>11}{:>13}{:>15}".format("rule", "controls", "precision",
                                                       "recall side", "recall player"))

        def fmt(v):
            return "{:>.3f}".format(v) if v is not None else "-"
        for name, r in list(summary["rules"].items()) + [
                ("confidence >= " + t, r) for t, r in summary["confidence"].items()]:
            print("{:<28}{:>9.0f}{:>11}{:>13}{:>15}".format(
                name, r["controls"], fmt(r["precision"]), fmt(r["recall_side"]),
                fmt(r["recall_player"])))
        print("precision by confidence bin:", {k: (round(v["n"]), v["precision"] and
                                                   round(v["precision"], 3))
                                               for k, v in summary["by_bin"].items()})
        print("precision when each score is low / high:",
              {k: (v["low"] and round(v["low"], 3), v["high"] and round(v["high"], 3))
               for k, v in summary["by_score"].items()})

    path = os.path.join(REPO, "experiments", "metrics", "control_probe.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=1, sort_keys=True)
    print("\nwrote", os.path.relpath(path, REPO))


if __name__ == "__main__":
    main()
