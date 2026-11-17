"""Tune the control confidence's ten scales (possession/controls.py HAND) on the bench's training
matches, then check them on validation.

Target: how well the confidence ranks controls after which the DFL's possession is the
controller's side above those after which it is not (ROC AUC, live play only), averaged over the
training matches under degradation v2, seeds 0-2: the same regime as `bench train`. A random search
around the hand-set scales, then a local refinement. Validation is scored once at the end, test
never. Writes possession/models/control_params.json and experiments/metrics/control_tuning.json.

    python scripts/tune_controls.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from possession import REPO, controls, external  # noqa: E402

SEEDS = (0, 1, 2)
SAMPLES = 400
REFINE = 300
PARAMS_PATH = os.path.join(REPO, "possession", "models", "control_params.json")
METRICS_PATH = os.path.join(REPO, "experiments", "metrics", "control_tuning.json")

# Search ranges per scale, and which way each score rises (zero above one for falling scores).
RANGES = {"prox_zero": (1.0, 2.5), "prox_one": (0.0, 1.5),
          "sep_zero": (-2.0, 1.0), "sep_one": (0.0, 4.0),
          "dwell_zero": (0.0, 5.0), "dwell_one": (2.0, 40.0),
          "together_zero": (2.0, 10.0), "together_one": (0.0, 3.0),
          "impulse_zero": (0.0, 6.0), "impulse_one": (3.0, 20.0)}
FALLING = ("prox", "together")


def valid(p):
    for name in ("prox", "sep", "dwell", "together", "impulse"):
        zero, one = p[name + "_zero"], p[name + "_one"]
        gap = zero - one if name in FALLING else one - zero
        if gap < 0.1:
            return False
    return True


def auc(score, y):
    """ROC AUC by ranks (ties averaged)."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def prepare(split):
    """[(episodes, outcome)] for every match and seed of a split, live play only."""
    out = []
    for seed in SEEDS:
        for match in external.split(split, degrade=seed):
            ep = controls.episodes(match)
            y = controls.side_outcome(ep, np.asarray(match.labels))
            keep = y >= 0
            out.append(({k: v[keep] for k, v in ep.items()}, y[keep]))
    return out


def evaluate(data, params):
    """Mean AUC, and pooled precision / share kept at a few confidence cuts."""
    aucs, rows = [], {t: [0, 0, 0] for t in (0.5, 0.7, 0.9)}
    for ep, y in data:
        controls.score(ep, params)
        c = ep["confidence"]
        aucs.append(auc(c, y))
        for t in rows:
            k = c >= t
            rows[t][0] += int(y[k].sum())
            rows[t][1] += int(k.sum())
            rows[t][2] += len(y)
    return {"auc": float(np.mean(aucs)),
            "cuts": {str(t): {"precision": r[0] / max(r[1], 1), "kept": r[1] / max(r[2], 1)}
                     for t, r in rows.items()}}


def search(data, rng, say=print):
    best = dict(controls.HAND)
    best_auc = evaluate(data, best)["auc"]
    say("hand-set: train AUC {:.4f}".format(best_auc))
    for i in range(SAMPLES):
        p = {k: float(rng.uniform(*RANGES[k])) for k in RANGES}
        if not valid(p):
            continue
        a = evaluate(data, p)["auc"]
        if a > best_auc:
            best, best_auc = p, a
            say("  sample {:>3}: {:.4f}".format(i, a))
    # Local refinement: nudge one or two scales at a time, keep what helps.
    for i in range(REFINE):
        p = dict(best)
        for k in rng.choice(list(RANGES), size=rng.integers(1, 3), replace=False):
            lo, hi = RANGES[k]
            p[k] = float(np.clip(p[k] + rng.normal(0, 0.1 * (hi - lo)), lo, hi))
        if not valid(p):
            continue
        a = evaluate(data, p)["auc"]
        if a > best_auc:
            best, best_auc = p, a
            say("  refine {:>3}: {:.4f}".format(i, a))
    return best


def main(say=print):
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import check_data
    check_data.check_data()
    train = prepare("train")
    say("train: {} live episodes over {} match-seeds".format(sum(len(y) for _, y in train),
                                                              len(train)))
    best = search(train, np.random.default_rng(0), say)

    val = prepare("val")
    report = {"regime": "degraded v2 seeds {}".format(list(SEEDS)), "target": "mean ROC AUC",
              "hand": controls.HAND, "tuned": best,
              "train": {"hand": evaluate(train, controls.HAND), "tuned": evaluate(train, best)},
              "val": {"hand": evaluate(val, controls.HAND), "tuned": evaluate(val, best)}}
    for split in ("train", "val"):
        for which in ("hand", "tuned"):
            r = report[split][which]
            say("{:<6}{:<6} AUC {:.4f}  ".format(split, which, r["auc"]) + "  ".join(
                ">= {}: precision {:.3f} kept {:.2f}".format(t, c["precision"], c["kept"])
                for t, c in r["cuts"].items()))
    say("tuned scales: " + json.dumps({k: round(v, 2) for k, v in best.items()}))

    with open(PARAMS_PATH, "w") as handle:
        json.dump({"about": "control confidence scales, tuned by scripts/tune_controls.py on the "
                            "bench's training matches (ADR 0003)", "params": best}, handle, indent=1)
        handle.write("\n")
    with open(METRICS_PATH, "w") as handle:
        json.dump(report, handle, indent=1, sort_keys=True)
    say("wrote {} and {}".format(os.path.relpath(PARAMS_PATH, REPO),
                                 os.path.relpath(METRICS_PATH, REPO)))


if __name__ == "__main__":
    main()
