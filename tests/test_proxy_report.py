"""Step 7.2 and 7.4: the proxy report on hand-made labels with known answers, and the official
validator. No match data, and no ground truth."""

import json

import numpy as np
import pytest

from possession import checks, output
from possession.checks import RunError


def report(text, blocks, fps=1, **kwargs):
    return checks.proxy_report(np.array(list(text)), blocks=blocks, fps=fps, **kwargs)


def test_known_answers_in_one_half():
    # frames 2..9 graded: H H H D D A A H
    r = report("XXHHHDDAAHXX", [(2, 9)])
    assert r["graded_frames"] == 8
    assert r["n_spells"] == 3                           # HHH, AA, H
    assert r["spell_length_s"]["min"] == 1.0
    assert r["spell_length_s"]["max"] == 3.0
    assert r["spell_length_s"]["mean"] == 2.0
    assert r["spell_length_s"]["median"] == 2.0
    assert r["label_changes_per_min"] == round(3 / (8 / 60.0), 4)   # H>D, D>A, A>H in 8 s
    assert r["spells_per_min"] == round(3 / (8 / 60.0), 4)
    assert r["dead_ball_share"] == 0.25
    assert r["home_share"] == round(4 / 6.0, 4)
    assert r["label_share"] == {"H": 0.5, "A": 0.25, "D": 0.25, "X": 0.0}


def test_frames_outside_the_halves_are_ignored():
    inside = "HHHDDAAH"
    a = report("XX" + inside + "XX", [(2, 9)])
    b = report("AD" + inside + "DH", [(2, 9)])
    assert a == b


def test_nothing_is_carried_across_halftime():
    # Two halves of H with halftime between: two possessions, no label change.
    r = report("HHDHH", [(0, 1), (3, 4)])
    assert r["n_spells"] == 2
    assert [h["n_spells"] for h in r["halves"]] == [1, 1]
    assert r["label_changes_per_min"] == 0.0
    assert r["dead_ball_share"] == 0.0


def test_a_half_with_no_real_label_warns():
    r = report("HAHAXXXX", [(0, 3), (4, 7)])
    assert [h["all_x"] for h in r["halves"]] == [False, True]
    assert "half 2 has no real label: every frame is X" in r["warnings"]


def test_jittery_labels_warn_but_do_not_raise():
    r = report("HA" * 300, [(0, 599)], fps=25)
    joined = " ".join(r["warnings"])
    assert "spells_per_min" in joined and "short_spell_share" in joined
    assert "dead_ball_share" in joined


def test_plausible_labels_do_not_warn():
    # 25 fps; 8 s possessions alternating teams, 4 s dead every third: about 5 per minute.
    one = "H" * 200 + "A" * 200 + "D" * 100
    r = report(one * 30, [(0, len(one) * 30 - 1)], fps=25)
    assert r["warnings"] == []


def test_smart_tagging_side_and_agreement():
    codes = np.array(["BUILD UP", "DEFENDING IN MIDDLE THIRD", "PROGRESSION"])
    team = np.array([["H", "A", ""],    # defending code ignored -> H
                     ["H", "", "A"],    # in-possession codes disagree -> none
                     ["", "", "A"],
                     ["?", "", ""],     # overlapping phases disagree -> none
                     ["", "H", ""]])    # only a defending code -> none
    side = checks.smart_tagging_side(codes, team)
    assert side.tolist() == ["H", "", "A", "", ""]
    r = report("HHHHH", [(0, 4)], smart_tagging=side)
    assert r["smart_tagging"] == {"coverage": 0.4, "agreement": 0.5}


def test_write_metrics_is_deterministic(tmp_path):
    r = report("HHDAA", [(0, 4)])
    path = checks.write_metrics("demo", r, repo=str(tmp_path))
    first = open(path, "rb").read()
    checks.write_metrics("demo", r, repo=str(tmp_path))
    assert open(path, "rb").read() == first
    record = json.loads(first)
    assert record["experiment"] == "demo" and record["n_spells"] == 2
    assert "thresholds" in record


def test_format_report_prints_warnings():
    text = checks.format_report(report("HAHAXXXX", [(0, 3), (4, 7)]))
    assert "warning: half 2 has no real label" in text


def test_official_validator_passes_and_fails(tmp_path):
    good = output.write_csv(["X"] * checks.N_FRAMES, str(tmp_path / "good.csv"))
    assert "OK" in checks.run_official_validator("demo", good)
    bad = output.write_csv(["X"] * 10, str(tmp_path / "bad.csv"))
    with pytest.raises(RunError, match="demo: evaluation/validate_submission.py fails"):
        checks.run_official_validator("demo", bad)
