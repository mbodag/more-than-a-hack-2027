"""Steps 4, 6 and 7 and the command itself, with a fake experiment and no match data."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import possession.__main__ as runner  # noqa: E402
from possession import N_FRAMES, checks, output  # noqa: E402
from possession.checks import RunError  # noqa: E402


# -- step 4 ------------------------------------------------------------------------------------

def test_check_output_accepts_good_labels():
    out = checks.check_output("demo", ["H", "A", "D", "X"], n_frames=4)
    assert out.tolist() == ["H", "A", "D", "X"]


@pytest.mark.parametrize("labels, message", [
    (["H", "A", "D"], "returned 3 labels; expected 4"),
    ([["H", "A"], ["D", "X"]], "shape"),
    (["H", "a", "D", "X"], "outside"),
    (["H", "A", None, "X"], "first at frame 2"),
    (np.array([0, 1, 2, 3]), "outside"),
    (["H", "A", "D", ""], "outside"),
])
def test_check_output_names_the_experiment_and_the_fault(labels, message):
    with pytest.raises(RunError, match="demo: .*" + message):
        checks.check_output("demo", labels, n_frames=4)


# -- step 6 ------------------------------------------------------------------------------------

def test_csv_bytes_are_exact():
    assert output.csv_bytes(np.array(["H", "X", "D"])) == b"frame,label\n0,H\n1,X\n2,D\n"


def test_output_paths(tmp_path):
    repo = str(tmp_path)
    assert output.output_path("nearest", repo) == os.path.join(repo, "outputs", "nearest.csv")
    assert output.output_path("north_london_tactics", repo) == os.path.join(
        repo, "North London Tactics.csv")


def test_write_csv_overwrites_and_leaves_no_tmp(tmp_path):
    path = str(tmp_path / "outputs" / "x.csv")
    output.write_csv(["H"], path)
    output.write_csv(["A", "D"], path)
    assert open(path, "rb").read() == b"frame,label\n0,A\n1,D\n"
    assert os.listdir(str(tmp_path / "outputs")) == ["x.csv"]


# -- step 7 (until #12) ------------------------------------------------------------------------

def test_check_file_passes_a_written_csv(tmp_path):
    path = output.write_csv(["H", "A", "D", "X"], str(tmp_path / "a.csv"))
    assert len(checks.check_file("demo", path, n_frames=4)) == 4


def test_check_file_fails_on_a_bad_file(tmp_path):
    path = str(tmp_path / "bad.csv")
    with open(path, "w") as handle:
        handle.write("frame,label\n0,H\n2,A\n")
    with pytest.raises(RunError, match="demo: .*official format check.*not 0..1 in order"):
        checks.check_file("demo", path, n_frames=2)


# -- the command, end to end -------------------------------------------------------------------

def _predict(match):
    rng = np.random.RandomState(0)
    return rng.choice(["H", "A", "D", "X", "X"], size=N_FRAMES)


@pytest.fixture
def fake(monkeypatch):
    calls = {"data": 0, "rebuild": []}

    def check_data():
        calls["data"] += 1

    def make_match(rebuild):
        calls["rebuild"].append(rebuild)
        return object()

    monkeypatch.setattr(runner, "check_data", check_data)
    monkeypatch.setattr(runner, "make_match", make_match)
    monkeypatch.setattr(runner, "list_experiments", lambda: ["demo", "north_london_tactics"])
    monkeypatch.setattr(runner, "load_experiment", lambda name: _predict)
    return calls


def test_run_twice_gives_byte_identical_csvs(fake, tmp_path):
    say = lambda *a: None  # noqa: E731
    path = runner.run("demo", repo=str(tmp_path), say=say)
    first = open(path, "rb").read()
    runner.run("demo", repo=str(tmp_path), say=say)
    assert open(path, "rb").read() == first
    assert path == os.path.join(str(tmp_path), "outputs", "demo.csv")
    assert fake["data"] == 2


def test_run_fills_before_writing(fake, tmp_path):
    path = runner.run("demo", repo=str(tmp_path), say=lambda *a: None)
    rows = open(path).read().splitlines()[1:]
    labels = np.array([row.split(",")[1] for row in rows])
    assert not (labels[11185:90652] == "X").any()
    assert not (labels[110808:190198] == "X").any()
    assert (labels[:11185] == "X").any()  # outside the halves: left alone


def test_submission_goes_to_the_repo_root(fake, tmp_path):
    path = runner.run("north_london_tactics", repo=str(tmp_path), say=lambda *a: None)
    assert path == os.path.join(str(tmp_path), "North London Tactics.csv")


def test_unknown_experiment_lists_the_existing_ones(fake, tmp_path):
    with pytest.raises(RunError, match="no experiment 'nope'.*demo, north_london_tactics"):
        runner.run("nope", repo=str(tmp_path))
    assert fake["data"] == 0


def test_failed_data_check_stops_before_the_experiment(fake, monkeypatch, tmp_path):
    def broken():
        raise runner._check_data.DataCheckError("missing: atd")

    monkeypatch.setattr(runner, "check_data", broken)
    monkeypatch.setattr(runner, "make_match", lambda rebuild: pytest.fail("ran past step 1"))
    with pytest.raises(runner._check_data.DataCheckError):
        runner.run("demo", repo=str(tmp_path))


def test_bad_output_stops_before_writing(fake, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "load_experiment", lambda name: lambda match: ["H"])
    with pytest.raises(RunError, match="demo: predict\\(\\) returned 1 labels"):
        runner.run("demo", repo=str(tmp_path), say=lambda *a: None)
    assert not os.path.exists(str(tmp_path / "outputs" / "demo.csv"))


def test_main_returns_1_with_the_message(fake, capsys):
    assert runner.main(["run", "nope"]) == 1
    assert "no experiment 'nope'" in capsys.readouterr().err


@pytest.mark.parametrize("argv, rebuild", [
    ([], None),
    (["--rebuild-cache"], True),
    (["--rebuild-cache", "ball_xy", "track_team"], ("ball_xy", "track_team")),
])
def test_rebuild_cache_argument(argv, rebuild):
    assert runner.parse_args(["run", "demo"] + argv).rebuild_cache == rebuild


def test_real_experiment_listing_runs():
    assert isinstance(runner.list_experiments(), list)
