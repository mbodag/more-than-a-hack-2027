"""check_data on fake files in a temporary folder; never touches the real match data."""

import hashlib
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import check_data as cd  # noqa: E402
import games  # noqa: E402

REQUIRED = {"tactical": ("atd_meta", "homography")}


@pytest.fixture
def data_dir(tmp_path):
    paths = games.asset_paths(data_dir=str(tmp_path))
    os.makedirs(os.path.dirname(paths["atd_meta"]))
    for asset in REQUIRED["tactical"]:
        with open(paths[asset], "wb") as handle:
            handle.write(asset.encode())
    return str(tmp_path)


def _expected():
    return {"tactical": {a: hashlib.sha256(a.encode()).hexdigest() for a in REQUIRED["tactical"]}}


def test_matching_files_pass(data_dir):
    checked = cd.check_data(data_dir, _expected(), REQUIRED)
    assert len(checked) == 2


def test_missing_file_is_named(data_dir):
    path = games.asset_paths(data_dir=data_dir)["homography"]
    os.remove(path)
    with pytest.raises(cd.DataCheckError, match="missing: " + path):
        cd.check_data(data_dir, _expected(), REQUIRED)


def test_changed_file_is_named(data_dir):
    path = games.asset_paths(data_dir=data_dir)["atd_meta"]
    with open(path, "ab") as handle:
        handle.write(b"!")
    with pytest.raises(cd.DataCheckError, match="mismatch: " + path):
        cd.check_data(data_dir, _expected(), REQUIRED)


def test_write_then_check_round_trips(data_dir, tmp_path):
    out = str(tmp_path / "sums.json")
    cd.write_checksums(data_dir, out, REQUIRED)
    assert cd.check_data(data_dir, cd.load_checksums(out), REQUIRED)


def test_committed_checksums_cover_every_required_file():
    table = cd.load_checksums()
    for source, assets in cd.REQUIRED.items():
        for asset in assets:
            assert len(table[source][asset]) == 64
