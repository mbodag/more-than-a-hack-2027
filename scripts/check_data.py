"""Check that the match data on this machine is byte-identical to everyone else's.

Every file is hashed with SHA-256 and compared with the checksums committed in
scripts/data_checksums.json. Checksums are not data, so committing them is safe; the files
themselves never are (AGENTS.md, hard rule 1).

Paths come from games.asset_paths(), so MTAH_DATA_DIR is respected and no file name is spelt out
here. The checksum file is keyed by source and asset name for the same reason.

    python scripts/check_data.py             # check; exit 1 naming any missing or mismatched file
    python scripts/check_data.py --write     # record checksums from the files on this machine

Run the check before any experiment. Other code can call check_data() directly, which raises
DataCheckError instead of exiting.

Standard library only, like games.py, so it runs before any dependency is installed.
"""

import argparse
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import games  # noqa: E402  (needs the repo root on sys.path first)

CHECKSUMS = os.path.join(REPO, "scripts", "data_checksums.json")

# What we use. The videos and the broadcast folder are out of scope for now (issue #2); to cover
# more, add their asset names here and run --write on a machine that has the files.
REQUIRED = {
    "tactical": ("atd_meta", "atd_raw", "homography", "smart_tagging"),
}

_CHUNK = 1 << 20


class DataCheckError(Exception):
    """The match data is missing, or differs from the committed checksums."""


def sha256(path):
    """Hex SHA-256 of a file, read in chunks so the 200 MB tracking file stays out of memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_paths(data_dir=None, required=None):
    """[(source, asset, path)] for every file we depend on."""
    rows = []
    for source, assets in sorted((required or REQUIRED).items()):
        paths = games.asset_paths(data_dir=data_dir, source=source)
        rows.extend((source, asset, paths[asset]) for asset in assets)
    return rows


def load_checksums(path=CHECKSUMS):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError) as error:
        raise DataCheckError("cannot read the committed checksums {}: {}".format(path, error))


def check_data(data_dir=None, checksums=None, required=None):
    """Raise DataCheckError naming every missing or mismatched file; return the paths checked."""
    expected = checksums if checksums is not None else load_checksums()
    problems = []
    checked = []
    for source, asset, path in _required_paths(data_dir, required):
        want = expected.get(source, {}).get(asset)
        if want is None:
            problems.append("no committed checksum for {}/{} ({})".format(source, asset, path))
        elif not os.path.isfile(path):
            problems.append("missing: {}".format(path))
        else:
            got = sha256(path)
            if got != want:
                problems.append("mismatch: {}\n    expected {}\n    got      {}".format(
                    path, want, got))
            else:
                checked.append(path)
    if problems:
        raise DataCheckError(
            "match data check failed:\n  {}\n\nDownload the files again (see issue #2 and "
            "data/README.md), keeping the names exactly, into\n  {}\nor point MTAH_DATA_DIR at "
            "wherever they are.".format("\n  ".join(problems), data_dir or games.DATA_DIR))
    return checked


def write_checksums(data_dir=None, path=CHECKSUMS, required=None):
    """Hash the files on this machine and record them. Every file must exist."""
    table = {}
    for source, asset, file_path in _required_paths(data_dir, required):
        if not os.path.isfile(file_path):
            raise DataCheckError("cannot record a checksum, file missing: {}".format(file_path))
        table.setdefault(source, {})[asset] = sha256(file_path)
    with open(path, "w") as handle:
        json.dump(table, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true",
                        help="record checksums from the files on this machine instead of checking")
    args = parser.parse_args(argv)
    try:
        if args.write:
            table = write_checksums()
            count = sum(len(assets) for assets in table.values())
            print("wrote {} checksums to {}".format(count, os.path.relpath(CHECKSUMS)))
        else:
            checked = check_data()
            print("match data OK: {} files match the committed checksums".format(len(checked)))
    except DataCheckError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
