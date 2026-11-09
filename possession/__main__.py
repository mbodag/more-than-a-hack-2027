"""The experiment runner (ADR 0002): data files in, one checked CSV out.

    python -m possession run <experiment>                          # normal run
    python -m possession run <experiment> --rebuild-cache          # rebuild every item it uses
    python -m possession run <experiment> --rebuild-cache tracking # rebuild only the named items
    python -m possession run north_london_tactics                  # the submission

Steps: 1 check the data, 2 create the Match, 3 run the experiment, 4 check its output, 5 fill,
6 write the CSV, 7 check the file on disk and report label-free proxy metrics.
"""

import argparse
import importlib
import os
import pkgutil
import sys
import time

from possession import REPO, checks, output
from possession.cache import CacheError
from possession.checks import RunError
from possession.fill import fill
from possession.match import Match

sys.path.insert(0, os.path.join(REPO, "scripts"))
import check_data as _check_data  # noqa: E402  (scripts/ is not a package)

EXPERIMENTS_PACKAGE = "possession.experiments"


def list_experiments():
    """Names of every experiment: each module in possession/experiments not starting with _."""
    package = importlib.import_module(EXPERIMENTS_PACKAGE)
    return sorted(name for _, name, is_package in pkgutil.iter_modules(package.__path__)
                  if not is_package and not name.startswith("_"))


def load_experiment(name):
    """The experiment's predict function."""
    module = importlib.import_module("{}.{}".format(EXPERIMENTS_PACKAGE, name))
    predict = getattr(module, "predict", None)
    if not callable(predict):
        raise RunError("{}: possession/experiments/{}.py has no predict(match) function".format(
            name, name))
    return predict


def check_data():
    """Step 1. Raises check_data.DataCheckError naming every missing or changed file."""
    return _check_data.check_data()


def make_match(rebuild):
    """Step 2. `rebuild` is None (use the cache), True (rebuild every item used) or a tuple of
    item names to rebuild. An unknown item name stops the run with the list of known ones."""
    try:
        return Match(rebuild=rebuild)
    except CacheError as error:
        raise RunError(str(error))


def run(experiment, rebuild=None, repo=REPO, say=print):
    """Steps 1 to 7 for one experiment; returns the path of the CSV written."""
    known = list_experiments()
    if experiment not in known:
        raise RunError("no experiment {!r}. Existing experiments: {}".format(
            experiment, ", ".join(known) or "none yet"))

    say("1 · checking the match data")
    check_data()

    say("2 · creating the Match")
    match = make_match(rebuild)

    say("3 · running {}".format(experiment))
    predict = load_experiment(experiment)
    started = time.time()
    labels = predict(match)
    say("    predict() took {:.1f} s".format(time.time() - started))

    say("4 · checking its output")
    labels = checks.check_output(experiment, labels)

    say("5 · filling X inside each half")
    filled = fill(labels)
    say("    {} X filled".format(int((labels != filled).sum())))

    path = output.output_path(experiment, repo)
    say("6 · writing {}".format(os.path.relpath(path, repo)))
    output.write_csv(filled, path)

    say("7 · checking the file on disk")
    table = checks.check_file(experiment, path)
    say("    official format check passed")
    side = checks.smart_tagging_side(match.smart_tagging_codes, match.smart_tagging_team)
    report = checks.proxy_report(table["label"].astype(str).values, smart_tagging=side)
    say(checks.format_report(report))
    metrics = checks.write_metrics(experiment, report, repo)
    say("    proxy report saved to {}".format(os.path.relpath(metrics, repo)))
    if experiment == output.SUBMISSION_EXPERIMENT:
        say("    python evaluation/validate_submission.py \"{}\"".format(output.SUBMISSION_FILE))
        say(checks.run_official_validator(experiment, path).rstrip())
    say("done: {}".format(os.path.relpath(path, repo)))
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m possession",
                                     description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="run one experiment and write its CSV")
    run_parser.add_argument("experiment", help="a file name in possession/experiments, no .py")
    run_parser.add_argument("--rebuild-cache", nargs="*", metavar="ITEM", default=None,
                            help="rebuild input items even if cached: every item the experiment "
                                 "uses, or only those named")
    args = parser.parse_args(argv)
    if args.rebuild_cache is not None:
        args.rebuild_cache = tuple(args.rebuild_cache) or True
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args.experiment, rebuild=args.rebuild_cache)
    except (RunError, _check_data.DataCheckError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
