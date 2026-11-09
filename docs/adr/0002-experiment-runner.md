# 0002: One command per experiment: cached match data, a `predict` per file, a fixed tail

- Status: proposed
- Date: 2026-11-09
- Issue: #4

## Context

We will try many possession methods in parallel sessions over a week, and the submitted CSV must
regenerate exactly from one documented command (RULES.md, hard rule 5). Parsing the 199,319-line
tracking file takes a minute or two, and any video processing will take far longer, so inputs must
be prepared once and reused. Adding an experiment has to be cheap, and two sessions adding
experiments must never edit the same file.

The organisers' baseline mixes everything in one script: reading the data, the method, writing the
CSV. We separate the parts that vary (the method) from the parts that must be identical for every
experiment (loading, filling, writing, checking).

## Decision

### The command

```bash
python -m possession run <experiment> [--rebuild-cache [ITEM ...]]
```

`possession/__main__.py` reads the arguments with `argparse`. One run does, in order:

1. `check_data()`: the match data matches the committed checksums, or stop.
2. Build a `Match` (lazy; see below).
3. `labels = possession.experiments.<experiment>.predict(match)`.
4. **Fill** the labels (see below).
5. Write the CSV: `outputs/<experiment>.csv`, or `North London Tactics.csv` at the repo root for
   the submission experiment.
6. **Check** the file as written on disk (see below).

### An experiment is one file

`possession/experiments/<experiment>.py` exposes:

```python
def predict(match) -> np.ndarray   # 199,319 labels, each "H", "A", "D" or "X", indexed by frame
```

- **The file name is the experiment's name**, and names its CSV. Adding an experiment adds one
  file; nothing central is edited, so parallel sessions don't collide.
- **No parameters on the command line.** Parameters live in the file. A variant is a new small file
  that calls the other with different values. Every CSV is then reproducible from its name alone.
- **Smoothing, minimum spell lengths and similar post-processing belong to the experiment.** They
  are the method (the baseline's smoothing window alone moves spellF1 fivefold), so they are shared
  functions an experiment chooses to call, never a step the runner forces on everyone.

### The submission is an experiment too

`possession/experiments/north_london_tactics.py` re-exports the chosen experiment's `predict` and
nothing else, so changing what we submit is a one-line diff. The runner writes its CSV under the
team name, at the repo root (the one CSV we commit), and also runs the organisers'
`evaluation/validate_submission.py` on it. The reproduction command for the technical document is
therefore:

```bash
python -m possession run north_london_tactics
```

### `Match`: the inputs, prepared once

- **Lazy.** Each item (parsed tracking, Smart Tagging per frame, homography, later video features)
  is computed the first time an experiment reads it, then cached. An experiment that never touches
  the video never pays for it.
- **Frame-indexed.** Every per-frame item is a numpy array whose first axis is the frame of the
  tactical video, length 199,319: `x[i]` is submission row `i`. ATD's 1-based ids are converted
  once, inside `Match`.
- **Additive only.** New items are added freely. An existing item's name and format never change:
  a different format gets a new name (e.g. `ball_xy` and `ball_xy_filled`). Existing experiments
  therefore never break.
- **The video is a path**, not decoded frames. Anything derived from it is an item like any other.

### The cache

- **Where:** `outputs/cache/` (gitignored with the rest of `outputs/`; it is derived from the match
  data, so it must never be committed).
- **Key:** `<item>__v<N>`, where `N` is a version number in the code that builds the item, bumped
  whenever that code changes. The data itself is pinned by `check_data()`. A code change therefore
  never silently loads a stale result.
- **Writes are atomic.** A build writes `<file>.tmp` and only swaps it into place with
  `os.replace` once complete. Stopping a build (Ctrl-C, a crash, a full disk) leaves the old cache
  untouched; the leftover `.tmp` is removed on the next run.
- **One backup.** Before a rebuild at the same version replaces a file, the current one is renamed
  to `<item>__v<N>.prev`. Older versions sit side by side under their own version numbers. Nothing
  is ever deleted automatically.
- **`--rebuild-cache`** recomputes and overwrites. With no names it rebuilds every item this
  experiment uses; with names (`--rebuild-cache tracking`) only those. Before starting it prints
  what it will rebuild and how long that took last time, so an accidental video rebuild can be
  stopped at once. It is a safety net; version numbers are what keep the cache current.
- **The cache is only a speed-up.** Before the method freeze, the submission is regenerated with an
  empty cache and must come out byte-identical.

### Fill

Every `X` inside a graded half is replaced by the last non-`X` label in the same half, or by the
first one at the head of a half. Halves come from `games.half_video_blocks()`; nothing is carried
across halftime. Ungraded frames are left as they are.

This mirrors the scorer's own `fill_abstentions`, so it leaves spellF1 unchanged and can only help
macroF1 (an `X` on a graded frame is always a miss; a filled label is sometimes right). It is
mechanical and the same for every experiment.

### Check

The runner reads the CSV back from disk, as the scorer will, and:

1. calls the official format check, `possession_metrics.validate_submission`, which the scorer
   itself uses. A failure stops the run with its reason. This is the only check that can fail a
   run;
2. computes our label-free proxy report: number of spells, spell-length distribution and the
   longest and shortest spells, flips per minute, dead-ball share, agreement with Smart Tagging.
   Anything unusual is printed as a warning, never a failure. Which metrics and thresholds exactly
   is for #5;
3. writes the proxy metrics to `experiments/metrics/<experiment>.json`, which is committed.

Proxy metrics are computed only inside the two graded halves. `evaluation/` is imported, never
edited; the public ground truth is never read.

## Considered options

- **Parameters on the command line, with sweeps (`--param window=25,50`).** Rejected: one file
  would produce many CSVs, and a CSV's name would no longer say how to regenerate it.
- **A separate `submit` command.** Rejected in favour of `north_london_tactics.py`: the submission
  is then an ordinary experiment, and choosing it is a reviewed one-line diff.
- **A preprocessing stage that builds everything before the experiment runs.** Rejected: the first
  run of any experiment would pay for every item, video included.
- **Shared post-processing (smoothing) in the runner.** Rejected: it is the main lever on spellF1
  and so part of each method, not of the pipeline.
- **Extending `evaluation/validate_submission.py`.** Rejected: it mirrors the organisers' scorer,
  which scores with its own copy, and `evaluation/` is never edited. We wrap it instead.
- **Experiments as configurations over shared components.** Deferred: modules first; shared
  components are extracted once two experiments duplicate code.

## Consequences

- Adding an experiment is one file and one command.
- `Match` only grows. Retiring an item means leaving it in place, unused.
- A variant of an experiment is a new file, which adds a little repetition.
- The cache can grow large with video; clearing it is a deliberate `rm -r outputs/cache`.
