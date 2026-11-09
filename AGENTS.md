# AGENTS.md

Shared context for every session on this repo. Stable facts only: **current status lives on the
GitHub board and in issue comments, never here.**

## The challenge in five lines

- More than a Hack 2027, Phase 1: for every frame of one match (Barça vs Athletic, `BAR-ATH`),
  say who has the ball. Output `frame,label`, exactly 199,319 rows, frames `0…199318` in order.
- Labels: `H` home in possession, `A` away, `D` dead ball, `X` abstain. Terms: [CONTEXT.md](CONTEXT.md).
- Ranked on `spellF1` (whole possessions matched by temporal IoU); `macroF1` is the first tie-break.
  Baseline to beat: `spellF1` 0.2433.
- Inputs: automatic, uncorrected tracking (ATD, ball missing on 41% of graded frames), Smart
  Tagging, homography, video. Organiser docs: [CHALLENGE.md](CHALLENGE.md), [DATA.md](DATA.md),
  [EVALUATION.md](EVALUATION.md), [RULES.md](RULES.md), [SUBMISSION.md](SUBMISSION.md).
- Deadline: **17 November 2026**. Method frozen 15 November; 16 November is for the technical
  document and the reproducibility check.

## Our team

We are **North London Tactics**, the name registered with the organisers. It must match exactly
(spaces and capitals included) in the submission form's Team Name field and in the deliverable
file names, `North London Tactics.csv` and `North London Tactics.pdf` (or `.md`), because the
scorer labels the board row with the CSV's file name (SUBMISSION.md). Quote the file name in shell
commands.

## Hard rules (breaking one can disqualify us)

1. **Never commit match data.** This repo is public and the data is not redistributable. `data/` is
   gitignored except its README; CI fails if anything under it, or any video, is tracked.
2. **No hand annotation of the provided match**, for any purpose. Watching clips to debug is fine;
   writing down labels from them is not.
3. **No tuning or training on the public 5-minute ground truth** (`evaluation/gt/`). It is a
   pipeline sanity check only. Judge changes with label-free proxy metrics.
4. **No human-produced data about this match** (Opta, StatsBomb, etc.).
5. **The submitted CSV must regenerate exactly** from this repo with one documented command.

## Repo layout

| Path | What | May we edit it? |
|---|---|---|
| `games.py` | match constants; the only place file paths are spelt out | no (organisers') |
| `starter/` | organisers' loaders, baseline, notebooks | no: import, don't edit |
| `evaluation/` | the official scorer and validator | no, never |
| `data/` | gated match data, gitignored; or point `MTAH_DATA_DIR` elsewhere | never commit |
| `possession/` | our code | yes |
| `scripts/` | our helper scripts (e.g. the data check) | yes |
| `experiments/` | one results record per experiment; see its README | yes |
| `docs/adr/` | decisions that are hard to reverse | yes |
| `docs/TECHNICAL.md` | the technical document deliverable, kept current | yes |
| `outputs/` | generated CSVs, clips, figures; gitignored | never commit |

Organiser files stay untouched so merges from `upstream` (metrica-sports/more-than-a-hack-2027)
stay clean.

## Commands

```bash
pip install -r requirements-starter.txt                  # numpy, pandas, kloppy, matplotlib
python scripts/check_data.py                             # data matches committed SHA-256s; run before any experiment
python evaluation/validate_submission.py <file>.csv      # before trusting any CSV
python evaluation/run_evaluation.py                      # scores evaluation/results/*.csv on the public 5 min
python starter/baseline.py                               # rebuilds the baselines (needs data)
python -m possession.viz                                 # match viewer: open outputs/viewer/index.html (needs data)
```

## How we work

- **One issue, one session, one PR.** Start by reading your issue and its latest status comment.
  Your first comment on the issue names your branch.
- **PRs:** title starts with `#<n>`, body says `Closes #<n>`. The owner merges; never push to `main`.
- **Small, atomic commits.** One logical change each.
- **Status comment** at the end of every session and when a PR opens:
  `Done / Results / Next / Blockers or decisions needed / Branch or PR`.
- **Labels:**
  - `ready-for-agent`: clear acceptance criteria, no open design decisions, dependencies closed,
    a concrete way to verify.
  - `ready-for-human`: design decisions or planning. Done when there's an ADR, new terms are in
    `CONTEXT.md`, and follow-up `ready-for-agent` issues exist.
  - `needs-triage`: new and unsorted. `blocked`: waiting on something.
  - `needs-data`: needs the match data, so it must run in a **local** session on a laptop that
    has it. Cloud sessions can't download the data (the source is restricted).
- **Hit an unsettled design decision?** Stop. Comment with the question, options and your
  recommendation; relabel `ready-for-human` + `blocked`; commit only work that doesn't depend on
  the answer. Never quietly pick an answer.
- **Run `python scripts/check_data.py` before any experiment.** It exits non-zero if a data file
  is missing or differs from the checksums in `scripts/data_checksums.json`. The experiment runner,
  once it exists, must call `check_data()` itself.
- **Experiments** are recorded in `experiments/` (see its README). Commit metrics, not CSVs. Only
  the final `North London Tactics.csv` is ever committed.
- **Method changes** update `docs/TECHNICAL.md` in the same PR.
- **Writing:** British English spelling.
