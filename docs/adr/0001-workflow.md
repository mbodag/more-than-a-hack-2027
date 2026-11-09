# 0001: How we work: issues, sessions and shared context

- Status: accepted
- Date: 2026-11-09
- Issue: #1

## Context

A team of four, plus many short-lived agent sessions (cloud and local), has eight days to build and
submit. Sessions start with no memory of earlier ones, the match data is gated and cannot be
downloaded in cloud sessions, and the repo must stay public with no data in it.

## Decision

- **Unit of work:** one GitHub issue, one session, one PR. The PR title starts with `#<n>`; the
  body says `Closes #<n>`. The owner merges every PR.
- **Board:** GitHub Projects, *Backlog → Ready → In progress → In review → Done*, moved by PR
  automations.
- **Shared context:** `AGENTS.md` holds stable facts only (about 100 lines). Status lives in issue
  comments (`Done / Results / Next / Blockers / Branch or PR`), posted at the end of each session
  and when a PR opens. `CONTEXT.md` is the glossary; `docs/adr/` records hard-to-reverse decisions.
- **Issue types by label:** `ready-for-agent` vs `ready-for-human`, plus `needs-triage`, `blocked`
  and `needs-data`. An agent that hits an unsettled design decision escalates instead of guessing.
- **Data:** stays on laptops. `needs-data` issues run in local sessions; cloud sessions take the
  rest. A checksum check guarantees identical files.
- **Experiments:** one record per experiment in `experiments/`; commit metrics, not CSVs.
- **CI:** unit tests, a guard against tracked data or video, and format validation of the final
  CSV. CI never scores against the public ground truth, so that score can't become a tuning signal.
- **Organiser files** (`games.py`, `starter/`, `evaluation/`) are not edited; our code is in
  `possession/`. Their updates come in from an `upstream` remote.

## Consequences

- Any session can start from its issue alone.
- Data-dependent work is limited by which laptops have the data and are open.
- There is no automatic score signal; judging progress relies on proxy metrics (see the
  method-choice issue, #5).
