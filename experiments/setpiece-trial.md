# Set pieces from Smart Tagging (`hmm_setpiece`, `hmm_setpiece_ms`), a trial

**Issue:** none yet (a trial). **Branch:** `setpiece-trial`.

## Method

`hmm_final`, unchanged, with Smart Tagging's `SET PIECES` tags as extra evidence before decoding.
The motive is a failure pattern spotted while debugging in the viewer: while a set piece is being
set up the ball is often not tracked, so the model can label the stoppage as short H and A spells;
and at a delivery into the box the defenders are nearest the ball, so the restart can go to the
defending side. The fix is general (every set-piece tag, nothing tied to a frame) and is still to
be judged on the bench.

For each set-piece tag:

- the restart is where the ball reappears (the end of the nearest ball gap of 2 s or more, within
  10 s of the tag start), or the tag start if there is no such gap;
- before the restart, back to the start of that gap (30 s at most), `D` gets +1.0 per frame;
- for 5 s after it, the tagged team gets +1.0 per frame (not for throw-ins, which are all tagged
  `N/A`).

The bonuses are soft: strong tracking evidence still wins. The windows and bonus sizes are guesses,
**not fitted and not benched**: the bench matches have no Smart Tagging. Benching it needs set-piece
tags made from the bench matches' own event data, made to look like Smart Tagging (`N/A` on
throw-ins, its timing, some tags dropped).

`hmm_setpiece_ms` adds the same evidence to the submission's model, `hmm_final_ms`, with its models
and settings unchanged. On a bench match (no Smart Tagging) each gives exactly its base model's
labels, so on BAR-ATH every difference comes from the set-piece evidence alone.

## Command

```bash
python -m possession run hmm_setpiece
python -m possession run hmm_setpiece_ms
```

## Proxy metrics (BAR-ATH, label-free)

| | `hmm_final` | `hmm_setpiece` | `hmm_final_ms` (submitted) | `hmm_setpiece_ms` |
|---|---|---|---|---|
| possessions | 384 | 377 | 408 | 406 |
| dead-ball share | 0.403 | 0.417 | 0.355 | 0.372 |
| possessions under 1 s | 8.1% | 8.5% | 6.6% | 6.7% |
| Smart Tagging agreement | 0.803 | 0.815 | 0.814 | **0.821** |
| warnings | none | none | none | none |

Smart Tagging agreement uses its in-possession phases, not set pieces, so it is not circular.

## Validation

The runner's format check and the organisers' validator passed.

## Verdict

Trial. On both base models it raises Smart Tagging agreement and the dead-ball share. Nothing was
tuned by watching BAR-ATH: the windows and bonuses were set once, before the first run. `hmm_setpiece_ms` is the
version to compare with the submission. Neither is a candidate until benched with set-piece tags
made from the bench matches' own events.

## The submission: `hmm_setpiece_islands`

Chosen by the team as the submission. It is `hmm_setpiece` (on `hmm_final`) followed by
`hmm.drop_dead_islands` at 3 s (`island_min` 75), the rule from #35: any play stretch shorter than
3 s with a stoppage on both sides becomes `D`. `north_london_tactics` runs it. Nothing is refitted.

Bench test, scored once (degradation v2, seeds 0-2): **spellF1 0.479, macroF1 0.837**, against
`hmm_final`'s 0.479 / 0.838. The bench has no Smart Tagging and few long breaks, so neither layer
does anything measurable there; the bench says they do no harm, not that they help.

It is built on `hmm_final`, not `hmm_final_ms`, so it keeps the ball-speed units bug (see
`docs/TECHNICAL.md`, limitations). That was the team's choice, made knowingly.
