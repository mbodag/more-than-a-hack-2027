# Set pieces from Smart Tagging (`hmm_setpiece`), a trial

**Issue:** none yet (a trial). **Branch:** `setpiece-trial`.

## Method

`hmm_final`, unchanged, with Smart Tagging's `SET PIECES` tags as extra evidence before decoding.
The motive is one failure seen in the viewer: during a corner's setup the ball is not tracked, and
`hmm_final` labels the stoppage as short H and A spells, then gives the restart to the defending
side (the defenders are nearest the ball in the box).

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

## Command

```bash
python -m possession run hmm_setpiece
```

## Proxy metrics (BAR-ATH, label-free)

| | `hmm_final` | `hmm_setpiece` | `hmm_final_ms` (submitted) |
|---|---|---|---|
| possessions | 384 | 377 | 408 |
| dead-ball share | 0.403 | 0.417 | 0.355 |
| possessions under 1 s | 8.1% | 8.5% | 6.6% |
| Smart Tagging agreement | 0.803 | 0.815 | 0.814 |
| warnings | none | none | none |

Smart Tagging agreement uses its in-possession phases, not set pieces, so it is not circular.

## Validation

The runner's format check and the organisers' validator passed.

## Verdict

Trial. It looks right on the corner that motivated it, seen side by side in the viewer. It is not a
candidate until benched, and it is built on `hmm_final`, not `hmm_final_ms`: the fair next version
puts the same evidence on `hmm_final_ms`.
