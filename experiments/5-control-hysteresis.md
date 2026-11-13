# 5 · control_hysteresis (candidate for option A)

- **Issue and commit:** #5 (option A), the commit that adds this file. A candidate only:
  `north_london_tactics` still points at `nearest_smooth50` until #5 is decided.
- **Method** (`possession/experiments/_control.py`): control evidence for a side when its nearest
  player is within 2 m of the ball and the other side's nearest is at least 1 m further; possession
  changes after 6 evidence frames in a row (gaps of up to 10 frames allowed), dated to the first of
  them; dead ball while the ball is off the pitch (≥ 12 frames) or untracked for more than 3 s;
  shorter gaps bridged; after a dead ball the first side with 3 evidence frames gets the restart.
  Every setting was chosen before any run, from football and from how the tracking behaves.
- **Variants:** `control_hysteresis_strict` (12-frame confirm, 15-frame tolerance) and
  `control_hysteresis_loose` (3 m reach, 5 s bridge).
- **Command:** `python -m possession run control_hysteresis` (and the two variants).
- **Proxy metrics:** _not run yet: the cloud session that wrote this has no match data._
- **Validation:** _pending the local run._
- **Verdict:** _pending._

## Checked without data

`tests/test_control.py` covers interception, deflection, duel, short and long tracking gaps, the
ball going out, a tracking blip off the pitch and abstaining with no evidence, on synthetic play.
A full-length synthetic match runs in about half a second.

## To do in the local run

1. `python scripts/inspect_tracking.py` first: if the ball's coordinates do not span the pitch
   from 0 to its length and width, the off-the-pitch rule is wrong and must be fixed before
   anything else.
2. Run all three and compare their proxy reports with `nearest_smooth50`'s: spells per minute,
   mean spell length, very short spells, dead-ball share, agreement with Smart Tagging.
3. Watch a few minutes in the viewer with each CSV loaded to see where it goes wrong. Fix general
   rules, never specific frames (RULES.md).

## Known gaps, in order of what to try next

- The ball's height is not tracked, so a high ball passing over a player looks like control. The
  6-frame confirm only partly covers it; the ball's speed relative to the player would help.
- Restarts: Smart Tagging's set-piece phases could mark the dead ball before a throw-in, corner
  or free kick, and who takes it.
- A tracking gap longer than 3 s in open play becomes a false D, which splits a spell in two.
  Bridging when the same side has the ball on both sides of the gap, as #5 suggests, is the fix
  to try.
