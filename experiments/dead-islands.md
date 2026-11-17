# hmm_combo_islands: no short possession between two stoppages

**Question.** In long breaks (cooling breaks, injuries, VAR) the decoder sometimes gives one side
the ball for a second or two between two dead-ball stretches. On spellF1 each of those is a whole
false possession: unmatchable, so a false positive at every IoU threshold. Does removing them help?

**Method** (`possession/experiments/hmm_combo_islands.py`, `hmm.drop_dead_islands`). `hmm_combo`
as it is (same models, no refit), then every play stretch shorter than `island_min` frames with
dead ball directly before and after it, inside one half, becomes `D`. A play stretch is a maximal
run of anything but `D`, so H and A flickering together count as one. Stretches touching a half's
edge are kept.

What it can cost: on the training matches' ground truth, 36 of 405 play stretches between two
stoppages are under 4 s (59 possessions).

## Bench (degradation v2, seeds 0-2)

| `island_min` | train spellF1 | train macroF1 | val spellF1 | val macroF1 |
|---|---|---|---|---|
| off (= `hmm_combo`) | **0.511** | 0.866 | 0.495 | 0.848 |
| 50 (2 s) | 0.511 | 0.865 | | |
| **75 (3 s)** | 0.508 | 0.865 | **0.496** | 0.848 |
| 100 (4 s) | 0.506 | 0.864 | 0.491 | 0.847 |
| 150 (6 s) | 0.501 | 0.862 | | |

Train picks off; every setting is within 0.01 of it, well inside the noise. Val at 3 s, per seed:
0.480 / 0.517 / 0.493 (`hmm_combo`: 0.479 / 0.516 / 0.491). The bench has few long breaks: our
validation output has about 7 short stretches between stoppages.

## BAR-ATH (label-free)

`hmm_combo`'s output has 28 short (< 4 s) play stretches between stoppages, about four times the
validation rate, many inside long breaks: three around 24:10-24:34 of the first half between 19-64 s
stoppages; 1.2 s of A at 31:32 before 124 s of dead ball; 0.8 s of A at 33:17 of the second half
with about 95 s dead either side. The submitted `hmm_final_ms` has 29.

| | possessions | median | dead share | ST agreement |
|---|---|---|---|---|
| `hmm_combo` | 362 | 4.64 s | 39.7% | 0.832 |
| `hmm_combo_islands` (3 s) | 335 | 5.32 s | 40.3% | 0.832 |

    python -m possession.bench train hmm_combo_islands
    python -m possession.bench val hmm_combo_islands
    python -m possession run hmm_combo_islands

Official format check passed.

## Verdict

**Kept, at 3 s, set by hand.** The bench cannot see the case it is aimed at and calls it a draw
(train −0.003, val +0.001). On BAR-ATH it removes 27 possessions, mostly in long breaks, where a
possession is very unlikely to be real, without changing the agreement with Smart Tagging. To go
into the submission it needs the `hmm_final` treatment (refit on train + validation) and a test
run, together with `hmm_combo`.
