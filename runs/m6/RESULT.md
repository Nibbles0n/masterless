# Milestone 6: Muon reordered so the momentum holds the rounding leftover — FAIL

**What was run.** `m6_accum.py`: orthogonalise each gradient first, then average (`A ← 0.95·A + 0.05·NS(G)`, step
`lr·A`), with the leftover `e = target − stored` added to `A` as `(1 − 1/0.95)·e/lr` and nearest rounding. Arms `float`,
`q4` and `q4_m8` (the momentum in 8 bits). Baselines rerun in the same Colab T4 session: milestones 1, 2, 3 and
milestone 4's `muon_post`. fp16 autocast, CPU preset, seeds 1–3.

**Learning rate (float arm, seed 1, tuning).** 0.01 → 1.925, 0.02 → 1.800, 0.04 → 1.728, **0.08 → 1.696**,
0.16 → 1.777. As fixed in advance, `q4` and `q4_m8` used 0.08, untuned. Seed 1 of `float` is the tuning run.

| arm | bytes per weight | seed 1 | seed 2 | seed 3 | mean ± standard error |
|---|---|---|---|---|---|
| m1 Muon, float | 8 | 1.6343 | 1.6291 | 1.6311 | 1.6315 ± 0.0015 |
| m2 4-bit + float copy | 8 | 1.6524 | 1.6444 | 1.6406 | 1.6458 ± 0.0035 |
| m3 4-bit, random rounding | 4.75 | 1.7626 | 1.7528 | 1.7494 | 1.7549 ± 0.0039 |
| m4c leftover buffer after NS | 6.75 | 1.6754 | 1.6754 | 1.6980 | 1.6829 ± 0.0075 |
| m6 `float` | 8 | 1.6962 | 1.6998 | 1.7047 | 1.7002 ± 0.0025 |
| m6 `q4` | 4.75 | 2.0435 | 2.0334 | 2.0642 | 2.0470 ± 0.0091 |
| m6 `q4_m8` | 1.78 | 2.0509 | 2.0386 | 2.0534 | 2.0476 ± 0.0046 |

The T4 baselines repeat study B's T4 numbers for m1 and m3 to four digits.

**Pass lines (PLAN.md, written before the runs).**
- `q4` ≤ m2 + 0.03 in each seed: **FAIL**, 0 of 3 (+0.391, +0.389, +0.424).
- `q4_m8`, the same line: **FAIL**, 0 of 3 (+0.399, +0.394, +0.413).
- `float` ≤ m1 + 0.03 in each seed: **FAIL**, 0 of 3 (+0.062, +0.071, +0.074).

**Also reported.** `q4` − m3 = +0.292 ± 0.011. `q4` − m4c = +0.364 ± 0.003. `q4_m8` − `q4` = +0.001 ± 0.006: storing
the momentum in 8 bits cost nothing measurable, so the failure is the rule, not the storage.

**Why it failed (found after the runs, `report/accum_sim.py`).** The rule hands a leftover back slowly: the momentum
releases about 1/20 of what it holds each step. The rounding sees only one step's share. So a weight moves one level
only when a single step's push passes half a gap, and by then about 20 times as much, roughly 9–10 gaps, is waiting
in the momentum. In the one-weight simulation (steady push of 0.05 gaps per step), the m6 weight first moves at step
201 and then trails its float twin by about 9 gaps for good. A separate leftover buffer (milestone 4c) first moves at
step 24 and never trails by more than half a gap. The momentum is an accumulator, as hoped, but its threshold sits 20
times too high. Separately, reordering alone (float weights) costs +0.069: orthogonalising each noisy gradient before
averaging is worse than Muon's order.

**Not tested.** A rule that compares the whole stored amount (`lr·A·0.95/0.05`) with the gap instead of one step's
share; random instead of nearest rounding; a learning rate tuned for the 4-bit arms; the warm-up (the injection divides
by the learning rate, which is 1/100 of its peak at step 0); longer runs; bigger models.
