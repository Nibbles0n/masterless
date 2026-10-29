# Milestone 5: spread the rounding error inside each block (C3s), CPU preset: **FAIL** (as predicted)

**What was run.** Milestone 3 (no float copy, random rounding), with one change. Inside each block of 16 weights
that share a scale, the weights are rounded one at a time. Each weight's rounding error is pushed onto the later
weights of its block, weighted by how the layer's inputs move together. This is GPTQ's rule, with a running average of
the inputs' 16 x 16 covariance blocks. Script: `m5_spread.py`. CPU preset, 2000 steps, Muon learning rate 0.01,
seeds 1–3. Extra memory: 0.29 bytes per weight for the covariance blocks. Time: about 455 s per run, against about
250 s for milestone 3 (three runs at once on this CPU).

**Pass line (written before any run):** each seed at least 0.02 *below* milestone 3's CE for the same seed.

| seed | milestone 3 | milestone 5 | difference |
|---|---|---|---|
| 1 | 1.7658 | 1.8582 | +0.092 |
| 2 | 1.7551 | 1.8432 | +0.088 |
| 3 | 1.7699 | 1.8540 | +0.084 |
| **mean ± standard error** | **1.7636 ± 0.0044** | **1.8518 ± 0.0045** | **+0.088 ± 0.002** |

Worse in every seed, so it fails the line. The gap grows during training: +0.02 at step 250, +0.06 at step 1000,
+0.09 at step 2000.

**Why (from the offline check, `report/spread_probe.py`, before the run).** At steps 250, 1000 and 1900 of a
milestone 3 run, we measured how much rounding noise reaches each layer's output, relative to the output change the
step intended. Results relative to plain random rounding:

| method | step 250 | step 1000 | step 1900 |
|---|---|---|---|
| block of 16 (this milestone) | 1.28x | 1.54x | 1.46x |
| full GPTQ (whole matrix) | 1.22x | 3.25x | 5.59x |
| top-8 directions + diagonal | 1.00x | 1.92x | 3.06x |
| carry error to the next weight | 7.8x | 16.9x | 24.5x |

Every method was worse. Based on the inputs' statistics alone, theory said spreading could cut the noise to 0.16–0.39x.
The reason is that without a float copy, the weight we want is always within a tiny step of a grid point (0.1–7% of
a gap). Random rounding of such a value is nearly exact: it rarely flips, and its noise per weight is about
step x gap, not gap squared. Spreading hands each later weight a correction about one rounding error in size, which
moves it to the middle of a gap, where random rounding is noisiest. On an evenly spaced test grid, spreading did cut
noise to 0.3–0.5x, but that grid's plain random rounding is already about 4.5x noisier than on the 4-bit float grid.
Spreading the error of round-to-nearest instead (no randomness) changed nothing at step 1000: the step-sized errors
never add up to a flip, which is the same stall as milestone 2's round-to-nearest.

**Not tested.** Spreading across the whole matrix during training (the offline check says worse still); spreading
only part of the error (offline at step 1000: 0.5 of the error gave 2–3x the noise of plain random rounding, still
worse); other block sizes; the GPU preset; ternary. Error spreading in *time* (milestone 4's error feedback) is the
other half of the idea and also failed.
