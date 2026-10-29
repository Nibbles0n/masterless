# Side study B2: bigger batches with bigger learning rates, measure only

**What was run.** As study B (milestone 1 = Muon with float weights, milestone 3 = 4-bit with no copy, the same 1.5
million characters per run, CPU-preset model, fp16 on a Colab T4, seeds 1–3), with every learning rate (Muon 0.01,
AdamW 1e-3 and its floor 1e-4) multiplied by m: m = 2, 4 at 1x the batch, m = 2, 4, 8 at 2x, 4x and 8x. The m = 1
cells are study B's runs. 68 runs over two T4 sessions (the second ran with 1 CPU thread per job instead of 2; the
first session reproduced study B's seed-1 runs to all five digits: m1 1.6343, m3 1.7626).

Final validation loss, mean ± standard error over 3 seeds:

| batch | arm | rate x1 | x2 | x4 | x8 |
|---|---|---|---|---|---|
| 12 (1x) | m1 | 1.6315 ± 0.0015 | **1.6306 ± 0.0061** | 1.7759 ± 0.0039 | |
| | m3 | **1.7549 ± 0.0039** | 1.7849 ± 0.0065 | 1.9911 ± 0.0390 | |
| 24 (2x) | m1 | 1.6492 ± 0.0021 | **1.6350 ± 0.0046** | 1.6764 ± 0.0029 | 1.9849 ± 0.0101 |
| | m3 | 1.7606 ± 0.0051 | **1.7451 ± 0.0052** | 1.8027 ± 0.0038 | 2.1352 ± 0.0131 |
| 48 (4x) | m1 | 1.7324 ± 0.0039 | 1.6693 ± 0.0008 | **1.6687 ± 0.0083** | 1.7912 ± 0.0029 |
| | m3 | 1.8126 ± 0.0059 | **1.7588 ± 0.0046** | 1.7709 ± 0.0046 | 1.9150 ± 0.0035 |
| 96 (8x) | m1 | 1.9380 ± 0.0036 | 1.7607 ± 0.0042 | **1.7185 ± 0.0019** | 1.7703 ± 0.0077 |
| | m3 | 1.9915 ± 0.0026 | 1.8429 ± 0.0021 | **1.7917 ± 0.0026** | 1.8755 ± 0.0092 |

**The four questions fixed before the runs.**
1. m3's best rate per batch size: 1x → x1 (1.755), 2x → x2 (1.745), 4x → x2 (1.759), 8x → x4 (1.792). Only 2x at
   rate x2 beats 1x at the old rate, by 0.010 ± 0.009 seed-matched (−0.025, −0.010, +0.006), so within the noise. Bigger batches with the
   right rate get about as far as 1x in 2–4 times fewer steps, but no further.
2. m1's best rates: 1x → x2, 2x → x2, 4x → x4, 8x → x4. The m3 − m1 gap with each arm at its own best rate:
   +0.124 ± 0.010, +0.110 ± 0.002, +0.090 ± 0.005, +0.073 ± 0.005.
3. At a fixed batch size, a bigger rate makes the gap **bigger**, not smaller: at 1x +0.123, +0.154, +0.215 for
   x1, x2, x4; at 8x +0.053, +0.082, +0.073, +0.105 for x1–x8. The owner's idea that bigger steps suffer less from
   rounding does not hold here.
4. Runs that blew up: none (no loss above 3, no NaN). The worst cell, m3 at 1x and x4, was erratic (1.94–2.07).

**Reading (after the runs).** Study B found that the gap mostly tracks how far training has got. Compared with that
reference (m1 and m3 at 1x and the old rate, read along their training curves at the same m1 loss), well-matched
batch-and-rate pairs sit a little below it: 8x at x4 has a gap of 0.073 where the reference has about 0.10 at the same
m1 loss (1.72), and 4x at x2 has 0.090 against about 0.11. Rates too high for the batch sit well above it (1x at x4:
0.215 against about 0.09). So bigger batches buy a slightly smaller gap, about a fifth, only at a rate matched to
the batch. Most of the effect is still how far training got. The comparison with the reference is rough: a
mid-training point of a 2000-step run has a higher learning rate than the last point of a short run.

**Not tested.** Rates between the doubling steps; schedules other than the preset's (100-step warm-up scaled by 1/k,
cosine to 10%); more characters (the same steps at a bigger batch); bigger models; milestone 2 or other 4-bit
methods at bigger batches.
