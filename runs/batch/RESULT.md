# Side study B: batch size, measure only

**What was run.** Milestone 1 (Muon, float weights) and milestone 3 (4-bit, no copy, random rounding) at 1x, 2x, 4x
and 8x the CPU preset's batch of 12 sequences. Every run reads the same 1.5 million characters (steps, warm-up and
eval interval divided by the batch factor). Learning rates unchanged (Muon 0.01, AdamW parts 1e-3). Seeds 1-3. All 24
runs on one Colab T4 with fp16 autocast, 3 at a time (runner: `batch_runs.py`, summary: `report/batch_summary.py`).

| batch | steps | m1 (float) | m3 (4-bit, no copy) | m3 - m1, seed-matched |
|---|---|---|---|---|
| 12 (1x) | 2000 | 1.6315 ± 0.0015 | 1.7549 ± 0.0039 | +0.123 ± 0.003 |
| 24 (2x) | 1000 | 1.6492 ± 0.0021 | 1.7606 ± 0.0051 | +0.112 ± 0.004 |
| 48 (4x) | 500 | 1.7324 ± 0.0039 | 1.8126 ± 0.0059 | +0.080 ± 0.007 |
| 96 (8x) | 250 | 1.9380 ± 0.0036 | 1.9915 ± 0.0026 | +0.053 ± 0.005 |

(mean ± standard error over seeds 1-3)

**Questions fixed before the runs.**
1. Does the m3 - m1 gap shrink as the batch grows? **Yes, from +0.123 to +0.053.**
2. Does a bigger batch help m3 itself, with the same characters? **No.** m3 gets worse (1.755 → 1.992), and so does
   full-precision Muon (1.632 → 1.938).

**Reading (after the runs).** The gap shrinks only because, at a fixed learning rate, bigger batches get less far on
the same data. Compared at the same full-precision loss, the gap is about the same for every batch size. For example,
with m1 near 1.76, it is +0.091 at 1x, +0.100 at 2x and +0.082 at 4x. Near 1.93, it is +0.059 at 1x, +0.079 at 2x,
+0.061 at 4x and +0.053 at 8x. So the rounding damage tracks how far training has gone, not the batch size. Bigger
batches are not a free fix here.

**Not tested.** Learning rates retuned per batch size (bigger batches usually want bigger steps, which would also make
each step a bigger fraction of a grid gap); the same number of steps (more data); longer training; bigger models.
