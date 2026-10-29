# Side study T: the ticker, measure only — level with a float copy at 2.28 bytes per weight

**What was run.** `side_ticker.py`, milestone 7's `q4_b4_m8` with the owner's tick rule: target = stored 4-bit weight
+ carry − lr·update; the new weight is the nearest 4-bit value to the target (a weight ticks once its carry passes half
the gap on that side, so each weight has its own thresholds); the new carry is target − new weight, stored in units of
the gap on its own side as 4-bit integers −7..7 (k/14 of the gap), rounded at random, with no scale of its own. 8-bit
momentum as in milestone 7. Colab T4, fp16 autocast, CPU preset, seeds 1–3, Muon learning rate 0.01, untuned.
Memory 2.28 bytes per weight (2.09 with FP8 block scales), the same as milestone 7's `fp8_m8`.

| arm | bytes per weight | seed 1 | seed 2 | seed 3 | mean ± standard error |
|---|---|---|---|---|---|
| m2 4-bit + float copy | 8 | 1.6524 | 1.6444 | 1.6406 | 1.6458 ± 0.0035 |
| m4c 4-bit + bf16 buffer | 6.75 | 1.6754 | 1.6754 | 1.6980 | 1.6829 ± 0.0075 |
| m7 `q4_b4_m8` (random rounding, 4-bit buffer) | 2.53 | 1.7158 | 1.7099 | 1.7102 | 1.7120 ± 0.0019 |
| m7 `fp8_m8→fp4` | (2.28 to train) | 1.6632 | 1.6773 | 1.6745 | 1.6717 ± 0.0043 |
| **ticker `q4_t4_m8`** | 2.28 | 1.6423 | 1.6579 | 1.6501 | **1.6501 ± 0.0045** |

**The three questions fixed before the run (seed-matched).**
1. Against milestone 7's `q4_b4_m8`: **−0.062 ± 0.006**, better in every seed (−0.074, −0.052, −0.060).
2. Against `fp8_m8→fp4`: **−0.022 ± 0.002**, better in every seed (−0.021, −0.019, −0.024). Training in 4 bits with
   the ticker beats training in FP8 and rounding afterwards.
3. Within 0.03 of milestone 2 in each seed: **yes, 3 of 3** (−0.010, +0.014, +0.010; mean +0.004 ± 0.007). This was
   written down as reference only, so it is not yet the paper test's pass; see below.

**Also.** Against milestone 4c (16-bit buffer, 32-bit momentum, 6.75 bytes): −0.033 ± 0.009, better in every seed.
Along training the ticker stays level with m2 (+0.012, −0.003, −0.002, +0.005, +0.004 at steps 250, 500, 1000, 1500,
2000), with none of the late fall-off of `q4_b4_m8`. It took about 280 s per run on the T4 against about 190 s for
`q4_b4_m8` (the gap lookup is not optimised).

**Reading (after the run).** The ticker differs from `q4_b4_m8` in two ways at once, so which one matters is not
known: nearest rounding with the carry (no random rounding of the weight), and a carry measured in units of its own
gap (fine for small weights, where `q4_b4_m8`'s step of 1/7 of the block's largest leftover was coarse). The
simulation in `report/noise_sim.py` (sized to a real Muon run) puts the ticker with a 4-bit carry at 1.3 gaps of drift,
below an FP8-like grid's 2.2 and far below plain random rounding's 12.

**Not tested.** Which of the two differences does the work; an 8-bit carry; FP8 block scales in training (counted
only); longer runs (study L found plain 4-bit falling further behind with more steps), bigger batches and bigger
models; other learning rates; the paper test itself as a pass line on new seeds.
