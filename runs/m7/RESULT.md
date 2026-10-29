# Milestone 7: about 2 bytes per weight, no float copy — 4-bit arm FAIL, FP8 arm PASS

**What was run.** `m7_small_state.py`, a copy of milestone 4 with two arms, on one Colab T4 (fp16 autocast), CPU
preset, seeds 1–3, Muon learning rate 0.01 (untuned):
- `q4_b4_m8`: milestone 4's `muon_post` (4-bit E2M1 weights, random rounding, a leftover buffer added after
  Newton-Schulz), with the buffer stored as 4-bit integers (one fp32 scale per 16) and the momentum as 8-bit integers
  (one fp32 scale per 128). 2.53 bytes per weight (2.16 with FP8 scales).
- `fp8_m8`: FP8 E4M3 weights (each block of 16 scaled so its largest weight sits on 448), random rounding, no buffer,
  8-bit momentum. 2.28 bytes per weight (2.09 with FP8 scales).
- `fp8_m8→fp4`: `fp8_m8`'s final weights rounded to nearest on the 4-bit grid once, after training.

Baselines are milestone 6's T4 runs (`runs/m6/base_*`), which a later T4 session reproduced to all digits.

| arm | bytes per weight | seed 1 | seed 2 | seed 3 | mean ± standard error |
|---|---|---|---|---|---|
| m1 Muon, float | 8 | 1.6343 | 1.6291 | 1.6311 | 1.6315 ± 0.0015 |
| m2 4-bit + float copy | 8 | 1.6524 | 1.6444 | 1.6406 | 1.6458 ± 0.0035 |
| m3 4-bit, random rounding | 4.75 | 1.7626 | 1.7528 | 1.7494 | 1.7549 ± 0.0039 |
| m4c 4-bit + bf16 buffer | 6.75 | 1.6754 | 1.6754 | 1.6980 | 1.6829 ± 0.0075 |
| **`q4_b4_m8`** | 2.53 | 1.7158 | 1.7099 | 1.7102 | **1.7120 ± 0.0019** |
| **`fp8_m8`** | 2.28 | 1.6464 | 1.6578 | 1.6520 | **1.6521 ± 0.0033** |
| **`fp8_m8→fp4`** | (2.28 to train; a 4-bit model) | 1.6632 | 1.6773 | 1.6745 | **1.6717 ± 0.0043** |

**Pass lines (PLAN.md, written before the runs).**
- `q4_b4_m8` ≤ m2 + 0.03 in each seed: **FAIL**, 0 of 3 (+0.063, +0.066, +0.070).
- `fp8_m8` ≤ m1 + 0.03 in each seed: **PASS**, 3 of 3 (+0.012, +0.029, +0.021).
- `fp8_m8→fp4` against `q4_b4_m8` (measure only): **−0.040 ± 0.006**, better in every seed (−0.053, −0.033, −0.036).

**Also reported.** `fp8_m8→fp4` − `fp8_m8` = +0.020 ± 0.002 (the cost of rounding to 4 bits after training).
`q4_b4_m8` − m4c = +0.029 ± 0.009 (the cost of shrinking the buffer to 4 bits and the momentum to 8 bits).
`q4_b4_m8` − m3 = −0.043 ± 0.002 (the 4-bit buffer still helps a lot). `fp8_m8` − m2 = +0.006 ± 0.006: FP8 weights
with no copy are as good as 4-bit weights with a float copy, at 2.28 bytes against 8. `fp8_m8→fp4` − m2 =
+0.026 ± 0.008 (within 0.03 in 1 of 3 seeds; not a pass line).

**Reading (after the runs).** The 4-bit arm stays about 0.02 behind m2 up to step 1000 (+0.028, +0.023, +0.023 at
steps 250, 500, 1000), then falls behind as the learning rate decays (+0.053 at 1500, +0.066 at 2000). A likely
reason, not measured: the 4-bit buffer is rounded at random every step with a step size of 1/7 of the block's largest
leftover, so it adds noise that does not shrink when the steps do. The FP8 model's 4-bit rounding cost also grows
during training (+0.003 at step 250, +0.020 at step 2000). So at equal training memory, spending the 8 bits as one FP8
weight beats a 4-bit weight plus a 4-bit buffer, even when the end product must be a 4-bit model.

**Not tested.** Which of the two changes in `q4_b4_m8` (4-bit buffer, 8-bit momentum) costs the +0.029 against m4c
(milestone 6 found 8-bit momentum free in its own setting); nearest rather than random rounding for the buffer;
an 8-bit buffer; FP8 with a leftover buffer; FP8 with 32-bit momentum; smarter rounding after training (GPTQ-style)
or a short 4-bit fine-tune after it; FP8 scales in training (counted in the bytes only); learning rates other than
0.01; longer runs, bigger batches and bigger models.
