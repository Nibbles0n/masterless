# Side study 2B: the ticker at 2 bits — PASS with an FP8 carry, FAIL with a 4-bit carry

**What was run.** `side_2bit.py`, a copy of `side_ticker.py` with 2-bit weights. The grid has four values, −1.5, −0.5,
+0.5 and +1.5, times one fp32 scale per block of 16. The scale is 1.25 × the block's mean absolute weight, which gives
about 30% less rounding error than absmax/1.5 on bell-shaped blocks.

The runs used a Colab T4 with fp16 autocast, the CPU preset and seeds 1–3. The Muon learning rate was 0.01, untuned.
Five arms (bytes per weight of the four hidden matrices, with fp32 scales; FP8 scales in brackets):

| arm | bytes per weight | seed 1 | seed 2 | seed 3 | mean ± standard error |
|---|---|---|---|---|---|
| `q2_ste_muon`: 2-bit + float copy, Muon (the target) | 8 | 1.7253 | 1.7278 | 1.7228 | **1.7253 ± 0.0015** |
| `q2_t8_m8`: 2-bit ticker, FP8 carry, 8-bit momentum | 2.53 (2.34) | 1.7396 | 1.7441 | 1.7590 | **1.7476 ± 0.0059** |
| `q2_t4_m8`: 2-bit ticker, 4-bit carry, 8-bit momentum | 2.03 (1.84) | 1.7958 | 1.8026 | 1.8096 | **1.8027 ± 0.0040** |
| `q2_ste_adam`: 2-bit + float copy, AdamW (lr 1e-3) | 12 | 1.9599 | 1.9447 | 1.9362 | 1.9469 ± 0.0069 |
| `fp8_m8→2-bit`: FP8-trained, rounded to 2 bits after | (2.28 to train) | 2.0566 | 2.1936 | 2.1964 | 2.1488 ± 0.0461 |
| `fp8_m8` (FP8 weights, re-run of milestone 7) | 2.28 (2.09) | 1.6464 | 1.6578 | 1.6520 | 1.6521 ± 0.0033 |
| reference: m1 Muon, float | 8 | 1.6343 | 1.6291 | 1.6311 | 1.6315 ± 0.0015 |
| reference: m2 4-bit + float copy | 8 | 1.6524 | 1.6444 | 1.6406 | 1.6458 ± 0.0035 |
| reference: 4-bit ticker (side study T) | 2.28 | 1.6423 | 1.6579 | 1.6501 | 1.6501 ± 0.0045 |

**Pass line, fixed before the runs: each ticker arm within 0.05 of `q2_ste_muon` in each seed.**
- `q2_t8_m8` (FP8 carry): **PASS**, 3 of 3 (+0.014, +0.016, +0.036; mean +0.022 ± 0.007).
- `q2_t4_m8` (4-bit carry): **FAIL**, 0 of 3 (+0.071, +0.075, +0.087; mean +0.077 ± 0.005).

**Questions (measure only, seed-matched).**
1. FP8 carry against 4-bit carry: **−0.055 ± 0.002**, better in every seed. The 4-bit carry stays within 0.02 of the
   float copy until step 1000, then falls behind as the learning rate shrinks: +0.035 at step 1250, +0.055 at 1500 and
   +0.077 at 2000. The FP8 carry is level until step 1250 and ends +0.022 behind.
2. Cost of 2 bits with a float copy:
   - Muon: +0.080 ± 0.003 against milestone 2 (4-bit), and +0.094 against float Muon.
   - AdamW is 0.222 ± 0.007 worse than Muon at 2 bits.
   - Against milestone 0 (float AdamW, run on CPU in fp32), AdamW loses only +0.032 ± 0.004 from 2 bits. Muon's lead
     narrows from 0.28 at full precision to 0.22 at 2 bits, but it stays large.
3. Training in FP8 and rounding to 2 bits afterwards does badly: 2.149 ± 0.046, which is 0.35 behind the 4-bit-carry
   ticker and 0.40 behind the FP8-carry ticker, in every seed. Unlike 4 bits, where this gap was 0.02, at 2 bits the
   model has to be trained in 2 bits.

**Code check.** `fp8_m8` repeated milestone 7's three numbers to every digit.

**Reading (after the run).** For most weights the 2-bit gap is about 3 to 6 times wider than the 4-bit gap at the same weight, so late
steps are an even smaller fraction of it. A carry in 1/14ths of a gap can't hold them, and its own random rounding
becomes the noise. That is the same failure the zig-zag theory predicts for too coarse a carry. An FP8 carry has fine
steps near zero, so it holds them, at 0.5 bytes more per weight. The ticker with an FP8 carry is the first no-copy
method in this project within 0.03 of its float-copy twin at 2 bits. It is still 0.10 behind milestone 2 and the 4-bit
ticker, so at this model size 4 bits remains the better trade for quality.

**Not tested.**
- Other 2-bit grids: absmax scale, ternary (that is milestone 8), or a learned scale.
- An 8-bit carry stored as int8 (k/254 of a gap) instead of FP8.
- 8-bit block scales in training (counted only).
- Other learning rates. Everything ran at 0.01, and AdamW at 1e-3 untuned. Bigger steps might suit the wider gap.
- Longer runs, bigger models, a GPU preset.
- AdamW without a float copy at 2 bits.
- The m0 reference was run on CPU in fp32, not on a T4.
- The memory counts cover weights and optimizer state only; gradients and activations are extra.
- The 2-bit and FP8 formats are simulated, with fp16 matrix multiplies, so there is no speed measurement.
