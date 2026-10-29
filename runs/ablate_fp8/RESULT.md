# Side study F: FP8 block scales (measure only)

**What changed from milestone 3.** Each block of 16 weights shares one scale. In milestone 3 that scale is a full
32-bit number, recomputed every step as (largest weight in the block) / 6, so the largest weight always lands exactly
on ±6, at full precision. Here the scale is stored the way NVFP4 stores it: an 8-bit float (FP8, E4M3), with one
32-bit scale per whole matrix. Values that the cruder scale would push past ±6 are clipped to ±6. Script:
`ablate_fp8_scales.py`. CPU preset, 2000 steps, Muon learning rate 0.01, seeds 1–3.

| seed | milestone 3 (fp32 scales) | FP8 scales | difference |
|---|---|---|---|
| 1 | 1.7658 | 1.7685 | +0.003 |
| 2 | 1.7551 | 1.7763 | +0.021 |
| 3 | 1.7699 | 1.7699 | +0.000 |
| **mean ± standard error** | **1.7636 ± 0.0044** | **1.7716 ± 0.0024** | **+0.008 ± 0.007** |

**Reading.** FP8 scales cost about 0.008 nats per character, which is about the size of the seed-to-seed spread (one
seed +0.021, the other two about 0). So the "hidden full-precision weight" that fp32 scales give each block explains
little of milestone 3's result. Memory: 0.5 + 1/16 bytes per weight for codes and scales, instead of 0.5 + 0.25.

**Not tested.** 2000 steps only; the CPU preset only; no other scale formats (MXFP4's power-of-two scale per 32 weights);
no smarter way of choosing the scale (for example, minimising the rounding error instead of taking the largest weight).
