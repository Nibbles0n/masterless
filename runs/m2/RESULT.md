# Milestone 2 (arm C1), CPU preset: format usable (2026-10-25)

Milestone 1 (Muon lr 0.01) with fake 4-bit weights on the four Muon matrices. In the forward pass each matrix is
rounded to nearest on the NVFP4 value set (E2M1: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6), with one scale per block of 16
weights along the input dimension, `scale = absmax / 6`. The gradient passes straight through the rounding. Muon
updates the fp32 float copy as in milestone 1. Embeddings, head and norms stay fp32. Script: `m2_fake4.py`.

| Seed | Muon (m1) | C1 fake 4-bit (m2) | m2 − m1 | SE over 256 windows |
|---|---|---|---|---|
| 1 | 1.6216 | 1.6493 | +0.0278 | 0.0186 |
| 2 | 1.6294 | 1.6469 | +0.0174 | 0.0182 |
| 3 | 1.6323 | 1.6428 | +0.0105 | 0.0180 |
| **Mean ± SE over seeds** | 1.6278 ± 0.0032 | **1.6463 ± 0.0019** | **+0.019** | |

Measure-only line: a gap below +0.10 means the format is usable. Gap **+0.019**, so it is **usable**.

Training-state bytes per weight (four hidden matrices): 8, the fp32 float copy plus the fp32 momentum. The 4-bit forward
weight is recomputed every step. A real kernel that caches it would add 0.75 (0.5 for the code, 0.25 for the scale),
for 8.75.

Simplification: the block scale is kept in fp32. Real NVFP4 stores it in FP8 (E4M3), with a second fp32 scale per
tensor, so the real format rounds the scale too and may lose a little more.

Not tested: FP8 scales; activations or gradients in 4 bits (only the weights are rounded); the embeddings and head in
4 bits; the GPU preset; retuning the learning rate for the 4-bit model.
