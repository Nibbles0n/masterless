# Milestone 4 (arm C3b), CPU preset: FAIL (2026-10-25)

Milestone 3's storage (4-bit grid values only, random rounding, no float copy), plus ECO's error feedback
([arXiv 2601.22101](https://arxiv.org/abs/2601.22101)). After each step the rounding error `e = w̃ − ŵ` (the weight the
optimizer wanted, minus the grid weight actually stored) is fed back. Script: `m4_eco.py`.

- **(a) ECO with Adam** (prior-art control): AdamW with milestone 0's settings on the four matrices, injection exactly
  as Algorithm 3, line 7 of the paper: `m ← m + ((1 − β1^t)/η)(1 − 1/β1)(√(v/(1 − β2^t)) + ε) ⊙ e`. `v` is unchanged.
- **(b) Muon, error before Newton-Schulz:** ECO's equations (6)–(7), `m ← m + (1/η)(1 − 1/β) e`, with η replaced by
  Muon's step size per unit of momentum, `η_eff = η · ‖update‖ / ‖momentum input‖`. This is the Muon version of the
  paper's Adam rule, which swaps 1/η for each weight's own step size. **This adaptation is ours, not the paper's.** The
  injected error then passes through the orthogonalisation.
- **(c) Muon, error after Newton-Schulz:** the error is kept in a bf16 buffer and added straight to the next step's
  weight change, after Newton-Schulz, so it bypasses the orthogonalisation. The buffer costs 2 bytes per weight.

## Results (final val CE, nats per character)

| Seed | C1 (m2) | C3a (m3) | (a) ECO Adam | (b) Muon, before NS | (c) Muon, after NS | (c) − m2 |
|---|---|---|---|---|---|---|
| 1 | 1.6493 | 1.7658 | 1.9249 | 1.9094 | 1.6656 | +0.0163 |
| 2 | 1.6469 | 1.7551 | 1.9360 | 1.8941 | 1.6803 | +0.0335 |
| 3 | 1.6428 | 1.7699 | 1.9239 | 1.8909 | 1.6739 | +0.0310 |
| **Mean ± SE over seeds** | 1.6463 ± 0.0019 | 1.7636 ± 0.0044 | 1.9283 ± 0.0039 | 1.8981 ± 0.0057 | **1.6733 ± 0.0043** | +0.027 |
| Bytes per weight | 8 | 4.75 | 8.75 | 4.75 | 6.75 | |

SE over the 256 windows of one run: about 0.015 for (a) and (b), 0.018 for (c).

Pass line: the best Muon arm is within 0.03 of milestone 2 in 3 of 3 seeds. The best Muon arm is (c), which made it in
1 of 3 seeds (seeds 2 and 3 missed by 0.0035 and 0.0010). **FAIL.** The misses are tiny next to the run-to-run
noise, but the line was fixed before the run, so the result is a fail.

## How much of the injected error survives the orthogonalisation (arm b)

Every 100 steps, for each of the 16 matrices, d is the injected error term and M the momentum. We measure the cosine
between d and `NS(M + d) − NS(M)`, and the gain `‖NS(M + d) − NS(M)‖ / ‖d‖` divided by `‖NS(M)‖ / ‖M‖`, so that 1
means "scaled like the momentum itself". Averages over the 16 matrices:

| | cosine | gain | size of injection / size of momentum |
|---|---|---|---|
| step 100 (seed 1) | 0.71 | 1.99 | 0.15 |
| step 1000 (seed 1) | 0.69 | 1.33 | 0.31 |
| step 1900 (seed 1) | 0.68 | 1.07 | 0.50 |
| all steps, seeds 1 / 2 / 3 | 0.691 / 0.692 / 0.692 | 1.35 / 1.36 / 1.35 | 0.44 / 0.43 / 0.43 |

In plain words: the change Newton-Schulz makes points about 46° away from the error we fed in (cosine 0.69). About
half of the error's energy (0.69² ≈ 0.48) keeps its direction; the other half is turned into a sideways push of about
the same size. Meanwhile the injected error keeps growing compared with the momentum, because the steps shrink with the
learning rate while the rounding error stays about one grid step. By the end, the error is half the size of the
momentum. So in late training Muon's direction is set largely by rounding noise. In seed 1 the loss rises again after
step 1500 (1.888 → 1.909).

## What this means

- ECO with Adam (a) works as the paper says: 4-bit weights with no float copy, only 0.013 worse than milestone 0's
  full-precision AdamW (1.915). But Adam itself is 0.29 behind Muon here, so the control lands far from the Muon line.
- Feeding the error into Muon's momentum (b) is **worse than no feedback** (1.898 against milestone 3's 1.764).
  Newton-Schulz keeps only a direction, and it scrambles about half of the fed-back error.
- Feeding the error in after Newton-Schulz (c) recovers 77% of milestone 3's loss gap (1.673 against 1.764 and
  1.646). But it needs a separate 2-byte error buffer, and the weight plus that buffer is a bf16 float copy in all but
  name. It saves 1.25 bytes against milestone 2 (6.75 against 8). It is not the masterless method.

Not tested: storing the error in fewer bits (8-bit or 4-bit, measured in grid steps), which would make (c) much cheaper;
injecting into momentum only the part of the error that survives Newton-Schulz; ECO's exact form with the previous
error (Appendix A); round to nearest instead of random rounding; scaling the Muon learning rate for the 4-bit arms;
the GPU preset. The survival numbers measure one perturbation on the plain momentum, not on the Nesterov input that
Muon actually orthogonalises.
