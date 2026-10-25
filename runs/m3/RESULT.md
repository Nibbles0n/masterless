# Milestone 3 (arm C3a), CPU preset: measured, worse than milestone 2 as expected (2026-10-25)

No float copy. The four Muon matrices are stored only as grid values: 4-bit E2M1 codes times one fp32 scale per block
of 16 weights along the input dimension. Each step computes `w_new = w − lr · muon_update` as a temporary, recomputes
the block scale from `w_new` (absmax / 6), and rounds `w_new` to one of its two grid neighbours at random, with the
probability that makes the rounding unbiased on average. Muon's momentum is the only state. The weights start as
milestone 2's initial weights rounded to nearest. A check at the end of every run confirmed that all four matrices were
still exactly on the grid. Script: `m3_nomaster_sr.py`.

| Seed | C1 (m2) | C3a (m3) | m3 − m2 | SE over 256 windows |
|---|---|---|---|---|
| 1 | 1.6493 | 1.7658 | +0.1165 | 0.0174 |
| 2 | 1.6469 | 1.7551 | +0.1082 | 0.0170 |
| 3 | 1.6428 | 1.7699 | +0.1270 | 0.0172 |
| **Mean ± SE over seeds** | 1.6463 ± 0.0019 | **1.7636 ± 0.0044** | **+0.117** | |

Measure-only line (expected worse than milestone 2, as MuonQ found): **confirmed**, about 0.12 worse in every seed.
That is about six times the cost of the 4-bit format itself (+0.019, milestone 2).

The gap grows during training (seed 1, m3 − m2): +0.04 at step 250, +0.06 at step 1000, +0.10 at step 1500, +0.12 at
step 2000. As the learning rate decays, each step gets smaller next to the grid spacing. Random rounding then mostly
adds noise instead of moving the weight. This is the effect ECO's theory predicts for dropping the float copy.

Training-state bytes per weight (four hidden matrices): **4.75** = 0.5 (4-bit code) + 0.25 (fp32 scale per 16) + 4
(fp32 momentum), against 8 for milestone 2. With a bf16 momentum it would be 2.75.

Not tested: round to nearest without a float copy (it would freeze most weights once steps fall below half a grid
step); a bf16 or 8-bit momentum; FP8 scales; the GPU preset. The simulation keeps the grid values in an fp32 tensor and
does not pack real 4-bit codes, so no memory was actually measured, only computed.
