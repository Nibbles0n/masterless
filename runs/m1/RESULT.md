# Milestone 1, CPU preset: PASS (2026-10-25)

Milestone 0's baby GPT and schedule, with Muon on the four hidden matrices of each block (`qkv`, `proj`, `fc`,
`fc_out`) and AdamW, unchanged from milestone 0, on the embeddings, the tied head and the norms. Muon: Nesterov momentum
0.95, 5 Newton-Schulz steps (3.4445, −4.7750, 2.0315) in fp32, scaled by `max(1, rows/cols) ** 0.5`, same warmup and
cosine shape as AdamW, no weight decay. Script: `m1_muon.py`.

**Tuning, not evidence** (seed 1, final val CE):

| Muon lr | 0.01 | 0.02 | 0.04 |
|---|---|---|---|
| CE | **1.6216** | 1.6386 | 1.9137 (got stuck near 2.2 early, then slowly recovered) |

0.01 was the best value, and it is the smallest one tried, so a smaller value might do even better (not tested). Seed 1
at 0.01 is reused from the sweep.

| Seed | AdamW (m0) | Muon lr 0.01 | Muon − AdamW | SE over 256 windows |
|---|---|---|---|---|
| 1 | 1.9195 | 1.6216 | −0.2980 | 0.0186 |
| 2 | 1.9171 | 1.6294 | −0.2877 | 0.0178 |
| 3 | 1.9084 | 1.6323 | −0.2761 | 0.0186 |
| **Mean ± SE over seeds** | 1.9150 ± 0.0034 | **1.6278 ± 0.0032** | −0.287 | |

Pass line: Muon at least 0.02 below AdamW in 3 of 3 seeds. **PASS** (3 of 3, by about 0.29).

Training-state bytes per weight (the four hidden matrices, about 99% of the weights): AdamW 12 (fp32 weight + two
fp32 moments), Muon 8 (fp32 weight + one fp32 momentum).

The gap is roughly 14 times the bar. That is large, but in line with Muon's known speed-up on short runs: 2000 steps is
too short for AdamW at learning rate 1e-3 to finish. Some of the gap is probably AdamW being under-trained, not Muon
being better in the long run.

Not tested: the GPU preset; a retuned AdamW (higher learning rate or more steps); Muon learning rates below 0.01;
bf16 Newton-Schulz; weight decay on the Muon matrices.
