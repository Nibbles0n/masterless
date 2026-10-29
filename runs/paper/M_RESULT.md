# Step M: how much of an error fed into the momentum reaches the update — measured (no pass line)

**What was run.** `report/ns_survival.py`, no training. Pretrained GPT-2 small, float32, its 48 block matrices
(attention `c_attn` and `c_proj`, MLP `c_fc` and `c_proj`). Real gradients on 80 fixed FineWeb windows of 1,024 tokens
(the validation shard), 2 windows per step, 40 steps. Muon's Nesterov momentum (β = 0.95, as in `side_gpt2.py`) and
Adam's moments (β1 = 0.95, sharing Muon's buffer; β2 = 0.95; bias-corrected) were built over those steps. At steps
20 and 40 an error was added into the momentum. Its direction is each matrix's actual 4-bit rounding leftover,
`w − quantize_rtn(w)` (E2M1, one absmax/6 scale per 16 weights); its size was set to 0.01, 0.03, 0.1, 0.3, 1, 3 and 10
times the momentum's size. A random direction (sizes 0.1 and 1) is the control. One Colab T4, fp32 (no TF32),
56 seconds. The CPU attempts were killed twice when the shared machine ran out of memory, so the run moved to a T4.
An earlier 10-step run on the same T4 gave the same numbers to about ±0.02 (not kept).

**What is measured, per matrix.** The "intended correction" is the error passed straight through, at the optimizer's
own average gain (the size of the update divided by the size of the momentum). This is what ECO's rule relies on.
- **cosine**: between (update with error − update without) and the error. 1 means the direction is kept.
- **share arriving**: the part of that change that points along the error, as a share of the intended correction.
  1 means the correction arrives whole.
- **kept**: the cosine between the update with the error and the update without it. It says how much of the
  gradient's own direction is left.

Numbers are means over the 48 matrices ± standard error over matrices, at step 40 (step 20 agrees within 0.02 on
every cosine).

| optimizer, error size (× momentum) | cosine | share arriving | kept |
|---|---|---|---|
| SGD-momentum, any size | 1.000 | 1.000 | – |
| Adam with ECO's per-weight rule, any size | 1.000 | 1.000 | – |
| Adam, plain injection (no per-weight factor), any size | 0.80 ± 0.02 (0.21–0.95) | 4.9 ± 0.7 | – |
| Muon, 0.01 | 0.77 ± 0.01 | 7.6 ± 0.5 | 0.995 |
| Muon, 0.1 | 0.71 ± 0.01 | 5.1 ± 0.2 | 0.76 |
| Muon, 0.3 | 0.67 ± 0.01 | 2.24 ± 0.04 | 0.47 |
| Muon, 1 | 0.71 ± 0.01 | 0.90 ± 0.01 | 0.20 |
| Muon, 3 | 0.67 ± 0.01 | 0.307 ± 0.003 | 0.08 |
| Muon, 10 | 0.65 ± 0.01 | 0.093 ± 0.001 | 0.03 |
| Muon, size ECO's rule gives at learning rate 1e-3 (22 ± 1.2 × momentum, range 13–53) | 0.65 ± 0.01 | 0.046 ± 0.002 | 0.015 |
| Muon, random direction, 0.1 / 1 | 0.71 / 0.72 | 5.2 / 0.94 | 0.75 / 0.19 |

Across all Muon rows, single matrices range from cosine 0.48 to 0.91.

The size ECO's rule gives is (1 − β) × |error| ÷ (learning rate × |Muon's step|). This is milestone 4's adaptation of
ECO to Muon. For this error it is 7.4, 22, 74 and 222 times the momentum at learning rates 3e-3, 1e-3, 3e-4 and 1e-4.
Caveat: during real ECO training the error is each step's rounding error, not the full leftover of the pretrained
weights, so these sizes are an example, not a measurement of an ECO run.

**Why the direction bends (one SVD per matrix, step 40).** On average 22% of the error's energy (7–50% by matrix) lies
in the "symmetric" part. That part only changes the momentum's singular values, and orthogonalising sets them all to
about 1, so it is thrown away. The rest is reweighted by about (typical singular value ÷ that direction's singular
value). Directions where the momentum is weak get amplified, by up to about 20 times for 768-wide matrices, and the
strong ones are damped. This is why small errors arrive 5–8 times too large and at about 40° off.

**Muon's step size is nearly fixed.** Over all 40 steps × 48 matrices:
- The momentum's typical size (its RMS) spans a factor of 49 (16 after step 20).
- The RMS singular value of Newton–Schulz(momentum) is 0.905 (standard deviation 0.066, range 0.69–0.99). The spread
  is 7% of the mean, against 72% for the momentum.
- For one matrix over steps 21–40, the momentum's size changes by a median factor of 1.31 and the update's by 1.026.
- Multiplying the momentum by 1,000 or by 1/1,000 changes the update by less than 0.02%.

So each step moves each matrix by learning rate × a constant set by its shape. Mean RMS step per weight, before the
learning rate, by kind: `c_attn` 0.033, `attn c_proj` 0.030, `c_fc` 0.034, `mlp c_proj` 0.017. The spread within a
kind is 1–8%.

**In plain words.** For SGD-momentum, and for Adam with ECO's per-weight factor, an error put into the momentum comes
out in the weights exactly as intended. That is why ECO works for them. Muon is different. It squashes the momentum
into a matrix of fixed size, so it does not let the error through. A small error comes out bent by about 40° and
5–8 times too big. A large error comes out bent by the same amount and cut down to a fixed size. At the size ECO's
rule would use for GPT-2 at learning rate 0.001, under 5% of the correction arrives, and the update has almost
nothing left of the gradient's direction (cosine 0.015). A random error behaves the same way. This is a property of
the orthogonalisation, not of the 4-bit rounding leftover. It matches milestone 4's measurement during training on
the small Shakespeare model (cosine about 0.69).

**Not tested.**
- How the error spreads over the following steps. Only the next update is measured, while the error stays in the
  momentum and fades over about 20 steps.
- Newton–Schulz in bf16; only fp32 was used.
- 8-bit momentum.
- ECO's exact rule, which uses the previous step's error.
- Errors taken from an actual ECO-Muon run.
- Other models, and momentum built over more than 40 steps or during real training (here the weights stay fixed at
  the pretrained values while the momentum builds).
- Larger batches.
