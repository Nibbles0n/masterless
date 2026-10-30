# P2: GPT-2 6 × 512 from scratch, 0.9B tokens (2026-10-29)

**Verdict: both pass lines pass: P2a and P2c.** Trained from random weights, ours ends 0.005 to 0.013 behind the 4-bit
float copy (line: 0.03) on every seed, with no carry overflow, at 2.78 bytes per weight instead of 8. A bf16 model
trained normally and then rounded to 4 bits is 1.13 worse than ours.

Script: `p1_paper_gpt2.py --init scratch --n-layer 6 --n-embd 512 --n-head 8`, one A100 40GB shared by 3–4 runs, fp16
with loss scaling, strict deterministic mode. 45.2M parameters, 27,466 steps × 32 × 1,024 tokens (0.90B, about 20 per
parameter), 5% warm-up then cosine to 10%, Muon lr 0.04 with AdamW 4e-3 for the other weights, validation every 500
steps on 256 windows. Analysis: `report/p2_analysis.py` → `runs/paper/P2_summary.json`. Runs: `runs/paper/P2/`.

## Pass lines

| line | pass if | measured | result |
|---|---|---|---|
| **P2a** | ours ≤ 4-bit float copy + Muon + 0.03, each of seeds 1–3 | seed 1 +0.0049 ± 0.0019, seed 2 +0.0128 ± 0.0019, seed 3 +0.0101 ± 0.0019 | **PASS** |
| **P2c** | ours (mean of 3 seeds) below bf16-SR rounded to 4 bits by more than 2 × ours' seed standard deviation | 5.0515 − 3.9195 = 1.132, against 2 × 0.0067 = 0.013 | **PASS** |

± is the paired standard error over the 256 validation windows (evaluation noise only).

## All final losses (validation cross-entropy)

| arm | bytes per weight | seed 1 | seed 2 | seed 3 | mean ± seed s.d. |
|---|---|---|---|---|---|
| ours (4-bit, 8-bit carry, 8-bit momentum) | 2.78 | 3.9127 | 3.9262 | 3.9196 | 3.9195 ± 0.0067 |
| 4-bit float copy + Muon | 8 | 3.9078 | 3.9133 | 3.9095 | 3.9102 ± 0.0028 |
| bf16-SR + Muon, in bf16 | — | 3.8894 | | | |
| bf16-SR + Muon, rounded to 4 bits at the end | — | 5.0515 | | | |

Start: 10.93 (seed 1). Every ours run: 0 carry violations, 8–11 steps skipped by the fp16 loss scaler (the float copy:
9–10).

## Learning-rate tuning (10% length, float copy, seed 1)

| Muon, AdamW lr | loss after 2,746 steps |
|---|---|
| 0.01, 1e-3 | 4.3055 |
| 0.02, 2e-3 | 4.3015 |
| **0.04, 4e-3** (picked) | **4.2678** |

The picked rate is the top of the grid, so a larger one might be better still. Short tuning runs also tend to favour
larger rates. The same rate is used for every arm, so this does not favour ours.

## What else it shows

- **Ours is a little behind the float copy on every seed** (mean +0.009), and each seed's gap is several standard
  errors from zero. So the cost is small but real, and inside the 0.03 line, not zero. In P1 (continued training,
  3,050 steps) it was +0.0013.
- **Seed spread is larger for ours** (0.0067 against 0.0028). The random rounding of the carry adds a little
  seed-to-seed noise.
- **Rounding to nearest at the end destroys a normally trained model**: bf16-SR is the best model in float (3.889) but
  loses 1.16 when rounded to 4 bits. GPTQ recovers almost all of it (see below).

## GPTQ instead of rounding to nearest (change log sixth entry, 2026-10-30)

bf16-SR seed 1 was retrained with `--save` and matches the logged run exactly (3.88939; rounded to nearest 5.05152).
GPTQ (`gptq_paper.py`, 128 calibration windows from the training data) on the 24 block matrices:

| | loss |
|---|---|
| bf16-SR in bf16 | 3.88940 |
| bf16-SR rounded to nearest | 5.05151 |
| **bf16-SR with GPTQ** | **3.91548** |
| ours, seeds 1 / 2 / 3 | 3.9127 / 3.9262 / 3.9196 |

- GPTQ recovers 98% of the 1.16 lost by rounding to nearest. All weights on the 4-bit grid (checked).
- GPTQ − ours (seed 1 against seed 1) = **+0.0028 ± 0.0021**. That is not more than twice the standard error, so by the
  reading written before the run, **the claim narrows**: from scratch, training in 4 bits beats training in bf16 and
  rounding to nearest, but not training in bf16 and rounding with GPTQ. The GPTQ model also lies inside the spread of
  our three seeds.
- Prediction "GPTQ recovers most of the 1.16" held; "stays above ours" did not (above by 0.003, not resolved).
- Memory: bf16-SR is 3.03 bytes per weight, ours 2.78. So from scratch at this size, bf16-SR plus GPTQ gives about the
  same 4-bit model for about 9% more memory. What the ticker still has that it does not: the weights being trained are
  the 4-bit weights (needed in RL, P5), and the advantage in continued training (P1).

## Not tested

- Learning rates above 0.04, and the batch size (fixed at 32 × 1,024; the plan named it as a tuning knob, but only the
  rate was tuned).
- Seeds 2–3 of bf16-SR, and GPTQ on them.
- GPTQ variants (column reordering, scale search), other calibration sizes, and why GPTQ recovers 98% here but about
  half in P1.
- Other model sizes. The 124M from-scratch run is future work.
- Kahan, naive rounding and ECO-Muon from scratch (they were compared in P1 only).
