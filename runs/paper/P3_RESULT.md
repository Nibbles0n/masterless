# P3: the noise model (2026-10-28)

**Verdict: P3a FAIL (rank correlation 0.795 against 0.8). P3b FAIL (at lr 1e-4 the model predicts a 12-bit carry, 6 bits
was enough). P3c prediction written below, before any P5 result: 12 bits.** The pass lines are not moved. What to do
next is the owner's decision.

Script: `p1_paper_gpt2.py`, A100, fp16, TF32 off, deterministic, seed 1. Pretrained GPT-2 small, 10M tokens per run
(305 steps × 32 × 1,024, full schedule: 5% warm-up, cosine to 0.1). Carry bits {4, 6, 8, 12} with 8-bit momentum, and
the 4-bit float copy, at Muon learning rates {1e-4, 3e-4, 1e-3, 3e-3}: 20 runs, none diverged, 0 carry violations.
Analysis: `report/p3_analysis.py` → `runs/paper/P3_summary.json`.

## The measured gaps (cell minus the float copy at the same rate)

| carry bits | lr 1e-4 | lr 3e-4 | lr 1e-3 | lr 3e-3 |
|---|---|---|---|---|
| 4 | +0.0425 ± 0.0013 | +0.0518 ± 0.0012 | +0.0513 ± 0.0012 | +0.0452 ± 0.0010 |
| 6 | **−0.0013 ± 0.0004** | +0.0017 ± 0.0005 | +0.0041 ± 0.0005 | +0.0022 ± 0.0006 |
| 8 | **−0.0005 ± 0.0002** | −0.0002 ± 0.0003 | −0.0000 ± 0.0003 | −0.0006 ± 0.0005 |
| 12 | −0.0001 ± 0.0001 | +0.0001 ± 0.0002 | +0.0004 ± 0.0003 | +0.0007 ± 0.0005 |
| float copy (loss) | 3.3404 | 3.3272 | 3.3167 | 3.3274 |

± is the paired standard error over the 256 validation windows. Bold: better than the float copy by more than 2 errors.

**The plain pattern: the damage depends on the carry bits and hardly on the learning rate.** 4 bits costs about 0.045 at
every rate. 6 bits costs 0.002–0.004, except at the smallest rate, where it is slightly better than the float copy.
8 and 12 bits cost nothing measurable at any rate.

## The model's numbers

- Mean 4-bit gap around a GPT-2 weight: 0.0386. Muon's update per weight (root-mean-square, measured on CPU from one
  batch of 8 × 1,024 tokens at the start): 0.0283. Sum of the schedule's learning-rate fractions over 305 steps: 167.95.
- ρ = [(gap ÷ 2^b) × √305 ÷ 2] ÷ [lr × 167.95 × 0.0283], from 44 (4 bits, lr 1e-4) down to 0.006 (12 bits, lr 3e-3).

## Pass lines

- **P3a:** Spearman rank correlation between ρ and the gap, with cells within 2 errors of zero tied at the lowest ranks:
  **0.795 → FAIL** (line 0.8). The plan did not say where cells clearly *better* than the float copy go. Ranking them
  below the tied cells gives 0.44, and tying every cell without damage gives 0.65, so every reading fails.
- **P3b:** ρ* fitted on rates 3e-4 and 1e-3 (fewest misclassified cells, then the geometric midpoint): **ρ* = 1.01**, with
  0 of 8 misclassified. Held out:
  - lr 3e-3: predicted 6 bits, measured 8. One level apart: OK.
  - lr 1e-4: predicted 12 bits, measured 6. Two levels apart: **miss**.
  - **FAIL.**

**Why it fails:** ρ assumes each step adds a fixed amount of random rounding to the carry. So small learning rates,
which need many small steps to travel, should hurt most. They don't. A small step barely moves the carry, and random
rounding then adds noise in proportion to the step (the round-5 rule: variance ≈ spacing × |step|), not a fixed amount.
A post-hoc model with that rule ranks the cells in the same order, so it does no better here. The carry bits alone rank
them at 0.92. This is a finding made after seeing the data, not a tested claim.

## P3c: the predicted carry for RL (written before any P5 result)

- Qwen3.5-0.8B decoder matrices (497.6M weights): mean 4-bit gap 0.00399.
- RL update per weight per step (AdamW, lr 1e-6, from the P5 pilot's float run, mean of its last 10 of 30 steps):
  2.70e-7 root-mean-square. 200 steps: travel 5.40e-5.
- ρ for RL: 4 bits 32.7, 6 bits 8.2, 8 bits 2.04, 12 bits 0.128, 16 bits 0.008. ρ* = 1.01.
- **Prediction: the smallest carry under ρ* is 12 bits.** By the plan, P5 arm (c) runs a 12-bit carry (2 bytes), and the RL
  result checks the prediction.

**Caveat, for the owner:** the model that makes this prediction failed both of its own tests above. Read the other way,
the GPT-2 data say 8 bits was enough at every rate. RL's step relative to its gap (6.8e-5 per step) is about the same as
GPT-2's at lr 1e-4 (7.3e-5), where 6 bits was enough. So 12 bits is very likely safe but may be more than needed. Both
the plan's rule (12) and the simple reading (8) are written here before any P5 run.

**Owner's decision (2026-10-28, before any P5 run):** P5 runs an 8-bit carry. The 12-bit prediction stays on record but
is not tested. See the change log in `PAPER_PLAN.md`.

## Not tested

- More than one seed per cell (the seed noise of a 10M-token run is not measured; the errors above are evaluation noise
  only).
- Carry bits between 6 and 8, or above 12.
- Longer runs: drift grows with steps, so a carry that is enough at 305 steps may not be at 3,050. P1's 8-bit carry did
  hold for 3,050 steps.

## After the data: flip rates (2026-10-30)

Not part of the registered analysis. The share of 4-bit values that change at each step (`changed4`, logged during the
runs), averaged over the logged steps of the second half of training (%):

| | 1e-4 | 3e-4 | 1e-3 | 3e-3 |
|---|---|---|---|---|
| 4-bit carry | 1.57 | 1.62 | 1.61 | 1.63 |
| 6-bit carry | 0.35 | 0.36 | 0.39 | 0.43 |
| 8-bit carry | 0.088 | 0.095 | 0.111 | 0.150 |
| 12-bit carry | 0.008 | 0.012 | 0.028 | 0.076 |
| float copy | 0.003 | 0.010 | 0.034 | 0.101 |

With a float copy the flip share grows 30-fold with the learning rate. With a narrow carry it barely depends on the
rate, and each two extra carry bits divide it by about four (at the smaller rates). Reading (not tested): after a tick
the carry sits next to the rounding boundary, and the carry's own random rounding (up to 1/14 of a gap at 4 bits) is
much larger than one step, so the weight flips back and forth at random. This matches the flat 4-bit row of the gap
table and side study G's 1.6–1.9%.
