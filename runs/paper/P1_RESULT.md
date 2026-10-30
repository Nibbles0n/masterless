# P1: GPT-2 small, continued training, 100M tokens (2026-10-28)

**Verdict: all five pass lines pass: P1a, P1b, P1c, P1d, and P1e (added 2026-10-29, ECO with AdamW).** P1b's ECO-Muon margin (0.037) is just above the 1-seed
threshold (0.033), and uses a rerun with Newton–Schulz in fp32: the first ECO-Muon run overflowed because of how our
code handles fp16 (see below).

Script: `p1_paper_gpt2.py`, A100 40GB, fp16 with loss scaling, TF32 off, strict deterministic mode. Settings as side
study G stage 2: pretrained GPT-2 small, Muon lr 0.001 (AdamW 1e-4 for the other weights), 3,050 steps × 32 × 1,024
tokens, 256 validation windows. Analysis: `report/p1_analysis.py` → `runs/paper/P1_summary.json`. Runs: `runs/paper/P1/`.

## Pass lines

| line | pass if | measured | result |
|---|---|---|---|
| **P1a** (headline) | ours ≤ 4-bit float copy + Muon + 0.03, each of seeds 1–3 | seed 1 +0.0011 ± 0.0005, seed 2 +0.0006 ± 0.0005, seed 3 +0.0023 ± 0.0005; mean +0.0013 ± 0.0005 | **PASS** |
| **P1b** | ours ≥ 0.02 below naive and below ECO-Muon (seed 1), each gap above the 1-seed threshold (0.033) | naive 0.160 ± 0.002 above ours; ECO-Muon (fp32 Newton–Schulz) 0.037 ± 0.002 above ours | **PASS** (ECO narrowly) |
| **P1c** | ours below bf16-SR rounded to 4 bits, each seed | ours lower by 0.046, 0.049, 0.043 (± 0.001) | **PASS** |
| **P1d** | ours within 0.01 of Kahan (seed 1) | +0.0012 ± 0.0005 | **PASS** |
| **P1e** (added after an outside review, before its runs) | ours below ECO-AdamW in each of seeds 1–3, by more than 2 × the paired standard error | ours lower by 0.0794 ± 0.0013, 0.0801 ± 0.0014, 0.0797 ± 0.0013 | **PASS** |

± is the paired standard error over the 256 validation windows (evaluation noise only).

## All final losses

| arm | bytes per weight | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|
| **ours** (`q4_t8_m8`) | 2.78 | 3.28841 | 3.28921 | 3.28890 |
| 4-bit float copy + Muon | 8 | 3.28733 | 3.28860 | 3.28661 |
| bf16-SR, rounded to 4 bits | 3.03 | 3.33458 | 3.33869 | 3.33217 |
| bf16-SR, scored in bf16 | 3.03 | 3.26900 | 3.27195 | 3.26951 |
| Kahan (`q4_kahan_m8`) | 5.78 | 3.28724 | | |
| float Muon (fp32) | 8 | 3.26697 | | |
| float Muon, rounded to 4 bits | 8 | 3.33030 | | |
| 4-bit float copy + AdamW | 12 | 3.28397 | 3.28588 | 3.28464 |
| ECO-AdamW (`eco_adamw_q4`, random rounding, fp32 moments) | 8.75 | 3.36785 | 3.36930 | 3.36865 |
| naive (`q4_sr_m8`) | 1.78 | 3.44829 | | |
| ECO-Muon (`eco_muon_q4`), Newton–Schulz in fp32 | 1.78 | 3.32534 | | |
| ECO-Muon, first run, Newton–Schulz in fp16 | 1.78 | overflowed by step 10 | | |
| start: pretrained GPT-2, float | | 3.40253 | | |
| start: pretrained GPT-2 rounded to 4 bits (every 4-bit arm's step 0) | | 3.46200 | | |

Seed standard deviation (the largest among the 3-seed arms, bf16-SR rounded to 4 bits): 0.0033, so the 1-seed threshold
is 0.033. Ours alone: 0.0004.

## Measured only

- **Ours against float Muon in fp32:** +0.021 ± 0.001. Not resolved by the 1-seed rule (below 0.033). This is the
  price of 4-bit weights themselves: the 4-bit float copy pays the same (+0.020).
- **Ours against float Muon rounded to 4 bits after training:** ours is 0.042 lower. Resolved. Training in 4 bits
  beats training in float and rounding at the end.
- **Ours against bf16-SR scored in bf16:** +0.017 to +0.019. bf16-SR is then a 16-bit model, not a 4-bit one.
- **Ours against the 4-bit float copy + AdamW (standard 4-bit training, 12 bytes):** +0.0044 ± 0.0007 (seed 1);
  over seeds 1–3, +0.0040 ± 0.0003. At this learning rate AdamW's float copy is slightly ahead of Muon's float copy
  (3.2840 against 3.2873, seed 1).
- **Share of 4-bit weights that change per step:** ours 0.14%, Kahan 0.053%, naive 0.046%, ECO-Muon 0.0004%,
  ECO-AdamW 1.6% (mean over logged steps, seed 1). The ticker changes more weights than Kahan because its carry is rounded at random.
- **Carry check:** 0 carry violations in every ours run; every matrix on the 4-bit grid at the end.

## P1e: ECO with AdamW, as the ECO paper runs it

Added after an outside review asked for ECO in its own setting (Adam), not only our adaptation of it to Muon. The arm,
the pass line and the readings were written into `PAPER_PLAN.md` (change log, fifth entry) before the runs.

**Setup.** ECO's Adam version (Algorithm 3 of the ECO paper, our implementation in `StackECOAdam`): 4-bit weights on
the same grid as ours, rounded at random (the ECO paper finds this works best), the rounding error injected into
Adam's first moment, both moments in fp32 (8.75 bytes per weight). Same learning rate, betas and weight decay as the
4-bit float copy with AdamW (1e-4, (0.9, 0.95), 0.1), the same 100M tokens, seeds 1–3. The learning rate was not tuned
for ECO, as written in the plan.

**Result.** ECO-AdamW ends at 3.368–3.369, 0.080 behind ours in every seed (paired standard error 0.0013), and
0.084 ± 0.0002 (mean ± standard error over 3 seeds) behind the 4-bit float copy trained with the same AdamW. Ours is
0.0040 ± 0.0003 behind that AdamW float copy, while storing 2.78 bytes per weight against 12.

**How it fails.** ECO-AdamW drops from 3.462 to 3.359 in the first 300 steps and then stays flat (3.360–3.367) while
the float copy keeps improving to 3.284. About 1.4–1.9% of its 4-bit values change each step (ours: 0.14%), so its
weights are not frozen, as ECO-Muon's are; they keep moving, but the moves do not add up to progress. A likely reading,
not measured: each AdamW step is much smaller than the gap between 4-bit values, so random rounding adds noise about
the size of a gap at every step, and ECO's injection puts back only a (1 − β1) = 10% share of each step's error. The
ticker keeps the whole error.

**What the planned reading says:** pass, so ours beats ECO even in ECO's own setting (Adam with fp32 moments, more
than 3 times our memory), and no byte-matched (8-bit moment) ECO run is needed.

## GPTQ instead of rounding to nearest (change log sixth entry, 2026-10-29)

`gptq_paper.py` on the saved models; `report/gptq_analysis.py` writes `runs/paper/GPTQ_summary.json`. Raw files in
`runs/paper/GPTQ/`. The float Muon and bf16-SR seed-1 runs were retrained with `--save`; both match the logged runs
exactly (3.26697 and 3.269, rounded 3.3303 and 3.33458), so GPTQ was applied to the models in the tables. Every GPTQ
weight is on the 4-bit grid (checked).

| model | float | rounded to nearest | GPTQ | share of the rounding loss GPTQ recovers | GPTQ − ours (seed 1, 3.28841) |
|---|---|---|---|---|---|
| pretrained GPT-2 (start) | 3.40250 | 3.46193 | 3.42850 | 56% | |
| float Muon, seed 1 | 3.26697 | 3.33031 | 3.29754 | 52% | +0.0091 ± 0.0007 |
| bf16-SR, seed 1 | 3.26899 | 3.33463 | 3.30060 | 52% | +0.0122 ± 0.0007 |

- Predictions, written before the runs: start between 3.40 and 3.46 (**held**, 3.4285); both GPTQ models above ours by
  more than twice the paired standard error (**held**, 13 and 16 standard errors).
- In continued training, the claim "training in 4 bits beats training in higher precision and rounding afterwards"
  stands against GPTQ too. GPTQ recovers about half of what rounding to nearest loses.

## ECO-Muon: the first run overflowed, the rerun freezes the 4-bit weights

**First run (fp16 Newton–Schulz, like every other arm): overflowed by step 10.** ECO adds the rounding error, divided
by the step size, into the momentum. A 4-bit rounding error is far larger than one Muon step, so the momentum becomes
huge. Our code casts the momentum to fp16 *before* scaling it down inside Newton–Schulz, and the huge values overflow
fp16's range. That is a limit of our code's number format, not of ECO. The other arms' momenta stay small, so they are
not affected.

**Check, then rerun with Newton–Schulz in fp32 (the same seed and settings).** A 300-step check trained normally, so
the full run was redone. It is the ECO-Muon run P1b uses. It differs from the other arms only in Newton–Schulz's
number format, which can only help it.

**What it shows: ECO with Muon freezes the 4-bit weights.** It ends at 3.3253, 0.037 behind ours, and better than naive
(3.448). But only 0.0004% of its 4-bit weights change per step, falling to 0 by the end, against 0.14% for ours. The
injected error dominates the momentum, so each step mostly points back toward undoing the last rounding, and the
matrices stay where they started. Its gain over its 4-bit start (3.462 → 3.325) comes from the small float parts of the
model (embeddings, norms, biases), which train with AdamW in every arm. A run with the 4-bit matrices frozen on purpose
was not made, so this reading is not measured directly.

## Not tested

- A run with the 4-bit matrices frozen (the likely floor that ECO-Muon matches).
- ECO-Muon at other learning rates or with its injection capped (it is our own adaptation of ECO to Muon; the paper
  gives SGD and Adam only).
- Seeds 2–3 of the 1-seed arms (the AdamW float copy now has 3).
- ECO-AdamW at other learning rates, with 8-bit moments, or from scratch (P2 size): proposed as an extension.
- Other learning rates for any arm (P3 covers ours and the float copy at four rates over 10M tokens).
- GPTQ on seeds 2–3, GPTQ variants (column reordering, scale search) and other calibration sizes.

**Correction, 2026-10-29.** The table first listed 3.40253 as the 4-bit start. That is the float GPT-2 before
rounding; the 4-bit start (step 0 of every 4-bit arm) is 3.46200. No pass line used the start.

**Compute for P1e (2026-10-29):** 20.2 Colab units on one A100 (5 runs, 4 at a time). About 10 of them were wasted: the
runs finished around 18:50, but the watcher waited for a "done" flag that one queue never wrote, and the session sat
idle until 20:20. Total for the paper so far: about 164 units against the 150 cap.

**Compute for GPTQ (2026-10-29/30):** 9.5 units on one A100 (P1 and P2 retrains plus GPTQ, about 1 h 20 min). Total
about 174 units.
