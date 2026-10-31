# P5: RL in 4 bits on Qwen3.5-0.8B, GSM8K (2026-10-29)

**Verdict: P5b PASS, P5a FAIL.** After 200 steps of RL, ours (every weight stored in 4 bits, no full-precision copy)
gains 20.4 ± 2.4 points over its 4-bit start, more than normal 16-bit training gains over its own start (17.6 ± 2.4).
It ends 3.7 points behind normal training; the 95% interval, −5.2 to −2.2, has its lower end below the −4 line, so P5a
fails on seeds 1–3 pooled, as the plan's rule says. Most of the gap is the 6.5 points lost by rounding the model to 4
bits before any training.

**Against LoRA (measured only):** scored the way LoRA is used (a frozen 4-bit model plus a 16-bit add-on), LoRA ends
2.6 ± 1.6 points above ours, with less memory. As a pure 4-bit model it is worth nothing: merged and rounded to the
4-bit grid, **not one 4-bit value changes**, and the model scores 47.7%, its 48.1% start. Ours, which is a pure 4-bit
model, is 20.9 ± 2.2 points above that. The prediction written before the runs held.

Script: `p5_rl.py`, one A100 40GB per run, vLLM 0.32.0 as the answer writer. Qwen3.5-0.8B (post-trained), GRPO with 8
answers per question, reward 1 if the final number is right. 200 steps × 32 questions × 8 answers, up to 768 new
tokens in training and 1,024 in the exam, Muon 4e-5 on the decoder matrices (AdamW 1e-6 for the rest, AdamW 1e-5 for
(b)'s LoRA). Exam: the 1,319 GSM8K test questions, greedy. Analysis: `report/p5_analysis.py main 4e-5` →
`runs/paper/P5main_summary.json`. Runs: `runs/paper/P5/`.

## Pass lines (200-step design, seeds 1–3 pooled)

| line | pass if | measured | result |
|---|---|---|---|
| **P5a** | (c) − (a), 95% interval above −4 points | −3.71, interval −5.22 to −2.21 | **FAIL** |
| **P5b** | (c) gains ≥ 5 points over its 4-bit start, 95% interval above 0 | +20.44, interval +17.99 to +22.89 | **PASS** |

On seeds 1–2 alone (when the third-seed rule was checked) P5a was −3.34, interval −5.19 to −1.48. Seed 3 of ours ended
4.5 points below seed 3 of normal training, so the middle moved away from the line.

Intervals: each question's 0/1 result is averaged over seeds, the arms are paired question by question, and the
interval is 1.96 × the paired standard error over the 1,319 questions.

## All runs

| arm | seed | start | after 200 steps | gain | peak GPU memory | seconds per step |
|---|---|---|---|---|---|---|
| (a) 16-bit, float copy | 1 | 54.7% | 72.9% | +18.3 | 29.3 GB | 14.4 |
| (a) 16-bit, float copy | 2 | 54.7% | 71.1% | +16.5 | 29.3 GB | 14.3 |
| (a) 16-bit, float copy | 3 | 54.7% | 72.9% | +18.2 | 26.8 GB | 18.0 |
| (c) ours, 4-bit, 8-bit carry | 1 | 48.1% | 69.7% | +21.5 | 29.8 GB | 18.0 |
| (c) ours, 4-bit, 8-bit carry | 2 | 48.1% | 67.7% | +19.6 | 29.8 GB | 14.9 |
| (c) ours, 4-bit, 8-bit carry | 3 | 48.1% | 68.4% | +20.2 | 29.8 GB | 17.2 |
| (b) 4-bit frozen + LoRA rank 32 | 1 | 48.1% | 70.6% | +22.4 | 23.9 GB | 15.6 |
| (b) 4-bit frozen + LoRA rank 32 | 2 | 48.1% | 70.2% | +22.1 | 23.9 GB | 15.3 |
| (b) 4-bit frozen + LoRA rank 32 | 3 | 48.1% | 72.8% | +24.6 | 23.9 GB | 16.9 |
| (a) at 1e-4 (guard run, not used) | 1 | 54.7% | 36.6% | −18.1 | 29.3 GB | |

Seconds per step vary with the A100 the run landed on and with answer lengths.

## LoRA rounded to 4 bits (measured only, seeds 2–3)

At the end of (b) seeds 2 and 3 the add-on is merged into the 4-bit weights (4-bit value + B·A) and rounded to nearest
on the same 4-bit grid, block scales recomputed, as ours is. Then the exam is taken again.

| | seed 2 | seed 3 |
|---|---|---|
| exam, 4-bit + 16-bit add-on | 70.2% | 72.8% |
| exam, merged and rounded to 4 bits | 47.6% | 47.8% |
| share of 4-bit values that change | 0 | 0 |
| add-on size, mean of \|B·A\| / gap between 4-bit values | 0.0052 | 0.0051 |
| add-on size, root mean square | 1.83e-5 | 1.79e-5 |

Pooled: rounded − its own 16-bit score = −23.8 ± 2.3; rounded − the 4-bit start = −0.4 ± 2.0 (no change); ours −
rounded LoRA = +20.9 ± 2.2. The add-on is about 1/200 of a gap on average, so every weight rounds back to its start;
the small difference from 48.1% comes from block scales that shift very slightly when the add-on changes a block's
largest value. **Prediction, written before the runs:** "most weights round back to their start and the rounded model
scores close to the 48.1% start, well below ours." It held, more strongly than written (not most: all).

What this does and does not say: LoRA's gain is real, but it lives entirely in the 16-bit add-on. It is not a 4-bit
model; it is a 4-bit model plus a 16-bit one, and it must be served that way (as QeRL's own evaluation does). Ours
produces a model whose weights are only the 4-bit values. Other ways to bring LoRA's gain into 4 bits (training the
block scales, as Scale-QLoRA does, or a longer healing run) were not tested.

## The learning-rate guard fired

The plan set Muon 1e-4 with a guard written in advance: if (a) seed 1's 20-step mean training reward ends more than
0.04 below its highest, rerun at 4e-5 and use 4e-5 everywhere. At 1e-4 the reward peaked at 0.663 (around step 35)
and ended at 0.357; the exam score fell to 36.6%, below the start. So every run above uses 4e-5.

## What else it shows

- **The gap is mostly the starting gap.** Rounding the model to 4 bits costs 6.5 ± 2.8 points before any training.
  After 30 steps (the shorter design) ours was 6.6 behind; after 200 steps, 3.7 behind. Ours gains about 3 points more
  than normal training, as if it partly recovers what rounding lost, but not fast enough to pass the line in 200 steps.
- **(c) against (b), as LoRA is served** (measured only): −2.60, interval −4.19 to −1.01 (three seeds each). LoRA
  learns more slowly early (at 30 steps ours led by 12.1 ± 2.5 points; by step 80 ours' 20-step mean training reward
  was 0.63 against 0.50) and passes ours by step 200. LoRA's learning rate (AdamW 1e-5) was not tuned; neither was ours
  beyond the guard. Peak memory: LoRA 23.9 GB, ours 29.8 GB.
- **Weights really move in 4 bits:** 0.11% of the 4-bit weights change value each step (mean over steps).
- **Answer writer vs trainer:** the mean probability gap between vLLM and the trainer for the same answer is 0.0030
  for (a), 0.0033 for (c) and 0.0052 for (b). The ticker adds no rounding mismatch of its own.
- **Memory:** the peak GPU memory is about the same for (a) and (c) (27–29 against 29.8 GB). It is set by vLLM's
  reserved share (35% of the card) and the training activations; the 4-bit values are held in fp32 tensors here. The
  ticker's saving is measured in P6, not here.

## The 30-step design (reported, not used for the verdict)

30 steps × 64 questions × 8 answers, Muon 1e-4; (a) seed 1 is the re-tune run. `runs/paper/P5short/`,
`runs/paper/P5short_summary.json`.

| | measured | as a pass line |
|---|---|---|
| P5a: (c) − (a) | −6.60, interval −8.53 to −4.66 | would FAIL |
| P5b: (c) gain over its start | +16.38, interval +13.74 to +19.01 | would PASS |
| (a) gain over its start | +16.45 ± 2.48 | |
| (c) − (b) | +12.13 ± 2.54 | |

Runs: (a) 70.3%, 71.9%; (c) 62.5%, 66.6%; (b) 52.4%.

## Disclosures

- The 30-step (c) seed 1 was read before the third change-log entry (the switch to the 200-step design) was written.
- The guard used training reward only. The test set was not read to pick the learning rate for the 200-step runs.
- Seeds 1–2 were fully read before the fourth change-log entry (third seeds, LoRA seeds 2–3, rounded LoRA) was written.
- The rounded-LoRA exam was added to `p5_rl.py` after (b) seed 1 ran, so it covers seeds 2–3 only. The add-ons of seeds
  2–3 are saved (`*.pt`, not committed).

## Not tested

- A healing run before RL (the rule did not call for one: rounding lost 6.5 of 54.7 points, under half).
- Longer training (does the gap keep closing?), larger models, other tasks.
- The 4-bit answer writer: vLLM holds a 16-bit copy of the 4-bit values for writing answers.
- Bringing LoRA's gain into 4 bits by other means (training block scales, healing).
- P3c's 12-bit carry prediction (P5 runs the 8-bit carry, see the change log).

Compute: this round (third seeds, LoRA seeds 2–3) used 23.6 Colab units; about 144 in all against the 150 cap.

## After the runs: a GPTQ start (2026-10-30, change log seventh entry)

A measurement only, no RL. `gptq_qwen.py` rounds the same 186 decoder matrices to the same 4-bit format with GPTQ
(calibrated on 128 GSM8K training questions with their reference solutions, in the RL chat format) and scores the model
with P5's evaluation on one A100 (vLLM, bf16, greedy, all 1,319 test questions). The same run re-scored the float model
and round to nearest, and both reproduced the logged numbers exactly. Result file: `runs/paper/GPTQ/qwen_start.json`.

| starting model | test accuracy | share of answers cut off at 1,024 tokens |
|---|---|---|
| bf16 (float) | 54.7% | 8.5% |
| 4-bit, GPTQ | **51.2%** | 9.9% |
| 4-bit, round to nearest (every P5 4-bit run) | 48.1% | 17.4% |

- GPTQ − round to nearest: **+3.0 ± 2.8 points** (95% interval, paired over questions); bf16 − GPTQ: 3.5 ± 2.7.
- GPTQ recovers 47% of the 6.5 points that rounding to nearest costs; the prediction written before the run was "about
  half". Rounded to nearest, the model also runs into the length limit almost twice as often (17.4% against 9.9%).
- What it means for P5: the ticker's 20.4-point gain (P5b) is measured from the round-to-nearest start. A GPTQ start
  would begin 3 points higher. Whether RL from it keeps that lead, and whether the ticker would then meet P5a, is not
  tested.
- Cost: 1.8 Colab units (A100, about 12 minutes including a 4-minute GPTQ).

## Setup (d): 4-bit weights with an fp32 master copy, DeepSeek-V4 style (2026-10-31, change log eighth entry)

The recipe the ticker replaces. DeepSeek-V4 (arXiv 2606.19348, Sec. 5.2.1) trains its expert weights in FP4 during
post-training, RL included: an fp32 master copy is rounded to 4 bits for each forward pass, the gradient passes straight
through to the master, and RL answers are written with the 4-bit weights. `p5_rl.py --arm copy4` does this on our format
(E2M1, fp32 scale per 16, recomputed from the master each step) with Muon at 4e-5 and fp32 momentum; everything else as
(c). Start: the same round-to-nearest model (48.1%). Seeds 1–3, one A100-40GB each; 8.75 bytes of training state per
decoder weight against the ticker's 2.78. Result files: `runs/paper/P5/p5_copy4_muon_vllm_s{1,2,3}_lr4e-5.json`.

| setup | end, seeds 1, 2, 3 | mean | gain over start (95% interval) |
|---|---|---|---|
| (c) ticker | 69.7, 67.7, 68.4 | 68.6 | +20.4 ± 2.4 |
| (d) 4-bit + fp32 copy | 68.1, 67.3, 64.3 | 66.6 | +18.4 ± 2.5 |

- **P5d: PASS.** (c) − (d) = **+2.0 points, interval [+0.5, +3.6]**; the line was "lower end above −3".
- The prediction ("within 1 point of each other, both near 68–69%") did not hold: the ticker ended 2.0 points ahead,
  and (d) ended near 66.6%. The ticker is ahead on every seed (+1.6, +0.4, +4.1 with the same seed), but the seeds
  spread widely ((d)'s standard deviation is 2.0 points, (c)'s 1.0), so the seed-level standard error of the difference
  is 1.3 points. We read this as "matches", not as "better".
- (d) − (a) = −5.7 ± 1.6: the master-copy recipe is further from bf16 training than the ticker (−3.7 ± 1.5).
- Share of 4-bit values changing per step: 0.033% for (d), 0.11% for (c). The ticker flips about three times as many
  values for the same learning rate; after the data, our reading is the boundary jitter of P3 (the carry's own random
  rounding flips weights that sit next to a boundary back and forth). Not tested further.
- Generator–trainer token-probability gap: 0.0033 for (d), as for (c) and (a).
- Peak GPU memory 31.2 GB (fp32 tensors for the 4-bit values, a master and fp32 momentum); P6 measures packed memory.
- Cost: 27.6 Colab units, about 11 of them lost to a session that ended mid-run (seeds 1 and 3 were run again).

Not tested: (d) in MXFP4 (blocks of 32, power-of-two scales), with FP8 compute, or with AdamW; 16-bit RL rounded at
the end; RL from a GPTQ start for either setup.
