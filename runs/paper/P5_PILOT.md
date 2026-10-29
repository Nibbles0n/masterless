# P5 pilot: RL set-up, model choice and cost

Date 2026-10-28. Code: `p5_rl.py`. Numbers: `P5_pilot.json` and the `p5_*_pilot`, `p5_*_sel`, `p5_*_calib` files
here. This is tuning, not evidence: one seed, 30 steps.

## The model

- The Qwen3.5 small models are on Hugging Face as `Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.5-2B` (post-trained) and
  `Qwen/Qwen3.5-0.8B-Base` and `Qwen/Qwen3.5-2B-Base`. Licence Apache 2.0, not gated. Each checkpoint also holds a
  vision encoder and a multi-token-prediction head; we load only the text model.
- Support: `transformers` 5.2.0 or newer (5.1.0 does not have it); vLLM 0.17.0 or newer. We used vLLM 0.32.0, which
  needs `transformers` below 5.20, so 5.19.0; torch 2.13.
- The fast Gated DeltaNet kernels (`flash-linear-attention` 0.6.0) install with pip and run on both the A100 and the
  T4. The first call tunes the kernels, which takes about 3 minutes, once per machine. `causal-conv1d` is not needed.
  One environment fix is needed on Colab: two cuDNN packages overwrite each other's files (see the top of `p5_rl.py`).
- vLLM runs on the T4 too, in fp16, but the full-parameter arms of the 0.8B model do not fit in a T4's 15 GB.

**Choice: Qwen3.5-0.8B (post-trained).** The rule takes the smaller model if its float accuracy is between 25% and
70%. It is, so the 2B model was not measured.

| GSM8K test, 1,319 questions, greedy, non-thinking, up to 1,024 new tokens | accuracy | answers cut off at 1,024 tokens |
|---|---|---|
| 0.8B, float (bf16) | **54.7% ± 1.4** | 8.5% |
| 0.8B, rounded to 4 bits (the start of arms b and c) | **48.1% ± 1.4** | 17.4% |
| difference, question by question | −6.5 ± 2.8 points (95%) | |

The 4-bit rounding loses 6.5 of 54.7 points, far less than half, so **no healing run is needed**.

## Generator: vLLM

Arm (a), AdamW, 64 questions × 8 answers per step, answers up to 768 new tokens:

| | seconds per step | of which: writing answers | training | weight copy and optimizer |
|---|---|---|---|---|
| vLLM in the same process, steps 2–8 | 59 | | | |
| vLLM, last 10 of 30 steps | **32** | 16.4 | 15.4 | 0.1 |
| the model's own `generate`, steps 2–8 (batch 256; 512 ran out of memory) | **157** | 124 | 37 | 0.1 |

vLLM is about 2.7 times faster per step on the same answers (6 times on the writing part), so **every arm uses vLLM**.
Copying the trained weights into vLLM after each step takes about 0.1 s.

Generator against trainer: both run bf16 copies of the same weights, but their arithmetic differs. Over the 30 AdamW
steps the probability the two give a sampled token differs by 0.003 on average; the average log-probability difference
is 0.0003; the largest single-token log-probability difference seen was 5.1.

## Optimizer: AdamW

| 30 steps, arm (a) | learning rate | GSM8K test after 30 steps |
|---|---|---|
| AdamW | 1e-6 | **62.2%** (+7.6 ± 2.7 points over the start) |
| Muon | 1.37e-5 | 56.9% |
| Muon − AdamW, question by question | | **−5.4 ± 2.5 points** (95%) |

Muon's learning rate was set so that its average step per weight matched AdamW's average over the 30 AdamW steps
(Muon moves each weight by about 0.027 × learning rate per step; a 2-step calibration run measured that). AdamW's
early steps are about three times larger than its average, and the model learned faster with them, mostly to give
shorter answers (from about 510 to 290 tokens) that are cut off less often. By the plan's rule ("AdamW if the pilot
shows Muon does clearly worse in float"), **P5 uses AdamW**, and the ticker carries AdamW's update (8-bit moments).

Not tested: other Muon learning rates (a larger one may close the gap), more than one seed, more than 30 steps.

## Cost (A100-40GB, 5.30 units per hour, read from `colab usage`)

- At 64 questions per step, the measured steps project to 1.95 A100-hours per run for arm (a) and 2.00 for the
  ticker (its rounding step adds about 0.9 s), plus about 0.15 h to install packages on a fresh machine: 2.15 h, over
  the plan's 2-hour line. **The rule halves the questions per step to 32 and keeps 200 steps.** This is a close call:
  without the install time the ticker run is just under 2 hours.
- At 32 questions per step: about **1.15 A100-hours, about 6 units per run** (projected from the 64-question run, not
  measured).
- All of P5 (arms a ×2, b ×1, c ×2): about **35 units**; with both third seeds, about **48 units**. The plan expected
  about 30.

Settings for the main runs: `--gen vllm --opt adamw --lr 1e-6 --questions 32 --answers 8 --steps 200
--max-new-tokens 768 --eval-max-new-tokens 1024` (arm b: `--lora-lr 1e-5`, rank 32, alpha 32; arm c: `--carry-bits`
from P3c).

## Checks done

- Every arm (float, lora4, ticker) with both optimizers ran 2–3 steps on the T4 (a tiny random Qwen3.5 for every arm,
  the real 0.8B for the ticker with Muon and for LoRA) and on the A100 in bf16 (real 0.8B: ticker with Muon and with
  AdamW, LoRA, float with Muon), with both generators. Coin-flip rewards were used there so every step trains.
- The ticker's weights and the LoRA arm's frozen weights are checked to lie exactly on the 4-bit grid at the end of
  every run.

## Units spent

T4: about 2.7 (smoke tests, 2.5 hours). A100: about 9.0 (1.7 hours). Total about 11.7, within the 3 + 12 allowed.

## Muon re-tune (owner's request, 2026-10-28, before any P5 run)

The pilot gave Muon one learning rate, set to match AdamW's average step. The owner asked for a fair try first. Three
more float runs, otherwise as the pilot (arm (a), 64 questions × 8 answers, 30 steps, vLLM, seed 1, AdamW 1e-6 for the
weights Muon does not handle, constant learning rate). Runs: `runs/paper/P5tune/`. Analysis:
`report/p5tune_analysis.py` → `runs/paper/P5tune_summary.json`.

| 30 steps, float | Muon learning rate | GSM8K test | vs start (54.7%) | vs AdamW, question by question (95%) | answer length (test) | cut off at 1,024 |
|---|---|---|---|---|---|---|
| AdamW 1e-6 (pilot) | | 62.2% | +7.6 ± 2.7 | | 282 | 2.2% |
| Muon (pilot) | 1.37e-5 | 56.9% | +2.2 ± 2.6 | −5.4 ± 2.6 | 340 | 4.7% |
| Muon | 4e-5 | 66.9% | +12.3 ± 2.8 | **+4.7 ± 2.6** | 387 | 2.6% |
| Muon | **1e-4** | **70.3%** | +15.6 ± 2.8 | **+8.0 ± 2.7** | 496 | 9.0% |
| Muon | 2.5e-4 | 59.6% | +4.9 ± 3.0 | −2.7 ± 3.0 | 390 | 5.9% |

Training reward, mean of 5-step blocks (steps 1–5 … 26–30):
- AdamW 1e-6: 0.42, 0.45, 0.50, 0.50, 0.53, 0.58
- Muon 4e-5: 0.41, 0.46, 0.50, 0.54, 0.58, 0.63
- Muon 1e-4: 0.44, 0.51, 0.58, 0.63, 0.64, 0.68 (training answers grow again after step 15: 431 → 551 tokens, 22% cut
  off at 768)
- Muon 2.5e-4: 0.49, 0.60, 0.64, 0.64, 0.62, 0.60 (peaks near step 15, then falls: too large)

**What it shows:** the pilot's Muon learning rate was too small. Muon's best step per weight is about 4 to 10 times
AdamW's average (1.1e-6 and 2.7e-6 against 2.7e-7, root-mean-square, last 10 steps). At 4e-5 and 1e-4, Muon is ahead
of AdamW at 30 steps, by more than the 95% interval.

**By the plan's rule** ("Muon … or AdamW if the pilot shows Muon does clearly worse in float"), Muon does not do worse,
so **P5 uses Muon** on the decoder matrices, and the ticker carries Muon's update (2.78 bytes per weight, not 3.81 with
AdamW's 8-bit moments). Seconds per step are about the same (35–44 at 64 questions), so the cost per run is unchanged.

**Not a claim:** AdamW was tried at one learning rate, Muon at four. "Muon beats AdamW at RL" is not tested and is not
claimed. P5 compares arms with the same optimizer, so it does not need it.

**Open, for the owner:** the learning rate for 200 steps at 32 questions per step. These runs are 30 steps, and the
learning rate is constant. 2.5e-4 peaked by step 15; 200 steps at 1e-4 travels about 5 times as far as that peak, and
32 questions per step make each step noisier. 4e-5 is the cautious choice; 1e-4 was best at 30 steps.

The RL step relative to Qwen's 4-bit gap (mean 0.00399): 2.7e-4 per step at 4e-5, 6.8e-4 at 1e-4. P1's GPT-2 runs
(lr 1e-3) were at 7.3e-4, where the 8-bit carry held for 3,050 steps.

Units: 8.5 (A100, about 1.6 hours with installs). Balance 323.03.
