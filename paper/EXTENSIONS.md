# Masterless Muon: proposed extensions (for reviewers, 2026-10-29)

The paper's experiments are done and every pass line is reported (`paper/main.pdf`). Before spending more compute, we
would like to know which of the extensions below would change your assessment. Costs are in Google Colab compute
units on one A100 40GB (about 6.7 units per hour, roughly US$0.10 per unit).

## What is already shown

- GPT-2 small, continued training (100M tokens, 3 seeds): the ticker (4-bit weights, 8-bit carry, 2.78 bytes per
  weight) ends 0.0013 behind a 4-bit float copy with Muon (8 bytes) and 0.0040 behind one with AdamW (12 bytes).
- ECO in its own setting (AdamW, random rounding, fp32 moments, 8.75 bytes per weight), same model, 3 seeds: 0.080
  behind the ticker in every seed. ECO with Muon (our adaptation): 0.037 behind, its 4-bit weights frozen.
- From scratch, 45M parameters, 0.9B tokens (3 seeds): the ticker is within 0.005–0.013 of the 4-bit float copy.
- RL on Qwen3.5-0.8B (GSM8K, 200 steps, 3 seeds): 3.7 points behind full precision (the pre-registered line, −4,
  fails); most of the gap is the 6.5 points lost by rounding the start to 4 bits.

## Proposed extensions

| # | what | why it matters | cost |
|---|---|---|---|
| **E1** | ECO-AdamW and the AdamW float copy **from scratch** (45M, 0.9B tokens) | P1e is continued training at a small learning rate, where each step is far smaller than a 4-bit gap. From scratch the steps are larger, and ECO may do better there. | 1 seed each: about 35–45 units. 3 seeds each: about 100–130. |
| **E2a** | **GPTQ** on the pretrained GPT-2 (P1's starting point), against round-to-nearest (3.462) | Shows how much a careful post-training rounding recovers on this grid. | Free (runs on a CPU). |
| **E2b** | Train in float (float Muon, bf16 with random rounding) on P1, **then GPTQ**, against the ticker (3.2884) | The paper's "train in float, round at the end" baselines use round-to-nearest. GPTQ is the strong version. | About 3 units. |
| **E2c** | The same on P2 (from scratch, bf16 then GPTQ) | P2's margin (1.13) is against round-to-nearest of a small model; GPTQ will likely close much of it. | About 20 units. |
| **E3** | RL: a short healing run before RL, and/or 400 steps instead of 200 | Tests whether the 3.7-point gap is only the starting gap, and whether it keeps closing. | About 30–40 units. |
| **E4** | Scale: GPT-2 small from scratch (124M), or continued training of a 1B model | The main open question. | 150+ units, may need a larger GPU. |

Our own order, if all were wanted: E2a, E2b, E1 (one seed), then the rest.

## Questions for reviewers

1. Which of E1–E4 would change your view of the paper's main claim (4-bit weights trained with Muon, no full-precision
   copy, at about 2.8 bytes per weight)?
2. Is anything missing that is not listed here?

Contact: Malcolm Frei, malcolmf830@gmail.com. Code, pre-registration with its dated change log, and every run log:
https://github.com/Nibbles0n/masterless (to be made public before submission).
