# P6: memory, measured (T4)

**P6a: PASS.** Ours peaks at 3.74 GiB for one full-parameter training step of Qwen3.5-0.8B. The standard setup
(bf16 + AdamW with an fp32 master copy) peaks at 11.74 GiB. 3.74 ÷ 11.74 = **0.318**, and the pass line was 0.35 or
less.

Script: `p6_memory.py`, using the release optimizer `release/ticker_muon.py`. Raw records are in
`runs/paper/p6_raw.jsonl` and the summary in `runs/paper/P6_memory.json`. Run on one Colab Tesla T4 (15 GB) with
torch 2.11.0, transformers 5.20.2 and no flash-linear-attention, so the Gated DeltaNet layers use transformers' plain
PyTorch fallback.

## Setup, the same for every row

- **Model.** `Qwen/Qwen3.5-0.8B`, text only (`AutoModelForCausalLM` loads `Qwen3_5ForCausalLM` and drops the vision
  tower and the multi-token-prediction layer). It has 752.4M parameters:
  - 497.6M in the 186 decoder matrices, where Muon is used;
  - 254.8M elsewhere, almost all of it the 248k × 1024 embedding, which is shared with the output layer.
- **Batch.** 2 sequences × 1,024 tokens of random token ids from a fixed seed.
- **Activation checkpointing** on every decoder layer.
- **Chunked loss.** The output layer and loss run 256 tokens at a time under checkpointing, so the 248k-wide scores
  are never made all at once.
- **Number format.** Compute is in bf16. Newton–Schulz runs in fp16, because a T4 has no bf16 tensor cores.
- **Steps.** 3 training steps per setup. Peak is `torch.cuda.max_memory_allocated` over a step, after a first step.
  Steps 1–3 gave the same peak to within 4 KB.
- **Steady state** is the memory held between steps: the weights and the optimizer state, with no gradients.

| setup | what is stored | peak (GiB) | steady state (GiB) | peak ÷ (a) | seconds per step |
|---|---|---|---|---|---|
| (a) standard | bf16 weights, fp32 master copy, fp32 AdamW moments for every parameter; bf16 gradients | **11.74** | 9.83 | 1.00 | 6.3 |
| (b) float Muon | fp32 weights and gradients; fp32 Muon momentum on the matrices; fp32 AdamW on the rest | 10.33 | 6.56 | 0.88 | 7.3 |
| (c) bf16-SR | bf16 weights, no copy, randomly rounded; 8-bit Muon momentum; rest bf16 + 8-bit AdamW; full gradients, then step | 4.31 | 2.40 | 0.37 | 11.5 |
| (c′) bf16-SR, layer by layer | as (c), but each matrix is updated during the backward pass and its gradient freed at once | 3.96 | 2.40 | 0.34 | 12.2 |
| **(d) ours** | 4-bit codes (2 per byte), FP8 block scales, int8 carry, int8 momentum, updated during the backward pass; rest as (c) | **3.74** | **2.18** | **0.318** | 13.5 |

The steady-state numbers match a byte count:

| setup | bytes per parameter | total |
|---|---|---|
| (a) | 14 for all 752M | 9.81 GiB |
| (b) | 8 for the matrices, 12 for the rest | 6.55 GiB |
| (c) | 3.03 for the matrices, 4.03 for the rest | 2.37 GiB |
| (d) | 2.59 for the matrices, 4.03 for the rest | 2.17 GiB |

**Measured only.** Ours ÷ bf16-SR:
- **0.87** at the peak against (c), which keeps the full gradients until the step;
- **0.95** at the peak against (c′), which uses the same layer-by-layer trick as ours (the fair comparison);
- **0.91** in steady state, close to the expected 0.9.

The decoder matrices alone shrink by 2.59 ÷ 3.03 = 0.86. The embedding is a third of this model and is stored the same
way in both, which pulls the whole-model ratio towards 1. About 1.6 GiB of each peak is not weights or optimizer
state, and it is the same in every setup: the embedding's gradient, the loss chunks, the recomputed layer and the 8-bit
AdamW's working chunks. Most of the saving against (a) therefore comes from dropping the master copy and the 32-bit
optimizer state. Against bf16 weights with random rounding, the saving is small.

**Time the update adds.** Ours takes 13.5 s per step. The same setup with the 4-bit matrices frozen (no weight
gradient, no update) takes 8.6 s. So the weight gradients and the ticker updates add **4.8 s per step**, and 4.1 s of
that is inside the update itself, timed with CUDA events. For comparison, bf16-SR's Muon step takes about 3.3 s (4.9 s
after the backward pass, minus 1.66 s for the 8-bit AdamW).

The update code is plain PyTorch:
- no `torch.compile`;
- no fused kernels;
- the random numbers come from an integer hash in int64;
- Newton–Schulz runs in fp16 on a T4.

The times are a ranking on this setup, not a measure of what an optimised kernel would do. Setup (a) is fastest
because its step is a few elementwise passes. The forward and backward passes cost about the same in every setup
(6.0–7.0 s), so rebuilding the 4-bit weights adds about 0.4 s.

## Demo, not evidence: Qwen3.5-2B on one T4

Qwen3.5-2B, text only, has 1.88B parameters, 1.37B of them in the decoder matrices. It was run with the same batch,
sequence length, checkpointing and loss.

- **Ours fits.** A full-parameter step (every decoder matrix in the ticker format, the embedding and the small
  parameters with 8-bit AdamW) peaks at **8.34 GiB**, with 5.25 GiB of steady state and 32.9 s per step. The loss fell
  over the 3 steps (14.73 → 13.81 → 13.46, on random tokens), and every layer updated exactly once per step.
- **Setup (a) does not fit.** It ran out of memory at 14.4 GiB while setting up its fp32 master copy and moments.
  By the byte count it needs about 1.88B × 14 bytes = 24.5 GiB before activations.

## What was not tested

- **Training quality.** These are memory and time measurements of 3 steps on random tokens. Nothing here says the
  model learns well. The losses only show that each setup's step changes the model.
- **One batch shape** (2 × 1,024) and one GPU type. The share of memory that is not weights or state grows with the
  batch.
- **Other memory savers.** No fused output-layer-and-loss kernel and no CPU offload. The baselines use the same
  chunked loss and checkpointing, and their optimizers process 16M values at a time, so their peaks are about as low as
  plain PyTorch gets.
- **Timing caveats.**
  - Without flash-linear-attention, the Gated DeltaNet layers are slower than they would be in practice, which inflates
    every step time equally.
  - The step times are means of steps 2–3 and are not tuned.
- **8-bit AdamW is our own.** It is a simple implementation in `release/ticker_muon.py`, not bitsandbytes, so a
  bitsandbytes-based baseline could differ slightly in memory and speed.
- **The release code was checked separately.** Its packed format matches a plain dense reference exactly in
  `release/test_ticker_muon.py`, which passes on this machine's CPU and on the Colab VM. No convergence run of the
  packed path was made here.
