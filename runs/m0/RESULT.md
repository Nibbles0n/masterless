# Milestone 0, CPU preset: PASS (2026-10-25)

Baby GPT (4 layers, width 128, context 64, 0.80M parameters), AdamW, fp32, 2000 steps of 12 × 64 characters (1.5M
characters seen), on this machine's CPU (2 threads per run, about 2 minutes each).

| Seed | Final val CE (nats/char) | SE over 256 windows |
|---|---|---|
| 1 | 1.9195 | 0.0144 |
| 2 | 1.9171 | 0.0145 |
| 3 | 1.9084 | 0.0152 |
| **Mean** | **1.9150 ± 0.0034** (SE over seeds) | |

Pass line ≤ 1.95: **PASS**. nanoGPT's README reports about 1.88 for this setup; our evaluation uses fixed windows and
a slightly different schedule, so a gap of a few hundredths is expected.

Not tested: the GPU preset (Kaggle T4), fp16 on CUDA, any other optimizer.
