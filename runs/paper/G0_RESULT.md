# G0: checks before the paper runs (2026-10-28)

**Verdict: 4 of 6 checks pass. (a) passes for float Muon but fails for the 4-bit carry. (e) fails for the T4.**
- Diagnosis of (a): random-rounding noise, not a bug. The old script misses its own line by the same amount.
- (e) does not affect any result: every paper table runs on the A100.
- The pass lines are not moved. Whether to go on is the owner's decision.

Script: `p1_paper_gpt2.py`. Settings: side study G stage 2, seed 1 (muon lr 0.001, a 3,050-step schedule stopped at
step 300, 32 × 1,024 tokens per step), 256 validation windows. Results are in `runs/paper/G0/`.

| check | pass line | measured | result |
|---|---|---|---|
| (a) reproduction, T4 fp16, step 300 | within 0.005 of side study G | float Muon 3.29941 vs 3.29987 (0.0005); 4-bit carry `q4_t4_m8` 3.36396 vs 3.37368 (0.0097) | **float Muon PASS, 4-bit carry FAIL** |
| (b) limit test, A100 | Kahan (`q4_kahan_m32`) within 0.003 of the 4-bit float copy | 3.32609 vs 3.32653 (0.00044) | **PASS** |
| (c) carry in range, A100 | no leftover above half a gap; every weight on the grid | 0 violations (8-bit and 4-bit carry, every step); on the grid at every check. Also 0 over the full 3,050-step P1 run | **PASS** |
| (d) determinism, A100 | same seed twice, step-50 loss within 1e-4 | default mode: 3.5685 vs 3.5681 (4e-4). Strict deterministic mode: 3.5684 vs 3.5684, bit-identical (same change counts to all digits) | **PASS in strict mode**, which every paper run now uses |
| (e) speed | T4 ≥ 1.5 × 12.7k tokens/s; A100 tokens per unit measured | T4: 16.8k (4-bit carry, 1.32×), 17.0k (float Muon, 1.24× its old 13.7k). A100: 145k tokens/s, 134k in strict mode | **T4 FAIL**; A100 measured |
| (f) budget | projection ≤ 120 units | about 84 (below) | **PASS** |

## (a) diagnosis: the old script does not reproduce itself either

The old script (`side_gpt2.py`) was run again on a T4: the same seed 1, then seed 2. Seed 2 has the same data and
different random rounding. Validation loss at step 300:

| run | step 100 | step 200 | step 300 |
|---|---|---|---|
| side study G, the original | 3.39355 | 3.37882 | **3.37368** |
| old script, seed 1 again | 3.40783 | 3.36891 | **3.38150** |
| old script, seed 2 | 3.39659 | 3.38198 | **3.36695** |
| new script, T4 | 3.42149 | 3.37362 | **3.36396** |
| new script, A100 | 3.40590 | 3.38357 | **3.36422** |

- **The same old code and seed land 0.0078 apart.** The GPU sums in a different order on each run, and with a 4-bit
  carry each small change flips rounding decisions.
- **The old script's three runs spread over 3.367–3.382.** The new script's 3.364 is 0.003 below that range.
- **The new script draws its random rounding on the GPU, the old one on the CPU.** So the two never take the same path.
- **New T4 and new A100 agree within 0.0003 at step 300**, but differ by 0.016 at step 100. That is the same kind of
  swing.
- **Float Muon has no random rounding, and it reproduces within 0.0005.**

Conclusion: the 0.005 line was too tight for a randomly rounded 4-bit carry. The float arm and checks (b)–(d) show
the training code is right.

## Two GPU problems found and fixed during G0

- **bf16 on the A100.** The first A100 runs used bf16 arithmetic. Each check is now fp16 with loss scaling on every GPU
  (`--amp fp16`, the default), as in side study G.
- **TF32 on the A100.** With TF32 on, compiled fp16 matrix products on the raw weights raised the validation loss by
  0.014. That would hit ours, Kahan and bf16-SR, but not the float-copy arms. A grid of compile × TF32 × fp16/fp32 ×
  4-bit on both GPUs found it: every other combination agreed within 0.0005. TF32 is now off.
- The first A100 runs, made with these settings, are not used.

## Prices and tokens per unit

- **A100 40GB:** 5.30 units per hour (`colab usage` with and without the session). Strict mode: 134k tokens/s, about
  91M tokens per unit. A full P1 run (100M tokens plus 31 checks) took 770–800 s, about 1.15 units.
- **T4:** 1.07 units per hour, 16.8k tokens/s, about 57M tokens per unit.

So every P1–P4 run goes on the A100. It is 1.6× cheaper per token.

**Projection (f).**
- Spent so far: 9.6 units, including the helpers' T4 sessions.
- Still to come: P1 12 runs (14), P3 (about 3.5), P4 (about 1), P2 (about 26, at a guessed 3× speed for the 45M
  model), P5 (about 30).
- Total about 84 units, under the 120 line.

## Not tested

- Strict deterministic mode on the T4.
- The 6-, 12- and 16-bit carries on a GPU (on the CPU only, in a smoke test with a tiny model).
- ECO-Muon, naive and bf16-SR beyond the CPU smoke test.
- The from-scratch mode on a GPU.
