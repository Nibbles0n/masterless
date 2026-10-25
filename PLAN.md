# Masterless: the plan

**Question.** Can Muon train a model whose weights live *only* on a low-bit grid (4-bit, then ternary), with no
full-precision copy of the weights, by feeding the rounding error back into Muon's momentum?

**Why it is open (checked 2026-10-25).**
- ECO ([arXiv 2601.22101](https://arxiv.org/abs/2601.22101), ICML 2026) removes the full-precision copy by injecting the
  rounding error into momentum. It was tested with SGD-momentum and Adam only, at FP8 and INT4 weight-only fine-tuning.
  It does not test Muon, ternary or binary.
- MuonQ ([arXiv 2605.11396](https://arxiv.org/html/2605.11396v1)) and 4-bit-Muon-GRASP shrink Muon's *momentum*, not
  its weights. MuonQ found random rounding hurt 4-bit Muon: Muon keeps only an update's direction, so rounding noise
  does not average out.
- The new part: does Muon's orthogonalisation step scramble the fed-back error, and how do we fix it if it does?
  Re-check the literature before writing the paper.

**Why it matters.** A float copy plus momentum costs about 4–8 bytes per weight. Masterless Muon costs about 1.5–2.5:
the 4-bit weights plus one momentum. That is less memory per GPU, and it is step C3 of the SwitchNet swap ladder.

## Rules for every result

- One change per milestone. Each is judged against the one before.
- Fixed evaluation: validation cross-entropy (nats per character) on the same 256 windows for every run, with a
  standard error. Report the final step.
- 3 seeds for a claim; 1 seed is a look. Write the pass line down *before* the run.
- Honest negatives: a failed milestone is a finding. Record it, don't tune around it.

## Milestones

| # | Change | Pass line (written before the run) | Where |
|---|---|---|---|
| 0 | Baby GPT, characters of Shakespeare, AdamW, fp32/fp16 | CPU preset: mean val CE ≤ 1.95 over 3 seeds (nanoGPT reports about 1.88 for this setup). GPU preset: ≤ 1.52 (nanoGPT: 1.4697) | CPU here; Kaggle T4 for the GPU preset |
| 1 | Muon on the hidden matrices, AdamW on embeddings, head and norms | Muon's final CE at least 0.02 below AdamW's at the same steps, 3 of 3 seeds | Kaggle T4 |
| 2 | Fake 4-bit weights (values on a 4-bit grid with a scale per block of 16), float copy kept (C1) | Measure only: the gap to milestone 1. Below +0.10 means the format is usable | Kaggle T4 |
| 3 | No float copy: the weight is the grid value, each step rounds `w − lr·update` to the grid at random (C3a) | Measure only. Expected worse than milestone 2 (MuonQ) | Kaggle T4 |
| 4 | No float copy, rounding error fed back into Muon's momentum, ECO-style (C3b). Two arms: error added before or after the orthogonalisation step. ECO with Adam as the prior-art control | **The paper test:** the best arm is within 0.03 of milestone 2, 3 of 3 seeds | Kaggle T4 |
| 5 | Milestones 2–4 at ternary, and a second model size | The order of the arms holds at both sizes | Kaggle T4 |

**How the pass lines are read (added 2026-10-25 before the milestone 1 claim runs and before any milestone 2–4
run; the lines themselves are unchanged).** Milestones 1–4 run first at the CPU preset on this machine.
- Milestone 1: seed k of Muon is compared with seed k of milestone 0's AdamW; each of seeds 1, 2, 3 must be at least
  0.02 lower. The Muon learning rate is picked from {0.01, 0.02, 0.04} on seed 1 (tuning, not evidence).
- Milestone 2: gap = mean of milestone 2 minus mean of milestone 1 over seeds 1–3, at milestone 1's learning rate.
- Milestone 4: the "best Muon arm" is the Muon arm (error before or after the orthogonalisation) with the lower mean
  final CE. It passes if, for each of seeds 1, 2, 3, its CE is at most milestone 2's CE for the same seed + 0.03.
  The "after" arm keeps the error in an extra bf16 buffer (2 bytes per weight); its memory is reported beside it.

After milestone 4 passes: a real-text model (FineWeb-Edu, about 30M parameters) on a rented GPU, and the paper draft.
If milestone 4 fails: find out *why* (measure how much of the fed-back error survives the orthogonalisation step),
then decide.

## Status

- [x] Milestone 0, CPU preset (3 seeds): PASS, mean 1.915 ± 0.003 (`runs/m0/RESULT.md`)
- [ ] Milestone 0, GPU preset on Kaggle (1 seed is enough to confirm the setup)
- [x] Milestone 1, CPU preset (3 seeds): PASS, Muon 1.628 ± 0.003 vs AdamW 1.915, 0.28–0.30 lower in every seed
  (`runs/m1/RESULT.md`)
- [x] Milestone 2, CPU preset (3 seeds): usable, C1 1.646 ± 0.002, gap +0.019 to milestone 1 (`runs/m2/RESULT.md`)
- [x] Milestone 3, CPU preset (3 seeds): measured, C3a 1.764 ± 0.004, +0.117 worse than milestone 2, 4.75 bytes per
  weight against 8 (`runs/m3/RESULT.md`)
- [x] Milestone 4, CPU preset (3 seeds): **FAIL**. Best Muon arm (error added after Newton-Schulz, with a 2-byte error
  buffer) 1.673 ± 0.004, within 0.03 of milestone 2 in 1 of 3 seeds. Error fed into momentum before Newton-Schulz:
  1.898, worse than no feedback; only about half the error survives orthogonalisation (cosine 0.69). ECO with Adam:
  1.928 (`runs/m4/RESULT.md`)
