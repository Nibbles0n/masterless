# P4: ablations (2026-10-28), looks only

GPT-2 small continued training, 25M tokens (763 steps × 32 × 1,024, full schedule), seed 1, Muon lr 0.001, A100, fp16,
TF32 off, deterministic. Each run is ours (`q4_t8_m8`) with one thing changed. Script: `p1_paper_gpt2.py` (the P1 script
plus the three ablation arms; the earlier arms' code is unchanged). Runs: `runs/paper/P4/`.

| run | final loss | minus ours | share of 4-bit weights changing per step | carry violations |
|---|---|---|---|---|
| ours (`q4_t8_m8`) | 3.30465 | | 0.133% | 0 |
| carry in absolute units (`_abs`): k/254 of the block's largest gap, not of the weight's own gap | 3.30588 | +0.0012 ± 0.0005 | 0.101% | 0 |
| weight rounded at random, not to nearest (`_srw`); the carry spans a whole gap | 3.37594 | **+0.0713 ± 0.0012** | **31.2%** | 0 |
| Error Diffusion's cap: each step moves a weight at most 0.25 of its gap (`_cap`) | 3.30426 | −0.0004 ± 0.0004 | 0.133% | 0 |

± is the paired standard error over the 256 validation windows (evaluation noise only, one seed).

## What it shows

- **Rounding the weight to nearest is the part that matters most.** With random rounding, a third of the weights flip
  every step, and the loss is 0.071 worse: about the cost of the 4-bit carry in P3. The carry still holds the exact
  leftover, but each flip changes the forward pass, so the model is noisy at every step.
- **Measuring the carry in units of each weight's own gap helps a little** (0.0012, about 2.7 errors). An absolute unit
  wastes resolution on the small values, where the gaps are 4 times smaller than the largest.
- **Error Diffusion's cap does nothing here.** Muon's steps at this rate are far below a quarter of a gap (about
  2.8e-5 against a mean gap of 0.039), so the cap never binds: the share of changing weights is the same to 3 digits.
  It may matter at much larger learning rates, which were not tested.

## Not tested

- More than one seed. These are looks, as planned, not pass lines.
- The cap at a learning rate large enough for it to bind.
