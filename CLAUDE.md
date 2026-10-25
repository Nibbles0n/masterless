# Masterless working rules

A small PyTorch research repo: Muon without a full-precision weight copy, for low-bit training. The plan and
milestones are in `PLAN.md`; work on the first unchecked milestone only.

- **One file per milestone** (`m0_baby_gpt.py`, `m1_...`). Copy the previous one and change one thing. Keep files
  self-contained so they run on Kaggle with a single command. No framework, no package.
- **Pass lines go into `PLAN.md` before the run** and are never moved after seeing a number.
- **Compute:** this machine's CPU for smoke tests and the CPU preset; free Kaggle or Colab T4s for everything else.
  No paid GPUs without the owner's yes.
- **Results:** `runs/<milestone>/*.json` are committed; checkpoints and `data/` are not. Each milestone ends with a
  short note in `runs/<milestone>/RESULT.md`: the numbers (mean ± standard error over seeds), pass or fail, and what was
  not tested.
- **Environment:** `~/.venvs/masterless/bin/python`.
- **Commit only when asked.**
- **Explain to the owner in plain words** (grade-12 maths, no unexplained acronyms).
