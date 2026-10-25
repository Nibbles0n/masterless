# Masterless

Muon without a master copy: quantization-aware training where the weights live only on a low-bit grid.
The question and milestones are in [PLAN.md](PLAN.md).

## Run locally (CPU)

```bash
uv venv ~/.venvs/masterless && uv pip install --python ~/.venvs/masterless/bin/python torch numpy
~/.venvs/masterless/bin/python m0_baby_gpt.py --preset cpu --seed 1
```

The CPU preset (0.8M parameters, 2000 steps) takes a few minutes. The first run downloads tiny Shakespeare (1 MB)
into `data/`. Results go to `runs/m0/`: a `.jsonl` line per evaluation and a `.json` summary.

## Run on Kaggle (free T4)

1. New notebook. Settings: Accelerator **GPU T4**, Internet **on**.
2. In a cell, get the code (push this repo to GitHub first, or upload `m0_baby_gpt.py` as a dataset):
   `!git clone https://github.com/<you>/masterless && cd masterless && python m0_baby_gpt.py --preset gpu --seed 1`
3. Download `runs/m0/*.json` from the output panel and put them in `runs/m0/` here.

Kaggle gives about 30 GPU hours a week. Colab's free T4 works the same way. A T4 has no bf16, so the script uses fp16
with loss scaling on CUDA automatically.
