# Side study L: 10,000 steps (5x the CPU preset), measure only

**What was run.** The committed milestone scripts, unchanged, at 10,000 steps instead of 2000 (runner:
`long_runs.py`). Same model, data and 256 validation windows. Run on a free Colab T4 with 16-bit (fp16) arithmetic,
so the numbers are not comparable digit for digit with the CPU runs. 10,000 steps is 7.7 million characters, about
7.6 passes over the training text.

**Learning rates (seed 1, tuning, not evidence).** AdamW: 0.001 → 1.583, **0.002 → 1.558**, 0.004 → 1.570. Muon:
**0.005 → 1.517**, 0.01 → 1.532, 0.02 → 1.521. Muon's best was the smallest value tried. As fixed in advance, m3 used
Muon's (0.005) and ECO-Adam used AdamW's (0.002).

| arm | seed 1 | seed 2 | seed 3 | mean ± standard error |
|---|---|---|---|---|
| AdamW, lr 0.002 | 1.5584 | 1.5463 | 1.5509 | 1.5519 ± 0.0035 |
| Muon, lr 0.005 | 1.5172 | 1.5210 | 1.5164 | 1.5182 ± 0.0014 |
| m3 (4-bit, no copy), lr 0.005 | 1.6813 | 1.6842 | 1.6882 | 1.6845 ± 0.0020 |
| ECO-Adam (4-bit, no copy), lr 0.002 | 1.5720 | 1.5672 | 1.5800 | 1.5731 ± 0.0037 |

No arm overfits: every arm's best validation loss is at step 9,750 or 10,000.

**The five questions written before the runs.**
1. AdamW − Muon gap smaller than 0.29 at 2000 steps? **Yes: +0.034 ± 0.005** (+0.041, +0.025, +0.035). Most of
   Muon's lead at 2000 steps was speed.
2. m3 still below AdamW in 3 of 3 seeds? **No, 0 of 3.** m3 − AdamW = +0.123, +0.138, +0.137. m3 is below AdamW
   (3-seed mean) only until step 3,000.
3. m3 − Muon gap larger or smaller than +0.136? **Larger: +0.166 ± 0.003.**
4. First step at which the 3-seed mean reaches 2.0 / 1.8 / 1.7 (evaluated every 250 steps):

   | arm | 2.0 | 1.8 | 1.7 |
   |---|---|---|---|
   | AdamW | 1,250 | 3,000 | 4,750 |
   | Muon | 750 | 1,500 | 2,500 |
   | m3 | 1,000 | 2,750 | 7,250 |
   | ECO-Adam | 1,250 | 3,250 | 5,250 |

5. ECO-Adam within 0.03 of AdamW? **Yes: +0.021 ± 0.004**, all three seeds (+0.014, +0.021, +0.029).

**Extra, decided after seeing the results (seed 1 only, tuning, not evidence).** m3 at higher learning rates, to
test whether bigger steps (relative to the grid gap) help: lr 0.01 → 1.741 (worse than 1.681 at 0.005); lr 0.02
stalled near 2.0, then diverged (loss NaN from step 2,750; the script's final on-grid check then failed and no .json
summary was written). Full-precision Muon was stable at 0.02.

**Reading.** m3 descends faster than AdamW early but levels off about 0.13 higher, and its gap to Muon grows with
training length. This fits random rounding's noise growing with the total distance the weights travel.

**Not tested.** Single-pass training (no character seen twice), which is what "descent speed" means for large-scale
training; this text is repeated 7.6 times. More than one seed for the learning-rate choices. The GPU-preset model.
Whether the lr 0.02 divergence is caused by fp16 arithmetic or by the 4-bit rounding itself.
