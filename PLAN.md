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
| 5 | No float copy, random rounding as in milestone 3, but each step's rounding error is spread over the rest of the matrix so the layer's *output* changes as Muon intended (noise shaping, as GPTQ does once after training) | Each of seeds 1, 2, 3 at least 0.02 below milestone 3's CE for the same seed (written 2026-10-25 after the offline check, before any run; the check predicts FAIL) | CPU here |
| 6 | Muon reordered: orthogonalise each gradient first, then average (`A = 0.95·A + 0.05·NS(G)`, step `lr·A`). The step is now a plain multiple of the momentum, so the momentum can hold the rounding leftover exactly (ECO's rule for momentum SGD). No float copy, nearest rounding: a weight moves one grid level once its pushes add up past half a gap. Arms: `float` (reordered Muon, float weights), `q4` (4-bit, leftover in the momentum), `q4_m8` (as `q4`, momentum stored in 8 bits) | `q4`: each of seeds 1, 2, 3 at most milestone 2's CE for the same seed + 0.03 (the milestone 4 paper test). `q4_m8`: the same line. `float`: each seed at most milestone 1's + 0.03 | Colab T4 |
| 7 | Small state at about 2 bytes per weight, no float copy. Arm `q4_b4_m8`: milestone 4's `muon_post` (4-bit weights, random rounding, the rounding leftover kept in a buffer and added after Newton-Schulz) with the buffer stored in 4 bits and the momentum in 8 bits. Arm `fp8_m8`: FP8 (E4M3) weights, random rounding, no buffer, momentum in 8 bits. Third evaluation: `fp8_m8`'s final weights rounded to 4 bits after training (`fp8_m8→fp4`) | `q4_b4_m8`: each of seeds 1, 2, 3 at most milestone 2's CE for the same seed + 0.03 (the milestone 4 paper test). `fp8_m8`: each seed at most milestone 1's + 0.03. `fp8_m8→fp4`: measure only, compared with `q4_b4_m8` seed by seed | Colab T4 |
| 8 | Milestones 2–7 at ternary, and a second model size. **Only after a no-copy 4-bit arm without a hidden float copy passes the milestone 4 line** (owner's decision, 2026-10-25) | The order of the arms holds at both sizes | T4 |

**How the pass lines are read (added 2026-10-25 before the milestone 1 claim runs and before any milestone 2–4
run; the lines themselves are unchanged).** Milestones 1–4 run first at the CPU preset on this machine.
- Milestone 1: seed k of Muon is compared with seed k of milestone 0's AdamW; each of seeds 1, 2, 3 must be at least
  0.02 lower. The Muon learning rate is picked from {0.01, 0.02, 0.04} on seed 1 (tuning, not evidence).
- Milestone 2: gap = mean of milestone 2 minus mean of milestone 1 over seeds 1–3, at milestone 1's learning rate.
- Milestone 4: the "best Muon arm" is the Muon arm (error before or after the orthogonalisation) with the lower mean
  final CE. It passes if, for each of seeds 1, 2, 3, its CE is at most milestone 2's CE for the same seed + 0.03.
  The "after" arm keeps the error in an extra bf16 buffer (2 bytes per weight); its memory is reported beside it.
- Milestone 5 (written 2026-10-25, before any milestone 5 run): the arm is the 16 x 16 block version
  (`m5_spread.py`), the only one cheap enough to keep online (0.29 extra bytes per weight, about 0.1 s per step on
  this CPU). Same Muon learning rate as milestone 3 (0.01). Offline check (`report/spread_probe.py`, m3 seed 1): every
  spreading method put more noise on the layers' outputs than plain random rounding at steps 250, 1000 and 1900
  (block version 1.28x, 1.54x, 1.46x; full GPTQ 1.22x, 3.25x, 5.59x), because without a float copy the target sits
  within a tiny step of a grid point, where random rounding is already nearly exact, and spread error pushes later
  weights to the middle of a gap. So the run is a check that the offline measure predicts training, not a hope.

- Milestone 6 (written 2026-10-25, before any milestone 6 run; the owner asked for this milestone, and the old
  milestone 6 is now 7). File `m6_accum.py`, a copy of milestone 3. All runs on one Colab T4 session with fp16
  autocast, with milestones 1, 2, 3 and milestone 4's `muon_post` arm rerun there on seeds 1-3 as the baselines (so
  the comparisons stay within one machine). Momentum 0.95, plain (heavy-ball) momentum, no Nesterov. The leftover
  `e = target − stored` is added to the momentum as `(1 − 1/0.95) · e / lr`. Averaging already-orthogonalised steps
  gives a smaller step than Muon's, so the learning rate is picked for `float` on seed 1 from
  {0.01, 0.02, 0.04, 0.08, 0.16} (tuning, not evidence); `q4` and `q4_m8` use the same rate, untuned. `q4_m8` stores
  the momentum as 8-bit integers with one fp32 scale per 128 weights along a row, rounded at random so small
  leftovers are not lost. Memory per weight: `q4` 4.75 bytes (as milestone 3), `q4_m8` 1.78 (1.59 with FP8 block
  scales). Also reported, without a pass line: `q4` against milestone 3 and against milestone 4's `muon_post`
  (which needs an extra 2-byte buffer for the same job), seed-matched.

- Milestone 7 (written 2026-10-26, before any milestone 7 run; the owner asked for it, and the old milestone 7 is
  now 8). File `m7_small_state.py`, a copy of milestone 4 with the `muon_post` arm only, one Colab T4 session, fp16
  autocast, seeds 1-3, Muon learning rate 0.01 for both arms (milestone 1's, untuned, as for milestones 3-5).
  Baselines: milestones 1 and 2 from milestone 6's T4 runs (`runs/m6/base_m1`, `base_m2`; a new T4 session
  reproduced them to all digits in study B2), and milestone 4's `muon_post` from `runs/m6/base_m4post`.
  - `q4_b4_m8`: the leftover buffer is stored as signed 4-bit integers (-7..7) times one fp32 scale per 16 weights
    (the block's largest leftover / 7), rounded at random so small leftovers are not lost on average. The momentum
    is stored as in milestone 6's `q4_m8` (8-bit integers, one fp32 scale per 128 weights of a row, random
    rounding). Weights and their scales as in milestone 3.
  - `fp8_m8`: each block of 16 weights is scaled so its largest weight lands on 448, E4M3's largest value, and the
    weight is rounded at random to its two E4M3 neighbours. Momentum as in `q4_b4_m8`.
  - `fp8_m8→fp4`: after training (and at every evaluation, for the curve), the Muon matrices are copied and rounded
    to nearest on milestone 3's 4-bit grid, the usual one-shot rounding after training; nothing else changes.
  - Memory per weight of the hidden matrices, with fp32 scales: `q4_b4_m8` 0.5 + 0.25 + 0.5 + 0.25 + 1 + 0.03 =
    2.53 bytes; `fp8_m8` 1 + 0.25 + 1 + 0.03 = 2.28 (2.16 and 2.09 with FP8 block scales). Both well below the
    8 bytes of milestones 1-2 and the 6.75 of milestone 4's `muon_post`.
  - Also reported, without a pass line: `q4_b4_m8` against milestone 4's `muon_post` (what shrinking the buffer and
    momentum cost), and `fp8_m8→fp4` against `fp8_m8` (what the rounding after training costs).

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
- [x] Side study F, FP8 block scales (3 seeds): +0.008 ± 0.007 against milestone 3, about the seed spread
  (`runs/ablate_fp8/RESULT.md`)
- [x] Side study L, 10,000 steps on a T4 (3 seeds): AdamW 1.552, Muon 1.518, m3 1.685, ECO-Adam 1.573. Muon's lead
  over AdamW shrinks to 0.034; m3 falls behind AdamW after about step 3,000 (`runs/long/RESULT.md`)
- [x] Milestone 6, on a T4 (3 seeds): **FAIL**. Reordered Muon, 4-bit with the leftover in the momentum: 2.047 ± 0.009,
  +0.40 behind milestone 2 and +0.29 behind milestone 3; 8-bit momentum the same (2.048). With float weights the
  reordering alone costs +0.069 against Muon. Cause: the momentum releases a leftover over about 20 steps, so a weight
  moves only after about 10 gaps of wanted movement have piled up (`runs/m6/RESULT.md`)
- [x] Milestone 7, on a T4 (3 seeds): 4-bit weights + 4-bit leftover buffer + 8-bit momentum (2.53 bytes) **FAIL**, 1.712 ± 0.002, +0.066 behind milestone 2 (but 0.043 ahead of milestone 3). FP8 weights + 8-bit momentum (2.28 bytes) **PASS**, 1.652 ± 0.003, +0.021 behind milestone 1 and level with milestone 2. The FP8 model rounded to 4 bits after training: 1.672 ± 0.004, 0.040 better than the 4-bit-trained model in every seed (`runs/m7/RESULT.md`)
- [x] Side study T, the ticker, on a T4 (3 seeds, measure only): 4-bit weights, nearest rounding with a 4-bit carry in units of its own gap, 8-bit momentum (2.28 bytes): 1.650 ± 0.005, level with milestone 2 (+0.004 ± 0.007, within 0.03 in 3 of 3 seeds), 0.022 better than FP8 rounded to 4 bits after training and 0.062 better than milestone 7's 4-bit buffer, in every seed. Reference only, not yet the paper test (`runs/ticker/RESULT.md`)
- [x] Side study 2B, the ticker at 2 bits, on a T4 (3 seeds): FP8 carry (2.53 bytes) **PASS**, 1.748 ± 0.006, +0.022 behind 2-bit with a float copy (1.725); 4-bit carry (2.03 bytes) **FAIL**, 1.803 ± 0.004, +0.077, falling behind after step 1000. The FP8 carry beats the 4-bit carry by 0.055 in every seed. 2 bits with a float copy costs +0.080 against milestone 2; AdamW at 2 bits is 0.22 worse than Muon; FP8 rounded to 2 bits after training is 2.149, far behind (`runs/twobit/RESULT.md`)
- [x] Side study G, GPT-2 small (124M, pretrained) on Colab T4s (1 seed, paired SE over 256 windows): both 2-byte
  systems stable (no divergence, every run improves and ends below pretrained float 3.402). Close: 25M tokens FP8
  **PASS** +0.0142 ± 0.0004 vs float Muon, ticker **FAIL** +0.057 ± 0.001 vs 4-bit + float copy; 100M tokens FP8 **FAIL**
  +0.035 ± 0.001 (ended worse than at 25M, 3.302 vs 3.296; stalled at the peak learning rate; gap growth +0.021 against
  the 0.02 line, FAIL), ticker **FAIL** +0.047 ± 0.001 (gap shrinking). Stage 3 (measure only): float Muon rounded
  to FP8 at the end costs +0.003, so training in FP8 loses +0.032 to per-step random rounding; in 4 bits the ticker
  is level with float Muon rounded at the end (+0.003 ± 0.001), FP8 rounded to 4 bits +0.039 behind. The ticker's flat
  change rate is a carry flaw (6.6% of weights parked on the rounding line), not the learning rate; an 8-bit carry
  fixes it in a probe, not yet run on GPT-2 (`runs/gpt2/RESULT.md`)
- [x] Side study B, batch size 1x-8x on a T4 (3 seeds): the m3 - m1 gap falls from +0.123 to +0.053, but only
  because bigger batches got less far on the same data; at equal progress the gap is about the same
  (`runs/batch/RESULT.md`)
- [x] Side study B2, bigger learning rates with bigger batches on a T4 (3 seeds, 68 runs): at a fixed batch a bigger
  rate makes the m3 - m1 gap bigger, not smaller. The best m3 per batch is about the same (1.745-1.759 up to 4x, in
  2-4 times fewer steps). A batch-matched rate trims the gap at equal progress by about a fifth (`runs/batch_lr/RESULT.md`)
- [x] Milestone 5, CPU preset (3 seeds): **FAIL**, as the offline check predicted. Rounding error spread inside each
  block of 16: 1.852 ± 0.005, +0.088 worse than milestone 3 in every seed (`runs/m5/RESULT.md`)

## Side studies (not milestones; measure only)

**L. Longer training (written 2026-10-25, before any long run).** The milestone arms at 10,000 steps (5x the CPU
preset): same model, data and 256 windows, run on a Colab T4 with fp16 autocast (so not comparable digit for digit with
the CPU runs). Runner: `long_runs.py`. Learning rates are picked on seed 1 (tuning, not evidence) from AdamW
{1e-3, 2e-3, 4e-3} and Muon {0.005, 0.01, 0.02}; milestone 3 uses Muon's, ECO with Adam uses AdamW's. Then seeds 1-3
of AdamW, Muon, milestone 3 and ECO with Adam. Questions fixed now:
1. Floors: is the AdamW - Muon gap at 10,000 steps smaller than the 0.29 at 2,000 steps?
2. Is milestone 3 still below AdamW at 10,000 steps, in 3 of 3 seeds?
3. Is the milestone 3 - Muon gap at 10,000 steps larger or smaller than the +0.136 at 2,000 steps?
4. Speed: the first step at which each arm's 3-seed mean reaches val CE 2.0, 1.8 and 1.7.
5. Does ECO with Adam stay within 0.03 of AdamW?

**F. FP8 block scales (written 2026-10-25, before the run).** Milestone 3 recomputes each block's fp32 scale from the
unrounded weights, so the largest weight of every block lands exactly on ±6 and trains at full precision (1 weight in
16). Ablation: milestone 3 with block scales stored in FP8 (E4M3) under one fp32 scale per matrix, as NVFP4 does;
3 seeds, CPU preset. Measure only: the change against milestone 3 (expected: worse).

**B. Batch size (written 2026-10-25, before any run).** Theory from milestone 5: without a float copy, random
rounding's noise grows with the total distance the weights travel. Bigger batches give fewer, less zig-zagging steps
for the same data, so they should cut the noise. Runs: milestone 1 (Muon, float) and milestone 3 (4-bit, no copy) at
2x, 4x and 8x the CPU preset's batch of 12, with the same number of characters (steps, warm-up and eval interval
divided by the same factor), the same learning rates (Muon 0.01, AdamW parts 1e-3), seeds 1-3, all on a Colab T4
with fp16 autocast, including 1x (so every number comes from the same machine; moved from CPU before any result was
seen, for speed). Runner: `batch_runs.py`. Measure only. Questions fixed now:
1. The m3 - m1 gap at each batch size (mean ± standard error, seed-matched). Does it shrink as the batch grows?
2. m3's own final loss at each batch size, same characters: does a bigger batch help it even though it takes fewer
   steps?
Not a fair test of large batches in general: the learning rate is not retuned per batch size.

**T. The ticker (written 2026-10-26, before any run; the owner asked for it).** Milestone 7's `q4_b4_m8` with the
owner's tick rule instead of random rounding: target = stored weight + carry − lr·update, the new weight is the
nearest 4-bit value to the target (so a weight ticks once its carry passes half the gap on that side, and the
threshold differs per weight because the gaps differ), and the new carry is target − new weight. The carry is
stored in units of the gap on its own side (always between −1/2 and +1/2), as 4-bit integers −7..7 standing for
k/14 of that gap, rounded at random; it needs no scale of its own. Momentum in 8 bits as in milestone 7. File
`side_ticker.py`, a copy of `m7_small_state.py` with one arm `q4_t4_m8`. Colab T4, fp16, seeds 1-3, Muon learning
rate 0.01. Memory 0.5 + 0.25 + 0.5 + 1 + 0.03 = 2.28 bytes per weight, the same as `fp8_m8`. Measure only.
Questions fixed now, all seed-matched:
1. Against milestone 7's `q4_b4_m8` (1.712): does the tick rule with a gap-unit carry do better?
2. Against `fp8_m8→fp4` (1.672): does training in 4 bits with a ticker beat training in FP8 and rounding after?
3. Is it within 0.03 of milestone 2 in each seed (the milestone 4 line, for reference only; no claim)?

**2B. The ticker at 2 bits (written 2026-10-26, before any run; the owner asked for it).** How low can the ticker
go? File `side_2bit.py`, a copy of `side_ticker.py`; CPU preset on a Colab T4, fp16, seeds 1-3, Muon learning rate
0.01, untuned. The 2-bit grid: the four values −1.5, −0.5, +0.5, +1.5 times one fp32 scale per block of 16, scale =
1.25 × the block's mean absolute weight (about 30% less rounding error than absmax/1.5 on bell-shaped blocks; not
ternary, which stays milestone 8). Five arms (bytes per weight of the hidden matrices, fp32 scales):
- `q2_t4_m8`: 2-bit weights, the ticker with a 4-bit carry (k/14 of the gap), 8-bit momentum. 2.03 bytes.
- `q2_t8_m8`: the same with an 8-bit carry, 512 × (carry in gaps) stored as FP8 E4M3, random rounding. 2.53 bytes.
- `q2_ste_muon`: 2-bit forward weights with a float copy (straight-through), Muon, fp32 momentum (m2 at 2 bits). 8.
- `q2_ste_adam`: the same with AdamW on every parameter, as milestone 0 (lr 1e-3). 12.
- `fp8_m8`: milestone 7's arm re-run, also evaluated after rounding to the 2-bit grid at every eval. 2.28.

Pass line, fixed now: a ticker arm passes if it is within 0.05 of `q2_ste_muon` in each seed (seed-matched).
Measure only, all seed-matched:
1. FP8 carry against 4-bit carry: called better only if better in all 3 seeds.
2. How far the 2-bit straight-through arms fall behind milestone 2 (4-bit with a float copy, 1.646), and Muon
   against AdamW at 2 bits.
3. `fp8_m8→2-bit` (train in FP8, round after) against the two ticker arms.
Code check: `fp8_m8` should repeat milestone 7's numbers (within 0.002 in each seed).

**G. The two 2-byte systems at GPT-2 small scale (written 2026-10-26, before any stage-1 run; the owner asked for
it).** Do FP8 weights + 8-bit momentum (`fp8_m8`, milestone 7) and the ticker (`q4_t4_m8`, side study T) stay stable
on a real model, and do they keep improving a model that is already good? File `side_gpt2.py`: the pretrained OpenAI
GPT-2 small (124M parameters) continues training on FineWeb text (GPT-2 tokens, the modded-nanogpt shards; validation
on 256 fixed windows of 1024 tokens from the separate val shard, the same windows for every arm). Muon gets the 48
matrices inside the blocks (84.9M weights, 68% of the parameters); AdamW, in float, the embeddings (tied to the
output head), biases and LayerNorms, learning rate 1e-4 in every arm. At step 0 the pretrained weights are rounded
to nearest onto the arm's grid; the ticker's carry starts as that rounding's leftover. Every arm sees the same
batches (fixed data seed); seed 1 only, so differences between arms get a paired standard error over the 256
windows. Colab T4s, fp16 autocast, `torch.compile`; 32 sequences of 1024 tokens per step (32,768 tokens), 5% warm-up,
cosine down to 10%. Arms (bytes per weight of the block matrices, weights + optimizer state):
- `adamw_float`: AdamW on everything (12 bytes), learning rate 1e-4 (best of 3e-5, 1e-4, 3e-4 in a 200-step check:
  3.373, 3.370, 3.381).
- `muon_float`: Muon in float (8 bytes), learning rate 0.001, the best in a 200-step check (0.01: 3.52 at step 100; 0.003: 3.381; 0.001: 3.371; 0.0003, added
  because 0.001 was at the edge: 3.380).
- `q4_float_copy`: 4-bit forward weights with a float copy (milestone 2; 8 bytes), Muon at 0.001.
- `fp8_m8` (2.28 bytes) and `q4_t4_m8` (2.28 bytes): Muon at 0.001, untuned.
Pilot (30 steps): rounding the pretrained model costs +0.003 (FP8) and +0.058 (4-bit) on 64 windows; speed about
12k tokens per second on a T4 for every Muon arm.

Stage 1: 760 steps (24.9M tokens), all five arms, about 40 minutes per arm. Pass lines, fixed now, for each of the
two 2-byte arms:
1. Stable: no loss that is not a number, and its largest rise of validation loss above its own best so far is at
   most `muon_float`'s + 0.05.
2. Improves: its final validation loss is below its own step-0 loss (after rounding).
3. Close: `fp8_m8` ≤ `muon_float` + 0.03; `q4_t4_m8` ≤ `q4_float_copy` + 0.03 (the milestone 4 line at this scale).
Measure only: `q4_t4_m8` against `muon_float`; each arm against the pretrained float model's loss (does a 4-bit
model end better than float GPT-2 started?); `muon_float` and `fp8_m8` rounded to 4 bits after training against
`q4_t4_m8`; the share of weights whose grid value changes per step; skipped (overflow) steps.

Stage 2, run if neither 2-byte arm fails line 1 in stage 1: 3,050 steps (100M tokens), `muon_float`,
`q4_float_copy`, `fp8_m8`, `q4_t4_m8`, about 2.5 hours per arm. The same three lines, plus:
4. No growing gap: each 2-byte arm's gap to its reference (line 3) is at most 0.02 larger than in stage 1.

Stage 3, measure only (written 2026-10-26 after stage 2, before any stage-3 run; the owner asked for it): how much
of each 2-byte arm's gap is the storage format itself, and how much is training in it? Re-run `muon_float` and
`fp8_m8` with stage 2's settings, and every 500 steps and at the end also score the weights rounded to nearest:
`muon_float` to FP8 and to 4 bits, `fp8_m8` to 4 bits. Compare `fp8_m8` with `muon_float` rounded to FP8 (the price
of training in FP8, beyond storing in it), and `q4_t4_m8` with `muon_float` rounded to 4 bits. Also log, for every
Muon arm, the share of weights whose 4-bit and FP8 round-to-nearest codes change per step: `muon_float`'s gives the
curve an exact system would show, for comparison with the ticker's flat 1.6–1.9%. Code check: the re-runs should
repeat stage 2's final losses (3.2666 and 3.3015) to within 0.002. Final weights are saved for later rounding tests.

**B2. Batch size with bigger learning rates (written 2026-10-25, before any run).** The owner's idea: bigger steps
might suffer less from rounding, and bigger batches might allow bigger steps. Same setup as B (same characters, T4,
fp16, seeds 1-3, runner `batch_runs.py`), with every learning rate (Muon, AdamW and its floor) multiplied by 2 or 4
at 1x, and by 2, 4 or 8 at 2x, 4x and 8x. Plus seed 1 of both arms at 1x and the old rates, to check that a new
Colab session reproduces study B. The old rates come from study B. Measure only. Questions fixed now:
1. For each batch size, m3's best rate multiplier (lowest 3-seed mean) and that loss. Does any bigger batch, at its
   best rate, beat m3 at 1x with the old rates (1.755)?
2. The same for m1, and the m3 - m1 gap with each arm at its own best rate per batch size.
3. At a fixed batch size, does a bigger rate shrink the m3 - m1 gap?
4. Runs that blow up (loss not a number, or above 3): counted per setting.
Expected from side study L (at 1x, 10,000 steps, m3 got worse from rate 0.005 to 0.01 and blew up at 0.02): at 1x,
bigger rates hurt m3.

## Paper sprint

The paper sprint (written 2026-10-28) has its own plan, pass lines and checklist in `PAPER_PLAN.md`.
