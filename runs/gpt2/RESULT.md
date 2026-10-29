# Side study G: the two 2-byte systems on GPT-2 small (124M)

Pretrained OpenAI GPT-2 small continues training on FineWeb (modded-nanogpt GPT-2 shards). Muon (lr 0.001) on the
48 block matrices (84.9M weights), float AdamW (lr 1e-4) on the rest. 32 × 1024 tokens per step, 5% warm-up, cosine to
10%, fp16 autocast, Colab T4. Same batches for every arm (fixed data seed), seed 1 only. Validation: 256 fixed
windows of 1024 tokens from the separate val shard, loss in nats per token. Differences carry a paired standard error
over those 256 windows (the only error bar available with one seed). Pass lines: `PLAN.md`, section G, written before
stage 1. Report: https://claude.ai/artifact/J6Us6XRJtvHQrCsauDhaxa (`report/build_gpt2.py`).

Pretrained GPT-2 small, float, on these windows: **3.4024**.

**Summary.** Stability: PASS for both 2-byte systems at both lengths. No run diverged, every run improved on its
own start, and every run ended below the pretrained float model. Closeness: FAIL for both at 100M tokens. FP8 is
+0.035 behind float Muon (it passed at 25M tokens with +0.014, then stalled while the learning rate was high). The
ticker is +0.047 behind 4-bit with a float copy (+0.057 at 25M tokens; its gap shrinks with longer training, but
stays noisy). Against rounding a float Muon model once at the end (stage 3): FP8 storage costs only +0.003, so the
FP8 arm loses +0.032 to per-step random rounding; in 4 bits the ticker is level with float Muon rounded at the end
(+0.003 ± 0.001), and FP8 rounded to 4 bits is +0.039 behind. The ticker's flat change rate is a carry flaw (weights
parked on the rounding line), not the learning rate; an 8-bit carry fixes it in a probe.

## Stage 1: 760 steps (24.9M tokens)

| arm | bytes/weight | step 0 | final | largest rise | 4-bit after training | skipped | max grad | minutes |
|---|---|---|---|---|---|---|---|---|
| adamw_float | 12 | 3.4024 | 3.2824 | +0.0000 | 3.3442 | 0 | 2.84 | 28 |
| muon_float | 8 | 3.4024 | 3.2820 | −0.0002 | 3.3425 | 0 | 3.23 | 33 |
| q4_float_copy | 8 | 3.4618 | 3.3042 | +0.0002 | – | 0 | 3.42 | 43 |
| fp8_m8 | 2.28 | 3.4053 | 3.2962 | −0.0001 | 3.3592 | 0 | 3.26 | 43 |
| q4_t4_m8 | 2.28 | 3.4618 | 3.3611 | +0.0159 | – | 1 | 7.57 | 35 |

(Largest rise: the largest amount a validation check sat above the best earlier check; negative means every check
was a new best.)

Pass lines:
- `fp8_m8`: line 1 (stable) PASS, line 2 (improves, 3.405 → 3.296) PASS, line 3 (≤ muon_float + 0.03) PASS:
  **+0.0142 ± 0.0004**.
- `q4_t4_m8`: line 1 PASS (rise +0.016 against the allowed +0.048), line 2 PASS (3.462 → 3.361), line 3
  (≤ q4_float_copy + 0.03) **FAIL: +0.0568 ± 0.0013**.

Measured only: ticker − muon_float +0.0790 ± 0.0013; ticker − fp8 +0.0649 ± 0.0014; fp8 − q4_float_copy
−0.0080 ± 0.0006; q4_float_copy − muon_float +0.0222 ± 0.0005; muon_float − adamw_float −0.0003 ± 0.0003. Every arm
ends below the pretrained float model; the ticker by 0.041. The ticker's best check was step 400 (3.3513); it then rose
to 3.3611 while the learning rate fell. It ends behind float Muon rounded to 4 bits after training (3.3425) by 0.019.
Share of stored codes that change per step: FP8 about 1.3% early, falling with the learning rate to 0.2%; the ticker
about 1.6–2.0% the whole run. Ticker gradient norm, median over the second half: 1.81 (the others 0.73–0.83).

## Stage 2: 3,050 steps (99.9M tokens)

| arm | bytes/weight | step 0 | final | largest rise | 4-bit after training | skipped | max grad | hours |
|---|---|---|---|---|---|---|---|---|
| muon_float | 8 | 3.4024 | 3.2666 | +0.0000 | 3.3296 | 0 | 3.37 | 2.1 |
| q4_float_copy | 8 | 3.4618 | 3.2862 | +0.0002 | – | 0 | 3.55 | 2.7 |
| fp8_m8 | 2.28 | 3.4053 | 3.3015 | +0.0018 | 3.3701 | 0 | 3.40 | 2.8 |
| q4_t4_m8 | 2.28 | 3.4618 | 3.3327 | +0.0175 | – | 1 | 7.61 | 2.3 |

Pass lines:
- `fp8_m8`: line 1 PASS, line 2 PASS (3.405 → 3.302), line 3 **FAIL: +0.0350 ± 0.0007** (line 0.03), line 4
  **FAIL: the gap grew by +0.0208** (line 0.02; +0.0142 in stage 1 → +0.0350; missed by 0.001, about one standard
  error).
- `q4_t4_m8`: line 1 PASS (rise +0.018), line 2 PASS (3.462 → 3.333), line 3 **FAIL: +0.0465 ± 0.0010**, line 4
  PASS (the gap shrank by 0.010, from +0.0568 to +0.0465).

Measured only: ticker − muon_float +0.0662 ± 0.0010; ticker − fp8 +0.0312 ± 0.0011; fp8 − q4_float_copy
+0.0153 ± 0.0008; q4_float_copy − muon_float +0.0197 ± 0.0005.

**FP8 ended worse after 100M tokens (3.3015) than after 25M (3.2962).** Its curve went flat at 3.309–3.311 from step
300 to step 1,200, while the learning rate was near its peak and float Muon kept improving (3.300 → 3.285); it only
started improving again as the learning rate fell. Float Muon, the float-copy run and the ticker all improved from
stage 1 to stage 2 (−0.015, −0.018, −0.028). A likely cause, not tested: random rounding to FP8 adds a little noise
every step, in proportion to the step size; at the peak learning rate it cancels the gain, and over 4× more steps
the noise piles up. A smaller learning rate for FP8, or a carried leftover instead of random rounding, would test it.

The ticker kept improving but stays noisy (validation checks swing by ±0.01: 3.3347 at step 2,400, 3.3460 at step
3,000, 3.3327 at step 3,050). It ends level with float Muon rounded to 4 bits after training (3.3296, +0.003) and
ahead of FP8 rounded to 4 bits after training (3.3701). Gradient norm median over the second half: 1.59 (others
0.62–0.64). Share of codes changed per step: ticker 1.6–1.9% throughout; FP8 1.3% at the peak learning rate, 0.24% at
the end.

## Stage 3: rounding after training (re-runs of muon_float and fp8_m8)

Same seed and settings as stage 2, with checks every 500 steps of the weights rounded to nearest (block matrices
only). Code check: final losses repeat stage 2 (muon_float 3.2666 vs 3.2666; fp8_m8 3.3010 vs 3.3015, FP8 rounds at
random so it is not bit-identical). Measured only; no pass line. Final losses, and paired differences over the
256 windows:

| Trained as | Stored as | Final loss |
|---|---|---|
| float Muon | float | 3.2666 |
| float Muon | FP8, rounded at the end | 3.2693 |
| FP8 + 8-bit momentum | FP8 | 3.3010 |
| float Muon | 4-bit, rounded at the end | 3.3294 |
| FP8 + 8-bit momentum | 4-bit, rounded at the end | 3.3679 |
| 4-bit + float copy (stage 2) | 4-bit | 3.2862 |
| 4-bit ticker (stage 2) | 4-bit | 3.3327 |

- float Muon rounded to FP8 − float Muon: **+0.0027 ± 0.0002**
- trained in FP8 − float Muon rounded to FP8: **+0.0318 ± 0.0007**
- float Muon rounded to 4 bits − float Muon: +0.0628 ± 0.0010
- ticker − float Muon rounded to 4 bits: **+0.0033 ± 0.0012**
- FP8-trained rounded to 4 bits − float Muon rounded to 4 bits: **+0.0385 ± 0.0013**
- ticker − FP8-trained rounded to 4 bits: −0.0352 ± 0.0014
- 4-bit + float copy − float Muon rounded to 4 bits: −0.0432 ± 0.0009

So FP8 storage is nearly free (0.003); the FP8 arm's loss comes from random rounding at every step. The FP8-trained
model rounded to 4 bits stayed at 3.368–3.373 from step 500 to the end, while float Muon rounded to 4 bits improved
from 3.354 to 3.329. Share of 4-bit values changing per step: float Muon peaks at 0.12% and ends at 0.012%; the FP8
arm stays at 0.65–0.83% throughout (0.77% at the end). Share of FP8 values changing: float Muon 1.25% peak, 0.17% at
the end; FP8 arm 1.33% peak, 0.23% at the end. Final weights saved in `data/gpt2_s3/` (not committed).

## Why the ticker's change rate is flat

The ticker changes 1.6–1.9% of its 4-bit values per step for the whole of stage 2, from warm-up to the end of the
cosine. The learning rate does reach it; the carry design causes this. After a tick the leftover is about half a gap,
which the carry stores as ±7/14, its largest value, so the weight sits exactly on the rounding line. A tiny outward
step ticks it back across; a tiny inward step leaves the line only by random rounding. Parked weights flip a whole gap
with the sign of each update, whatever its size.

Probe (`report/ticker_probe.py`, output `runs/gpt2/ticker_probe.txt`): four pretrained layer-5 matrices, Muon-like
steps on random gradients, 35 steps measured after 15, six fixed learning rates, the same updates for every variant.
Share of 4-bit values changing per step:

| lr | exact weight, rounded | ticker k/14 (as run) | ticker k/15 | ticker 8-bit carry |
|---|---|---|---|---|
| 1e-5 | 0.001% | 1.595% | 0.000% | 0.090% |
| 1e-4 | 0.015% | 1.600% | 0.000% | 0.095% |
| 1e-3 | 0.147% | 1.667% | 0.005% | 0.164% |
| 3e-3 | 0.437% | 1.824% | 0.016% | 0.337% |

Ticker k/14: 6.6% of weights parked on a line, 44–46% of changes undone the next step (exact weight: 7% at 1e-3).
The 8-bit carry follows the learning rate and nearly matches the exact rate at the runs' peak (1e-3). Reading 4 bits
as fifteenths freezes the weights: the clamp discards any push past the largest carry. Earlier estimate, for
scale: a step at lr 0.001 moves a weight about 0.0005–0.001 of a 4-bit gap, against a carry unit of 1/14.

## Not tested

- One seed; one Muon learning rate (0.001, chosen for float Muon) for every Muon arm, untuned for the 2-byte arms.
- Ticker fixes on GPT-2: the 8-bit carry was only tested by the probe (random gradients, 50 steps, 4 matrices).
  A dead zone on the rounding line, and FP8 with a smaller learning rate or a carried leftover, are untested.
- Rounding after training for the 4-bit + float copy arm (already stored in 4 bits) and the AdamW arm.
- Downstream tasks and text samples; training from scratch; larger models.
- Byte counts cover block-matrix weights and optimizer state only. Matrix multiplies run in fp16, so no speed gain.
- `side_gpt2.py` fixes an FP8 stochastic-rounding edge case: a block's largest value scaled to 447.99997 could round
  down to 416 and shrink the block scale. `m7_small_state.py` has the same edge case; it was harmless there (the next
  step repaired it), and m7 was not re-run.
- The 200-step learning-rate check for Muon at 0.003 and 0.01 was run on a Colab session that was reclaimed; their logs
  were lost (numbers recorded in PLAN.md from the printout).

## Time and cost

T4, about 12.7k tokens per second for the Muon arms (AdamW 15k, the FP8 and float-copy arms 10k and 8k). Colab bills
1.07 compute units per T4-hour. Stage 1: 3.0 T4-hours; stage 2: 9.8; stage 3: 5.4 (two re-runs side by side,
about 2.8 hours of waiting). The whole study, pilots and learning-rate checks included, used about 21 compute units
(balance 393 → 372). All Colab sessions are stopped.
