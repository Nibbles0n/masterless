# Zig-zag and low-bit weights: the theory

Written 2026-10-26 to design against. Not a milestone and changes no pass line. The numbers come from
`report/path_probe.py`, `report/noise_sim.py` and `report/zigzag_probe.py`, all run on milestone 1 (Muon with float
weights, CPU preset, seed 1).

## 1. Words

For one weight:

| symbol | meaning |
|---|---|
| g | the **gap**: the distance between the two storable values on either side of the weight |
| δₜ | the **step**: the change the optimizer asks for at step t |
| L = Σ\|δₜ\| | the **path**: the total distance travelled, counting every move back and forth |
| D = \|Σδₜ\| | the **net move** from start to finish |
| Z = L / D | the **zig-zag factor**. Z = 1 means a straight line, and Z ≥ 1 always |

Measured on a real Muon run: L ≈ 183 gaps, D ≈ 24 gaps, so **Z ≈ 7.6**. The weights travel 7.6 times further than
they end up going.

## 2. The rounding law

Store a number on a grid with spacing r using **random rounding**. If the wanted value is a fraction f of the way from
one grid point to the next, store the upper point with probability f and the lower one otherwise.

- On average the stored value is exactly right.
- Its spread (variance) is r²·f·(1 − f).

For a small step δ that starts on a grid point, f = |δ| / r. Each rounding then adds variance

  **r²·f·(1 − f) ≈ r·|δ|  when |δ| ≪ r,  and never more than r²/4.**

Roundings at different steps are independent, so their variances add. If nothing pulls the weight back (Section 4),
the stored weight drifts away from where float training would have put it by about

  **error ≈ √(r · L)**  (small steps),  **error ≈ √(T · r²/5)**  (steps comparable to r; T = number of steps).

Divide by the signal, the net move D:

  **error / signal = √(r / D) · √Z**

This is why the zig-zag factor matters. Low-bit storage charges for the path, but training only gains from the net
move. For plain 4-bit (r = g, L = 183g, D = 24g) the error is √183 ≈ 13.5 gaps against a signal of 24 gaps, more than
half the signal.

## 3. What each way of storing weights pays

| storage | what is randomly rounded, at what spacing r | noise grows with | prediction | simulation |
|---|---|---|---|---|
| plain 4-bit (m3) | the weight, r = g | the path L | √183 ≈ 13.5 gaps | 12.2 |
| FP8 (m7 `fp8_m8`) | the weight, r ≈ g/8 | steps, capped at r²/4 each | about 2.4 | 2.2 |
| ticker (`q4_t4_m8`) | only the carry, r = g/14; the weight itself is rounded to nearest with the leftover kept | the number of steps T | √(2000·(1/14)²/5) ≈ 1.4 | 1.3 |
| float copy (m2) | nothing | — | 0 | 0 |

The simulation is `report/noise_sim.py`: 20,000 weights, with steps sized to match the real run.

What follows for design:

- **Plain random rounding pays for every bit of zig-zag.** Halving Z halves the noise variance, the same as one extra
  bit.
- **The ticker mostly does not care about zig-zag.** Its rounding happens in the carry, and once a step is bigger
  than about r/4 = g/56 it costs the same whatever its size. Steps fall below that only late in training: Muon's push
  ends at about g/60. So less zig-zag would help the ticker only at the end, and a finer carry would help it
  throughout.
- More bits lower r, and the noise falls only as √r. Each extra bit cuts the noise by a factor of 1.41, so bits are
  an expensive way to buy accuracy.

## 4. The pull-back, which this law leaves out

Errors do not add up forever. Where the loss curves up around the weight, the gradient pulls a misplaced weight back.
A simple model of one weight, with pull-back strength κ per step:

  eₜ₊₁ = (1 − κ)·eₜ + εₜ  →  the error settles at a variance of Var(ε) / (2κ)

Most directions in a network are nearly flat (κ ≈ 0). There the √ law holds and the error keeps growing. Side study B
agrees: the damage from plain 4-bit tracked how far training had gone, not the batch size. Muon complicates the
picture because its step size does not shrink as the error shrinks, so κ is not a constant. This is left unmodelled.

## 5. Where the zig-zag comes from

Three sources, each with its own fix.

**5a. Gradient noise.** Model one weight's gradient as μ + σ·(noise): a steady signal μ plus noise of size σ.
Momentum averages over about 1/(1 − β) steps, which shrinks the noise to σₘ = σ·√((1 − β)/(1 + β)). That is 0.16σ
for β = 0.95 and 0.07σ for β = 0.99. If each step is proportional to the momentum and noise dominates, the average
step size is about 0.8·σₘ while the net move per step is μ:

  **Z ≈ 0.8 · σₘ / |μ|**  →  raising β from 0.95 to 0.99 should cut Z by 2.2 times, if μ holds still meanwhile.

**5b. Bouncing.** In a sharply curved direction, a fixed-size step overshoots the bottom and the weight hops from side
to side. This shows up as **negative** correlation between one step and the next. Pure noise passed through momentum
gives **positive** correlation instead. Muon is prone to bouncing because its step size does not shrink near the
bottom.

**5c. Muon spreads its movement evenly.** Newton–Schulz sets the strength of every direction of the momentum matrix
to 1. The momentum's strong directions carry the signal and its weak directions are mostly noise, so Muon boosts noise
to full size. In maths terms:

- Muon's step is the **steepest descent step under the spectral norm**: the step that goes downhill fastest among all
  steps with largest stretch η. Its exact answer is −η·U·Vᵀ, where M = U·S·Vᵀ is the momentum's singular value
  decomposition. Bernstein & Newhouse, "Old Optimizer, New Norm", 2024.
- Sign descent (and roughly Adam) is steepest descent under the **largest-entry norm**. It also spreads movement over
  every weight.
- Low-bit storage pays in the **sum of |moves|**, the L1 norm. Steepest descent under L1 does the opposite: it puts
  all the movement into the few weights with the strongest signal.

So Muon's design (equal movement in every direction) and low-bit storage's bill (paid per unit of movement, everywhere)
pull against each other. This is the precise form of "Muon doesn't handle quantization noise".

## 6. The design problem as maths

Plain Muon solves:

  choose δ to make ⟨G, δ⟩ as negative as possible, subject to ‖δ‖_spectral ≤ η

Here ⟨G, δ⟩ = Σᵢ Gᵢ·δᵢ is the first-order change in the loss and G is the momentum. The low-bit version adds the bill
from Section 2:

  **choose δ to minimise ⟨G, δ⟩ + λ · Σᵢ rᵢ·|δᵢ|, subject to ‖δ‖_spectral ≤ η**

- rᵢ is the rounding spacing at weight i: the gap for plain rounding, the carry's spacing for the ticker, about |w|/8
  for FP8.
- λ is the cost in loss of one unit of rounding variance. It is unknown; tune it.

Facts that follow:

1. **Without the spectral limit**, the bill is a lasso (L1) penalty. Its exact answer per weight is the **soft
   threshold**: shrink every move toward zero by λ·rᵢ, and moves smaller than that become zero.
2. **As the bits go up (r → 0), the problem becomes plain Muon.** Any method built from this objective turns back into
   Muon at high precision, which is a good property.
3. **With both terms**, there is no closed-form answer. Cheap approximations:
   (a) Muon first, then soft-threshold each weight by λ·rᵢ;
   (b) shrink Muon's weak directions before orthogonalising (fix 5c), then threshold.
4. **G is noisy.** A weight whose true μ is tiny looks like noise, and a threshold stops exactly those weights. The
   cost is that weights with a small but real signal move later. That trade-off is what λ sets.

## 7. Candidate designs

| family | what it does | extra memory per weight | fixes |
|---|---|---|---|
| longer momentum (β = 0.99) | averages away more noise | 0 | 5a, with a lag |
| soft threshold after NS | skip small moves, shrink the rest by λ·r | 0 | 5c, the L1 bill directly |
| top-k moves | move only the largest k% of the step each time | 0 | 5c |
| cautious mask | drop moves whose sign disagrees with this batch's gradient | 0 | 5a |
| damped / low-rank orthogonalisation | U·f(S)·Vᵀ with f small for weak directions (for example s/(s + s_median)), or keep only the top k directions | 0 (an SVD or a modified NS polynomial) | 5c at its source |
| sign-consistency gate | a 1–2-bit record of the last step signs; shrink weights that keep flipping | 0.125–0.25 byte | 5a, 5b |
| carry as confidence (ticker only) | a carry that keeps changing sign is zig-zag; use it as the gate at no cost | 0 | 5a, 5b |
| tick hysteresis (ticker only) | tick only past (½ + h) of the gap | 0 | churn in the forward weights (not rounding noise) |

Low-rank orthogonalised updates already exist, for example Dion (2025), built to save communication. As far as I
know nobody has tested them for low-bit storage.

## 8. Checks on a real run (`report/zigzag_probe.py`)

One float Muon run, seed 1, measured at three momentum settings. At steps 100, 500, 1000, 1500 and 1990 of the
β = 0.95 run, the true gradient was estimated from 64 fresh batches.

| β | path (gaps) | net (gaps) | Z | step-to-step correlation | final loss |
|---|---|---|---|---|---|
| 0.9 | 179 | 16.1 | 11.1 | +0.57 | 1.627 |
| 0.95 | 183 | 24.2 | 7.6 | +0.71 | 1.622 |
| 0.99 | 186 | 60.3 | 3.1 | +0.92 | 1.690 |

1. **The noise model holds.** Section 5a predicts Z ratios of 1.43 (0.9 vs 0.95) and 2.27 (0.95 vs 0.99). Measured:
   1.47 and 2.45. The correlation is positive, so the zig-zag is noise, not bouncing (5b).
2. **Under Muon the path is fixed by the learning rate** (179–186 gaps at every β), because Newton–Schulz fixes each
   step's size. Momentum changes only the net movement, so it cannot cut plain rounding's bill, which is the path. A
   longer average also lags: float training at β = 0.99 got worse by 0.068.
3. **Progress is concentrated, movement is not.** The momentum's 8 strongest directions out of 128 give 35–93% of
   Muon's downhill progress (93% at step 100, about 50% late). Muon moves all 128 equally far.
4. **Downhill progress per unit of path, as a multiple of Muon's** (same moment, same momentum, median over the 16
   matrices, range over the 5 checkpoints):

   | step | per unit of path (L1) | per unit of length (L2) |
   |---|---|---|
   | exact orthogonalisation | 0.94–1.00 | 0.95–1.00 |
   | keep only the 8 strongest directions | 1.5–3.8 | 1.4–3.5 |
   | weak directions shrunk, s/(s + median) | 1.3–1.7 | 1.3–1.6 |
   | Muon's signs, sizes from the momentum | 2.1–3.9 | 1.4–2.4 |
   | only where this batch agrees (cautious) | 1.8–6.1 | 1.4–4.7 |
   | soft threshold (smaller half stays put) | 1.4–1.5 | 0.93–1.00 |
   | largest quarter only | 1.4–1.6 | 0.88–0.97 |
   | sign of momentum | 0.96–2.5 | 1.2–3.2 |

   The soft threshold and the largest-quarter rule lose nothing per unit of length but gain about 1.5× per unit of
   path: the pure low-bit gain. A factor of 2 per unit of path is worth one extra bit.
5. **Caveat.** These are one-step, first-order scores. The current batch's raw gradient scores 2–13× by the same
   measure, and raw gradient steps train badly. So the scores rank candidates; only a training run shows whether a
   gain survives a whole run.


## 9. What to test next (a draft; the owner designs it)

**Predicted:** less zig-zag should help plain random rounding (m3) a lot and the ticker only a little (Section 3).
Testing both checks the theory as well as the fix.

A small first test on the CPU preset, seeds 1–3:

- m3 + the candidate fix, at 2–3 strengths;
- the ticker + the same fix;
- float Muon + the same fix, to check that the fix alone does not hurt training.

A pass line, to be fixed before running: m3 + fix closes at least half of m3's gap to m1. The gap is 0.117, so the
target is m3 + fix ≤ m1 + 0.06 in each seed.

## 10. The same law at GPT-2 scale (side study G, from saved weights only)

`report/gpt2_zigzag.py` and `report/gpt2_drift.py` use only what stage 2 and stage 3 saved: the pretrained weights,
float Muon's final weights and the FP8 arm's final weights. Nothing was trained. Path L is computed from the learning
rate schedule and the size of Muon's steps, which Newton–Schulz fixes. Units are each weight's 4-bit gap at the start.

| | CPU preset (milestone 1) | GPT-2, 100M tokens |
|---|---|---|
| path L | 183 gaps | **1.43 gaps** |
| net move D (mean) | 24 gaps | **0.16 gaps** |
| zig-zag Z | 7.6 | **8.8** (7.4–11 over the 48 matrices) |
| average step | about g/11 | **g/2100** (peak g/1300; 2-bit peak g/5000) |

**The zig-zag is the same; the signal is 150 times smaller.** Continued training of a pretrained model moves each
weight only a sixth of a 4-bit gap. By `error / signal = √(r/D)·√Z`, every storage pays about 13 times more noise per
unit of progress than on the CPU preset (√((8.8/0.16)/(7.6/24)) ≈ 13).

**The law predicts the FP8 arm's drift with no fitted constant.** FP8's spacing averages 0.21 of a 4-bit gap.
Predicted random walk: √(0.21 × 1.43) = 0.544 gaps (root mean square). Measured distance between the FP8 arm's and
float Muon's final weights: **0.547**. Float Muon's own net move is 0.30 (root mean square), so the FP8 arm's
rounding noise is 1.8 times its signal. It still made 85% of float Muon's progress in float Muon's direction, and the
cosine between the two moves is only 0.42. Caveat: the 0.547 also includes the ordinary divergence of two runs, which
was not measured separately.

The drift matching the √ law means no pull-back is visible in the total. Most directions are flat and keep their
noise. The loss is hurt mostly in the curved directions, where the noise settles at a level that scales with the
step size (Section 4). That fits FP8 stalling at the peak learning rate and improving again as it fell.

Predicted noise ÷ signal at GPT-2 (root mean square, 4-bit weights unless stated; ticker carries are randomly
rounded, so steps far below the carry unit pay r·|δ| as in Section 2):

| storage | spacing r (gaps) | noise | noise ÷ signal |
|---|---|---|---|
| plain 4-bit random rounding (m3) | 1 | 1.20 | 4.0 |
| FP8 weights (`fp8_m8`) | 0.21 | 0.54 (measured 0.55) | 1.8 |
| ticker, 4-bit carry k/14 (as run) | 0.071 | 0.32 | 1.1, plus the parking churn |
| ticker, FP8 carry | about 0.02 | about 0.17 | about 0.6 |
| ticker, int8 carry k/254 | 0.0039 | 0.075 | 0.25 |
| 2-bit weights, int8 carry | 0.0039 of a 2-bit gap | 0.037 (2-bit gaps) | 0.5 |
| 2-bit weights, FP8 carry | about 0.02 | about 0.085 | about 1.1 |

Consequences for design:

1. **At GPT-2 the ticker is no longer immune to zig-zag.** Its steps are 1/90 of a 4-bit carry unit, so it is back
   in the r·L regime, and Section 3's "the ticker mostly does not care" no longer holds.
2. **The carry's resolution is the biggest lever.** Going from k/14 to k/254 cuts r by 18 times. That is worth as much
   as cutting Z by 18 times, far beyond any step rule measured in Section 8 (1.4–4 times, first-order). Near zero an
   int8 carry is also finer than an FP8 carry for typical carries: FP8 spacing grows with the carry's size.
3. **Step rules multiply on top.** A rule with k times more progress per unit of path cuts noise ÷ signal by √k. That
   is the main remaining lever wherever r cannot shrink: FP8 weights with no carry, and 2-bit weights.
4. **2 bits needs two more carry bits.** A 2-bit gap is about 4 times a 4-bit gap here, so D in gaps is 4 times
   smaller. Keeping the same noise ÷ signal needs r 4 times finer, or Z 4 times lower.
5. **Training from scratch should be easier than continued training.** D is much larger there (Section 8: 24 gaps).
   Continued training of a good model makes the smallest moves, so it is the hardest case for storage without a float
   copy.
