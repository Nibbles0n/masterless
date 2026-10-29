"""Why milestone 6 failed: one weight on a grid with gap 1, pushed by a steady step of 0.05 gaps per step
(about a Muon step at the peak learning rate). Three ways to store it:
  float    the weight is a float; it moves 0.05 every step (the target)
  m6       milestone 6's rule: momentum A <- 0.95 A + 0.05 o, target = w + lr A, round to nearest, leftover
           e = target - stored added back as A <- A + (1/0.95 - 1) e / lr (the code's (1 - 1/0.95) e / lr, with
           the sign flipped because here the step is +lr A, in the code -lr A)
  buffer   milestone 4c's rule: a separate leftover buffer r, target = w + step + r, round to nearest, r = target - w
Also the same with noisy pushes (mean 0.05, spread 0.2 per step), the realistic case.

Run:  python report/accum_sim.py   ->  prints how far each lags, writes report/data/accum_sim.json
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
BETA, LR, STEPS = 0.95, 1.0, 400


def run(pushes):
    out = {"float": [], "m6": [], "buffer": []}
    wf, mf = 0.0, 0.0  # float weight and its momentum (reordered Muon with float weights)
    w6, a6 = 0.0, 0.0
    wb, mb, rb = 0.0, 0.0, 0.0
    for o in pushes:
        mf = BETA * mf + (1 - BETA) * o
        wf += LR * mf
        a6 = BETA * a6 + (1 - BETA) * o
        t = w6 + LR * a6
        w6_new = float(round(t))
        a6 += (1 / BETA - 1) * (t - w6_new) / LR
        w6 = w6_new
        mb = BETA * mb + (1 - BETA) * o
        t = wb + LR * mb + rb
        wb_new = float(round(t))
        rb, wb = t - wb_new, wb_new
        for k, v in (("float", wf), ("m6", w6), ("buffer", wb)):
            out[k].append(v)
    return out


def main():
    steady = run([0.05] * STEPS)
    rng = random.Random(0)
    noisy = run([0.05 + rng.gauss(0, 0.2) for _ in range(STEPS)])
    for name, r in (("steady", steady), ("noisy", noisy)):
        for k in ("m6", "buffer"):
            gap = [abs(a - b) for a, b in zip(r[k], r["float"])]
            first = next((i + 1 for i, v in enumerate(r[k]) if v != 0), None)
            print(f"{name:6s} {k:6s} first move at step {first}, mean |stored - float| {sum(gap) / len(gap):.2f} gaps, max {max(gap):.2f}")
    json.dump(dict(steady=steady, noisy=noisy), open(os.path.join(HERE, "data", "accum_sim.json"), "w"))


if __name__ == "__main__":
    main()
