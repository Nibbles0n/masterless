"""Why storing weights in few bits is hard for any optimizer: 20,000 independent weights on a grid with gap 1, each
pushed every step by a Muon-sized step with the training schedule's shape (100-step warm-up, cosine down to 10% over
2000 steps). Each push is mostly noise with a small steady part, as real gradient steps are. The sizes match a real
milestone 1 run measured by report/path_probe.py (Muon weights travel about 183 gaps in all and move about 24 net):
at the peak the push's spread is 0.21 gaps and its steady part 0.022 gaps. Five ways to store the weight:
  float      no rounding (the target)
  random     random rounding to the grid each step, nothing remembered (milestone 3)
  fine8      random rounding to a grid 8 times finer (about 3 more bits, roughly what FP8 adds over 4-bit)
  tick4      the owner's ticker: round to nearest, carry the remainder in 4 bits (k/14 of a gap, rounded at random)
  tick       the ticker with an exact carry (stored weight + carry = the float weight, so this is a float copy)
For each, the typical distance from the float weight (root mean square, in gaps) at every step.

Run:  python report/noise_sim.py   ->  prints the distances, writes report/data/noise_sim.json
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N, STEPS, WARMUP = 20000, 2000, 100
SPREAD, DRIFT = 0.21, 0.022


def lr_frac(it):
    if it < WARMUP:
        return (it + 1) / WARMUP
    frac = (it - WARMUP) / (STEPS - WARMUP)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac))


def sr(x, rng, gap=1.0):
    lo = np.floor(x / gap)
    return (lo + (rng.random(x.shape) < x / gap - lo)) * gap


def main():
    rng = np.random.default_rng(0)
    w = {k: np.zeros(N) for k in ("float", "random", "fine8", "tick4", "tick")}
    carry4, carry = np.zeros(N), np.zeros(N)
    out = {k: [] for k in w if k != "float"}
    push_rms, path = [], np.zeros(N)
    for it in range(STEPS):
        push = lr_frac(it) * (DRIFT + SPREAD * rng.standard_normal(N))
        push_rms.append(float(np.sqrt((push ** 2).mean())))
        path += np.abs(push)
        w["float"] += push
        w["random"] = sr(w["random"] + push, rng)
        w["fine8"] = sr(w["fine8"] + push, rng, gap=1 / 8)
        t = w["tick"] + carry + push
        w["tick"] = np.round(t)
        carry = t - w["tick"]
        t = w["tick4"] + carry4 + push
        w["tick4"] = np.round(t)
        carry4 = sr((t - w["tick4"]) * 14, rng) / 14
        if (it + 1) % 25 == 0:
            for k in out:
                out[k].append([it + 1, float(np.sqrt(((w[k] - w["float"]) ** 2).mean()))])
    for k, v in out.items():
        print(f"{k:7s} distance from float at steps 500 / 1000 / 2000: " + " / ".join(f"{dict(v)[s]:.3f}" for s in (500, 1000, 2000)))
    print(f"push per step at the peak {max(push_rms):.3f} gaps, at the end {push_rms[-1]:.4f}")
    print(f"distance travelled {path.mean():.1f} gaps, net movement {np.abs(w['float']).mean():.1f} gaps,"
          f" sqrt(travelled) = {np.sqrt(path.mean()):.2f}")
    json.dump(dict(dist=out, push=[[i + 1, push_rms[i]] for i in range(0, STEPS, 25)]),
              open(os.path.join(HERE, "data", "noise_sim.json"), "w"))


if __name__ == "__main__":
    main()
