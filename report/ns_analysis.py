"""Why Newton-Schulz keeps only part of the fed-back error. Reads the momentum snapshots saved by diagnostics.py.

For each snapshot (momentum M and injected error d of one matrix of layer 1, arm muon_pre), work in M's own singular
directions (M = U S V^T, made wide like Newton-Schulz does) and split d into three parts:
  stretch  changes the strength of M's existing directions (the symmetric part of U^T d V on M's directions)
  rotate   turns M's directions into each other (the antisymmetric part)
  outside  points into directions M does not use at all (the columns past M's rank, rectangular matrices only)
Exact orthogonalisation (the polar factor U V^T) removes the stretch part and keeps the other two, divided by the
singular values (rotate by (s_i + s_j) / 2, outside by s_i), so small directions get amplified.
Compares four cosines with d: measured NS5, exact polar (finite step), exact polar to first order, and the "even
scaling" bound sqrt(share_rotate + share_outside). Also the NS5 transfer curve and where M's singular values sit on it.

Run:  python report/ns_analysis.py   (after diagnostics.py --mode muon_pre)
"""
import importlib.util
import json
import math
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("m4", os.path.join(os.path.dirname(HERE), "m4_eco.py"))
m4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m4)


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def polar(X):
    U, _, Vh = torch.linalg.svd(X, full_matrices=False)
    return U @ Vh


def analyse(M, d):
    M, d = M.double(), d.double()
    if M.size(0) > M.size(1):
        M, d = M.T, d.T
    m = M.size(0)
    U, S, Vh = torch.linalg.svd(M, full_matrices=True)
    V = Vh.T
    D = U.T @ d @ V
    sq, out = D[:, :m], D[:, m:]
    sym, anti = (sq + sq.T) / 2, (sq - sq.T) / 2
    tot = D.pow(2).sum()
    share = dict(stretch=(sym.pow(2).sum() / tot).item(), rotate=(anti.pow(2).sum() / tot).item(),
                 outside=(out.pow(2).sum() / tot).item())
    # first-order change of the polar factor U V1^T
    om_sq = (sq - sq.T) / (S[:, None] + S[None, :])
    om_out = out / S[:, None]
    dP = U @ torch.cat([om_sq, om_out], dim=1) @ V.T
    ns_delta = m4.newton_schulz(M.float() + d.float()).double() - m4.newton_schulz(M.float()).double()
    return dict(
        share=share,
        cos_ns5=cos(ns_delta, d),
        cos_polar=cos(polar(M + d) - polar(M), d),
        cos_polar_linear=cos(dP, d),
        cos_even_bound=math.sqrt(share["rotate"] + share["outside"]),
        inject_over_m=(d.norm() / M.norm()).item(),
        sv_normalised=(S / M.norm()).tolist(),
    )


def ns5_curve():
    """Newton-Schulz on a single singular value s (after the divide-by-Frobenius-norm step): 5 rounds of the quintic."""
    a, b, c = 3.4445, -4.7750, 2.0315
    xs = torch.logspace(-4, 0, 241, dtype=torch.float64)
    y = xs.clone()
    for _ in range(5):
        y = a * y + b * y ** 3 + c * y ** 5
    return dict(s=xs.tolist(), f=y.tolist())


def main():
    snaps = torch.load(os.path.join(HERE, "data", "snap_muon_pre_s1.pt"))
    rows = []
    for step, mats in sorted(snaps.items()):
        for name, t in mats.items():
            r = analyse(t["M"], t["d"])
            # the same split for a random error of the same size, to see whether the real error is "random-like"
            g = torch.Generator().manual_seed(step)
            rnd = torch.randn(t["d"].shape, generator=g) * t["d"].std()
            rr = analyse(t["M"], rnd)
            r.update(step=step, type=name.split(".")[-2], shape=list(t["M"].shape),
                     random=dict(share=rr["share"], cos_ns5=rr["cos_ns5"], cos_polar_linear=rr["cos_polar_linear"]))
            rows.append(r)
            print(step, r["type"], {k: round(v, 3) for k, v in r["share"].items()}, "ns5", round(r["cos_ns5"], 3),
                  "polar", round(r["cos_polar"], 3), "lin", round(r["cos_polar_linear"], 3),
                  "even", round(r["cos_even_bound"], 3), "| random ns5", round(rr["cos_ns5"], 3))
    json.dump(dict(rows=rows, curve=ns5_curve()), open(os.path.join(HERE, "data", "ns_analysis.json"), "w"))


if __name__ == "__main__":
    main()
