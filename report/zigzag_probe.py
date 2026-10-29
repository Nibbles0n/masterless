"""Where does Muon's zig-zag come from, and which step rules would cut it? Not a milestone; float weights, no rounding.
Checks the predictions of report/zigzag_theory.md on one milestone 1 run (Muon, CPU preset, seed 1).

Along the whole run, for every weight of the four hidden matrices (in units of its own E2M1 gap at the start):
  travelled, net, zig-zag Z = travelled / net, per phase (steps 0-500, 500-1500, 1500-2000)
  lag-1 correlation of the steps (positive = smooth drift, negative = bouncing back and forth) and sign-flip rate
At steps 100, 500, 1000, 1500, 1990, before the step, from 64 fresh batches (their own generator, so the training run
is unchanged): the mean gradient mu (the "true" downhill direction) and the per-batch spread, giving
  snr2      (mean signal / noise)^2 of one batch's gradient, with the 1/64 noise of the mean removed
  top8_share the part of Muon's downhill progress that comes from the momentum's 8 strongest directions (of 128)
and for candidate step directions d (the minibatch gradient, momentum, Muon, sign, changed Muon steps, and three
changes to how Muon treats the momentum's weak directions, from its singular value decomposition):
  per_l1    <d, mu> / |d|_1   downhill progress per unit of path (what low-bit storage pays for)
  per_l2    <d, mu> / |d|_2   downhill progress per unit of step length (the usual cosine, times |mu|)
both as a ratio to Muon's, the median over the 16 matrices. <d, mu> uses fresh batches, so it is unbiased.

Run:  python report/zigzag_probe.py [--beta 0.95]   ->  report/data/zigzag_probe_b<beta>.json
"""
import argparse
import contextlib
import importlib.util
import json
import os
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("m1", os.path.join(os.path.dirname(HERE), "m1_muon.py"))
m1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m1)
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
CHECKS, N_REF, PHASES = (100, 500, 1000, 1500, 1990), 64, (500, 1500)


def gap_at(w):
    """The E2M1 gap above each weight's magnitude, scale = block-of-16 absmax / 6 (as report/path_probe.py)."""
    s = w.view(w.size(0), -1, 16).abs().amax(-1, keepdim=True) / 6
    a = (w.view(w.size(0), -1, 16) / s).abs()
    i = torch.bucketize(a, E2M1, right=True).clamp(1, len(E2M1) - 1)
    return ((E2M1[i] - E2M1[i - 1]) * s).view_as(w)


def candidates(g, buf, beta, scale):
    """Step directions (the weight moves by -lr * d). Muon's d is scale * NS(u); the changed ones keep its overall size."""
    m = buf.lerp(g, 1 - beta)
    u = g.lerp(m, beta)
    ns = scale * m1.newton_schulz(u)
    a = ns.abs()
    U, S, Vh = torch.linalg.svd(u, full_matrices=False)
    top = a >= a.flatten().kthvalue(int(0.75 * a.numel())).values
    return dict(
        grad=g, momentum=m, muon=ns, sign=torch.sign(u),
        muon_shrink=torch.sign(ns) * (a - a.median()).clamp(min=0),     # soft threshold: the smaller half stays put
        muon_top25=ns * top,                                            # move only the largest quarter
        muon_cautious=ns * (torch.sign(ns) == torch.sign(g)),           # move only where this batch agrees
        muon_by_mom=ns * u.abs() / u.abs().mean(),                      # Muon's signs, sizes from the momentum
        polar=U @ Vh,                                                   # exact orthogonalisation (what NS5 approximates)
        rank8=U[:, :8] @ Vh[:8],                                        # only the 8 strongest directions
        damped=U @ torch.diag(S / (S + S.median())) @ Vh,               # weak directions shrunk, strong ones kept
    )


def reference(model, train, cfg, gen, hidden):
    """Mean and variance over N_REF fresh batches of each hidden matrix's gradient (raw, before clipping)."""
    s1 = [torch.zeros_like(p) for p in hidden]
    s2 = [torch.zeros_like(p) for p in hidden]
    for _ in range(N_REF):
        x, y = m1.get_batch(train, cfg, gen, "cpu")
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x).view(-1, cfg.vocab_size), y.view(-1)).backward()
        for p, a, b in zip(hidden, s1, s2):
            a += p.grad
            b += p.grad ** 2
    model.zero_grad(set_to_none=True)
    mean = [a / N_REF for a in s1]
    var = [b / N_REF - mu ** 2 for b, mu in zip(s2, mean)]
    return mean, var


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.95)
    ap.add_argument("--no-checks", action="store_true", help="only the path numbers and the final loss")
    ap.add_argument("--max-iters", type=int, default=None, help="smoke tests only")
    args = ap.parse_args()
    torch.manual_seed(1)
    train, val, vocab = m1.load_data()
    cfg = m1.Config(**m1.PRESETS["cpu"], vocab_size=vocab)
    if args.max_iters:
        cfg.max_iters = args.max_iters
    gen, ref_gen = torch.Generator().manual_seed(1), torch.Generator().manual_seed(12345)
    model = m1.GPT(cfg)
    opts = m1.build_optimizer(model, cfg, "muon", 0.01)
    muon = opts[0][0]
    for g in muon.param_groups:
        g["momentum"] = args.beta
    hidden = [p for n, p in model.named_parameters() if n.endswith(m1.MUON_NAMES)]
    gaps = [gap_at(p.detach()) for p in hidden]
    marks = {0: [p.detach().clone() for p in hidden]}
    prev_w = [p.detach().clone() for p in hidden]
    prev_d = [torch.zeros_like(p) for p in hidden]
    trav = {k: [torch.zeros_like(p) for p in hidden] for k in ("all", 0, 1, 2)}
    lag1, sq, flips = 0.0, 0.0, [torch.zeros_like(p) for p in hidden]
    checks, t0 = [], time.time()
    for it in range(cfg.max_iters):
        for opt, peak in opts:
            for g in opt.param_groups:
                g["lr"] = peak * m1.lr_at(it, cfg) / cfg.lr
        ref = reference(model, train, cfg, ref_gen, hidden) if it in CHECKS and not args.no_checks else None
        x, y = m1.get_batch(train, cfg, gen, "cpu")
        loss = F.cross_entropy(model(x).view(-1, cfg.vocab_size), y.view(-1))
        model.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if ref is not None:
            row = dict(step=it, snr2=0.0, per_l1={}, per_l2={}, mag_corr=0.0)
            num = den = 0.0
            for p, mu, var in zip(hidden, *ref):
                num += (mu ** 2 - var / (N_REF - 1)).sum().item()
                den += var.sum().item()
                buf = muon.state[p].get("momentum", torch.zeros_like(p))
                c = candidates(p.grad, buf, args.beta, max(1, p.size(0) / p.size(1)) ** 0.5)
                l1 = {k: (d * mu).sum().item() / d.abs().sum().item() for k, d in c.items()}
                l2 = {k: (d * mu).sum().item() / d.norm().item() for k, d in c.items()}
                for k in c:
                    row["per_l1"].setdefault(k, []).append(l1[k] / l1["muon"])
                    row["per_l2"].setdefault(k, []).append(l2[k] / l2["muon"])
                u = p.grad.lerp(buf.lerp(p.grad, 1 - args.beta), args.beta)
                U, _, Vh = torch.linalg.svd(u, full_matrices=False)
                gain = ((U.T @ mu) * Vh).sum(1)  # downhill progress Muon makes along each of the momentum's directions
                row.setdefault("top8_share", []).append((gain[:8].sum() / gain.sum()).item())
                row["mag_corr"] += torch.corrcoef(torch.stack([u.abs().flatten(), c["muon"].abs().flatten()]))[0, 1].item() / len(hidden)
            row["snr2"] = num / den
            row["top8_share"] = torch.tensor(row["top8_share"]).median().item()
            for key in ("per_l1", "per_l2"):
                row[key] = {k: torch.tensor(v).median().item() for k, v in row[key].items()}
            checks.append(row)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else {a: round(b, 3) for a, b in v.items()} if isinstance(v, dict) else v) for k, v in row.items()}), flush=True)
        for opt, _ in opts:
            opt.step()
        with torch.no_grad():
            phase = sum(it >= b for b in PHASES)
            for i, (p, q, g) in enumerate(zip(hidden, prev_w, gaps)):
                d = (p - q) / g
                trav["all"][i] += d.abs()
                trav[phase][i] += d.abs()
                lag1 += (d * prev_d[i]).sum().item()
                sq += (d * d).sum().item()
                flips[i] += (d * prev_d[i] < 0).float()
                prev_d[i] = d
            prev_w = [p.detach().clone() for p in hidden]
            if it + 1 in PHASES:
                marks[it + 1] = [p.detach().clone() for p in hidden]
    marks[cfg.max_iters] = [p.detach().clone() for p in hidden]
    cat = lambda xs: torch.cat([x.flatten() for x in xs])
    net = lambda a, b: cat([(marks[b][i] - marks[a][i]).abs() / gaps[i] for i in range(len(hidden))])
    bounds = (0,) + PHASES + (cfg.max_iters,)
    phases = []
    for k in range(3):
        t, n = cat(trav[k]).mean().item(), net(bounds[k], bounds[k + 1]).mean().item()
        phases.append(dict(steps=[bounds[k], bounds[k + 1]], travelled=t, net=n, zigzag=t / n))
    t_all, n_all = cat(trav["all"]).mean().item(), net(0, cfg.max_iters).mean().item()
    ce, se = m1.evaluate(model, m1.fixed_windows(val, cfg), "cpu", contextlib.nullcontext)
    out = dict(beta=args.beta, travelled=t_all, net=n_all, zigzag=t_all / n_all, lag1_corr=lag1 / sq,
               flip_rate=cat(flips).mean().item() / (cfg.max_iters - 1), phases=phases, checks=checks,
               final_val_ce=ce, final_val_se=se, wall_s=time.time() - t0)
    print(json.dumps({k: v for k, v in out.items() if k != "checks"}))
    json.dump(out, open(os.path.join(HERE, "data", f"zigzag_probe_b{args.beta:g}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
