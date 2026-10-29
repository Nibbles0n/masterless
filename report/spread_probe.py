"""Offline check before milestone 5: does spreading the rounding error over the matrix cut the noise the layer's
output sees? Not a milestone; no training change.

Trains milestone 3 (seed 1, CPU preset, op for op as m3_nomaster_sr.py) and keeps, per hidden matrix, a running average
of its inputs' covariance H = mean(x x^T) (what an online method could keep). At steps 250, 1000 and 1900 it takes the
intended weight w~ = w - lr * update and rounds it to the grid in several ways, several times each, and measures on
held-out inputs X how much noise reaches the output, relative to the output change the step intended:
    noise = |(Q - w~) X|^2 / |(w~ - w) X|^2
All methods use random rounding, so all are right on average; lower noise is strictly better.
  sr        independent random rounding per weight (milestone 3)
  next      the error of each weight is carried to the next weight in its row (like dithering an image)
  full      GPTQ's update: round one input column at a time and spread its error over the later columns, using H
            (damped by 1% of its mean diagonal; full_damp10: 10%)
  block16   the same with H cut to 16 x 16 blocks matching the scale blocks (cheap to keep online)
  lowrank8  the same with H approximated by its top 8 directions plus a diagonal
Also reports the weight-space size of Q - w~ against sr (spreading moves weights more, in directions X hardly uses).

Run:  python report/spread_probe.py   ->  report/data/spread_probe.json
"""
import importlib.util
import json
import os
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("m3", os.path.join(ROOT, "m3_nomaster_sr.py"))
m3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m3)
E2M1, BLOCK = m3.E2M1, m3.BLOCK
TYPES = ("qkv", "proj", "fc", "fc_out")
MODES = ("sr", "next", "full", "full_damp10", "block16", "lowrank8")


def sr_values(x, gen):
    """Random rounding of grid-unit values x (any shape, |x| <= 6) to E2M1, unbiased."""
    grid = E2M1
    a = x.abs().clamp(max=6.0)
    lo_i = (torch.bucketize(a, grid, right=True) - 1).clamp(max=len(grid) - 2)
    lo, hi = grid[lo_i], grid[lo_i + 1]
    up = torch.rand(a.shape, generator=gen, dtype=a.dtype) < (a - lo) / (hi - lo)
    return torch.where(up, hi, lo) * x.sign()


def round_spread(T, H, gen, mode, damp=0.01):
    """Round T (out, in) column by column with random rounding; spread each column's error using H (in, in).
    mode 'next' carries the error unweighted to the next column of the same row."""
    T = T.double().clone()
    s = m3.block_scale(T.float()).double().squeeze(-1)          # (out, in/16), fixed from the target
    out, n = T.shape
    Q = torch.zeros_like(T)
    if mode == "next":
        carry = torch.zeros(out, dtype=T.dtype)
        for j in range(n):
            sj = s[:, j // BLOCK]
            v = T[:, j] + carry
            q = sr_values((v / sj).clamp(-6, 6), gen).double() * sj
            Q[:, j] = q
            carry = v - q
        return Q
    Hd = H.double()
    Hd = Hd + damp * Hd.diagonal().mean() * torch.eye(n, dtype=Hd.dtype)
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hd)), upper=True)
    for j in range(n):
        sj = s[:, j // BLOCK]
        q = sr_values((T[:, j] / sj).clamp(-6, 6), gen).double() * sj
        Q[:, j] = q
        err = (T[:, j] - q) / U[j, j]
        T[:, j + 1:] -= err[:, None] * U[j, j + 1:][None, :]
    return Q


def approx(H, kind):
    if kind == "full":
        return H
    if kind == "block16":
        mask = torch.block_diag(*[torch.ones(BLOCK, BLOCK)] * (H.size(0) // BLOCK))
        return H * mask
    if kind == "lowrank8":
        ev, V = torch.linalg.eigh(H.double())
        top = V[:, -8:] @ torch.diag(ev[-8:]) @ V[:, -8:].T
        return (top + torch.diag((H.double() - top).diagonal().clamp_min(0))).float()


def main():
    torch.set_num_threads(2)
    train, val, vocab = m3.load_data()
    cfg = m3.Config(**m3.PRESETS["cpu"], vocab_size=vocab)
    torch.manual_seed(1)
    gen = torch.Generator().manual_seed(1)
    model = m3.GPT(cfg)
    named = [(n, p) for n, p in model.named_parameters() if n.endswith(m3.MUON_NAMES)]
    with torch.no_grad():
        for _, p in named:
            p.copy_(m3.quantize_rtn(p))
    round_gen = torch.Generator().manual_seed(1001)
    params = [p for _, p in named]
    names = [n for n, _ in named]
    rest = [p for n, p in model.named_parameters() if not n.endswith(m3.MUON_NAMES)]
    adamw = torch.optim.AdamW([{"params": [p for p in rest if p.dim() >= 2], "weight_decay": cfg.weight_decay},
                               {"params": [p for p in rest if p.dim() < 2], "weight_decay": 0.0}],
                              lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    mods = {n + ".weight": m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n.split(".")[-1] in TYPES}
    H = {n: torch.zeros(mods[n].in_features, mods[n].in_features) for n in names}
    grab = {"on": True, "X": {}}

    def hook(name):
        def f(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].size(-1)).float()
            if grab["on"]:
                H[name].mul_(0.98).add_(x.T @ x / x.size(0), alpha=0.02)
            else:
                grab["X"].setdefault(name, []).append(x)
        return f
    for n in names:
        mods[n].register_forward_hook(hook(n))

    eval_gen = torch.Generator().manual_seed(777)
    snaps, momentum, out = tuple(int(v) for v in os.environ.get("SNAPS", "250,1000,1900").split(",")), [torch.zeros_like(p) for p in params], []
    t0 = time.time()
    for it in range(cfg.max_iters):
        lr = 0.01 * m3.lr_at(it, cfg) / cfg.lr
        for g in adamw.param_groups:
            g["lr"] = cfg.lr * m3.lr_at(it, cfg) / cfg.lr
        x, y = m3.get_batch(train, cfg, gen, "cpu")
        loss = F.cross_entropy(model(x).float().view(-1, cfg.vocab_size), y.view(-1))
        model.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        targets, befores = [], []
        with torch.no_grad():
            for i, p in enumerate(params):
                buf = momentum[i]
                buf.lerp_(p.grad, 0.05)
                u = p.grad.lerp(buf, 0.95)
                ns = m3.newton_schulz(u).to(p.dtype)
                w_tilde = p - lr * max(1, p.size(0) / p.size(1)) ** 0.5 * ns
                targets.append(w_tilde)
                befores.append(p.clone())
                p.copy_(m3.quantize_sr(w_tilde, round_gen))
        adamw.step()
        if it > max(snaps):
            break
        if it in snaps:
            grab["on"], grab["X"] = False, {}
            with torch.no_grad():
                for _ in range(4):
                    xe, _ = m3.get_batch(val, cfg, eval_gen, "cpu")
                    model(xe)
            grab["on"] = True
            for i, n in enumerate(names):
                X = torch.cat(grab["X"][n]).double()
                T = targets[i]
                row = dict(step=it, name=n, type=n.split(".")[-2], shape=list(T.shape))
                # signal: the output change the step intended, from the on-grid weight before this step
                sig = ((T.double() - befores[i].double()) @ X.T).pow(2).sum().item()
                rng = torch.Generator().manual_seed(it * 100 + i)
                for mode in MODES:
                    noise, wnoise = [], []
                    for _ in range(3):
                        if mode == "sr":
                            Q = (sr_values((T.view(T.size(0), -1, BLOCK) / m3.block_scale(T)).clamp(-6, 6), rng)
                                 * m3.block_scale(T)).view_as(T).double()
                        elif mode == "next":
                            Q = round_spread(T, None, rng, "next")
                        elif mode == "full_damp10":
                            Q = round_spread(T, H[n], rng, mode, damp=0.1)
                        else:
                            Q = round_spread(T, approx(H[n], mode), rng, mode)
                        D = Q - T.double()
                        noise.append((D @ X.T).pow(2).sum().item() / sig)
                        wnoise.append(D.pow(2).sum().item())
                    row[mode] = sum(noise) / len(noise)
                    row[mode + "_w"] = sum(wnoise) / len(wnoise)
                out.append(row)
                print(it, n, {k: round(row[k], 2) for k in MODES}, flush=True)
    json.dump(out, open(os.path.join(HERE, "data", "spread_probe.json"), "w"))
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
