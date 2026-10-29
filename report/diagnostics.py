"""Instrumented re-runs of milestones 1-4 (seed 1, CPU preset) for the report. Not a milestone.

Re-implements each arm's hidden-matrix step exactly as in m1_muon.py ... m4_eco.py, so the final losses must match the
committed runs, and logs what the step does inside:
  every 25 steps, per matrix type (qkv, proj, fc, fc_out; averaged over the 4 layers):
    step_over_gap   median over weights of |intended change| / local grid spacing
    flip_frac       fraction of 4-bit codes that changed this step (m2: of the forward weights; m1: none, 0)
    follow_cos      cosine between the stored change and the intended change
    follow_ratio    |stored change| / |intended change|
    err_over_step   |rounding error| / |intended change|
    survival        (muon_pre only) cos and gain of NS(M + d) - NS(M) against the injected d
  every 100 steps: the same cosine and ratio for the change summed over the last 100 steps (does noise average out?)
  every step: the training loss.
Also saves momentum snapshots of layer 1 (muon_pre) for report/ns_analysis.py.

Run:  python report/diagnostics.py --mode m3 ; modes: m1 m2 m3 eco_adam muon_pre muon_post
"""
import argparse
import importlib.util
import json
import math
import os
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m2 = load("m2_fake4")    # GPT with fake-quant QLinear
m4 = load("m4_eco")      # GPT with plain Linear, quantizers, newton_schulz
E2M1, BLOCK = m4.E2M1, m4.BLOCK
TYPES = ("qkv", "proj", "fc", "fc_out")


def local_gap(w):
    """Grid spacing (in weight units) around each weight of w, on w's own block-scaled E2M1 grid."""
    grid = E2M1.to(w.device)
    s = m4.block_scale(w)
    a = (w.view(w.size(0), -1, BLOCK) / s).abs()
    lo_i = (torch.bucketize(a, grid, right=True) - 1).clamp(max=len(grid) - 2)
    return ((grid[lo_i + 1] - grid[lo_i]) * s).view_as(w)


def codes(w):
    """Signed 4-bit code (-7..7) of each weight of an on-grid w, on its own block scale."""
    grid = E2M1.to(w.device)
    x = w.view(w.size(0), -1, BLOCK) / m4.block_scale(w)
    return (torch.bucketize(x.abs(), (grid[1:] + grid[:-1]) / 2) * x.sign()).view_as(w)


def cos(a, b):
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


class Hidden:
    """The hidden-matrix optimizer of one arm, copied op for op from the milestone scripts, with probes."""

    def __init__(self, mode, params, names, cfg, muon_lr, gen):
        self.mode, self.params, self.names, self.cfg, self.gen = mode, params, names, cfg, gen
        self.peak = cfg.lr if mode == "eco_adam" else muon_lr
        self.lr = self.peak
        self.state = [dict() for _ in params]
        self.probe = False
        self.rows = []
        self.win_int = [torch.zeros_like(p) for p in params]
        self.win_act = [torch.zeros_like(p) for p in params]
        self.snap = None  # set to a dict to collect momentum snapshots

    def stored(self, p):
        return m4.quantize_rtn(p) if self.mode == "m2" else p

    @torch.no_grad()
    def step(self):
        lr, beta = self.lr, 0.95
        rows = []
        for i, p in enumerate(self.params):
            st = self.state[i]
            before = self.stored(p).clone()
            code_before = codes(before) if self.mode != "m1" else None
            e = d = buf_pre = None
            if self.mode == "eco_adam":
                b1, b2, eps = self.cfg.beta1, self.cfg.beta2, 1e-8
                if not st:
                    st.update(t=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                st["t"] += 1
                t, mm, v = st["t"], st["m"], st["v"]
                mm.lerp_(p.grad, 1 - b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                denom = (v / (1 - b2 ** t)).sqrt() + eps
                w_tilde = p * (1 - lr * self.cfg.weight_decay) - lr * (mm / (1 - b1 ** t)) / denom
                intended = w_tilde - p
                w_hat = m4.quantize_sr(w_tilde, self.gen)
                p.copy_(w_hat)
                e = w_tilde - w_hat
                mm.add_((1 - b1 ** t) / lr * (1 - 1 / b1) * denom * e)
            else:
                buf = st.setdefault("momentum", torch.zeros_like(p))
                buf.lerp_(p.grad, 1 - beta)
                u = p.grad.lerp(buf, beta)
                scale = max(1, p.size(0) / p.size(1)) ** 0.5
                if self.mode in ("m1", "m2"):
                    ns = m4.newton_schulz(u).to(p.dtype)
                    intended = -lr * scale * ns
                    w_tilde = None
                    p.add_(ns, alpha=-lr * scale)
                else:
                    ns = m4.newton_schulz(u).to(p.dtype)
                    update = scale * ns
                    if self.mode == "m3":  # m3_nomaster_sr.py multiplies in this order; keep it bit-identical
                        w_tilde = p - lr * scale * ns
                    else:
                        w_tilde = p - lr * update
                    intended = -lr * update
                    if self.mode == "muon_post":
                        err = st.setdefault("error", torch.zeros_like(p, dtype=torch.bfloat16))
                        w_tilde += err.float()
                    w_hat = m4.quantize_sr(w_tilde, self.gen)
                    e = w_tilde - w_hat
                    p.copy_(w_hat)
                    if self.mode == "muon_post":
                        st["error"].copy_(e)
                    elif self.mode == "muon_pre":
                        lr_eff = lr * update.norm() / u.norm()
                        d = (1 - 1 / beta) / lr_eff * e
                        buf_pre = buf.clone()
                        buf.add_(d)
            after = self.stored(p)
            actual = after - before
            self.win_int[i] += intended
            self.win_act[i] += actual
            if self.snap is not None and self.names[i].startswith("blocks.1.") and buf_pre is not None:
                self.snap[self.names[i]] = dict(M=buf_pre.clone(), d=d.clone(), e=e.clone())
            if self.probe:
                ref = w_tilde if w_tilde is not None else p
                r = dict(type=self.names[i].split(".")[-2],
                         step_over_gap=(intended.abs() / local_gap(ref)).median().item(),
                         flip_frac=(codes(after) != code_before).float().mean().item() if code_before is not None else 0.0,
                         follow_cos=cos(actual, intended),
                         follow_ratio=(actual.norm() / intended.norm()).item())
                if e is not None:
                    r["err_over_step"] = (e.norm() / intended.norm()).item()
                if d is not None:
                    ns_m = m4.newton_schulz(buf_pre)
                    delta = m4.newton_schulz(buf_pre + d) - ns_m
                    r.update(surv_cos=cos(delta, d),
                             surv_gain=(delta.norm() / d.norm() / (ns_m.norm() / buf_pre.norm())).item(),
                             inject_over_m=(d.norm() / buf_pre.norm()).item())
                rows.append(r)
        if rows:
            self.rows.append(rows)

    def window(self):
        out = []
        for i, name in enumerate(self.names):
            out.append(dict(type=name.split(".")[-2], cos=cos(self.win_act[i], self.win_int[i]),
                            ratio=(self.win_act[i].norm() / self.win_int[i].norm()).item()))
            self.win_int[i].zero_()
            self.win_act[i].zero_()
        return out


def by_type(rows):
    """Average each metric over the 4 layers, per matrix type, and over all 16 matrices ("all")."""
    out = {}
    for t in TYPES + ("all",):
        sel = [r for r in rows if t == "all" or r["type"] == t]
        out[t] = {k: sum(r[k] for r in sel) / len(sel) for k in sel[0] if k != "type"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("m1", "m2", "m3", "eco_adam", "muon_pre", "muon_post"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--muon-lr", type=float, default=0.01)
    ap.add_argument("--max-iters", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    args = ap.parse_args()

    mod = m2 if args.mode == "m2" else m4
    train, val, vocab = mod.load_data()
    cfg = mod.Config(**mod.PRESETS["cpu"], vocab_size=vocab)
    if args.max_iters is not None:
        cfg.max_iters = args.max_iters
    amp = __import__("contextlib").nullcontext

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    model = mod.GPT(cfg)
    named = [(n, p) for n, p in model.named_parameters() if n.endswith(mod.MUON_NAMES)]
    if args.mode not in ("m1", "m2"):
        with torch.no_grad():
            for _, p in named:
                p.copy_(m4.quantize_rtn(p))
    round_gen = torch.Generator("cpu").manual_seed(1000 + args.seed)
    hidden = Hidden(args.mode, [p for _, p in named], [n for n, _ in named], cfg, args.muon_lr, round_gen)
    rest = [p for n, p in model.named_parameters() if not n.endswith(mod.MUON_NAMES)]
    adamw = torch.optim.AdamW([{"params": [p for p in rest if p.dim() >= 2], "weight_decay": cfg.weight_decay},
                               {"params": [p for p in rest if p.dim() < 2], "weight_decay": 0.0}],
                              lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    windows = mod.fixed_windows(val, cfg)

    rec = dict(mode=args.mode, seed=args.seed, muon_lr=args.muon_lr, probes=[], windows=[], evals=[], train_loss=[])
    snaps = {}
    snap_steps = (100, 500, 1000, 1500, 1900)
    t0 = time.time()
    for it in range(cfg.max_iters + 1):
        if it % 250 == 0 or it == cfg.max_iters:
            ce, se = mod.evaluate(model, windows, "cpu", amp)
            rec["evals"].append(dict(step=it, val_ce=round(ce, 5), val_se=round(se, 5)))
            print(json.dumps(rec["evals"][-1]), round(time.time() - t0, 1), flush=True)
        if it == cfg.max_iters:
            break
        hidden.lr = hidden.peak * mod.lr_at(it, cfg) / cfg.lr  # same float order as the milestone scripts
        for g in adamw.param_groups:
            g["lr"] = cfg.lr * mod.lr_at(it, cfg) / cfg.lr
        x, y = mod.get_batch(train, cfg, gen, "cpu")
        loss = F.cross_entropy(model(x).float().view(-1, cfg.vocab_size), y.view(-1))
        model.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        hidden.probe = it % 25 == 0
        hidden.snap = {} if (args.mode == "muon_pre" and it in snap_steps) else None
        hidden.step()
        adamw.step()
        rec["train_loss"].append(round(loss.item(), 4))
        if hidden.probe:
            rec["probes"].append(dict(step=it, lr=hidden.lr, **by_type(hidden.rows[-1])))
        if hidden.snap:
            snaps[it] = hidden.snap
        if (it + 1) % 100 == 0:
            rec["windows"].append(dict(step=it + 1, **by_type(hidden.window())))

    os.makedirs(args.out, exist_ok=True)
    json.dump(rec, open(os.path.join(args.out, f"diag_{args.mode}_s{args.seed}.json"), "w"))
    if snaps:
        torch.save(snaps, os.path.join(args.out, f"snap_{args.mode}_s{args.seed}.pt"))
    print("final", rec["evals"][-1], round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
