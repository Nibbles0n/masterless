"""Side study G: why does the 4-bit ticker change ~1.7% of its weights every step, whatever the learning rate?

On four pretrained GPT-2 block matrices (layer 5), apply Muon-like steps (Nesterov momentum 0.95 on random Gaussian
gradients, Newton-Schulz, the same size factor as side_gpt2.py) at several fixed learning rates, and count the share of
4-bit codes that change per step for:
  float + RTN : a float weight, rounded to nearest 4-bit each step (what a float copy shows the forward pass)
  ticker 4-bit    : side_gpt2.py's ticker, carry k/14 of a gap (k = -7..7) rounded at random
  ticker 4-bit /15: the same 4 bits read as k/15 of a gap, so the largest carry sits just inside the rounding line
  ticker 8-bit    : the carry in k/254 of a gap
Also: the share of weights parked exactly on a rounding line (|carry| at its largest value), and the share of ticks
undone the next step. Every variant gets the same updates.

Run:  python report/ticker_probe.py
"""
import os
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from side_gpt2 import BLOCK, block_scale, newton_schulz, quantize_rtn, side_gaps  # noqa: E402

torch.set_num_threads(2)
torch.manual_seed(0)
sd = load_file(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gpt2.safetensors"))
NAMES = ["h.5.attn.c_attn.weight", "h.5.attn.c_proj.weight", "h.5.mlp.c_fc.weight", "h.5.mlp.c_proj.weight"]
W0 = [sd[n].t().contiguous().float() for n in NAMES]  # HF Conv1D (in, out) -> (out, in)
STEPS, WARM = 50, 15
TICKERS = {"ticker 4-bit": (7, 14), "ticker 4-bit /15": (7, 15), "ticker 8-bit": (127, 254)}  # (largest k, k per gap)
LRS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3)


def code(w):
    return (2 * w.view(w.size(0), -1, BLOCK) / block_scale(w)).round()


def carry_code(r, x_new, L, gen, D):
    up, dn = side_gaps(x_new)
    c = D * r / torch.where(r > 0, up, dn)
    lo = c.floor()
    return (lo + (torch.rand(c.shape, generator=gen) < c - lo)).clamp(-L, L)


class Ticker:
    def __init__(self, w, L, gen, D=None):
        self.L, self.gen, self.D = L, gen, D or 2 * L
        w_hat = quantize_rtn(w)
        s = block_scale(w)
        x_new = w_hat.view(w.size(0), -1, BLOCK) / s
        self.k = carry_code((w - w_hat).view_as(x_new) / s, x_new, L, gen, self.D)
        self.p = w_hat

    def step(self, delta):
        p = self.p
        s_old = block_scale(p)
        x = p.view(p.size(0), -1, BLOCK) / s_old
        up, dn = side_gaps(x)
        carry = (self.k / self.D * torch.where(self.k > 0, up, dn) * s_old).view_as(p)
        w_tilde = p + delta + carry
        w_hat = quantize_rtn(w_tilde)
        s_new = block_scale(w_tilde)
        x_new = w_hat.view_as(x) / s_new
        self.k = carry_code((w_tilde - w_hat).view_as(x) / s_new, x_new, self.L, self.gen, self.D)
        self.p = w_hat
        return w_hat

    def parked(self):
        return (self.k.abs() == self.L).float().mean().item()


def run(lrs):
    """All learning rates side by side: each step's update is computed once and applied at every rate."""
    V = ("float + RTN",) + tuple(TICKERS)
    out = {(lr, v): dict(changed=[], undone=[], parked=[]) for lr in lrs for v in V}
    for w0 in W0:
        gen = torch.Generator().manual_seed(1)
        st = {}
        for lr in lrs:
            st[lr] = dict(wf=w0.clone(), tk={v: Ticker(w0, L, torch.Generator().manual_seed(2 + i), D)
                                             for i, (v, (L, D)) in enumerate(TICKERS.items())})
            st[lr]["cur"] = {"float + RTN": code(w0), **{v: code(t.p) for v, t in st[lr]["tk"].items()}}
            st[lr]["prev"] = {v: None for v in V}
        buf = torch.zeros_like(w0)
        for t in range(STEPS):
            g = torch.randn(w0.shape, generator=gen)
            buf.lerp_(g, 0.05)
            u = g.lerp(buf, 0.95)
            upd = max(1, w0.size(0) / w0.size(1)) ** 0.5 * newton_schulz(u).float()
            for lr in lrs:
                S = st[lr]
                S["wf"] = S["wf"] - lr * upd
                new = {"float + RTN": code(S["wf"]), **{v: code(tk.step(-lr * upd)) for v, tk in S["tk"].items()}}
                for v in V:
                    d = new[v] - S["cur"][v]
                    if t >= WARM:
                        r = out[(lr, v)]
                        r["changed"].append((d != 0).float().mean().item())
                        if S["prev"][v] is not None:
                            pc = S["prev"][v] != 0
                            r["undone"].append(((d == -S["prev"][v]) & pc).sum().item() / max(1, pc.sum().item()))
                        if v in S["tk"]:
                            r["parked"].append(S["tk"][v].parked())
                    S["prev"][v], S["cur"][v] = d, new[v]
    return {k: {m: sum(x) / len(x) if x else None for m, x in r.items()} for k, r in out.items()}


def main():
    s = block_scale(W0[0]).mean().item()
    print(f"mean block scale (layer 5 c_attn): {s:.4g}; a 4-bit gap near the middle of the grid is 0.5-1 scale")
    print(f"{'lr':>7s}  {'variant':16s} {'changed/step':>12s} {'undone next':>11s} {'parked':>7s}")
    for (lr, v), x in run(LRS).items():
        pk = "" if x["parked"] is None else f"{100 * x['parked']:6.2f}%"
        print(f"{lr:7.0e}  {v:16s} {100 * x['changed']:11.3f}% {100 * x['undone']:10.1f}% {pk:>7s}")


if __name__ == "__main__":
    main()
