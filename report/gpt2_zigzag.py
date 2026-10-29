"""How much did the weights zig-zag in the GPT-2 continued training (side study G, stage 2, float Muon)? Not a milestone.

Net move D: |final - pretrained| per weight, from the saved stage-3 weights (data/gpt2_s3/). Path L: Muon's steps have a
size fixed by Newton-Schulz, so L = sum over steps of lr_t * |update entry|. The mean |entry| of each matrix shape is
measured on Newton-Schulz outputs of 8 matrices with the spectrum of a real gradient direction (the net move itself),
and of 8 random ones, as a bracket. Both are in units of each weight's 4-bit (E2M1) gap and its 2-bit gap at the start.
Also: the ratio of a step to the gaps and the carry units, at the peak learning rate.

Run:  python report/gpt2_zigzag.py  ->  report/data/gpt2_zigzag.json
"""
import importlib.util
import json
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


g2 = load("side_gpt2")
b2 = load("side_2bit")
torch.manual_seed(0)
cfg = json.load(open(os.path.join(ROOT, "runs/gpt2/s2/muon_float_lr0.001_s1.json")))
steps, warmup, lr = cfg["steps"], cfg["warmup"], cfg["muon_lr"]
lr_sum = lr * sum(g2.lr_frac(t, steps, warmup) for t in range(steps))
model = g2.load_pretrained(g2.GPT())
start = {k: v.detach().clone() for k, v in model.state_dict().items() if k.endswith(g2.MUON_NAMES)}
end = torch.load(os.path.join(ROOT, "data/gpt2_s3/muon_float_lr0.001_s1.pt"), map_location="cpu", weights_only=True)


def gap4(w):
    """The E2M1 gap above each weight's magnitude (as report/zigzag_probe.py), with the run's absmax/6 block scale."""
    s = g2.block_scale(w)
    a = (w.view(w.size(0), -1, 16) / s).abs()
    E = g2.E2M1
    i = torch.bucketize(a, E, right=True).clamp(1, len(E) - 1)
    return ((E[i] - E[i - 1]) * s).view_as(w)


def gap2(w):
    """The 2-bit grid is evenly spaced, so the gap is the block scale (1.25 x mean |w|)."""
    return b2.block_scale2(w).expand(w.size(0), -1, 16).reshape_as(w)


def ns_entry(shape, like=None):
    """Mean |entry| of Muon's scaled update for a matrix of this shape."""
    out = []
    for _ in range(8):
        G = torch.randn(shape) if like is None else like + 0.3 * like.std() * torch.randn(shape)
        u = max(1, shape[0] / shape[1]) ** 0.5 * g2.newton_schulz(G)
        out.append(u.abs().mean().item())
    return sum(out) / len(out)


rows, tot = [], {k: 0.0 for k in ("n", "L4", "D4", "L4r", "L2", "D2", "step4", "step2")}
for k, w0 in start.items():
    d = (end[k] - w0)
    e_real, e_rand = ns_entry(w0.shape, d), ns_entry(w0.shape)
    G4, G2 = gap4(w0), gap2(w0)
    L4, L4r = (lr_sum * e_real / G4).mean().item(), (lr_sum * e_rand / G4).mean().item()
    D4 = (d.abs() / G4).mean().item()
    L2, D2 = (lr_sum * e_real / G2).mean().item(), (d.abs() / G2).mean().item()
    step4, step2 = (lr * e_real / G4).median().item(), (lr * e_real / G2).median().item()
    kind = "attn" if "attn" in k else "ffn"
    rows.append(dict(name=k, kind=kind, L4=L4, L4_random=L4r, D4=D4, Z=L4 / D4, Z_random=L4r / D4, L2=L2, D2=D2,
                     step4_peak=step4, step2_peak=step2))
    n = w0.numel()
    for a, v in zip(("L4", "D4", "L4r", "L2", "D2", "step4", "step2"), (L4, D4, L4r, L2, D2, step4, step2)):
        tot[a] += v * n
    tot["n"] += n
    print(f"{k:28s} Z {L4 / D4:6.1f} (random {L4r / D4:5.1f})  path {L4:.3f} net {D4:.4f} gaps4  step {step4:.5f}")
n = tot.pop("n")
avg = {a: v / n for a, v in tot.items()}
avg.update(Z=avg["L4"] / avg["D4"], Z_random=avg["L4r"] / avg["D4"])
for kind in ("attn", "ffn"):
    sub = [r for r in rows if r["kind"] == kind]
    avg["Z_" + kind] = sum(r["L4"] for r in sub) / sum(r["D4"] for r in sub)
print(json.dumps({a: round(v, 5) for a, v in avg.items()}))
json.dump(dict(avg=avg, rows=rows, lr_sum=lr_sum), open(os.path.join(HERE, "data", "gpt2_zigzag.json"), "w"), indent=1)
