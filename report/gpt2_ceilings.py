"""Two ceilings for rounding-aware Muon on GPT-2 small, measured without training. Not a milestone.

1. Rotated storage. Muon's step is unchanged if a matrix is stored rotated (NS(G Q) = NS(G) Q for any orthogonal Q), so
   only the rounding changes. How much smaller are the 4-bit (E2M1, blocks of 16, absmax / 6) gaps after a Hadamard
   rotation with random signs? Measured on the pretrained weights and on the float-Muon stage-2 weights. Random rounding
   adds variance in proportion to the gap, so gap ratio = noise-variance ratio.
2. Aimed rounding error. A weight error E costs this layer's output E x, with mean energy trace(E Sigma E^T), where Sigma
   is the second moment of the layer's inputs. Evenly spread error pays the average eigenvalue of Sigma (AM). Error
   shaped into the weak input directions can at best pay the geometric mean (GM), so AM / GM is the ceiling. Sequential
   feedback (GPTQ order: each column's error is absorbed by the later columns) pays the mean conditional variance;
   that ratio is reported too. Sigma comes from the pretrained model on fixed FineWeb validation windows.

Run:  python report/gpt2_ceilings.py  ->  report/data/gpt2_ceilings.json
"""
import importlib.util
import json
import math
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
torch.set_num_threads(2)


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


g2 = load("side_gpt2")
gen = torch.Generator().manual_seed(0)
model = g2.load_pretrained(g2.GPT()).eval()
weights = {"pretrained": {k: v.detach().clone() for k, v in model.state_dict().items() if k.endswith(g2.MUON_NAMES)},
           "float_muon_s2": torch.load(os.path.join(ROOT, "data/gpt2_s3/muon_float_lr0.001_s1.pt"), map_location="cpu",
                                       weights_only=True)}


# ---------------------------------------------------------------- 1. rotated storage

def hadamard(n):
    H = torch.ones(1, 1)
    while H.size(0) < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def block_rot(n, b):
    """Block-diagonal Hadamard (blocks of b) with random signs: an orthogonal n x n matrix."""
    s = torch.randint(0, 2, (n,), generator=gen).float() * 2 - 1
    return torch.block_diag(*[hadamard(b)] * (n // b)) * s


def random_rot(n):
    q, r = torch.linalg.qr(torch.randn(n, n, generator=gen))
    return q * torch.sign(torch.diagonal(r))


def gap4(w):
    """The E2M1 gap above each weight's magnitude, with the absmax / 6 block scale (as report/gpt2_zigzag.py)."""
    s = g2.block_scale(w)
    a = (w.view(w.size(0), -1, 16) / s).abs()
    E = g2.E2M1
    i = torch.bucketize(a, E, right=True).clamp(1, len(E) - 1)
    return ((E[i] - E[i - 1]) * s).view_as(w)


def rot_stats(w):
    rms = w.pow(2).mean().sqrt()
    return dict(gap=(gap4(w).mean() / rms).item(), scale=(g2.block_scale(w).mean() * 6 / rms).item(),
                rtn=((g2.quantize_rtn(w) - w).pow(2).mean().sqrt() / rms).item())


ROTS = {"none": lambda w: w,
        "hadamard16 (in)": lambda w: w @ block_rot(w.size(1), 16),
        "hadamard256 (in)": lambda w: w @ block_rot(w.size(1), 256),
        "hadamard256 (in and out)": lambda w: block_rot(w.size(0), 256).T @ w @ block_rot(w.size(1), 256),
        "random full (in)": lambda w: w @ random_rot(w.size(1))}

# Muon does not see the rotation: check NS(G Q) = NS(G) Q on one matrix.
G, Q = torch.randn(768, 3072, generator=gen), block_rot(3072, 256)
equiv = ((g2.newton_schulz(G @ Q) - g2.newton_schulz(G) @ Q).norm() / g2.newton_schulz(G).norm()).item()
print(f"Muon rotation check: relative difference {equiv:.2e}")

rot = {}
for wname, sd in weights.items():
    tot = {r: {"gap": 0.0, "scale": 0.0, "rtn": 0.0} for r in ROTS}
    kinds = {}
    n_all = 0
    for k, w in sd.items():
        if not k.endswith(g2.MUON_NAMES):
            continue
        kind = next(m for m in g2.MUON_NAMES if k.endswith(m)).removesuffix(".weight")
        n_all += w.numel()
        for r, f in ROTS.items():
            st = rot_stats(f(w.float()))
            for a, v in st.items():
                tot[r][a] += v * w.numel()
                kinds.setdefault(kind, {}).setdefault(r, {}).setdefault(a, []).append(v)
    rot[wname] = {"all": {r: {a: v / n_all for a, v in d.items()} for r, d in tot.items()},
                  "by_kind": {kd: {r: {a: sum(v) / len(v) for a, v in d.items()} for r, d in rs.items()}
                              for kd, rs in kinds.items()}}
    for r, d in rot[wname]["all"].items():
        print(f"{wname:14s} {r:26s} gap/rms {d['gap']:.4f}  scale/rms {d['scale']:.3f}  rtn/rms {d['rtn']:.4f}")


# ---------------------------------------------------------------- 2. lopsided inputs

N_WIN = 24
xs, _ = g2.fixed_windows(g2.load_shard("fineweb_val_000000.bin"), N_WIN, 1024)
mom, count = {}, {}


def hook(name):
    def f(mod, inp):
        x = inp[0].reshape(-1, inp[0].size(-1)).double()
        mom[name] = mom.get(name, 0) + x.T @ x
        count[name] = count.get(name, 0) + x.size(0)
    return f


for i, b in enumerate(model.h):
    for m in ("c_attn", "attn_c_proj", "c_fc", "mlp_c_proj"):
        getattr(b, m).register_forward_pre_hook(hook(f"h.{i}.{m}"))
with torch.no_grad():
    for j in range(0, N_WIN, 2):
        model(xs[j:j + 2])
        print(f"inputs: windows {j + 2}/{N_WIN}", flush=True)

rows = []
for name, M in mom.items():
    S = M / count[name]
    lam = torch.linalg.eigvalsh(S).flip(0).clamp_min(0)
    d, am = lam.numel(), lam.mean().item()
    gm = lambda floor: math.exp(torch.log(lam.clamp_min(floor * am)).mean().item())
    # Sequential feedback in column order: column j's error is absorbed by columns j+1..d, so it pays the variance of
    # input j left over after predicting it from inputs j+1..d (Cholesky of the reversed matrix).
    Lr = torch.linalg.cholesky(S.flip(0, 1) + 1e-10 * am * torch.eye(d, dtype=S.dtype))
    seq = (torch.diagonal(Lr) ** 2).mean().item()
    cum = lam.cumsum(0) / lam.sum()
    rows.append(dict(name=name, kind=name.split(".", 2)[2], layer=int(name.split(".")[1]), dim=d,
                     top1pct_share=cum[max(1, d // 100) - 1].item(), top10pct_share=cum[d // 10 - 1].item(),
                     top1_share=cum[0].item(), eff_rank_frac=(lam.sum() ** 2 / (lam ** 2).sum()).item() / d,
                     am_over_gm=am / gm(1e-12), am_over_gm_floor1e4=am / gm(1e-4), am_over_seq=am / seq,
                     spectrum=[lam[int(q * (d - 1))].item() / am for q in (0, .01, .1, .25, .5, .75, .9, .99, 1)]))
    r = rows[-1]
    print(f"{name:16s} top1% {r['top1pct_share']:.3f} top10% {r['top10pct_share']:.3f} "
          f"AM/GM {r['am_over_gm']:9.1f} (floor {r['am_over_gm_floor1e4']:7.1f})  AM/seq {r['am_over_seq']:7.1f}")

summary = {}
for kind in ("c_attn", "attn_c_proj", "c_fc", "mlp_c_proj"):
    sub = [r for r in rows if r["kind"] == kind]
    geo = lambda a: math.exp(sum(math.log(r[a]) for r in sub) / len(sub))
    summary[kind] = {a: geo(a) for a in ("am_over_gm", "am_over_gm_floor1e4", "am_over_seq")}
    summary[kind]["top10pct_share"] = sum(r["top10pct_share"] for r in sub) / len(sub)
    print(kind, {a: round(v, 2) for a, v in summary[kind].items()})

json.dump(dict(muon_rotation_check=equiv, rotation=rot, inputs=dict(tokens=N_WIN * 1024, rows=rows, by_kind=summary)),
          open(os.path.join(HERE, "data", "gpt2_ceilings.json"), "w"), indent=1)
