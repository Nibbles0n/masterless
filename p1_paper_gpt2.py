"""Paper sprint (PAPER_PLAN.md), steps G0, P1, P2, P3, P4: GPT-2 with 4-bit weights and no float copy, every arm.

A copy of side_gpt2.py, rebuilt for speed: the 48 block matrices live in four stacked tensors, one per shape
(c_attn, attn_c_proj, c_fc, mlp_c_proj: (layers, out, in)), so Newton-Schulz runs batched and each arm's update is
one compiled elementwise function per shape. Vocabulary padded to 50,304 (the extra logits are sliced off before the
loss, so the pretrained model scores as before). Training tokens live on the GPU.

Arms (bytes per weight of the block matrices: weights + optimizer state, gradients not counted):
  muon_float            Muon, fp32 weights and momentum                                                       8
  adamw_float           AdamW on everything, fp32                                                            12
  q4_float_copy         4-bit weights in the forward pass, fp32 copy takes Muon's updates (straight-through)   8
  q4_float_copy_adamw   the same with AdamW (standard 4-bit training)                                        12
  q4_t{b}_m8            ours, the ticker: 4-bit weights; target = weight + carry - lr*update; the new weight is
                        the nearest 4-bit value; the carry is an integer k in -L..L meaning k/(2L) of the gap on
                        its own side, rounded at random, L = 2^(b-1) - 1; 8-bit momentum      0.78 + b/8 + 1.03
  q4_kahan_m8           the ticker with the carry kept exactly (fp32): Kahan summation on 4-bit weights      5.78
  q4_kahan_m32          the same with fp32 momentum: mathematically the 4-bit float copy (G0 limit test)     8.75
  q4_sr_m8              naive: no copy, each step rounded at random onto the 4-bit grid, 8-bit momentum      1.78
  eco_muon_q4           ECO with Muon: as q4_sr_m8, the rounding error fed into the momentum before
                        Newton-Schulz (m4_eco.py's muon_pre rule), 8-bit momentum                            1.78
  bf16_sr_m8            bf16 weights, no copy, each step rounded at random to bf16, 8-bit momentum           3.03
  eco_adamw_q4          ECO with AdamW, as the ECO paper runs it (Algorithm 3 of arXiv 2601.22101): 4-bit weights
                        rounded at random, the rounding error fed into the first moment, fp32 moments         8.75
P4 ablations, each q4_t8_m8 with one thing changed:
  q4_t8_m8_abs          the carry in absolute units: k/(2L) of the block's largest gap (2 x scale), not of its own gap
  q4_t8_m8_srw          the new weight rounded at random instead of to nearest; the carry then spans a whole gap
                        (k/L of the gap on its side) so the leftover still fits
  q4_t8_m8_cap          Error Diffusion's cap: each step's move -lr*update is clipped to 0.25 of the gap in its direction
4-bit grid: E2M1 values with sign, times an fp32 scale per block of 16 inputs (absmax / 6), as in side_gpt2.py.

Modes: --init pretrained (OpenAI GPT-2 small, continued training on FineWeb) or --init scratch (GPT-2 architecture at
--n-layer / --n-embd / --n-head, GPT-2 initialisation). Seed k fixes the random rounding (seed 1000 + k) and the data
order (data seed k - 1, so seed 1 repeats side study G's batches).

Run:  python p1_paper_gpt2.py --arm q4_t8_m8 --steps 3050 --out runs/paper/P1
"""
import argparse
import contextlib
import json
import math
import os
import time
import urllib.request

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
HF = "https://huggingface.co"
GPT2_URL = HF + "/openai-community/gpt2/resolve/main/model.safetensors"
SHARD_URL = HF + "/datasets/kjj0/fineweb10B-gpt2/resolve/main/{}"
VOCAB, VOCAB_PAD, CTX = 50257, 50304, 1024
KINDS = ("c_attn", "attn_c_proj", "c_fc", "mlp_c_proj")


# ---------------------------------------------------------------- data

def fetch(url, name):
    path = os.path.join(HERE, "data", name)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


def load_shard(name):
    path = fetch(SHARD_URL.format(name), name)
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520 and header[1] == 1, "not a modded-nanogpt token shard"
    return np.memmap(path, dtype=np.uint16, mode="r", offset=256 * 4, shape=(int(header[2]),))


class Tokens:
    """Training tokens on the device. Batches are drawn exactly as side_gpt2.get_batch draws them (the same random
    starts from the same CPU generator), then gathered on the device."""

    def __init__(self, shards, device):
        data = np.concatenate([np.asarray(s) for s in shards]) if len(shards) > 1 else np.asarray(shards[0])
        self.n = len(data)
        self.tok = torch.from_numpy(data.astype(np.int32)).to(device)
        self.device = device

    def batch(self, n, T, gen):
        ix = torch.randint(self.n - T - 1, (n,), generator=gen)
        idx = ix.to(self.device)[:, None] + torch.arange(T + 1, device=self.device)
        xy = self.tok[idx].long()
        return xy[:, :-1], xy[:, 1:]


def fixed_windows(data, n, T):
    starts = np.linspace(0, len(data) - T - 2, n).astype(np.int64)
    xy = torch.from_numpy(np.stack([data[s:s + T + 1] for s in starts]).astype(np.int64))
    return xy[:, :-1], xy[:, 1:]


# ---------------------------------------------------------------- 4-bit grid (any number of leading dimensions)

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
BLOCK, MBLOCK = 16, 128
GRID = torch.cat([-E2M1.flip(0)[:-1], E2M1])  # the 15 signed values, -6..6
GRID_MID = (GRID[1:] + GRID[:-1]) / 2
GAP_UP = torch.cat([GRID[1:] - GRID[:-1], torch.tensor([2.0])])
GAP_DN = torch.cat([torch.tensor([2.0]), GRID[1:] - GRID[:-1]])
_dev_cache = {}


def dev(t, device):
    key = (id(t), str(device))
    if key not in _dev_cache:
        _dev_cache[key] = t.to(device)
    return _dev_cache[key]


def blocks(w, size=BLOCK):
    return w.reshape(*w.shape[:-1], w.shape[-1] // size, size)


def block_scale(w):
    return blocks(w).abs().amax(-1, keepdim=True).clamp_min(1e-12) / 6


def quantize_rtn(w):
    """Round to nearest on the scaled E2M1 grid (side_gpt2.quantize_rtn, for stacked matrices)."""
    grid = dev(E2M1, w.device)
    wb, s = blocks(w), block_scale(w)
    x = wb / s
    idx = torch.bucketize(x.abs(), (grid[1:] + grid[:-1]) / 2)
    return (grid[idx] * x.sign() * s).reshape(w.shape)


def quantize_sr4(w):
    """Round at random to one of the two neighbouring scaled E2M1 values (scale from w)."""
    grid = dev(E2M1, w.device)
    wb, s = blocks(w), block_scale(w)
    a = (wb / s).abs()
    i = (torch.bucketize(a, grid, right=True) - 1).clamp(0, 6)
    lo, hi = grid[i], grid[i + 1]
    q = torch.where(torch.rand_like(a) < (a - lo) / (hi - lo), hi, lo)
    return (q.clamp(max=6.0) * wb.sign() * s).reshape(w.shape)


def on_grid(w):
    v = (blocks(w) / block_scale(w)).abs()
    return bool(((v.unsqueeze(-1) - dev(E2M1, w.device)).abs().amin(-1) < 1e-4).all())


def side_gaps(x):
    """For codes x on the signed grid: the gaps to the next value up and down."""
    i = torch.bucketize(x, dev(GRID_MID, x.device))
    return dev(GAP_UP, x.device)[i], dev(GAP_DN, x.device)[i]


def store_int8(a):
    """Round a at random to 8-bit integers times one fp32 scale per 128 weights along the last dimension."""
    ab = blocks(a, MBLOCK)
    s = ab.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 127
    x = ab / s
    lo = x.floor()
    return ((lo + (torch.rand_like(x) < x - lo)) * s).reshape(a.shape)


def round_bf16_sr(x):
    """fp32 -> the nearest bf16 value below or above (in magnitude), at random in proportion."""
    xi = x.contiguous().view(torch.int32)
    r = torch.randint(0, 1 << 16, x.shape, device=x.device, dtype=torch.int32)
    return ((xi + r) & -65536).view(torch.float32)


def code4(w):
    return (2 * blocks(w) / block_scale(w)).round()


# ---------------------------------------------------------------- the arms' updates (compiled, one call per shape)

def ticker_update(p, k, update, lr, L: int, D: int, var: str = ""):
    """One ticker step for stacked matrices. Returns the new 4-bit weights, the new carry codes and the number of
    weights whose leftover (before clamping) was more than the carry can hold (must be 0). var: a P4 ablation."""
    s_old = block_scale(p)
    x = blocks(p) / s_old
    up, dn = side_gaps(x)
    kf = blocks(k.float())
    unit = 2.0 if var == "abs" else torch.where(kf > 0, up, dn)
    carry = (kf / D * unit * s_old).reshape(p.shape)
    move = -lr * update
    if var == "cap":
        mb = blocks(move) / s_old
        move = (torch.minimum(torch.maximum(mb, -0.25 * dn), 0.25 * up) * s_old).reshape(p.shape)
    w_tilde = p + move + carry
    w_hat = quantize_sr4(w_tilde) if var == "srw" else quantize_rtn(w_tilde)
    s_new = block_scale(w_tilde)
    x_new = blocks(w_hat) / s_new
    r = blocks(w_tilde - w_hat) / s_new
    up2, dn2 = side_gaps(x_new)
    c = D * r / (2.0 if var == "abs" else torch.where(r > 0, up2, dn2))
    viol = (c.abs() > L * (1 + 1e-5) + 1e-4).sum()  # float error grows with L
    lo = c.floor()
    k_new = (lo + (torch.rand_like(c) < c - lo)).clamp(-L, L).reshape(p.shape).to(k.dtype)
    return w_hat, k_new, viol


def kahan_update(p, carry, update, lr):
    w_tilde = p - lr * update + carry
    w_hat = quantize_rtn(w_tilde)
    return w_hat, w_tilde - w_hat


def sr4_update(p, update, lr):
    w_tilde = p - lr * update
    w_hat = quantize_sr4(w_tilde)
    return w_hat, w_tilde - w_hat


def bf16sr_update(p, update, lr):
    return round_bf16_sr(p - lr * update)


def newton_schulz_batched(G, dtype, steps=5):
    """Approximately orthogonalise each matrix of the stack G (layers, rows, cols), as Keller Jordan's Muon."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(dtype)
    tall = X.size(1) > X.size(2)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return (X.mT if tall else X).float()


# ---------------------------------------------------------------- arms

def arm_config(arm):
    var = ""
    if arm.startswith("q4_t8_m8_"):
        arm, var = "q4_t8_m8", arm[len("q4_t8_m8_"):]
    if arm.startswith("q4_t") and arm.endswith("_m8"):
        b = int(arm[4:-3])
        return dict(opt="muon", store="ticker", bits=b, mom="int8", fake4=False, var=var,
                    bytes=0.5 + 4 / 16 + b / 8 + 1 + 4 / MBLOCK)
    return {
        "muon_float": dict(opt="muon", store="float", mom="fp32", fake4=False, bytes=8),
        "adamw_float": dict(opt="adamw", store="float", mom=None, fake4=False, bytes=12),
        "q4_float_copy": dict(opt="muon", store="float", mom="fp32", fake4=True, bytes=8),
        "q4_float_copy_adamw": dict(opt="adamw", store="float", mom=None, fake4=True, bytes=12),
        "q4_kahan_m8": dict(opt="muon", store="kahan", mom="int8", fake4=False, bytes=0.5 + 0.25 + 4 + 1 + 4 / MBLOCK),
        "q4_kahan_m32": dict(opt="muon", store="kahan", mom="fp32", fake4=False, bytes=0.5 + 0.25 + 4 + 4),
        "q4_sr_m8": dict(opt="muon", store="sr4", mom="int8", fake4=False, bytes=0.5 + 0.25 + 1 + 4 / MBLOCK),
        "eco_muon_q4": dict(opt="muon", store="eco", mom="int8", fake4=False, bytes=0.5 + 0.25 + 1 + 4 / MBLOCK),
        "bf16_sr_m8": dict(opt="muon", store="bf16sr", mom="int8", fake4=False, bytes=2 + 1 + 4 / MBLOCK),
        "eco_adamw_q4": dict(opt="eco_adamw", store="eco", mom="fp32", fake4=False, bytes=0.5 + 0.25 + 8),
    }[arm]


ARMS = ("muon_float", "adamw_float", "q4_float_copy", "q4_float_copy_adamw", "q4_t4_m8", "q4_t6_m8", "q4_t8_m8",
        "q4_t12_m8", "q4_t16_m8", "q4_t8_m8_abs", "q4_t8_m8_srw", "q4_t8_m8_cap", "q4_kahan_m8", "q4_kahan_m32", "q4_sr_m8", "eco_muon_q4", "bf16_sr_m8",
        "eco_adamw_q4")


# ---------------------------------------------------------------- model (GPT-2, block matrices stacked by shape)

class GPT(nn.Module):
    def __init__(self, n_layer=12, n_head=12, d=768):
        super().__init__()
        self.n_layer, self.n_head, self.d = n_layer, n_head, d
        shapes = {"c_attn": (3 * d, d), "attn_c_proj": (d, d), "c_fc": (4 * d, d), "mlp_c_proj": (d, 4 * d)}
        self.W = nn.ParameterDict({k: nn.Parameter(torch.zeros(n_layer, *s)) for k, s in shapes.items()})
        self.B = nn.ParameterDict({k: nn.Parameter(torch.zeros(n_layer, s[0])) for k, s in shapes.items()})
        self.ln1 = nn.ModuleList(nn.LayerNorm(d) for _ in range(n_layer))
        self.ln2 = nn.ModuleList(nn.LayerNorm(d) for _ in range(n_layer))
        self.wte = nn.Embedding(VOCAB_PAD, d)
        self.wpe = nn.Embedding(CTX, d)
        self.ln_f = nn.LayerNorm(d)
        self.fake4 = False

    def forward(self, idx, targets, reduction="mean"):
        B_, T = idx.shape
        W = {k: (w + (quantize_rtn(w.detach()) - w).detach()) if self.fake4 else w for k, w in self.W.items()}
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        C, H = self.d, self.n_head
        for i in range(self.n_layer):
            q, k, v = F.linear(self.ln1[i](x), W["c_attn"][i], self.B["c_attn"][i]).split(C, dim=2)
            q, k, v = (t.view(B_, T, H, C // H).transpose(1, 2) for t in (q, k, v))
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B_, T, C)
            x = x + F.linear(a, W["attn_c_proj"][i], self.B["attn_c_proj"][i])
            h = F.gelu(F.linear(self.ln2[i](x), W["c_fc"][i], self.B["c_fc"][i]), approximate="tanh")
            x = x + F.linear(h, W["mlp_c_proj"][i], self.B["mlp_c_proj"][i])
        logits = F.linear(self.ln_f(x), self.wte.weight)[..., :VOCAB].float()
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1), reduction="none").view(B_, T)
        return loss.mean() if reduction == "mean" else loss.mean(dim=1)


def load_pretrained(model):
    from safetensors.torch import load_file
    sd = {k.removeprefix("transformer."): v for k, v in load_file(fetch(GPT2_URL, "gpt2.safetensors")).items()}
    ren = {"attn.c_attn": "c_attn", "attn.c_proj": "attn_c_proj", "mlp.c_fc": "c_fc", "mlp.c_proj": "mlp_c_proj"}
    with torch.no_grad():
        model.wte.weight.zero_()
        model.wte.weight[:VOCAB].copy_(sd["wte.weight"])
        model.wpe.weight.copy_(sd["wpe.weight"])
        model.ln_f.weight.copy_(sd["ln_f.weight"])
        model.ln_f.bias.copy_(sd["ln_f.bias"])
        for i in range(model.n_layer):
            for a, b in ren.items():
                model.W[b][i].copy_(sd[f"h.{i}.{a}.weight"].t())  # Conv1D stores (in, out)
                model.B[b][i].copy_(sd[f"h.{i}.{a}.bias"])
            for ln, mod in (("ln_1", model.ln1[i]), ("ln_2", model.ln2[i])):
                mod.weight.copy_(sd[f"h.{i}.{ln}.weight"])
                mod.bias.copy_(sd[f"h.{i}.{ln}.bias"])
    return model


def init_scratch(model, gen):
    """GPT-2 initialisation: normal(0, 0.02); the two residual projections scaled by 1/sqrt(2 * layers)."""
    with torch.no_grad():
        for k, w in model.W.items():
            std = 0.02 / math.sqrt(2 * model.n_layer) if k in ("attn_c_proj", "mlp_c_proj") else 0.02
            w.copy_(torch.randn(w.shape, generator=gen) * std)
        model.wte.weight.copy_(torch.randn(model.wte.weight.shape, generator=gen) * 0.02)
        model.wpe.weight.copy_(torch.randn(model.wpe.weight.shape, generator=gen) * 0.01)
    return model


# ---------------------------------------------------------------- optimizer

class StackMuon(torch.optim.Optimizer):
    """Muon on the four stacked block-matrix tensors, in the arm's storage."""

    def __init__(self, params, lr, cfg, ns_dtype, momentum=0.95, check=False, compile=True):
        super().__init__(params, dict(lr=lr, momentum=momentum))
        self.cfg, self.ns_dtype, self.check = cfg, ns_dtype, check
        wrap = torch.compile if compile else (lambda fn: fn)
        st = cfg["store"]
        b = cfg.get("bits")
        self.L, self.D = ((1 << (b - 1)) - 1, (1 << b) - 2) if b else (None, None)
        self.var = cfg.get("var", "")
        if self.var == "srw":
            self.D = self.L  # random rounding leaves up to a whole gap
        self.f = {"ticker": ticker_update, "kahan": kahan_update, "sr4": sr4_update, "eco": sr4_update,
                  "bf16sr": bf16sr_update, "float": None}[st]
        self.f = self.f and wrap(self.f)
        self.int8 = wrap(store_int8)
        self.viol = None
        self.count_changes = False
        self.changed4, self.total = 0, 0

    @torch.no_grad()
    def init_storage(self):
        """Round the starting weights onto the arm's grid (to nearest); the carry starts as the leftover."""
        st = self.cfg["store"]
        for p in self.param_groups[0]["params"]:
            if st in ("ticker", "kahan", "sr4", "eco"):
                w_hat = quantize_rtn(p)
                if st == "ticker":
                    s = block_scale(p)
                    x_new = blocks(w_hat) / s
                    r = blocks(p - w_hat) / s
                    up, dn = side_gaps(x_new)
                    c = self.D * r / (2.0 if self.var == "abs" else torch.where(r > 0, up, dn))
                    lo = c.floor()
                    k = (lo + (torch.rand_like(c) < c - lo)).clamp(-self.L, self.L).reshape(p.shape)
                    self.state[p]["carry"] = k.to(torch.int8 if self.cfg["bits"] <= 8 else torch.int16)
                elif st == "kahan":
                    self.state[p]["carry"] = p - w_hat
                p.copy_(w_hat)
            elif st == "bf16sr":
                p.copy_(p.to(torch.bfloat16).float())

    @torch.no_grad()
    def step(self):
        group = self.param_groups[0]
        lr, beta = group["lr"], group["momentum"]
        st = self.cfg["store"]
        lr_t = torch.tensor(lr, device=group["params"][0].device)
        for p in group["params"]:
            s = self.state[p]
            buf = s.setdefault("momentum", torch.zeros_like(p))
            buf.lerp_(p.grad, 1 - beta)
            u = p.grad.lerp(buf, beta)  # Nesterov
            if self.cfg["mom"] == "int8" and st != "eco":
                buf.copy_(self.int8(buf))
            update = max(1, p.size(1) / p.size(2)) ** 0.5 * newton_schulz_batched(u, self.ns_dtype)
            c4 = code4(p) if self.count_changes else None
            if st == "float":
                p.add_(update, alpha=-lr)
            elif st == "ticker":
                w_hat, k_new, viol = self.f(p, s["carry"], update, lr_t, self.L, self.D, self.var)
                s["carry"] = k_new
                self.viol = viol if self.viol is None else self.viol + viol
                p.copy_(w_hat)
            elif st == "kahan":
                w_hat, s["carry"] = self.f(p, s["carry"], update, lr_t)
                p.copy_(w_hat)
            elif st == "sr4":
                p.copy_(self.f(p, update, lr_t)[0])
            elif st == "eco":
                w_hat, e = self.f(p, update, lr_t)
                lr_eff = lr * update.norm(dim=(1, 2), keepdim=True) / u.norm(dim=(1, 2), keepdim=True).clamp_min(1e-30)
                buf.add_((1 - 1 / beta) / lr_eff.clamp_min(1e-30) * e)  # m4_eco.py's muon_pre rule (ECO eq. 6-7)
                buf.copy_(self.int8(buf))
                p.copy_(w_hat)
            elif st == "bf16sr":
                p.copy_(self.f(p, update, lr_t))
            if c4 is not None:
                self.changed4 += (code4(p) != c4).sum().item()
                self.total += p.numel()

    def read_changed(self):
        out = self.changed4 / self.total if self.total else None
        self.changed4, self.total = 0, 0
        return out


class StackECOAdam(StackMuon):
    """AdamW on the stacked block matrices with 4-bit weights and no float copy, ECO's rule for Adam (Algorithm 3 of
    arXiv 2601.22101, as m4_eco.py): the target is rounded at random onto the 4-bit grid and the rounding error goes
    into the first moment, scaled by Adam's per-weight step. fp32 moments, as in the ECO paper."""

    def __init__(self, params, lr, cfg, weight_decay, betas=(0.9, 0.95), eps=1e-8):
        torch.optim.Optimizer.__init__(self, params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.cfg, self.check, self.var = cfg, False, ""
        self.viol, self.count_changes, self.changed4, self.total = None, False, 0, 0

    @torch.no_grad()
    def step(self):
        group = self.param_groups[0]
        lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
        for p in group["params"]:
            st = self.state[p]
            if "m" not in st:
                st.update(t=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
            st["t"] += 1
            t, m, v = st["t"], st["m"], st["v"]
            m.lerp_(p.grad, 1 - b1)
            v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
            denom = (v / (1 - b2 ** t)).sqrt() + eps
            w_tilde = p * (1 - lr * wd) - lr * (m / (1 - b1 ** t)) / denom
            c4 = code4(p) if self.count_changes else None
            w_hat = quantize_sr4(w_tilde)
            m.add_((1 - b1 ** t) / max(lr, 1e-30) * (1 - 1 / b1) * denom * (w_tilde - w_hat))
            p.copy_(w_hat)
            if c4 is not None:
                self.changed4 += (code4(p) != c4).sum().item()
                self.total += p.numel()


def build_optimizers(model, cfg, muon_lr, adamw_lr, weight_decay, ns_dtype, check, compile=True):
    hidden = list(model.W.values())
    decay = [model.wte.weight, model.wpe.weight]
    rest = [p for n, p in model.named_parameters()
            if not n.startswith("W.") and n not in ("wte.weight", "wpe.weight")]
    out = []
    if cfg["opt"] == "adamw":
        decay = decay + hidden
    elif cfg["opt"] == "eco_adamw":
        out.append((StackECOAdam(hidden, adamw_lr, cfg, weight_decay), adamw_lr))
    else:
        out.append((StackMuon(hidden, muon_lr, cfg, ns_dtype, check=check, compile=compile), muon_lr))
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": rest, "weight_decay": 0.0}]
    out.append((torch.optim.AdamW(groups, lr=adamw_lr, betas=(0.9, 0.95), fused=torch.cuda.is_available()),
                adamw_lr))
    return out


def lr_frac(it, steps, warmup, min_frac=0.1):
    if it < warmup:
        return (it + 1) / warmup
    frac = (it - warmup) / max(1, steps - warmup)
    return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, frac)))


# ---------------------------------------------------------------- eval

@torch.no_grad()
def evaluate(fwd, windows, device, amp, bs):
    x_all, y_all = windows
    per_window = []
    for i in range(0, len(x_all), bs):
        x, y = x_all[i:i + bs].to(device), y_all[i:i + bs].to(device)
        with amp():
            per_window.append(fwd(x, y, reduction="none").float().cpu())
    ce = torch.cat(per_window).double()
    evaluate.last = ce
    return ce.mean().item(), (ce.std(unbiased=True) / math.sqrt(len(ce))).item()


@torch.no_grad()
def evaluate_rounded(fwd, params, windows, device, amp, bs):
    saved = [p.clone() for p in params]
    for p in params:
        p.copy_(quantize_rtn(p))
    out = evaluate(fwd, windows, device, amp, bs)
    for p, w in zip(params, saved):
        p.copy_(w)
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--init", choices=("pretrained", "scratch"), default="pretrained")
    ap.add_argument("--n-layer", type=int, default=12)
    ap.add_argument("--n-embd", type=int, default=768)
    ap.add_argument("--n-head", type=int, default=12)
    ap.add_argument("--muon-lr", type=float, default=0.001)
    ap.add_argument("--adamw-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3050)
    ap.add_argument("--stop-at", type=int, default=None, help="stop early (the schedule still follows --steps)")
    ap.add_argument("--seqs-per-step", type=int, default=32)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=None, help="default: 5%% of the steps")
    ap.add_argument("--eval-interval", type=int, default=100)
    ap.add_argument("--eval-windows", type=int, default=256)
    ap.add_argument("--eval-bs", type=int, default=8)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--train-shards", default="fineweb_train_000001.bin")
    ap.add_argument("--val-shard", default="fineweb_val_000000.bin")
    ap.add_argument("--ns-dtype", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    ap.add_argument("--amp", choices=("fp16", "bf16"), default="fp16",
                    help="fp16 (with loss scaling) by default on every GPU: bf16 moves the validation loss by about 0.013")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--check", action="store_true", help="G0: check carry range and grid every step")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "paper", "P1"))
    args = ap.parse_args()
    assert args.seqs_per_step % args.micro == 0
    accum = args.seqs_per_step // args.micro
    warmup = args.warmup if args.warmup is not None else max(1, args.steps // 20)
    stop = args.stop_at if args.stop_at is not None else args.steps
    cfg = arm_config(args.arm)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)  # strict: with warn_only, flash attention stays non-deterministic
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.amp]
    amp = (lambda: torch.autocast("cuda", dtype=amp_dtype)) if device == "cuda" else contextlib.nullcontext
    use_scaler = device == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    ns_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16,
                "auto": amp_dtype if device == "cuda" else torch.float32}[args.ns_dtype]
    # TF32 off: on an A100, compiled fp16 matmuls on the raw weights with TF32 on raised the validation loss by 0.014
    # (G0, 2026-10-28); every other setting agreed within 0.0005.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    train = Tokens([load_shard(s) for s in args.train_shards.split(",")], device)
    windows = fixed_windows(load_shard(args.val_shard), args.eval_windows, args.seq_len)

    torch.manual_seed(1000 + args.seed)  # the random rounding's stream (device RNG)
    data_gen = torch.Generator().manual_seed(args.seed - 1)
    model = GPT(args.n_layer, args.n_head, args.n_embd)
    if args.init == "pretrained":
        assert (args.n_layer, args.n_embd, args.n_head) == (12, 768, 12)
        load_pretrained(model)
    else:
        init_scratch(model, torch.Generator().manual_seed(args.seed))
    model = model.to(device)
    hidden = list(model.W.values())
    fwd = model if args.no_compile else torch.compile(model)

    val0, se0 = evaluate(fwd, windows, device, amp, args.eval_bs)
    opts = build_optimizers(model, cfg, args.muon_lr, args.adamw_lr, args.weight_decay, ns_dtype, args.check,
                            not args.no_compile)
    muon = opts[0][0] if isinstance(opts[0][0], StackMuon) else None
    if muon is not None:
        muon.init_storage()
    model.fake4 = cfg["fake4"]

    os.makedirs(args.out, exist_ok=True)
    lr_tag = args.adamw_lr if cfg["opt"] in ("adamw", "eco_adamw") else args.muon_lr
    stem = os.path.join(args.out, f"{args.arm}_lr{lr_tag:g}_s{args.seed}{args.tag}")
    log = open(stem + ".jsonl", "w")
    n_hidden = sum(p.numel() for p in hidden)
    header = dict(study="paper", script="p1_paper_gpt2.py", arm=args.arm, init=args.init, n_layer=args.n_layer,
                  n_embd=args.n_embd, n_head=args.n_head, muon_lr=args.muon_lr, adamw_lr=args.adamw_lr,
                  seed=args.seed, steps=args.steps, stop_at=stop, warmup=warmup, seqs_per_step=args.seqs_per_step,
                  micro=args.micro, seq_len=args.seq_len, tokens_per_step=args.seqs_per_step * args.seq_len,
                  eval_windows=args.eval_windows, weight_decay=args.weight_decay, train_shards=args.train_shards,
                  compile=not args.no_compile, device=device,
                  gpu=torch.cuda.get_device_name() if device == "cuda" else None,
                  amp=str(amp_dtype) if device == "cuda" else "fp32", ns_dtype=str(ns_dtype), torch=torch.__version__,
                  hidden_weights=n_hidden, params=sum(p.numel() for p in model.parameters()) - (VOCAB_PAD - VOCAB) *
                  args.n_embd, state_bytes_per_weight=round(cfg["bytes"], 4), check=args.check,
                  deterministic=args.deterministic, val_ce_start=round(val0, 5), val_se_start=round(se0, 5))
    print(json.dumps(header), flush=True)

    def write(row):
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()

    train_loss, grad_norm, skipped, evals = [], [], 0, []
    t0, t_log, tok_log, t_train = time.time(), time.time(), 0, 0.0
    diverged, ce, se, per_window = False, float("nan"), float("nan"), None
    for it in range(stop + 1):
        if it % args.eval_interval == 0 or it == stop:
            ce, se = evaluate(fwd, windows, device, amp, args.eval_bs)
            per_window = evaluate.last
            evals.append((it, ce))
            row = dict(step=it, val_ce=round(ce, 5), val_se=round(se, 5), wall_s=round(time.time() - t0, 1))
            if args.check and muon is not None and cfg["store"] in ("ticker", "kahan", "sr4", "eco"):
                row["on_grid"] = all(on_grid(p) for p in hidden)
                assert row["on_grid"], "a matrix left the 4-bit grid"
            write(row)
            if not math.isfinite(ce):
                diverged = True
                break
            t_log = time.time()
        if it == stop:
            break
        f = lr_frac(it, args.steps, warmup)
        for opt, peak in opts:
            for g in opt.param_groups:
                g["lr"] = peak * f
        if muon is not None:
            muon.count_changes = (it + 1) % args.log_interval == 0
        ts = time.time()
        loss_sum = torch.zeros((), device=device)
        for _ in range(accum):
            x, y = train.batch(args.micro, args.seq_len, data_gen)
            with amp():
                loss = fwd(x, y) / accum
            scaler.scale(loss).backward()
            loss_sum += loss.detach()
        for opt, _ in opts:
            scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scale_before = scaler.get_scale() if use_scaler else None
        for opt, _ in opts:
            scaler.step(opt)
        scaler.update()
        for opt, _ in opts:
            opt.zero_grad(set_to_none=True)
        if use_scaler:
            skipped += int(scaler.get_scale() < scale_before)
        if args.check and muon is not None and muon.viol is not None:
            v = int(muon.viol.item())
            assert v == 0, f"step {it + 1}: {v} carries were more than half a gap before clamping"
        tok_log += args.seqs_per_step * args.seq_len
        if (it + 1) % args.log_interval == 0 or it + 1 == stop:
            ls, g = loss_sum.item(), gn.item()
            train_loss.append(round(ls, 4))
            grad_norm.append(round(g, 4) if math.isfinite(g) else None)
            if device == "cuda":
                torch.cuda.synchronize()
            now = time.time()
            t_train += now - t_log
            write(dict(step=it + 1, train_loss=round(ls, 4), grad_norm=round(g, 4) if math.isfinite(g) else None,
                       changed4=muon.read_changed() if muon is not None else None, lr_frac=round(f, 4),
                       tok_per_s=round(tok_log / (now - t_log)), skipped=skipped,
                       viol=int(muon.viol.item()) if muon is not None and muon.viol is not None else None,
                       mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if device == "cuda" else None))
            if not math.isfinite(ls):
                diverged = True
                break
            t_log, tok_log = now, 0

    if not diverged and cfg["store"] in ("ticker", "kahan", "sr4", "eco"):
        assert all(on_grid(p) for p in hidden), "a matrix left the 4-bit grid"
    summary = dict(header, final_step=evals[-1][0], final_val_ce=round(ce, 5), final_val_se=round(se, 5),
                   diverged=diverged, wall_s=round(time.time() - t0, 1),
                   train_tokens=evals[-1][0] * args.seqs_per_step * args.seq_len, skipped_steps=skipped,
                   carry_violations=int(muon.viol.item()) if muon is not None and muon.viol is not None else None,
                   train_loss_every_log=train_loss, grad_norm_every_log=grad_norm, evals=evals,
                   final_per_window=[round(v, 5) for v in per_window.tolist()] if per_window is not None else None)
    if not diverged and cfg["store"] in ("float", "bf16sr"):
        ce4, se4 = evaluate_rounded(fwd, hidden, windows, device, amp, args.eval_bs)
        summary.update(final_val_ce_fp4=round(ce4, 5), final_val_se_fp4=round(se4, 5),
                       final_per_window_fp4=[round(v, 5) for v in evaluate.last.tolist()])
    json.dump(summary, open(stem + ".json", "w"))
    if args.save:
        torch.save(model.state_dict(), stem + ".pt")
    print(f"final val CE {ce:.4f} +- {se:.4f} at step {evals[-1][0]}, {summary['wall_s']} s"
          + (f"; rounded to 4 bits: {summary['final_val_ce_fp4']:.4f}" if "final_val_ce_fp4" in summary else ""))


if __name__ == "__main__":
    main()
