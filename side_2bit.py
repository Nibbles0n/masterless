"""Side study 2B: the ticker at 2 bits per weight. Five arms on the CPU preset:
  q2_t4_m8     2-bit weights, the ticker with a 4-bit carry (k/14 of the gap, -7..7, random rounding), 8-bit momentum
  q2_t8_m8     the same with an 8-bit carry: 512 x (carry in gaps) stored as FP8 E4M3, random rounding (the fixed
               power of two only moves where the format's fine steps start; no stored scale)
  q2_ste_muon  2-bit forward weights with a full float copy (straight-through), Muon with fp32 momentum (m2 at 2 bits)
  q2_ste_adam  the same with AdamW on every parameter, as milestone 0 (lr 1e-3)
  fp8_m8       milestone 7's FP8 weights + 8-bit momentum, re-run; at every eval its Muon matrices are also rounded
               to nearest on the 2-bit grid and evaluated (val_ce_q2)
The 2-bit grid: the four values -1.5, -0.5, +0.5, +1.5 times one fp32 scale per block of 16 weights along the input
dimension, scale = 1.25 x the block's mean absolute weight (about 30% less rounding error than absmax / 1.5 on
bell-shaped blocks; weights beyond 2 x scale are clipped to the end values). The grid is evenly spaced, so the gap is
the scale on both sides. The ticker keeps the scale it used (it cannot be read back from the codes); a new scale is
taken from the target each step.

Run:  python side_2bit.py --arm q2_t4_m8 --seed 1
Writes runs/twobit/<preset>_<arm>_s<seed>.jsonl (one line per eval) and .json.
"""
import argparse
import contextlib
import json
import math
import os
import time
import urllib.request
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    batch_size: int
    max_iters: int
    dropout: float
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_windows: int = 256  # fixed validation windows, the same for every seed and every milestone
    vocab_size: int = 0


PRESETS = {
    "cpu": dict(n_layer=4, n_head=4, n_embd=128, block_size=64, batch_size=12, max_iters=2000, dropout=0.0),
    "gpu": dict(n_layer=6, n_head=6, n_embd=384, block_size=256, batch_size=64, max_iters=5000, dropout=0.2),
}


# ---------------------------------------------------------------- data

def load_data():
    """Characters of tiny Shakespeare as uint8 ids; first 90% train, last 10% val (as nanoGPT)."""
    path = os.path.join(HERE, "data", "shakespeare.txt")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    text = open(path, encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.uint8)
    n = int(0.9 * len(ids))
    return ids[:n], ids[n:], len(chars)


def get_batch(data, cfg, gen, device):
    ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i + cfg.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + cfg.block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------- 2-bit grid

BLOCK = 16


def block_scale2(w):
    """One fp32 scale per block of 16 weights along the input dimension: 1.25 x mean |w|. Shape (out, in / 16, 1)."""
    return 1.25 * w.view(w.size(0), -1, BLOCK).abs().mean(-1, keepdim=True).clamp_min(1e-12)


def codes2(x):
    """Nearest of -1.5, -0.5, +0.5, +1.5 (values beyond the ends are clipped)."""
    return (x - 0.5).round().clamp(-2, 1) + 0.5


def quantize_rtn2(w, s=None):
    """Round w (out, in) to nearest on the 2-bit grid, with the scale s (default: the scale of w itself)."""
    s = block_scale2(w) if s is None else s
    return (codes2(w.view(w.size(0), -1, BLOCK) / s) * s).view_as(w)


def on_grid2(w, s):
    v = (w.view(w.size(0), -1, BLOCK) / s).abs()
    return bool((((v - 0.5).abs() < 1e-4) | ((v - 1.5).abs() < 1e-4)).all())


class QLinear(nn.Linear):
    """Linear layer whose forward pass sees the 2-bit weight; the gradient goes straight through to the float copy."""

    def forward(self, x):
        w = self.weight
        return F.linear(x, w + (quantize_rtn2(w.detach()) - w).detach())


# ---------------------------------------------------------------- FP8 (E4M3) grid (milestone 7)

FP8_MAX = 448.0  # E4M3's largest value; 3 mantissa bits, smallest normal 2^-6, subnormal spacing 2^-9


def fp8_scale(w):
    return w.view(w.size(0), -1, BLOCK).abs().amax(-1, keepdim=True).clamp_min(1e-30) / FP8_MAX


def quantize_fp8_rtn(w):
    s = fp8_scale(w)
    return ((w.view(w.size(0), -1, BLOCK) / s).to(torch.float8_e4m3fn).float() * s).view_as(w)


def sr_e4m3(x, gen):
    """x (already scaled into E4M3's range) rounded at random to one of its two E4M3 neighbours, without bias."""
    a = x.abs()
    gap = torch.exp2(torch.floor(torch.log2(a.clamp_min(2.0 ** -9))).clamp(min=-6) - 3)
    lo = torch.floor(a / gap) * gap
    return (lo + gap * (torch.rand(a.shape, generator=gen, device=x.device) < (a - lo) / gap)).clamp(max=FP8_MAX) * x.sign()


def quantize_fp8_sr(w, gen):
    """Round w (out, in) at random to one of its two E4M3 neighbours, after scaling each block of 16 to 448."""
    wb, s = w.view(w.size(0), -1, BLOCK), fp8_scale(w)
    return (sr_e4m3(wb / s, gen) * s).view_as(w)


def on_fp8_grid(w):
    x = w.view(w.size(0), -1, BLOCK) / fp8_scale(w)
    return bool(((x.to(torch.float8_e4m3fn).float() - x).abs() <= 1e-5 * x.abs() + 1e-6).all())


# ---------------------------------------------------------------- 8-bit momentum

MBLOCK = 128


def store_int8(a, gen):
    """a (out, in) rounded at random to 8-bit integers (-127..127) times one fp32 scale per 128 weights of a row;
    returned as the float values the integers stand for."""
    ab = a.view(a.size(0), -1, MBLOCK)
    s = ab.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 127
    x = ab / s
    lo = x.floor()
    q = lo + (torch.rand(x.shape, generator=gen, device=a.device) < x - lo)
    return (q * s).view_as(a)


# ---------------------------------------------------------------- model

class Block(nn.Module):
    def __init__(self, cfg, ste):
        super().__init__()
        Lin = QLinear if ste else nn.Linear
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.qkv = Lin(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = Lin(cfg.n_embd, cfg.n_embd, bias=False)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.fc = Lin(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.fc_out = Lin(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.n_head, self.dropout = cfg.n_head, cfg.dropout
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).contiguous().view(B, T, C)))
        return x + self.drop(self.fc_out(F.gelu(self.fc(self.ln2(x)))))


class GPT(nn.Module):
    def __init__(self, cfg, ste=False):
        super().__init__()
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg, ste) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # tied, as nanoGPT
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc_out.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


# ---------------------------------------------------------------- optimizer

MUON_NAMES = ("qkv.weight", "proj.weight", "fc.weight", "fc_out.weight")


def newton_schulz(G, steps=5):
    """Approximately orthogonalise G (push its singular values towards 1), as in Keller Jordan's Muon."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    tall = X.size(0) > X.size(1)
    if tall:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X


class Muon(torch.optim.Optimizer):
    """Muon for one arm: float (q2_ste_muon, fp32 weights and momentum), FP8 weights (fp8_m8), or 2-bit weights with
    the ticker (q2_t4_m8, q2_t8_m8); the last three store the momentum in 8 bits."""

    def __init__(self, params, lr, gen, arm, momentum=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))
        self.gen, self.arm = gen, arm

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, beta = group["lr"], group["momentum"]
            for p in group["params"]:
                st = self.state[p]
                buf = st.setdefault("momentum", torch.zeros_like(p))
                buf.lerp_(p.grad, 1 - beta)
                u = p.grad.lerp(buf, beta)  # Nesterov
                if self.arm == "q2_ste_muon":
                    p.add_(newton_schulz(u).to(p.dtype), alpha=-lr * max(1, p.size(0) / p.size(1)) ** 0.5)
                    continue
                buf.copy_(store_int8(buf, self.gen))
                update = max(1, p.size(0) / p.size(1)) ** 0.5 * newton_schulz(u).to(p.dtype)
                w_tilde = p - lr * update
                if self.arm == "fp8_m8":
                    p.copy_(quantize_fp8_sr(w_tilde, self.gen))
                    continue
                s_old = st["scale"]  # (out, in / 16, 1); the gap is the scale on both sides
                if self.arm == "q2_t4_m8":
                    k = st.setdefault("carry", torch.zeros(s_old.shape[:2] + (BLOCK,), dtype=torch.int8, device=p.device))
                    carry = k.float() / 14
                else:
                    k = st.setdefault("carry", torch.zeros(s_old.shape[:2] + (BLOCK,), dtype=torch.float8_e4m3fn, device=p.device))
                    carry = k.float() / 512
                w_tilde += (carry * s_old).view_as(p)
                s_new = block_scale2(w_tilde)
                x = w_tilde.view_as(carry) / s_new
                xq = codes2(x)
                c = x - xq  # the new carry in gaps: within -1/2..1/2, or beyond if the target was clipped
                if self.arm == "q2_t4_m8":
                    c = (14 * c).clamp(-7, 7)
                    lo = c.floor()
                    k.copy_(lo + (torch.rand(c.shape, generator=self.gen, device=c.device) < c - lo))
                else:
                    k.copy_(sr_e4m3((512 * c).clamp(-256, 256), self.gen))
                st["scale"] = s_new
                p.copy_((xq * s_new).view_as(p))


def build_optimizer(model, cfg, name, muon_lr, gen):
    """Returns a list of (optimizer, peak lr). q2_ste_adam: AdamW on everything, as milestone 0. Otherwise the arm's
    Muon gets the four hidden matrices and AdamW the rest."""
    if name == "q2_ste_adam":
        decay = [p for p in model.parameters() if p.dim() >= 2]
        no_decay = [p for p in model.parameters() if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        return [(torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2)), cfg.lr)]
    muon = [p for n, p in model.named_parameters() if n.endswith(MUON_NAMES)]
    rest = [p for n, p in model.named_parameters() if not n.endswith(MUON_NAMES)]
    decay = [p for p in rest if p.dim() >= 2]
    no_decay = [p for p in rest if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    adamw = torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    return [(Muon(muon, lr=muon_lr, gen=gen, arm=name), muon_lr), (adamw, cfg.lr)]


ARMS = ("q2_t4_m8", "q2_t8_m8", "q2_ste_muon", "q2_ste_adam", "fp8_m8")


def state_bytes_per_weight(name):
    """Training-state bytes per weight of the four hidden matrices: the weight (2-bit code + fp32 scale per 16, or
    FP8 + fp32 scale per 16, or a float copy), the carry, and the optimizer state (8-bit momentum + fp32 scale per
    128, or fp32 momentum, or AdamW's two fp32 averages)."""
    m8 = 1 + 4 / MBLOCK
    return {"q2_t4_m8": 0.25 + 4 / 16 + 0.5 + m8, "q2_t8_m8": 0.25 + 4 / 16 + 1 + m8, "q2_ste_muon": 4 + 4,
            "q2_ste_adam": 4 + 4 + 4, "fp8_m8": 1 + 4 / 16 + m8}[name]


def lr_at(it, cfg):
    if it < cfg.warmup_iters:
        return cfg.lr * (it + 1) / cfg.warmup_iters
    frac = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * min(1.0, frac))) * (cfg.lr - cfg.min_lr)


# ---------------------------------------------------------------- eval

def fixed_windows(data, cfg):
    """The same validation windows for every run: evenly spaced starts over the val split."""
    starts = np.linspace(0, len(data) - cfg.block_size - 2, cfg.eval_windows).astype(np.int64)
    x = np.stack([data[s:s + cfg.block_size] for s in starts]).astype(np.int64)
    y = np.stack([data[s + 1:s + 1 + cfg.block_size] for s in starts]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def evaluate(model, windows, device, amp):
    """Mean cross-entropy (nats per character) over the fixed windows, and its standard error across windows."""
    model.eval()
    x_all, y_all = windows
    per_window = []
    for i in range(0, len(x_all), 32):
        x, y = x_all[i:i + 32].to(device), y_all[i:i + 32].to(device)
        with amp():
            logits = model(x)
        ce = F.cross_entropy(logits.float().transpose(1, 2), y, reduction="none").mean(dim=1)
        per_window.append(ce.cpu())
    model.train()
    ce = torch.cat(per_window).double()
    return ce.mean().item(), (ce.std(unbiased=True) / math.sqrt(len(ce))).item()


@torch.no_grad()
def evaluate_q2(model, params, windows, device, amp):
    """The model with its Muon matrices rounded to nearest on the 2-bit grid (rounding after training)."""
    saved = [p.clone() for p in params]
    for p in params:
        p.copy_(quantize_rtn2(p))
    out = evaluate(model, windows, device, amp)
    for p, w in zip(params, saved):
        p.copy_(w)
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="cpu")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--muon-lr", type=float, default=0.01)  # best of the milestone 1 sweep
    ap.add_argument("--max-iters", type=int, default=None, help="override the preset (smoke tests)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "twobit"))
    args = ap.parse_args()

    train, val, vocab = load_data()
    cfg = Config(**PRESETS[args.preset], vocab_size=vocab)
    if args.max_iters is not None:
        cfg.max_iters = args.max_iters
        cfg.eval_interval = max(1, min(cfg.eval_interval, args.max_iters))

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    # T4 has no bf16: fp16 autocast with loss scaling on CUDA, plain fp32 on CPU.
    use_fp16 = device == "cuda"
    amp = (lambda: torch.autocast("cuda", dtype=torch.float16)) if use_fp16 else contextlib.nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    model = GPT(cfg, ste=args.arm.startswith("q2_ste")).to(device)
    muon_params = [p for n, p in model.named_parameters() if n.endswith(MUON_NAMES)]
    ticker = args.arm in ("q2_t4_m8", "q2_t8_m8")
    scales = []
    with torch.no_grad():
        for p in muon_params:
            if ticker:
                scales.append(block_scale2(p))
                p.copy_(quantize_rtn2(p, scales[-1]))
            elif args.arm == "fp8_m8":
                p.copy_(quantize_fp8_rtn(p))
    round_gen = torch.Generator(device).manual_seed(1000 + args.seed)  # the random rounding's own stream
    opts = build_optimizer(model, cfg, args.arm, args.muon_lr, round_gen)
    for p, s in zip(muon_params, scales):
        opts[0][0].state[p]["scale"] = s
    windows = fixed_windows(val, cfg)
    n_params = sum(p.numel() for p in model.parameters()) - model.pos.weight.numel()

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, f"{args.preset}_{args.arm}_s{args.seed}")
    log = open(stem + ".jsonl", "w")
    header = dict(milestone="twobit", arm=args.arm, preset=args.preset, muon_lr=args.muon_lr, seed=args.seed,
                  device=device, dtype="fp16-autocast" if use_fp16 else "fp32", params=n_params,
                  state_bytes_per_weight=state_bytes_per_weight(args.arm), torch=torch.__version__,
                  config=asdict(cfg))
    print(json.dumps(header))

    t0 = time.time()
    for it in range(cfg.max_iters + 1):
        if it % cfg.eval_interval == 0 or it == cfg.max_iters:
            ce, se = evaluate(model, windows, device, amp)
            row = dict(step=it, val_ce=round(ce, 5), val_se=round(se, 5), wall_s=round(time.time() - t0, 1))
            if args.arm == "fp8_m8":
                ce2, se2 = evaluate_q2(model, muon_params, windows, device, amp)
                row.update(val_ce_q2=round(ce2, 5), val_se_q2=round(se2, 5))
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
        if it == cfg.max_iters:
            break
        for opt, peak in opts:
            for g in opt.param_groups:
                g["lr"] = peak * lr_at(it, cfg) / cfg.lr
        x, y = get_batch(train, cfg, gen, device)
        with amp():
            loss = F.cross_entropy(model(x).float().view(-1, cfg.vocab_size), y.view(-1))
        model.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        for opt, _ in opts:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for opt, _ in opts:
            scaler.step(opt)
        scaler.update()

    if ticker:
        st = opts[0][0].state
        assert all(on_grid2(p, st[p]["scale"]) for p in muon_params), "a Muon matrix left the 2-bit grid"
        assert all(st[p]["carry"].float().abs().max() <= (7 if args.arm == "q2_t4_m8" else 256) for p in muon_params)
    elif args.arm == "fp8_m8":
        assert all(on_fp8_grid(p) for p in muon_params), "a Muon matrix left the FP8 grid"
    summary = dict(header, final_val_ce=round(ce, 5), final_val_se=round(se, 5), wall_s=round(time.time() - t0, 1),
                   train_tokens=cfg.max_iters * cfg.batch_size * cfg.block_size)
    if args.arm == "fp8_m8":
        summary.update(final_val_ce_q2=row["val_ce_q2"], final_val_se_q2=row["val_se_q2"])
    json.dump(summary, open(stem + ".json", "w"), indent=1)
    print(f"final val CE {ce:.4f} +- {se:.4f} nats/char after {cfg.max_iters} steps, {summary['wall_s']} s")


if __name__ == "__main__":
    main()
