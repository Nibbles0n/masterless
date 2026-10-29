"""Milestone 7: no float copy, about 2 bytes of training state per weight. Two arms on Muon's four hidden matrices:
  q4_b4_m8  milestone 4's muon_post (4-bit E2M1 weights with a scale per 16, random rounding, the rounding leftover
            e = w_tilde - w_hat kept in a buffer and added straight to the next step's weight change, after
            Newton-Schulz), with the buffer stored as 4-bit integers (-7..7, one fp32 scale per 16 weights) and the
            momentum as 8-bit integers (one fp32 scale per 128 weights of a row), both rounded at random.
  fp8_m8    FP8 E4M3 weights (each block of 16 scaled so its largest weight sits on 448), random rounding, no
            buffer; momentum in 8 bits as above.
Third evaluation: at every eval of fp8_m8, the Muon matrices are copied and rounded to nearest on the 4-bit grid
(rounding after training), and that model is evaluated too (val_ce_fp4).

Run:  python m7_small_state.py --preset cpu --seed 1 --arm q4_b4_m8
Writes runs/m7/<preset>_<arm>_s<seed>.jsonl (one line per eval) and .json.
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


# ---------------------------------------------------------------- 4-bit grid

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
BLOCK = 16


def block_scale(w):
    """One fp32 scale per block of 16 weights along the input dimension: absmax / 6. Shape (out, in / 16, 1)."""
    return w.view(w.size(0), -1, BLOCK).abs().amax(-1, keepdim=True).clamp_min(1e-12) / 6


def quantize_rtn(w):
    """Round w (out, in) to nearest on the scaled E2M1 grid."""
    grid = E2M1.to(w.device)
    wb, s = w.view(w.size(0), -1, BLOCK), block_scale(w)
    x = wb / s
    idx = torch.bucketize(x.abs(), (grid[1:] + grid[:-1]) / 2)
    return (grid[idx] * x.sign() * s).view_as(w)


def quantize_sr(w, gen):
    """Round w (out, in) at random to one of its two neighbours on the scaled E2M1 grid, without bias."""
    grid = E2M1.to(w.device)
    wb, s = w.view(w.size(0), -1, BLOCK), block_scale(w)
    x = wb / s
    a = x.abs()
    lo_i = (torch.bucketize(a, grid, right=True) - 1).clamp(max=len(grid) - 2)
    lo, hi = grid[lo_i], grid[lo_i + 1]
    up = torch.rand(a.shape, generator=gen, device=w.device) < (a - lo) / (hi - lo)
    return (torch.where(up, hi, lo) * x.sign() * s).view_as(w)


def on_grid(w):
    v = (w.view(w.size(0), -1, BLOCK) / block_scale(w)).abs()
    return bool(((v.unsqueeze(-1) - E2M1.to(w.device)).abs().amin(-1) < 1e-4).all())


# ---------------------------------------------------------------- FP8 (E4M3) grid, 8-bit and 4-bit integer stores

FP8_MAX = 448.0  # E4M3's largest value; 3 mantissa bits, smallest normal 2^-6, subnormal spacing 2^-9


def fp8_scale(w):
    return w.view(w.size(0), -1, BLOCK).abs().amax(-1, keepdim=True).clamp_min(1e-30) / FP8_MAX


def quantize_fp8_rtn(w):
    s = fp8_scale(w)
    return ((w.view(w.size(0), -1, BLOCK) / s).to(torch.float8_e4m3fn).float() * s).view_as(w)


def quantize_fp8_sr(w, gen):
    """Round w (out, in) at random to one of its two E4M3 neighbours, after scaling each block of 16 to 448."""
    wb, s = w.view(w.size(0), -1, BLOCK), fp8_scale(w)
    x = wb / s
    a = x.abs()
    gap = torch.exp2(torch.floor(torch.log2(a.clamp_min(2.0 ** -9))).clamp(min=-6) - 3)
    lo = torch.floor(a / gap) * gap
    q = (lo + gap * (torch.rand(a.shape, generator=gen, device=w.device) < (a - lo) / gap)).clamp(max=FP8_MAX)
    return (q * x.sign() * s).view_as(w)


def on_fp8_grid(w):
    x = w.view(w.size(0), -1, BLOCK) / fp8_scale(w)
    return bool(((x.to(torch.float8_e4m3fn).float() - x).abs() <= 1e-5 * x.abs() + 1e-6).all())


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


def store_int4(a, gen):
    """As store_int8, with 4-bit integers (-7..7) and one fp32 scale per 16 weights."""
    ab = a.view(a.size(0), -1, BLOCK)
    s = ab.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 7
    x = ab / s
    lo = x.floor()
    q = lo + (torch.rand(x.shape, generator=gen, device=a.device) < x - lo)
    return (q * s).view_as(a)


# ---------------------------------------------------------------- model

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.fc_out = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
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
    def __init__(self, cfg):
        super().__init__()
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
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
    """Muon whose weights live on a low-bit grid with no float copy, and whose state is stored in 8 and 4 bits."""

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
                buf.copy_(store_int8(buf, self.gen))
                update = max(1, p.size(0) / p.size(1)) ** 0.5 * newton_schulz(u).to(p.dtype)
                w_tilde = p - lr * update
                if self.arm == "fp8_m8":
                    p.copy_(quantize_fp8_sr(w_tilde, self.gen))
                    continue
                w_tilde += st.setdefault("error", torch.zeros_like(p))
                w_hat = quantize_sr(w_tilde, self.gen)
                st["error"].copy_(store_int4(w_tilde - w_hat, self.gen))
                p.copy_(w_hat)


def build_optimizer(model, cfg, name, muon_lr, gen):
    """Returns a list of (optimizer, peak lr). The arm's optimizer gets the four hidden matrices; AdamW the rest."""
    muon = [p for n, p in model.named_parameters() if n.endswith(MUON_NAMES)]
    rest = [p for n, p in model.named_parameters() if not n.endswith(MUON_NAMES)]
    decay = [p for p in rest if p.dim() >= 2]
    no_decay = [p for p in rest if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    adamw = torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    return [(Muon(muon, lr=muon_lr, gen=gen, arm=name), muon_lr), (adamw, cfg.lr)]


def state_bytes_per_weight(name):
    """Training-state bytes per weight of the four hidden matrices: the weight code and its fp32 scale per 16, the
    4-bit buffer and its fp32 scale per 16 (q4_b4_m8 only), the 8-bit momentum and its fp32 scale per 128."""
    return {"q4_b4_m8": 0.5 + 4 / 16 + 0.5 + 4 / 16 + 1 + 4 / MBLOCK, "fp8_m8": 1 + 4 / 16 + 1 + 4 / MBLOCK}[name]


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
def evaluate_fp4(model, params, windows, device, amp):
    """The model with its Muon matrices rounded to nearest on the 4-bit grid (rounding after training)."""
    saved = [p.clone() for p in params]
    for p in params:
        p.copy_(quantize_rtn(p))
    out = evaluate(model, windows, device, amp)
    for p, w in zip(params, saved):
        p.copy_(w)
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="cpu")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--arm", choices=("q4_b4_m8", "fp8_m8"), required=True)
    ap.add_argument("--muon-lr", type=float, default=0.01)  # best of the milestone 1 sweep
    ap.add_argument("--max-iters", type=int, default=None, help="override the preset (smoke tests)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "m7"))
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
    model = GPT(cfg).to(device)
    muon_params = [p for n, p in model.named_parameters() if n.endswith(MUON_NAMES)]
    with torch.no_grad():
        for p in muon_params:
            p.copy_(quantize_fp8_rtn(p) if args.arm == "fp8_m8" else quantize_rtn(p))
    round_gen = torch.Generator(device).manual_seed(1000 + args.seed)  # the random rounding's own stream
    opts = build_optimizer(model, cfg, args.arm, args.muon_lr, round_gen)
    windows = fixed_windows(val, cfg)
    n_params = sum(p.numel() for p in model.parameters()) - model.pos.weight.numel()

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, f"{args.preset}_{args.arm}_s{args.seed}")
    log = open(stem + ".jsonl", "w")
    header = dict(milestone="m7", arm=args.arm, preset=args.preset, muon_lr=args.muon_lr, seed=args.seed,
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
                ce4, se4 = evaluate_fp4(model, muon_params, windows, device, amp)
                row.update(val_ce_fp4=round(ce4, 5), val_se_fp4=round(se4, 5))
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

    check = on_fp8_grid if args.arm == "fp8_m8" else on_grid
    assert all(check(p) for p in muon_params), "a Muon matrix left its grid"
    summary = dict(header, final_val_ce=round(ce, 5), final_val_se=round(se, 5), wall_s=round(time.time() - t0, 1),
                   train_tokens=cfg.max_iters * cfg.batch_size * cfg.block_size)
    if args.arm == "fp8_m8":
        summary.update(final_val_ce_fp4=row["val_ce_fp4"], final_val_se_fp4=row["val_se_fp4"])
    json.dump(summary, open(stem + ".json", "w"), indent=1)
    print(f"final val CE {ce:.4f} +- {se:.4f} nats/char after {cfg.max_iters} steps, {summary['wall_s']} s")


if __name__ == "__main__":
    main()
