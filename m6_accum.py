"""Milestone 6: Muon reordered so its momentum can hold the rounding leftover. A copy of milestone 3.

Muon normally averages the gradients and then orthogonalises the average: step = NS(momentum). That step is not a
plain multiple of the momentum, so a rounding leftover fed into the momentum gets bent and partly deleted
(milestone 4: cosine 0.69). Here the order is swapped: each step's gradient is orthogonalised first, then averaged,
    A <- 0.95 A + 0.05 NS(G),    target = w - lr * A    (plain momentum, no Nesterov).
The step is now lr times the momentum, so the momentum is an exact accumulator: the leftover e = target - stored is
added to it as (1 - 1/0.95) * e / lr (ECO's rule for momentum SGD, arXiv 2601.22101), and it comes out again over
the next steps. With nearest rounding a weight stays put until its pushes add up past half a grid gap, then moves
one level, like Bop.
Arms:
  float   the reordered Muon with float weights (is the reordering itself any good?)
  q4      4-bit E2M1 weights as milestone 3 (block of 16, fp32 scale), no float copy, nearest rounding, leftover in A
  q4_m8   as q4, with A stored as 8-bit integers, one fp32 scale per 128 weights along a row, rounded at random
Embeddings, head and norms stay fp32 with AdamW, as before.

Run:  python m6_accum.py --preset cpu --seed 1 --arm q4 --muon-lr 0.04
Writes runs/m6/<preset>_<arm>_s<seed>.jsonl (one line per eval) and .json (final summary).
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


def on_grid(w):
    v = (w.view(w.size(0), -1, BLOCK) / block_scale(w)).abs()
    return bool(((v.unsqueeze(-1) - E2M1.to(w.device)).abs().amin(-1) < 1e-4).all())


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
    """Reordered Muon: orthogonalise each gradient, then average. For the 4-bit arms the weight is rounded to nearest
    and the leftover goes into the momentum."""

    def __init__(self, params, lr, gen, arm, momentum=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))
        self.gen, self.arm = gen, arm

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, beta = group["lr"], group["momentum"]
            for p in group["params"]:
                buf = self.state[p].setdefault("momentum", torch.zeros_like(p))
                buf.lerp_(max(1, p.size(0) / p.size(1)) ** 0.5 * newton_schulz(p.grad).to(p.dtype), 1 - beta)
                w_tilde = p - lr * buf
                if self.arm == "float":
                    p.copy_(w_tilde)
                    continue
                w_hat = quantize_rtn(w_tilde)
                p.copy_(w_hat)
                buf.add_((1 - 1 / beta) / lr * (w_tilde - w_hat))
                if self.arm == "q4_m8":
                    buf.copy_(store_int8(buf, self.gen))


def build_optimizer(model, cfg, name, muon_lr, gen):
    """Returns a list of (optimizer, peak lr). Muon (arm `name`) gets the four hidden matrices; AdamW gets the rest."""
    muon = [p for n, p in model.named_parameters() if n.endswith(MUON_NAMES)]
    rest = [p for n, p in model.named_parameters() if not n.endswith(MUON_NAMES)]
    decay = [p for p in rest if p.dim() >= 2]
    no_decay = [p for p in rest if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    adamw = torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    return [(Muon(muon, lr=muon_lr, gen=gen, arm=name), muon_lr), (adamw, cfg.lr)]


def state_bytes_per_weight(name):
    """Training-state bytes per weight of the four hidden matrices. float: fp32 weight + fp32 momentum. q4: 4-bit code
    (0.5) + fp32 scale per 16 (0.25) + fp32 momentum (4). q4_m8: the momentum as int8 plus one fp32 scale per 128."""
    return {"float": 8, "q4": 0.5 + 4 / 16 + 4, "q4_m8": 0.5 + 4 / 16 + 1 + 4 / MBLOCK}[name]


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


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="cpu")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--arm", choices=("float", "q4", "q4_m8"), required=True)
    ap.add_argument("--muon-lr", type=float, required=True)  # picked for the float arm on seed 1 (PLAN.md)
    ap.add_argument("--max-iters", type=int, default=None, help="override the preset (smoke tests)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "m6"))
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
    if args.arm != "float":
        with torch.no_grad():
            for p in muon_params:
                p.copy_(quantize_rtn(p))
    round_gen = torch.Generator(device).manual_seed(1000 + args.seed)  # the random rounding's own stream
    opts = build_optimizer(model, cfg, args.arm, args.muon_lr, round_gen)
    windows = fixed_windows(val, cfg)
    n_params = sum(p.numel() for p in model.parameters()) - model.pos.weight.numel()

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, f"{args.preset}_{args.arm}_s{args.seed}")
    log = open(stem + ".jsonl", "w")
    header = dict(milestone="m6", arm=args.arm, preset=args.preset, muon_lr=args.muon_lr, seed=args.seed,
                  device=device, dtype="fp16-autocast" if use_fp16 else "fp32", params=n_params,
                  state_bytes_per_weight=state_bytes_per_weight(args.arm), torch=torch.__version__,
                  config=asdict(cfg))
    print(json.dumps(header))

    t0 = time.time()
    for it in range(cfg.max_iters + 1):
        if it % cfg.eval_interval == 0 or it == cfg.max_iters:
            ce, se = evaluate(model, windows, device, amp)
            row = dict(step=it, val_ce=round(ce, 5), val_se=round(se, 5), wall_s=round(time.time() - t0, 1))
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

    assert args.arm == "float" or all(on_grid(p) for p in muon_params), "a Muon matrix left the 4-bit grid"
    summary = dict(header, final_val_ce=round(ce, 5), final_val_se=round(se, 5), wall_s=round(time.time() - t0, 1),
                   train_tokens=cfg.max_iters * cfg.batch_size * cfg.block_size)
    json.dump(summary, open(stem + ".json", "w"), indent=1)
    print(f"final val CE {ce:.4f} +- {se:.4f} nats/char after {cfg.max_iters} steps, {summary['wall_s']} s")


if __name__ == "__main__":
    main()
