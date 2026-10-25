"""Milestone 0: baby GPT on Shakespeare characters, AdamW, 16/32-bit. The reference every later milestone is measured against.

The model and recipe follow nanoGPT's shakespeare_char config. Two presets:
  cpu: 4 layers, width 128, context 64, 2000 steps (nanoGPT's README reports val loss about 1.88 for this setup).
  gpu: 6 layers, width 384, context 256, 5000 steps (nanoGPT reports val loss 1.4697 on an A100). Runs on a free T4.

Run:  python m0_baby_gpt.py --preset cpu --seed 1
Writes runs/m0/<preset>_s<seed>.jsonl (one line per eval) and runs/m0/<preset>_s<seed>.json (final summary).
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


# ---------------------------------------------------------------- optimizer (milestone 1 adds "muon" here)

def build_optimizer(model, cfg, name):
    if name != "adamw":
        raise ValueError(f"unknown optimizer {name}")
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))


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
    ap.add_argument("--opt", default="adamw")
    ap.add_argument("--max-iters", type=int, default=None, help="override the preset (smoke tests)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "m0"))
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
    opt = build_optimizer(model, cfg, args.opt)
    windows = fixed_windows(val, cfg)
    n_params = sum(p.numel() for p in model.parameters()) - model.pos.weight.numel()

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, f"{args.preset}_{args.opt}_s{args.seed}")
    log = open(stem + ".jsonl", "w")
    header = dict(milestone="m0", preset=args.preset, opt=args.opt, seed=args.seed, device=device,
                  dtype="fp16-autocast" if use_fp16 else "fp32", params=n_params, torch=torch.__version__,
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
        for g in opt.param_groups:
            g["lr"] = lr_at(it, cfg)
        x, y = get_batch(train, cfg, gen, device)
        with amp():
            loss = F.cross_entropy(model(x).float().view(-1, cfg.vocab_size), y.view(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()

    summary = dict(header, final_val_ce=round(ce, 5), final_val_se=round(se, 5), wall_s=round(time.time() - t0, 1),
                   train_tokens=cfg.max_iters * cfg.batch_size * cfg.block_size)
    json.dump(summary, open(stem + ".json", "w"), indent=1)
    print(f"final val CE {ce:.4f} +- {se:.4f} nats/char after {cfg.max_iters} steps, {summary['wall_s']} s")


if __name__ == "__main__":
    main()
