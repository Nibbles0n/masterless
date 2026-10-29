"""Side study G: the two 2-byte systems at GPT-2 small scale. Continue training the pretrained OpenAI GPT-2 (124M) on
FineWeb text and ask whether training with no float copy stays stable and keeps improving an already good model.
Muon gets the 48 matrices inside the 12 blocks (about 85M weights: attention c_attn and c_proj, MLP c_fc and c_proj);
AdamW, in float, gets the embeddings (tied to the output head), the biases and the LayerNorms. Arms:
  adamw_float    AdamW on everything, float (how GPT-2 is usually trained; a reference)
  muon_float     Muon in float, fp32 momentum (the baseline the 2-byte arms are judged against)
  q4_float_copy  milestone 2: 4-bit weights in the forward pass, float copy kept (straight-through)
  fp8_m8         milestone 7: FP8 E4M3 weights (block-of-16 scales), random rounding, 8-bit momentum, no float copy
  q4_t4_m8       side study T, the ticker: 4-bit E2M1 weights, 4-bit carry in units of its own gap, 8-bit momentum
At step 0 the pretrained weights are rounded onto each arm's grid (to nearest). The ticker's carry starts as the
rounding leftover (pretrained weight - 4-bit weight), stored like any later carry, since the carry is by definition
target - stored weight and the target here is the pretrained weight.

Data: the GPT-2-token FineWeb shards of modded-nanogpt (kjj0/fineweb10B-gpt2 on Hugging Face; 100M tokens each,
256 int32 header then uint16 tokens). Training windows are drawn at random with one fixed data seed, so every arm sees
the same batches; validation is a fixed set of evenly spaced windows of the separate val shard.

Run:  python side_gpt2.py --arm q4_t4_m8 --muon-lr 0.003 --steps 760
Writes runs/gpt2/<arm>_lr<lr>_s<seed>.jsonl (one line per log/eval) and .json.
"""
import argparse
import contextlib
import json
import math
import os
import time
import urllib.request

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch touches the GPU
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
HF = "https://huggingface.co"
GPT2_URL = HF + "/openai-community/gpt2/resolve/main/model.safetensors"
SHARD_URL = HF + "/datasets/kjj0/fineweb10B-gpt2/resolve/main/{}"
N_LAYER, N_HEAD, N_EMBD, VOCAB, CTX = 12, 12, 768, 50257, 1024


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


def get_batch(data, n, T, gen, device):
    ix = torch.randint(len(data) - T - 1, (n,), generator=gen).numpy()
    xy = torch.from_numpy(np.stack([data[i:i + T + 1] for i in ix]).astype(np.int64))
    if device == "cuda":
        xy = xy.pin_memory().to(device, non_blocking=True)
    return xy[:, :-1], xy[:, 1:]


def fixed_windows(data, n, T):
    """The same validation windows for every run: n evenly spaced starts over the val shard."""
    starts = np.linspace(0, len(data) - T - 2, n).astype(np.int64)
    xy = torch.from_numpy(np.stack([data[s:s + T + 1] for s in starts]).astype(np.int64))
    return xy[:, :-1], xy[:, 1:]


# ---------------------------------------------------------------- 4-bit grid

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
BLOCK = 16


def block_scale(w):
    """One fp32 scale per block of 16 weights along the input dimension: absmax / 6. Shape (out, in / 16, 1)."""
    return w.view(w.size(0), -1, BLOCK).abs().amax(-1, keepdim=True).clamp_min(1e-12) / 6


def code4(w):
    """4-bit round-to-nearest codes of w (twice the grid value, so whole numbers), for counting which weights change."""
    return (2 * w.view(w.size(0), -1, BLOCK) / block_scale(w)).round()


def quantize_rtn(w):
    """Round w (out, in) to nearest on the scaled E2M1 grid."""
    grid = E2M1.to(w.device)
    wb, s = w.view(w.size(0), -1, BLOCK), block_scale(w)
    x = wb / s
    idx = torch.bucketize(x.abs(), (grid[1:] + grid[:-1]) / 2)
    return (grid[idx] * x.sign() * s).view_as(w)


def on_grid(w):
    v = (w.view(w.size(0), -1, BLOCK) / block_scale(w)).abs()
    return bool(((v.unsqueeze(-1) - E2M1.to(w.device)).abs().amin(-1) < 1e-4).all())


# ---------------------------------------------------------------- FP8 (E4M3) grid

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
    a = torch.where(a > FP8_MAX - 1e-3, FP8_MAX, a)  # the block's largest weight can come out as 447.99997 in fp32;
    # rounded down (about once in a million blocks) it would shrink the block's scale and move every weight in it
    # the binade's exponent from frexp (exact; log2 can round up just below a power of two, or down just above one,
    # and then the upper neighbour is off the grid)
    gap = torch.exp2((torch.frexp(a.clamp_min(2.0 ** -9)).exponent - 1).clamp(min=-6).float() - 3)
    lo = torch.floor(a / gap) * gap
    q = (lo + gap * (torch.rand(a.shape, generator=gen, device=w.device) < (a - lo) / gap)).clamp(max=FP8_MAX)
    return (q * x.sign() * s).view_as(w)


def code8(w):
    """FP8 round-to-nearest codes of w (block scale included), for counting which weights change."""
    return (w.view(w.size(0), -1, BLOCK) / fp8_scale(w)).to(torch.float8_e4m3fn).float()


def on_fp8_grid(w):
    x = w.view(w.size(0), -1, BLOCK) / fp8_scale(w)
    return bool(((x.to(torch.float8_e4m3fn).float() - x).abs() <= 1e-5 * x.abs() + 1e-6).all())


# ---------------------------------------------------------------- 8-bit momentum, 4-bit carry

MBLOCK = 128
GRID = torch.cat([-E2M1.flip(0)[:-1], E2M1])  # the 15 signed 4-bit values, -6..6
GRID_MID = (GRID[1:] + GRID[:-1]) / 2
GAP_UP = torch.cat([GRID[1:] - GRID[:-1], torch.tensor([2.0])])  # gap to the next value up (2 above +6)
GAP_DN = torch.cat([torch.tensor([2.0]), GRID[1:] - GRID[:-1]])  # gap to the next value down (2 below -6)


def side_gaps(x):
    """For codes x already on the signed grid: the gaps to the next value up and down (bucketize, not a 15-way
    argmin, so no (weights x 15) tensor is built)."""
    i = torch.bucketize(x, GRID_MID.to(x.device))
    return GAP_UP.to(x.device)[i], GAP_DN.to(x.device)[i]


def carry_code(r, x_new, gen):
    """The leftover r (in code units of the new block scale) as 4-bit integers -7..7 standing for k/14 of the gap on
    its own side, rounded at random."""
    up, dn = side_gaps(x_new)
    c = 14 * r / torch.where(r > 0, up, dn)
    lo = c.floor()
    return (lo + (torch.rand(c.shape, generator=gen, device=c.device) < c - lo)).clamp(-7, 7).to(torch.int8)


def store_int8(a, gen):
    """a (out, in) rounded at random to 8-bit integers (-127..127) times one fp32 scale per 128 weights of a row;
    returned as the float values the integers stand for."""
    ab = a.view(a.size(0), -1, MBLOCK)
    s = ab.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 127
    x = ab / s
    lo = x.floor()
    q = lo + (torch.rand(x.shape, generator=gen, device=a.device) < x - lo)
    return (q * s).view_as(a)


# ---------------------------------------------------------------- model (GPT-2 small, with biases)

class QLinear(nn.Linear):
    """A linear layer that can use 4-bit weights in the forward pass while keeping a float copy (straight-through)."""
    fake4 = False

    def forward(self, x):
        w = self.weight
        if self.fake4:
            w = w + (quantize_rtn(w.detach()) - w).detach()
        return F.linear(x, w, self.bias)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(N_EMBD)
        self.c_attn = QLinear(N_EMBD, 3 * N_EMBD)
        self.attn_c_proj = QLinear(N_EMBD, N_EMBD)
        self.ln_2 = nn.LayerNorm(N_EMBD)
        self.c_fc = QLinear(N_EMBD, 4 * N_EMBD)
        self.mlp_c_proj = QLinear(4 * N_EMBD, N_EMBD)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(self.ln_1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.attn_c_proj(a.transpose(1, 2).contiguous().view(B, T, C))
        return x + self.mlp_c_proj(F.gelu(self.c_fc(self.ln_2(x)), approximate="tanh"))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB, N_EMBD)
        self.wpe = nn.Embedding(CTX, N_EMBD)
        self.h = nn.ModuleList(Block() for _ in range(N_LAYER))
        self.ln_f = nn.LayerNorm(N_EMBD)

    def forward(self, idx):
        x = self.wte(idx) + self.wpe(torch.arange(idx.shape[1], device=idx.device))
        for b in self.h:
            x = b(x)
        return F.linear(self.ln_f(x), self.wte.weight)  # head tied to the token embedding, as GPT-2


def load_pretrained(model):
    """Copy the OpenAI GPT-2 weights in. Their Conv1D matrices are stored (in, out); ours are (out, in), so the input
    dimension is the last one and the block-of-16 scales run along it, as in the other milestone files."""
    from safetensors.torch import load_file
    sd = {k.removeprefix("transformer."): v for k, v in load_file(fetch(GPT2_URL, "gpt2.safetensors")).items()}
    ren = {"attn.c_attn": "c_attn", "attn.c_proj": "attn_c_proj", "mlp.c_fc": "c_fc", "mlp.c_proj": "mlp_c_proj"}
    ours = model.state_dict()
    with torch.no_grad():
        for k, v in sd.items():
            if k.endswith(".attn.bias") or k.endswith(".attn.masked_bias"):
                continue  # causal-mask buffers
            name = k
            for a, b in ren.items():
                name = name.replace(f".{a}.", f".{b}.")
            if name.endswith(".weight") and any(f".{b}." in name for b in ren.values()):
                v = v.t()
            assert ours[name].shape == v.shape, (k, name, v.shape)
            ours[name].copy_(v)
    return model


# ---------------------------------------------------------------- optimizer

MUON_NAMES = ("c_attn.weight", "attn_c_proj.weight", "c_fc.weight", "mlp_c_proj.weight")


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
    """Muon with the arm's storage: float (muon_float, q4_float_copy), FP8 weights + 8-bit momentum (fp8_m8), or
    4-bit weights + 4-bit carry + 8-bit momentum (q4_t4_m8)."""

    def __init__(self, params, lr, gen, arm, momentum=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))
        self.gen, self.arm = gen, arm
        self.changed, self.total = 0, 0  # weights whose grid code changed, summed over steps since the last read
        self.changed4, self.changed8 = 0, 0  # the same for the weights' 4-bit and FP8 round-to-nearest codes

    @torch.no_grad()
    def init_storage(self):
        """Round the pretrained weights onto the arm's grid (to nearest); the ticker's carry starts as the leftover."""
        for group in self.param_groups:
            for p in group["params"]:
                if self.arm == "fp8_m8":
                    p.copy_(quantize_fp8_rtn(p))
                elif self.arm == "q4_t4_m8":
                    w_hat = quantize_rtn(p)
                    s = block_scale(p)
                    x_new = w_hat.view(p.size(0), -1, BLOCK) / s
                    self.state[p]["carry"] = carry_code((p - w_hat).view_as(x_new) / s, x_new, self.gen)
                    p.copy_(w_hat)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, beta = group["lr"], group["momentum"]
            for p in group["params"]:
                st = self.state[p]
                buf = st.setdefault("momentum", torch.zeros_like(p))
                buf.lerp_(p.grad, 1 - beta)
                u = p.grad.lerp(buf, beta)  # Nesterov
                if self.arm in ("fp8_m8", "q4_t4_m8"):
                    buf.copy_(store_int8(buf, self.gen))
                update = max(1, p.size(0) / p.size(1)) ** 0.5 * newton_schulz(u).to(p.dtype)
                c4, c8 = code4(p), code8(p)
                if self.arm in ("muon_float", "q4_float_copy"):
                    p.add_(update, alpha=-lr)
                    self.changed4 += (code4(p) != c4).sum().item()
                    self.changed8 += (code8(p) != c8).sum().item()
                    self.total += p.numel()
                    continue
                if self.arm == "fp8_m8":
                    w_hat = quantize_fp8_sr(p - lr * update, self.gen)
                    self.changed += (code8(w_hat) != c8).sum().item()
                else:
                    s_old = block_scale(p)
                    x = p.view(p.size(0), -1, BLOCK) / s_old
                    k = st["carry"].float()
                    up, dn = side_gaps(x)
                    carry = (k / 14 * torch.where(k > 0, up, dn) * s_old).view_as(p)
                    w_tilde = p - lr * update + carry
                    w_hat = quantize_rtn(w_tilde)
                    s_new = block_scale(w_tilde)
                    x_new = w_hat.view_as(x) / s_new
                    st["carry"] = carry_code((w_tilde - w_hat).view_as(x) / s_new, x_new, self.gen)
                    self.changed += ((2 * x_new).round() != (2 * x).round()).sum().item()  # codes, not values: the
                    # block scale moves a little every step with its largest weight
                self.total += p.numel()
                self.changed4 += (code4(w_hat) != c4).sum().item()
                self.changed8 += (code8(w_hat) != c8).sum().item()
                p.copy_(w_hat)

    def read_changed(self):
        """Shares of weights changed per step since the last read: the arm's own stored code (None for float arms),
        the 4-bit round-to-nearest code, the FP8 round-to-nearest code."""
        t = self.total
        out = [None if self.arm in ("muon_float", "q4_float_copy") or not t else self.changed / t,
               self.changed4 / t if t else None, self.changed8 / t if t else None]
        self.changed, self.changed4, self.changed8, self.total = 0, 0, 0, 0
        return out


def build_optimizers(model, arm, muon_lr, adamw_lr, weight_decay, gen):
    """Returns a list of (optimizer, peak lr). Muon (in the arm's storage) gets the 48 block matrices, AdamW the rest;
    for adamw_float AdamW gets everything."""
    hidden = [p for n, p in model.named_parameters() if n.endswith(MUON_NAMES)]
    rest = [p for n, p in model.named_parameters() if not n.endswith(MUON_NAMES)]
    out = []
    if arm == "adamw_float":
        rest = rest + hidden
    else:
        out.append((Muon(hidden, lr=muon_lr, gen=gen, arm=arm), muon_lr))
    groups = [{"params": [p for p in rest if p.dim() >= 2], "weight_decay": weight_decay},
              {"params": [p for p in rest if p.dim() < 2], "weight_decay": 0.0}]
    out.append((torch.optim.AdamW(groups, lr=adamw_lr, betas=(0.9, 0.95)), adamw_lr))
    return out


def state_bytes_per_weight(arm):
    """Training-state bytes per weight of the block matrices (weights and optimizer state; gradients not counted)."""
    return {"adamw_float": 4 + 8, "muon_float": 4 + 4, "q4_float_copy": 4 + 4,
            "fp8_m8": 1 + 4 / 16 + 1 + 4 / MBLOCK, "q4_t4_m8": 0.5 + 4 / 16 + 0.5 + 1 + 4 / MBLOCK}[arm]


def lr_frac(it, steps, warmup, min_frac=0.1):
    """Linear warm-up, then cosine down to min_frac of the peak."""
    if it < warmup:
        return (it + 1) / warmup
    frac = (it - warmup) / max(1, steps - warmup)
    return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, frac)))


# ---------------------------------------------------------------- eval

@torch.no_grad()
def evaluate(model, windows, device, amp, bs=4):
    """Mean cross-entropy (nats per token) over the fixed windows, and its standard error across windows."""
    model.eval()
    x_all, y_all = windows
    per_window = []
    for i in range(0, len(x_all), bs):
        x, y = x_all[i:i + bs].to(device), y_all[i:i + bs].to(device)
        with amp():
            logits = model(x)
        per_window.append(F.cross_entropy(logits.float().transpose(1, 2), y, reduction="none").mean(dim=1).cpu())
    model.train()
    ce = torch.cat(per_window).double()
    evaluate.last = ce  # per window, kept so arms can be compared window by window (a paired standard error)
    return ce.mean().item(), (ce.std(unbiased=True) / math.sqrt(len(ce))).item()


@torch.no_grad()
def evaluate_rounded(model, params, windows, device, amp, fn=quantize_rtn):
    """The model with its block matrices rounded to nearest by fn (4-bit by default; rounding after training)."""
    saved = [p.clone() for p in params]
    for p in params:
        p.copy_(fn(p))
    out = evaluate(model, windows, device, amp)
    for p, w in zip(params, saved):
        p.copy_(w)
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("adamw_float", "muon_float", "q4_float_copy", "fp8_m8", "q4_t4_m8"),
                    required=True)
    ap.add_argument("--muon-lr", type=float, default=0.003)
    ap.add_argument("--adamw-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1, help="seeds the random rounding only; the data order is fixed")
    ap.add_argument("--steps", type=int, default=760)
    ap.add_argument("--seqs-per-step", type=int, default=32)
    ap.add_argument("--micro", type=int, default=8, help="sequences per forward pass (gradient accumulation)")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=None, help="default: 5%% of the steps")
    ap.add_argument("--eval-interval", type=int, default=100)
    ap.add_argument("--eval-windows", type=int, default=256)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--train-shards", default="fineweb_train_000001.bin")
    ap.add_argument("--val-shard", default="fineweb_val_000000.bin")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--round-every", type=int, default=0,
                    help="also score the weights rounded to FP8 and to 4 bits every this many steps and at the end "
                         "(muon_float, adamw_float: both; fp8_m8: 4 bits); 0: only 4 bits, at the end")
    ap.add_argument("--save", action="store_true", help="save the final model's state_dict next to the json")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "gpt2"))
    args = ap.parse_args()
    assert args.seqs_per_step % args.micro == 0
    accum = args.seqs_per_step // args.micro
    warmup = args.warmup if args.warmup is not None else max(1, args.steps // 20)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    # T4 has no bf16: fp16 autocast with loss scaling on CUDA, plain fp32 on CPU.
    use_fp16 = device == "cuda"
    amp = (lambda: torch.autocast("cuda", dtype=torch.float16)) if use_fp16 else contextlib.nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    train = [load_shard(s) for s in args.train_shards.split(",")]
    train = train[0] if len(train) == 1 else np.concatenate(train)
    windows = fixed_windows(load_shard(args.val_shard), args.eval_windows, args.seq_len)

    torch.manual_seed(args.seed)
    data_gen = torch.Generator().manual_seed(0)  # the same batches for every arm and seed
    round_gen = torch.Generator(device).manual_seed(1000 + args.seed)  # the random rounding's own stream
    model = load_pretrained(GPT()).to(device)
    hidden = [p for n, p in model.named_parameters() if n.endswith(MUON_NAMES)]
    fwd = torch.compile(model) if args.compile else model

    val_float0, se_float0 = evaluate(fwd, windows, device, amp)  # the pretrained model, before any rounding
    opts = build_optimizers(model, args.arm, args.muon_lr, args.adamw_lr, args.weight_decay, round_gen)
    muon = opts[0][0] if isinstance(opts[0][0], Muon) else None
    if muon is not None:
        muon.init_storage()
    if args.arm == "q4_float_copy":
        for m in model.modules():
            if isinstance(m, QLinear):
                m.fake4 = True

    os.makedirs(args.out, exist_ok=True)
    lr_tag = args.adamw_lr if args.arm == "adamw_float" else args.muon_lr
    stem = os.path.join(args.out, f"{args.arm}_lr{lr_tag:g}_s{args.seed}{args.tag}")
    log = open(stem + ".jsonl", "w")
    header = dict(study="gpt2", arm=args.arm, muon_lr=args.muon_lr, adamw_lr=args.adamw_lr, seed=args.seed,
                  steps=args.steps, warmup=warmup, seqs_per_step=args.seqs_per_step, seq_len=args.seq_len,
                  tokens_per_step=args.seqs_per_step * args.seq_len, eval_windows=args.eval_windows,
                  weight_decay=args.weight_decay, compile=args.compile, device=device,
                  gpu=torch.cuda.get_device_name() if device == "cuda" else None,
                  dtype="fp16-autocast" if use_fp16 else "fp32", torch=torch.__version__,
                  hidden_weights=sum(p.numel() for p in hidden), params=sum(p.numel() for p in model.parameters()),
                  state_bytes_per_weight=state_bytes_per_weight(args.arm),
                  val_ce_pretrained=round(val_float0, 5), val_se_pretrained=round(se_float0, 5))
    print(json.dumps(header), flush=True)

    def write(row):
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()

    rounds = {"muon_float": ("fp8", "fp4"), "adamw_float": ("fp8", "fp4"), "fp8_m8": ("fp4",)}.get(args.arm, ())
    rounders = {"fp8": quantize_fp8_rtn, "fp4": quantize_rtn}
    train_loss, grad_norm, skipped, evals, rounded = [], [], 0, [], []
    t0, t_log, tok_log = time.time(), time.time(), 0
    diverged = False
    for it in range(args.steps + 1):
        if it % args.eval_interval == 0 or it == args.steps:
            ce, se = evaluate(fwd, windows, device, amp)
            per_window = evaluate.last
            evals.append((it, ce))
            row = dict(step=it, val_ce=round(ce, 5), val_se=round(se, 5))
            if args.round_every and (it % args.round_every == 0 or it == args.steps):
                rounded.append(dict(step=it))
                for r in rounds:
                    rounded[-1][r] = round(evaluate_rounded(fwd, hidden, windows, device, amp, rounders[r])[0], 5)
                    rounded[-1][r + "_per_window"] = [round(v, 5) for v in evaluate.last.tolist()]
                    row["val_ce_" + r] = rounded[-1][r]
            write(dict(row, wall_s=round(time.time() - t0, 1)))
            if not math.isfinite(ce):
                diverged = True
                break
        if it == args.steps:
            break
        f = lr_frac(it, args.steps, warmup)
        for opt, peak in opts:
            for g in opt.param_groups:
                g["lr"] = peak * f
        loss_sum = 0.0
        for _ in range(accum):
            x, y = get_batch(train, args.micro, args.seq_len, data_gen, device)
            with amp():
                loss = F.cross_entropy(fwd(x).float().view(-1, VOCAB), y.reshape(-1)) / accum
            scaler.scale(loss).backward()
            loss_sum += loss.item()
        for opt, _ in opts:
            scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        scale_before = scaler.get_scale()
        for opt, _ in opts:
            scaler.step(opt)
        scaler.update()
        for opt, _ in opts:
            opt.zero_grad(set_to_none=True)
        skipped += int(use_fp16 and scaler.get_scale() < scale_before)  # an overflow: the step was skipped
        train_loss.append(round(loss_sum, 4))
        grad_norm.append(round(gn, 4) if math.isfinite(gn) else None)
        tok_log += args.seqs_per_step * args.seq_len
        if not math.isfinite(loss_sum):
            diverged = True
            write(dict(step=it + 1, diverged=True, train_loss=loss_sum))
            break
        if (it + 1) % args.log_interval == 0:
            if device == "cuda":
                torch.cuda.synchronize()
            now = time.time()
            write(dict(step=it + 1, train_loss=round(float(np.mean(train_loss[-args.log_interval:])), 4),
                       grad_norm=round(float(np.nanmean([g for g in grad_norm[-args.log_interval:] if g is not None]
                                                          or [float("nan")])), 4),
                       **dict(zip(("changed", "changed4", "changed8"),
                                  muon.read_changed() if muon is not None else (None,) * 3)), lr_frac=round(f, 4),
                       tok_per_s=round(tok_log / (now - t_log)), skipped=skipped,
                       mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if device == "cuda" else None))
            t_log, tok_log = now, 0

    if not diverged:
        if args.arm == "fp8_m8":
            assert all(on_fp8_grid(p) for p in hidden), "a matrix left the FP8 grid"
        if args.arm == "q4_t4_m8":
            assert all(on_grid(p) for p in hidden), "a matrix left the 4-bit grid"
    summary = dict(header, final_val_ce=round(ce, 5), final_val_se=round(se, 5), diverged=diverged,
                   wall_s=round(time.time() - t0, 1), train_tokens=len(train_loss) * args.seqs_per_step * args.seq_len,
                   skipped_steps=skipped, val_ce_step0=round(evals[0][1], 5),
                   max_val_rise=round(max(c - min(e[1] for e in evals[:i]) for i, (_, c) in enumerate(evals)
                                          if i > 0), 5) if len(evals) > 1 else None,
                   train_loss=train_loss, grad_norm=grad_norm, evals=evals,
                   final_per_window=[round(v, 5) for v in per_window.tolist()], rounded=rounded)
    if rounded and rounded[-1]["step"] == args.steps:
        for r in rounds:
            summary.update({f"final_val_ce_{r}": rounded[-1][r]})
    elif args.arm in ("muon_float", "fp8_m8", "adamw_float") and not diverged:
        ce4, se4 = evaluate_rounded(fwd, hidden, windows, device, amp)
        summary.update(final_val_ce_fp4=round(ce4, 5), final_val_se_fp4=round(se4, 5))
    json.dump(summary, open(stem + ".json", "w"))
    if args.save:
        torch.save(model.state_dict(), stem + ".pt")
    print(f"final val CE {ce:.4f} +- {se:.4f} nats/token after {len(train_loss)} steps, {summary['wall_s']} s"
          + (f"; rounded to 4-bit after training: {summary['final_val_ce_fp4']:.4f}" if "final_val_ce_fp4" in summary
             else ""))


if __name__ == "__main__":
    main()
