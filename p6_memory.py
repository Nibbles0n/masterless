"""P6 (PAPER_PLAN.md): memory, measured. Peak GPU memory and time of one full-parameter training step of Qwen3.5-0.8B
(text only) on one T4, for four setups that differ only in how weights and optimizer state are stored:

  a      bf16 weights + bf16 gradients + fp32 master copy + AdamW (fp32 moments) on every parameter: the standard
  b      float Muon: fp32 weights and gradients, fp32 Muon momentum on the decoder matrices, fp32 AdamW on the rest
  c      bf16-SR: bf16 weights, no master copy, updates rounded to bf16 at random; 8-bit Muon momentum on the decoder
         matrices; the rest bf16 + 8-bit AdamW (random rounding). Standard step after backward (full gradients exist)
  c_lw   bf16-SR as c, but each decoder matrix is updated inside the backward pass as soon as its gradient is ready
         (post-accumulate-grad hook), and its gradient freed: the same layer-by-layer trick ours uses
  d      ours: decoder matrices as release/ticker_muon.TickerLinear (two 4-bit codes per byte, FP8 block scales,
         int8 carry, int8 momentum), updated inside the backward pass; the rest bf16 + 8-bit AdamW, as in c
  d_frozen  d with the 4-bit matrices frozen (no weight gradient, no update): only to time what the update adds

Same in every setup: batch, sequence length, random token ids from a fixed seed, activation checkpointing on every
decoder layer, and a chunked loss (logits made 256 tokens at a time under checkpointing, so the 248k-wide vocabulary
never makes a full logits tensor). Compute in bf16 (setup b: fp32 weights under bf16 autocast). Newton-Schulz in fp16
on a T4 (no bf16 tensor cores). Learning rates are arbitrary small values; only memory and time are measured.

Run one setup:   python p6_memory.py --setup d --model Qwen/Qwen3.5-0.8B --out runs/paper/p6_raw.jsonl
Run all + 2B:    python p6_memory.py --all              (each setup in a fresh process; writes runs/paper/P6_memory.json)
CPU smoke test:  python p6_memory.py --smoke            (tiny random Qwen3.5 config; checks every setup runs)
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "release"))
import ticker_muon as tm  # noqa: E402

SETUPS = ("a", "b", "c", "c_lw", "d", "d_frozen")
MUON_LR, REST_LR, BETA = 1e-4, 1e-5, 0.95
CE_CHUNK = 256
CH = 1 << 24  # elementwise optimizer work in chunks of this many values (no full-size fp32 temporaries)


# ---------------------------------------------------------------- model and loss

def load_model(repo, dtype, device, smoke=False):
    from transformers import AutoConfig, AutoModelForCausalLM
    if smoke:
        cfg = AutoConfig.from_pretrained(repo).text_config
        cfg.update(dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=2,
                        num_key_value_heads=1, head_dim=32, linear_key_head_dim=16, linear_value_head_dim=16,
                        linear_num_key_heads=2, linear_num_value_heads=2, vocab_size=512,
                        layer_types=["linear_attention"] * 3 + ["full_attention"]))
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype)
    model.config.use_cache = False
    model.to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model


def _ce_chunk(h, W, y):
    return F.cross_entropy(F.linear(h, W).float(), y, reduction="sum")


def loss_fn(model, ids):
    """Mean next-token loss with the (tied) head applied CE_CHUNK tokens at a time under checkpointing."""
    h = model.model(input_ids=ids).last_hidden_state
    H = h.size(-1)
    h, y = h[:, :-1].reshape(-1, H), ids[:, 1:].reshape(-1)
    W = model.lm_head.weight
    total = 0.0
    for i in range(0, y.numel(), CE_CHUNK):
        total = total + checkpoint(_ce_chunk, h[i:i + CE_CHUNK], W, y[i:i + CE_CHUNK], use_reentrant=False)
    return total / y.numel()


def hidden_linears(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and tm._default_include(n, m)]


# ---------------------------------------------------------------- the baseline optimizers

class MasterAdamW:
    """Setup a: fp32 master copy and fp32 AdamW moments for every parameter; the model holds bf16 weights."""

    def __init__(self, params, lr, betas=(0.9, 0.95), eps=1e-8):
        self.params = [p for p in params if p.requires_grad]
        self.master = [p.detach().float() for p in self.params]
        self.m = [torch.zeros_like(q) for q in self.master]
        self.v = [torch.zeros_like(q) for q in self.master]
        self.lr, self.betas, self.eps, self.t = lr, betas, eps, 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        b1, b2 = self.betas
        bc1, bc2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for p, w, m, v in zip(self.params, self.master, self.m, self.v):
            if p.grad is None:
                continue
            pf, gf, wf, mf, vf = p.view(-1), p.grad.view(-1), w.view(-1), m.view(-1), v.view(-1)
            for a in range(0, pf.numel(), CH):
                b = min(pf.numel(), a + CH)
                g = gf[a:b].float()
                mf[a:b].lerp_(g, 1 - b1)
                vf[a:b].mul_(b2).addcmul_(g, g, value=1 - b2)
                del g
                wf[a:b].addcdiv_(mf[a:b] / bc1, (vf[a:b] / bc2).sqrt_().add_(self.eps), value=-self.lr)
                pf[a:b].copy_(wf[a:b])
            p.grad = None


class FloatAdamW(MasterAdamW):
    """The rest of setup b: fp32 weights, fp32 moments, no separate master (the weights are fp32)."""

    def __init__(self, params, lr, betas=(0.9, 0.95), eps=1e-8):
        self.params = [p for p in params if p.requires_grad]
        self.master = [p.data for p in self.params]  # the fp32 weights themselves
        self.m = [torch.zeros_like(p, dtype=torch.float32) for p in self.params]
        self.v = [torch.zeros_like(p, dtype=torch.float32) for p in self.params]
        self.lr, self.betas, self.eps, self.t = lr, betas, eps, 0


class FloatMuon:
    """Setup b: fp32 momentum, Newton-Schulz, plain fp32 update."""

    def __init__(self, params, lr, ns_dtype):
        self.params = list(params)
        self.buf = [torch.zeros_like(p, dtype=torch.float32) for p in self.params]
        self.lr, self.ns_dtype = lr, ns_dtype

    @torch.no_grad()
    def step(self):
        for p, buf in zip(self.params, self.buf):
            if p.grad is None:
                continue
            g = p.grad.float()
            buf.lerp_(g, 1 - BETA)
            u = g.lerp(buf, BETA)
            del g
            p.grad = None
            p.add_(tm._orthogonal_update(u, 5, self.ns_dtype), alpha=-self.lr)


class SRMuon:
    """Setups c / c_lw: bf16 weights, 8-bit momentum, random rounding of the new weight to bf16. With layerwise=True
    each matrix is updated by a post-accumulate-grad hook during backward and its gradient freed at once."""

    def __init__(self, params, lr, ns_dtype, layerwise=False, seed=0):
        self.params = list(params)
        self.lr, self.ns_dtype, self.seed, self.t = lr, ns_dtype, seed, 0
        self.state = {}
        for i, p in enumerate(self.params):
            npad = tm._pad128(p.numel())
            self.state[p] = dict(i=i, step=0, q=torch.zeros(npad, dtype=torch.int8, device=p.device),
                                 s=torch.zeros(npad // tm.MBLOCK, device=p.device))
            if layerwise:
                p.register_post_accumulate_grad_hook(self._hook)
        self.update_times = []

    def _hook(self, p):
        self._update(p)

    @torch.no_grad()
    def _update(self, p):
        st = self.state[p]
        st["step"] += 1
        g = p.grad
        p.grad = None
        u, q, s = tm._momentum(g, st["q"], st["s"], BETA, True, tm.rng_key(self.seed, st["i"], st["step"], 2))
        del g
        st["q"].copy_(q)
        st["s"].copy_(s)
        upd = tm._orthogonal_update(u, 5, self.ns_dtype)
        del u
        y = p.float() - self.lr * upd
        del upd
        p.copy_(tm._sr_to_bf16(y, tm.rng_key(self.seed, st["i"], st["step"], 5), 0).view_as(p))

    def step(self):
        for p in self.params:
            if p.grad is not None:
                self._update(p)


# ---------------------------------------------------------------- one setup

def tensor_bytes(ts):
    seen, n = set(), 0
    for t in ts:
        if t is None or t.data_ptr() in seen:
            continue
        seen.add(t.data_ptr())
        n += t.numel() * t.element_size()
    return n


def build(setup, repo, device, smoke):
    dtype = torch.float32 if setup == "b" else torch.bfloat16
    model = load_model(repo, dtype, device, smoke)
    ns_dtype = tm._default_ns_dtype(torch.device(device))
    info = {}
    n_params = sum(p.numel() for p in model.parameters())
    hid = hidden_linears(model)
    n_hidden = sum(m.weight.numel() for _, m in hid)
    info.update(params=n_params, hidden_matrix_params=n_hidden, hidden_matrices=len(hid))
    opts, ticker = [], None
    if setup == "a":
        opts = [MasterAdamW(model.parameters(), REST_LR)]
    elif setup == "b":
        hp = [m.weight for _, m in hid]
        hid_ids = {id(p) for p in hp}
        opts = [FloatMuon(hp, MUON_LR, ns_dtype),
                FloatAdamW([p for p in model.parameters() if id(p) not in hid_ids], REST_LR)]
    elif setup in ("c", "c_lw"):
        hp = [m.weight for _, m in hid]
        hid_ids = {id(p) for p in hp}
        opts = [SRMuon(hp, MUON_LR, ns_dtype, layerwise=setup == "c_lw"),
                tm.AdamW8bit([p for p in model.parameters() if id(p) not in hid_ids], lr=REST_LR)]
    else:
        layers = tm.convert_linears(model, scales="fp8", carry_bits=8, seed=0)
        ticker = tm.TickerMuon(layers, lr=MUON_LR, ns_dtype=ns_dtype)
        if setup == "d_frozen":
            for m in layers:
                m.trainable = False
        opts = [tm.AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=REST_LR)]
        info["ticker_layers"] = len(layers)
    return model, opts, ticker, info


def state_tensors(model, opts, ticker):
    """Every tensor that persists between steps: weights (and packed buffers) and optimizer state."""
    ts = list(model.parameters()) + list(model.buffers())
    for o in opts:
        if isinstance(o, MasterAdamW):
            ts += o.master + o.m + o.v
        elif isinstance(o, FloatMuon):
            ts += o.buf
        elif isinstance(o, SRMuon):
            ts += [t for st in o.state.values() for t in (st["q"], st["s"])]
        else:
            ts += [t for st in o.state.values() for t in st.values() if torch.is_tensor(t)]
    return ts


def run_setup(setup, repo, batch, seq, steps, device="cuda", smoke=False):
    cuda = device == "cuda"
    torch.manual_seed(0)
    t0 = time.time()
    model, opts, ticker, info = build(setup, repo, device, smoke)
    info["build_seconds"] = round(time.time() - t0, 1)
    vocab = model.config.vocab_size
    ids = torch.randint(0, vocab, (batch, seq), generator=torch.Generator().manual_seed(0)).to(device)
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    st_bytes = tensor_bytes(state_tensors(model, opts, ticker))
    rec = dict(setup=setup, model=repo, batch=batch, seq=seq, **info,
               steady_state_bytes_counted=st_bytes,
               steady_state_allocated=torch.cuda.memory_allocated() if cuda else None)
    upd_events = []
    if ticker is not None and cuda:
        def hook(kind, mod):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            upd_events.append(ev)
        ticker.update_hook = hook
    use_amp = setup == "b"
    out = []
    for step in range(steps):
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
        upd_events.clear()
        t_start = time.perf_counter()
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
            loss = loss_fn(model, ids)
        loss.backward()
        if cuda:
            torch.cuda.synchronize()
        t_bwd = time.perf_counter()
        for o in opts:
            o.step()
        for p in model.parameters():
            p.grad = None
        if cuda:
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        r = dict(step=step + 1, loss=loss.item(), seconds=t_end - t_start, seconds_fwd_bwd=t_bwd - t_start,
                 seconds_after_backward=t_end - t_bwd)
        if cuda:
            r.update(peak=torch.cuda.max_memory_allocated(), before=base, after=torch.cuda.memory_allocated(),
                     reserved_peak=torch.cuda.max_memory_reserved())
            if upd_events:
                r["seconds_layer_updates"] = sum(upd_events[i].elapsed_time(upd_events[i + 1])
                                                 for i in range(0, len(upd_events), 2)) / 1000
        out.append(r)
        print(json.dumps(r), flush=True)
    rec["steps"] = out
    if ticker is not None:
        layer_steps = {m.step for m in ticker._modules_by_id.values()}
        rec["ticker_layer_steps"] = sorted(layer_steps)  # each layer must have updated exactly once per step
    if cuda:
        rec["steady_state_allocated_after"] = torch.cuda.memory_allocated()
        rec["gpu"] = torch.cuda.get_device_name()
    return rec


# ---------------------------------------------------------------- driver

def env_info():
    import transformers
    d = dict(torch=torch.__version__, transformers=transformers.__version__)
    try:
        import fla  # noqa: F401
        d["fla"] = True
    except Exception:  # noqa: BLE001
        d["fla"] = False
    return d


def child(args):
    try:
        rec = run_setup(args.setup, args.model, args.batch, args.seq, args.steps)
        rec["status"] = "ok"
    except torch.cuda.OutOfMemoryError as e:
        rec = dict(setup=args.setup, model=args.model, batch=args.batch, seq=args.seq, status="OOM",
                   error=str(e).split("\n")[0][:300], peak_before_oom=torch.cuda.max_memory_allocated())
    rec["env"] = env_info()
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run_all(args):
    raw = args.out
    jobs = [(s, args.model) for s in args.setups.split(",")] + [(s, args.model_2b) for s in args.setups_2b.split(",")]
    for s, repo in jobs:
        print(f"=== {s} {repo}", flush=True)
        subprocess.run([sys.executable, __file__, "--setup", s, "--model", repo, "--batch", str(args.batch),
                        "--seq", str(args.seq), "--steps", str(args.steps), "--out", raw], check=False)


def smoke():
    for s in SETUPS:
        rec = run_setup(s, "Qwen/Qwen3.5-0.8B", 2, 64, 2, device="cpu", smoke=True)
        assert all(math.isfinite(r["loss"]) for r in rec["steps"]), s
        if s == "d":
            assert rec["ticker_layer_steps"] == [2], rec["ticker_layer_steps"]
        print(s, "ok", [round(r["loss"], 4) for r in rec["steps"]], rec["steady_state_bytes_counted"])


def summarize(raw, out_json):
    """Collect the raw per-setup records into runs/paper/P6_memory.json (the last record per setup and model wins)."""
    recs = {}
    for line in open(raw):
        r = json.loads(line)
        recs[(r["model"], r["setup"])] = r
    GiB = 2 ** 30
    rows = {}
    for (model, setup), r in recs.items():
        row = dict(model=model, setup=setup, status=r["status"], batch=r["batch"], seq=r["seq"])
        if r["status"] == "ok":
            later = r["steps"][1:] or r["steps"]
            row.update(
                params=r["params"], hidden_matrix_params=r["hidden_matrix_params"],
                peak_GiB=max(x["peak"] for x in later) / GiB, peak_first_step_GiB=r["steps"][0]["peak"] / GiB,
                steady_state_GiB=r["steady_state_allocated"] / GiB,
                steady_state_counted_GiB=r["steady_state_bytes_counted"] / GiB,
                step_seconds=sum(x["seconds"] for x in later) / len(later),
                fwd_bwd_seconds=sum(x["seconds_fwd_bwd"] for x in later) / len(later),
                after_backward_seconds=sum(x["seconds_after_backward"] for x in later) / len(later),
                losses=[x["loss"] for x in r["steps"]], gpu=r.get("gpu"), env=r.get("env"))
            if "seconds_layer_updates" in later[0]:
                row["layer_update_seconds"] = sum(x["seconds_layer_updates"] for x in later) / len(later)
            if "ticker_layer_steps" in r:
                row["ticker_layer_steps"] = r["ticker_layer_steps"]
        else:
            row.update(error=r.get("error"), peak_before_oom_GiB=r.get("peak_before_oom", 0) / GiB, env=r.get("env"))
        rows[f"{model}|{setup}"] = row
    small = "Qwen/Qwen3.5-0.8B"
    g = lambda s, k: rows.get(f"{small}|{s}", {}).get(k)  # noqa: E731
    res = dict(settings=dict(
        gpu="Tesla T4 (Colab)", data="random token ids, fixed seed", activation_checkpointing="every decoder layer",
        loss=f"chunked cross-entropy, {CE_CHUNK} tokens per chunk under checkpointing, tied head",
        compute_dtype="bf16 (setup b: fp32 weights under bf16 autocast)", newton_schulz_dtype="fp16 on T4",
        peak="torch.cuda.max_memory_allocated over a training step, max over steps 2..n (state already allocated)",
        steady_state="torch.cuda.memory_allocated before a step: weights + optimizer state, no gradients",
        muon_lr=MUON_LR, rest_lr=REST_LR), rows=rows)
    if g("a", "peak_GiB") and g("d", "peak_GiB"):
        res["P6a_ratio_ours_to_standard"] = g("d", "peak_GiB") / g("a", "peak_GiB")
        res["P6a_pass"] = res["P6a_ratio_ours_to_standard"] <= 0.35
    if g("c", "peak_GiB") and g("d", "peak_GiB"):
        res["ours_to_bf16SR_peak"] = g("d", "peak_GiB") / g("c", "peak_GiB")
        res["ours_to_bf16SR_steady"] = g("d", "steady_state_GiB") / g("c", "steady_state_GiB")
    if g("c_lw", "peak_GiB") and g("d", "peak_GiB"):
        res["ours_to_bf16SR_layerwise_peak"] = g("d", "peak_GiB") / g("c_lw", "peak_GiB")
    if g("d", "step_seconds") and g("d_frozen", "step_seconds"):
        res["update_adds_seconds"] = g("d", "step_seconds") - g("d_frozen", "step_seconds")
    with open(out_json, "w") as f:
        json.dump(res, f, indent=1)
    for k, row in rows.items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in row.items() if a not in ("env",)})
    print({k: v for k, v in res.items() if k != "rows"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", choices=SETUPS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize", metavar="RAW_JSONL")
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--model-2b", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--setups", default="a,b,c,c_lw,d,d_frozen")
    ap.add_argument("--setups-2b", default="d,a")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "paper", "p6_raw.jsonl"))
    args = ap.parse_args()
    if args.summarize:
        summarize(args.summarize, os.path.join(HERE, "runs", "paper", "P6_memory.json"))
    elif args.smoke:
        torch.set_num_threads(2)
        smoke()
    elif args.all:
        run_all(args)
    else:
        child(args)


if __name__ == "__main__":
    main()
