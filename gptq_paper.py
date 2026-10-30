"""GPTQ on our 4-bit format, for the "train in higher precision, then round" baselines (PAPER_PLAN.md, sixth change-log
entry). Rounds the block matrices of a saved model (or the pretrained GPT-2) to E2M1 values with one fp32 scale per 16
inputs, two ways: round to nearest (as every paper table does) and GPTQ (Frantar et al., 2023). Scores both on the
same 256 validation windows with the evaluation code of p1_paper_gpt2.py.

Run:  python gptq_paper.py --pretrained --out runs/paper/GPTQ/pretrained.json
      python gptq_paper.py --run runs/paper/P1/muon_float_lr0.001_s1_save.json --out runs/paper/GPTQ/p1_muon_float_s1.json
"""
import argparse
import contextlib
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I leaves the script's folder off the path
import p1_paper_gpt2 as p1  # noqa: E402

GROUP = p1.BLOCK


def round_codes(x):
    """x already divided by its scale: the nearest signed E2M1 value, clipped to +-6."""
    grid = p1.dev(p1.E2M1, x.device)
    idx = torch.bucketize(x.abs().clamp(max=6.0), (grid[1:] + grid[:-1]) / 2)
    return grid[idx] * x.sign()


@torch.no_grad()
def gptq(W, H, damp=0.01, blocksize=128):
    """W (out, in), H (in, in) = sum of x x^T over the calibration inputs. Returns the rounded W and its scales."""
    W = W.clone().float()
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += torch.eye(H.shape[0], device=H.device) * damp * torch.diag(H).mean()
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    scales = torch.zeros(W.shape[0], W.shape[1] // GROUP, device=W.device)
    for i1 in range(0, W.shape[1], blocksize):
        i2 = min(i1 + blocksize, W.shape[1])
        W1, Hinv1 = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
        Err1 = torch.zeros_like(W1)
        for i in range(i2 - i1):
            col = i1 + i
            if col % GROUP == 0:  # the block's scale, from the error-updated weights of its 16 columns
                s = W1[:, i:i + GROUP].abs().amax(1).clamp_min(1e-12) / 6  # blocksize is a multiple of 16
                scales[:, col // GROUP] = s
            w, d = W1[:, i], Hinv1[i, i]
            q = round_codes(w / s) * s
            Q[:, col] = q
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q, scales


def check_grid(Q, scales):
    codes = Q.reshape(Q.shape[0], -1, GROUP) / scales.unsqueeze(-1)
    grid = p1.dev(p1.E2M1, Q.device)
    return bool(((codes.abs().unsqueeze(-1) - grid).abs().amin(-1) < 1e-4).all())


@torch.no_grad()
def gptq_model(model, calib, bs=8):
    """Sequential GPTQ: layer by layer, matrix by matrix, each from the inputs of the already-rounded model."""
    m = model
    C, Hh = m.d, m.n_head
    xs = []
    for i in range(0, len(calib), bs):
        idx = calib[i:i + bs]
        xs.append(m.wte(idx) + m.wpe(torch.arange(idx.shape[1], device=idx.device)))
    on_grid = True

    def hessian(fn, dim):
        H = None
        n = 0
        for x in xs:
            a = fn(x).reshape(-1, dim).float()
            H = a.T @ a if H is None else H + a.T @ a
            n += a.shape[0]
        return H / n

    def attn(x, i):
        B_, T, _ = x.shape
        q, k, v = F.linear(m.ln1[i](x), m.W["c_attn"][i], m.B["c_attn"][i]).split(C, dim=2)
        q, k, v = (t.view(B_, T, Hh, C // Hh).transpose(1, 2) for t in (q, k, v))
        return F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B_, T, C)

    for i in range(m.n_layer):
        steps = (("c_attn", lambda x: m.ln1[i](x), C),
                 ("attn_c_proj", lambda x: attn(x, i), C),
                 ("c_fc", lambda x: m.ln2[i](x + F.linear(attn(x, i), m.W["attn_c_proj"][i], m.B["attn_c_proj"][i])), C),
                 ("mlp_c_proj", lambda x: F.gelu(F.linear(m.ln2[i](x + F.linear(attn(x, i), m.W["attn_c_proj"][i],
                                                                                    m.B["attn_c_proj"][i])),
                                                         m.W["c_fc"][i], m.B["c_fc"][i]), approximate="tanh"), 4 * C))
        for name, fn, dim in steps:
            Q, s = gptq(m.W[name][i], hessian(fn, dim))
            on_grid &= check_grid(Q, s)
            m.W[name][i].copy_(Q)
        for j, x in enumerate(xs):
            x = x + F.linear(attn(x, i), m.W["attn_c_proj"][i], m.B["attn_c_proj"][i])
            h = F.gelu(F.linear(m.ln2[i](x), m.W["c_fc"][i], m.B["c_fc"][i]), approximate="tanh")
            xs[j] = x + F.linear(h, m.W["mlp_c_proj"][i], m.B["mlp_c_proj"][i])
    return on_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="a run's .json; its .pt (saved with --save) is loaded")
    ap.add_argument("--pretrained", action="store_true", help="the pretrained GPT-2, no training")
    ap.add_argument("--calib", type=int, default=128)
    ap.add_argument("--calib-seed", type=int, default=1234)
    ap.add_argument("--windows", type=int, default=256, help="validation windows (256 in every paper table)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert bool(args.run) != args.pretrained
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    amp = (lambda: torch.autocast("cuda", dtype=torch.float16)) if device == "cuda" else contextlib.nullcontext

    if args.pretrained:
        hdr = dict(n_layer=12, n_embd=768, n_head=12, train_shards="fineweb_train_000001.bin", final_val_ce=None,
                   final_val_ce_fp4=None)
    else:
        hdr = json.load(open(args.run))
    model = p1.GPT(hdr["n_layer"], hdr["n_head"], hdr["n_embd"])
    if args.pretrained:
        p1.load_pretrained(model)
    else:
        model.load_state_dict(torch.load(args.run[:-5] + ".pt", map_location="cpu"))
    model = model.to(device).eval()
    windows = p1.fixed_windows(p1.load_shard("fineweb_val_000000.bin"), args.windows, 1024)
    fwd = torch.compile(model) if device == "cuda" else model
    ebs = 8 if device == "cuda" else 1

    out = dict(script="gptq_paper.py", source=args.run or "pretrained GPT-2", calib=args.calib,
               calib_seed=args.calib_seed, logged_float=hdr["final_val_ce"], logged_rtn=hdr.get("final_val_ce_fp4"))
    with torch.no_grad():
        ce, se = p1.evaluate(fwd, windows, device, amp, ebs)
        out.update(float_ce=ce, float_se=se, float_per_window=p1.evaluate.last.tolist())
        ce, se = p1.evaluate_rounded(fwd, list(model.W.values()), windows, device, amp, ebs)
        out.update(rtn_ce=ce, rtn_se=se, rtn_per_window=p1.evaluate.last.tolist())
        print(f"float {out['float_ce']:.5f} (logged {out['logged_float']}), "
              f"round to nearest {out['rtn_ce']:.5f} (logged {out['logged_rtn']})", flush=True)

        shard = p1.load_shard(hdr["train_shards"].split(",")[0])
        g = np.random.default_rng(args.calib_seed)
        starts = g.integers(0, len(shard) - 1025, args.calib)
        calib = torch.from_numpy(np.stack([shard[s:s + 1024] for s in starts]).astype(np.int64)).to(device)
        out["on_grid"] = gptq_model(model, calib)
        ce, se = p1.evaluate(fwd, windows, device, amp, ebs)
        out.update(gptq_ce=ce, gptq_se=se, gptq_per_window=p1.evaluate.last.tolist())
    print(f"GPTQ {out['gptq_ce']:.5f}, on grid {out['on_grid']}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"))


if __name__ == "__main__":
    main()
