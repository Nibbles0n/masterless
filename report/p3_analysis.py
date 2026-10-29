"""P3 analysis (PAPER_PLAN.md): the noise model, pass lines P3a and P3b, and the P3c carry prediction for P5.

The model: a b-bit carry adds about (gap / 2^b) * sqrt(steps) / 2 of random drift to each weight; the weight's intended
travel is about sum over steps of (lr_t * u), u = the root-mean-square of Muon's update per weight. rho = drift / travel.
  gap: the mean over the block-matrix weights of the 4-bit gap around each weight (mean of the gaps up and down, times
       its block scale), measured on the starting weights rounded to 4 bits.
  u:   measured on CPU from one batch (8 x 1,024 tokens) of FineWeb: Muon's update (Newton-Schulz of the gradient, times
       its shape factor) at the start of training, root-mean-square over the 48 matrices' weights.
For RL (P3c) the travel is the per-step root-mean-square of lr * AdamW's update measured in the P5 pilot (mean of its
last 10 of 30 steps), times 200 steps, and the gap is measured on Qwen3.5-0.8B's decoder matrices.

Writes runs/paper/P3_summary.json. Run: python report/p3_analysis.py
"""
import glob
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import p1_paper_gpt2 as P  # noqa: E402

torch.set_num_threads(2)
D3 = os.path.join(HERE, "runs", "paper", "P3")
BITS, LRS, STEPS = (4, 6, 8, 12), (1e-4, 3e-4, 1e-3, 3e-3), 305
FIT, HELD = (3e-4, 1e-3), (1e-4, 3e-3)


def mean_gap(ws):
    """mean 4-bit gap around each weight (absolute units) over the given matrices"""
    tot, n = 0.0, 0
    for w in ws:
        w = w.float()
        w = w.reshape(-1, w.shape[-1])[:, : w.shape[-1] // P.BLOCK * P.BLOCK]
        wq = P.quantize_rtn(w)
        s = P.block_scale(w)
        up, dn = P.side_gaps(P.blocks(wq) / s)
        tot += float(((up + dn) / 2 * s).sum())
        n += w.numel()
    return tot / n


def gpt2_numbers():
    model = P.GPT()
    P.load_pretrained(model)
    hidden = list(model.W.values())
    gap = mean_gap([w.detach() for w in hidden])
    data = P.load_shard("fineweb_train_000001.bin")
    rng = np.random.default_rng(0)
    T, n = 1024, 8
    starts = rng.integers(0, len(data) - T - 1, n)
    x = torch.stack([torch.from_numpy(data[s:s + T].astype(np.int64)) for s in starts])
    y = torch.stack([torch.from_numpy(data[s + 1:s + T + 1].astype(np.int64)) for s in starts])
    for i in range(n):  # one sequence at a time, to fit in CPU memory
        loss = model(x[i:i + 1], y[i:i + 1])
        loss = loss[0] if isinstance(loss, tuple) else loss
        (loss / n).backward()
    sq, cnt = 0.0, 0
    for p in hidden:
        upd = max(1, p.size(1) / p.size(2)) ** 0.5 * P.newton_schulz_batched(p.grad, torch.float32)
        sq += float((upd ** 2).sum())
        cnt += upd.numel()
    return gap, math.sqrt(sq / cnt)


def qwen_gap():
    from safetensors import safe_open
    snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*/"))[0]
    ws = []
    for f in glob.glob(os.path.join(snap, "*.safetensors")):
        with safe_open(f, "pt") as fh:
            for k in fh.keys():
                if ".layers." in k and k.endswith(".weight") and "mtp" not in k and "visual" not in k:
                    t = fh.get_tensor(k)
                    if t.dim() == 2 and t.shape[-1] % P.BLOCK == 0:
                        ws.append(t)
    return mean_gap(ws), sum(w.numel() for w in ws)


def load(arm, lr):
    with open(os.path.join(D3, f"{arm}_lr{lr:g}_s1.json")) as f:
        return json.load(f)


def avg_ranks(v):
    order = np.argsort(v, kind="stable")
    r = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def main():
    gap, u = gpt2_numbers()
    warm = max(1, STEPS // 20)
    frac_sum = sum(P.lr_frac(i, STEPS, warm) for i in range(STEPS))
    cells = []
    for lr in LRS:
        ref = load("q4_float_copy", lr)
        rw = np.array(ref["final_per_window"])
        for b in BITS:
            r = load(f"q4_t{b}_m8", lr)
            d = np.array(r["final_per_window"]) - rw
            m, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
            drift = gap / 2 ** b * math.sqrt(STEPS) / 2
            travel = lr * frac_sum * u
            cells.append(dict(bits=b, lr=lr, loss=r["final_val_ce"], copy=ref["final_val_ce"], gap=m, se=se,
                              tied=abs(m) <= 2 * se, works=m <= 2 * se, rho=drift / travel))
    # P3a: rank correlation, tied cells (within 2 SE of zero) share the lowest ranks
    rho = np.array([c["rho"] for c in cells])
    g = np.array([-np.inf if c["tied"] else c["gap"] for c in cells])
    rg, rr = avg_ranks(g), avg_ranks(rho)
    spearman = float(np.corrcoef(rg, rr)[0, 1])
    # P3b: threshold fitted on the two middle rates, the fewest misclassified cells, then the geometric midpoint
    fit = [c for c in cells if c["lr"] in FIT]
    cand = sorted({c["rho"] for c in fit})
    cuts = [cand[0] / 2] + [math.sqrt(a * b) for a, b in zip(cand, cand[1:])] + [cand[-1] * 2]
    err = [sum((c["rho"] <= t) != c["works"] for c in fit) for t in cuts]
    best = min(err)
    good = [t for t, e in zip(cuts, err) if e == best]
    rho_star = math.sqrt(good[0] * good[-1])
    held = []
    for lr in HELD:
        cs = [c for c in cells if c["lr"] == lr]
        pred = next((c["bits"] for c in cs if c["rho"] <= rho_star), None)
        meas = next((c["bits"] for c in cs if c["works"] and all(d["works"] for d in cs if d["bits"] > c["bits"])), None)
        ok = pred is not None and meas is not None and abs(BITS.index(pred) - BITS.index(meas)) <= 1
        held.append(dict(lr=lr, predicted_bits=pred, measured_bits=meas, ok=ok))
    # P3c: RL
    qgap, qn = qwen_gap()
    pilot = [json.loads(l) for l in open(os.path.join(HERE, "runs", "paper", "p5_float_adamw_vllm_s1_pilot.jsonl"))]
    upd = [r["update_rms"] for r in pilot if r.get("update_rms") is not None][-10:]
    rl_steps = 200
    rl_travel = rl_steps * float(np.mean(upd))
    rl = []
    for b in (4, 6, 8, 12, 16):
        r_ = qgap / 2 ** b * math.sqrt(rl_steps) / 2 / rl_travel
        rl.append(dict(bits=b, rho=r_, works=r_ <= rho_star))
    pred_rl = next((c["bits"] for c in rl if c["works"]), None)
    out = dict(gpt2=dict(mean_gap=gap, muon_update_rms=u, lr_frac_sum=frac_sum, steps=STEPS), cells=cells,
               P3a=dict(spearman=spearman, result=spearman >= 0.8),
               P3b=dict(rho_star=rho_star, fit_misclassified=best, held_out=held, result=all(h["ok"] for h in held)),
               P3c=dict(qwen_mean_gap=qgap, qwen_matrix_weights=qn, rl_update_rms_per_step=float(np.mean(upd)),
                        rl_steps=rl_steps, rl_travel=rl_travel, ladder=rl, predicted_bits=pred_rl))
    with open(os.path.join(HERE, "runs", "paper", "P3_summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=1))
    for c in cells:
        print(f"b={c['bits']:2d} lr={c['lr']:g}  gap {c['gap']:+.5f} ± {c['se']:.5f}  works={c['works']}  rho={c['rho']:.3g}")


if __name__ == "__main__":
    main()
