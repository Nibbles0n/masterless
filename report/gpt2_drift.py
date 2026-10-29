"""How far did per-step random rounding push the FP8 arm away from float Muon in side study G (stage 3 weights)?
Compares the rounding law's prediction (random-walk drift sqrt(r * path)) with the measured distance between the two
final models, in 4-bit gaps, and how much of float Muon's net move the FP8 arm reproduced. Uses only saved weights.

Run:  python report/gpt2_drift.py  ->  report/data/gpt2_drift.json
"""
import importlib.util
import json
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("g2", os.path.join(ROOT, "side_gpt2.py"))
g2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g2)
zz = {r["name"]: r for r in json.load(open(os.path.join(HERE, "data", "gpt2_zigzag.json")))["rows"]}
w0 = {k: v for k, v in g2.load_pretrained(g2.GPT()).state_dict().items() if k.endswith(g2.MUON_NAMES)}
ld = lambda a: torch.load(os.path.join(ROOT, f"data/gpt2_s3/{a}_lr0.001_s1.pt"), map_location="cpu", weights_only=True)
wf, w8 = ld("muon_float"), ld("fp8_m8")


def gap4(w):
    s = g2.block_scale(w)
    a = (w.view(w.size(0), -1, 16) / s).abs()
    E = g2.E2M1
    i = torch.bucketize(a, E, right=True).clamp(1, len(E) - 1)
    return ((E[i] - E[i - 1]) * s).view_as(w)


def fp8_gap(w):
    """Spacing of the E4M3 grid at each weight (block scale absmax / 448)."""
    s = g2.fp8_scale(w)
    a = (w.view(w.size(0), -1, 16) / s).abs().clamp_min(2.0 ** -9)
    return (torch.exp2((torch.frexp(a).exponent - 1).clamp(min=-6).float() - 3) * s).view_as(w)


acc = {k: 0.0 for k in ("n", "r", "pred_var", "meas_sq", "net_f_sq", "net_8_sq", "dot", "rtn_sq")}
for k in w0:
    g = gap4(w0[k])
    r = fp8_gap(w0[k]) / g
    L = zz[k]["L4"]  # mean path in 4-bit gaps (same for the FP8 arm: Muon's step size is fixed)
    df, d8 = (wf[k] - w0[k]) / g, (w8[k] - w0[k]) / g
    n = g.numel()
    acc["n"] += n
    acc["r"] += r.sum().item()
    acc["pred_var"] += (r * L).sum().item()  # r * |delta| summed over the path, per weight
    acc["meas_sq"] += ((d8 - df) ** 2).sum().item()
    acc["net_f_sq"] += (df ** 2).sum().item()
    acc["net_8_sq"] += (d8 ** 2).sum().item()
    acc["dot"] += (df * d8).sum().item()
    acc["rtn_sq"] += (((g2.quantize_rtn(wf[k]) - wf[k]) / g) ** 2).sum().item()
n = acc["n"]
out = dict(
    fp8_gap_over_4bit_gap=acc["r"] / n,
    predicted_drift_rms=(acc["pred_var"] / n) ** 0.5,
    measured_fp8_vs_float_rms=(acc["meas_sq"] / n) ** 0.5,
    float_net_rms=(acc["net_f_sq"] / n) ** 0.5,
    fp8_net_rms=(acc["net_8_sq"] / n) ** 0.5,
    fp8_progress_along_float=acc["dot"] / acc["net_f_sq"],
    cos_fp8_float=acc["dot"] / (acc["net_f_sq"] * acc["net_8_sq"]) ** 0.5,
    float_rounded_to_4bit_rms=(acc["rtn_sq"] / n) ** 0.5,
)
print(json.dumps({a: round(v, 4) for a, v in out.items()}, indent=1))
json.dump(out, open(os.path.join(HERE, "data", "gpt2_drift.json"), "w"), indent=1)
