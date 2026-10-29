"""Bundle every number the report shows into report/data/report_data.json (read by build.py)."""
import glob
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ARMS = {  # key: (glob of committed runs, label)
    "m0": ("runs/m0/cpu_adamw_s{}.json", "m0 AdamW, full precision"),
    "m1": ("runs/m1/cpu_muon_lr0.01_s{}.json", "m1 Muon, full precision"),
    "m2": ("runs/m2/cpu_c1_s{}.json", "m2 4-bit + float copy"),
    "m3": ("runs/m3/cpu_c3a_s{}.json", "m3 4-bit only, random rounding"),
    "a": ("runs/m4/cpu_eco_adam_s{}.json", "m4a ECO with Adam"),
    "b": ("runs/m4/cpu_muon_pre_s{}.json", "m4b Muon, error before NS"),
    "c": ("runs/m4/cpu_muon_post_s{}.json", "m4c Muon, error after NS"),
}


def curve(path):
    rows = [json.loads(l) for l in open(path) if '"val_ce"' in l]
    return [[r["step"], r["val_ce"]] for r in rows]


def mean_se(xs):
    m = sum(xs) / len(xs)
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


out = dict(arms={}, sweep={}, survival={}, diag={}, ns=None)
for key, (pat, label) in ARMS.items():
    seeds = {}
    for s in (1, 2, 3):
        f = os.path.join(ROOT, pat.format(s))
        d = json.load(open(f))
        seeds[s] = dict(final=d["final_val_ce"], window_se=d["final_val_se"], curve=curve(f[:-5] + ".jsonl"))
        if "survival" in d:
            out["survival"][s] = d["survival"]
        bytes_ = d.get("state_bytes_per_weight", 12)
    m, se = mean_se([seeds[s]["final"] for s in seeds])
    out["arms"][key] = dict(label=label, seeds=seeds, mean=m, se=se, bytes=bytes_)

for lr in ("0.01", "0.02", "0.04"):
    f = os.path.join(ROOT, f"runs/m1/cpu_muon_lr{lr}_s1.jsonl")
    out["sweep"][lr] = curve(f)

for mode in ("m1", "m2", "m3", "eco_adam", "muon_pre", "muon_post"):
    f = os.path.join(HERE, "data", f"diag_{mode}_s1.json")
    if not os.path.exists(f):
        continue
    d = json.load(open(f))
    tl = d["train_loss"]
    smooth = [[i + 50, sum(tl[i:i + 100]) / 100] for i in range(0, len(tl) - 99, 50)]  # 100-step means
    out["diag"][mode] = dict(probes=d["probes"], windows=d["windows"], evals=d["evals"], train_loss=smooth)

f = os.path.join(HERE, "data", "ns_analysis.json")
if os.path.exists(f):
    out["ns"] = json.load(open(f))

# one real block of 16 weights for the grid diagram: blocks.1.fc at initialisation, seed 1, in grid units
import importlib.util
import torch
spec = importlib.util.spec_from_file_location("m4", os.path.join(ROOT, "m4_eco.py"))
m4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m4)
cfg = m4.Config(**m4.PRESETS["cpu"], vocab_size=65)
torch.manual_seed(1)
w = dict(m4.GPT(cfg).named_parameters())["blocks.1.fc.weight"].detach()[0, :16]
scale = (w.abs().max() / 6).item()
out["block"] = dict(scale=scale, x=(w / scale).tolist())

# follow-ups: milestone 5 (error spreading), side study F (FP8 scales), offline spreading check, side study L
follow = {}
for key, pat in (("m5", "runs/m5/cpu_c3s_s{}.json"), ("fp8", "runs/ablate_fp8/cpu_c3a_fp8_s{}.json")):
    seeds = {}
    for s in (1, 2, 3):
        f = os.path.join(ROOT, pat.format(s))
        seeds[s] = dict(final=json.load(open(f))["final_val_ce"], curve=curve(f[:-5] + ".jsonl"))
    m, se = mean_se([seeds[s]["final"] for s in seeds])
    follow[key] = dict(seeds=seeds, mean=m, se=se)
probe = json.load(open(os.path.join(HERE, "data", "spread_probe.json")))
follow["probe"] = {}
for step in sorted({r["step"] for r in probe}):
    rows = [r for r in probe if r["step"] == step]
    follow["probe"][step] = {mode: sorted(r[mode] / r["sr"] for r in rows)[len(rows) // 2 - 1: len(rows) // 2 + 1]
                             for mode in ("next", "full", "block16", "lowrank8")}
    follow["probe"][step] = {k: sum(v) / 2 for k, v in follow["probe"][step].items()}  # median of 16 matrices
    follow["probe"][step]["sr_noise"] = sorted(r["sr"] for r in rows)[7:9]
f = os.path.join(HERE, "data", "long_runs.json")
if os.path.exists(f):
    follow["long"] = json.load(open(f))
out["follow"] = follow

json.dump(out, open(os.path.join(HERE, "data", "report_data.json"), "w"))
print({k: (v["mean"], v["se"]) for k, v in out["arms"].items()}, "diag:", list(out["diag"]))
