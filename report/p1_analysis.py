"""P1 analysis (PAPER_PLAN.md): pass lines P1a-P1d and the measured-only numbers, from runs/paper/P1/*.json.

Writes runs/paper/P1_summary.json. Run: python report/p1_analysis.py
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(HERE, "runs", "paper", "P1")


def load(arm, seed, lr="0.001", tag=""):
    with open(os.path.join(D, f"{arm}_lr{lr}_s{seed}{tag}.json")) as f:
        return json.load(f)


def win(r, fp4=False):
    v = r.get("final_per_window_fp4" if fp4 else "final_per_window")
    return None if v is None else np.array(v)


def paired(a, b):
    """mean and standard error over the validation windows of a - b"""
    d = a - b
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def changed_share(arm, seed, lr="0.001", tag=""):
    rows = [json.loads(l) for l in open(os.path.join(D, f"{arm}_lr{lr}_s{seed}{tag}.jsonl"))]
    v = [r["changed4"] for r in rows if r.get("changed4") is not None]
    return float(np.mean(v)) if v else None


out = {"seeds": {}, "pass": {}, "measured": {}}
ours = {s: load("q4_t8_m8", s) for s in (1, 2, 3)}
copy = {s: load("q4_float_copy", s) for s in (1, 2, 3)}
bsr = {s: load("bf16_sr_m8", s) for s in (1, 2, 3)}
one = {a: load(a, 1) for a in ("q4_kahan_m8", "muon_float", "q4_sr_m8")}
# ECO-Muon: the first run (fp16 Newton-Schulz) overflowed in its first 10 steps, because the code casts the momentum to
# fp16 before normalising it and ECO's injected error makes the momentum huge. P1b uses the rerun with fp32
# Newton-Schulz; the first run is kept on record.
eco_fp16 = load("eco_muon_q4", 1)
one["eco_muon_q4"] = load("eco_muon_q4", 1, tag="_nsfp32")
one["q4_float_copy_adamw"] = load("q4_float_copy_adamw", 1, lr="0.0001")

# seed standard deviation: the largest among the 3-seed arms (bf16-SR counted both as scored in bf16 and in 4 bits)
three = {"ours": [ours[s]["final_val_ce"] for s in (1, 2, 3)],
         "float_copy": [copy[s]["final_val_ce"] for s in (1, 2, 3)],
         "bf16_sr_bf16": [bsr[s]["final_val_ce"] for s in (1, 2, 3)],
         "bf16_sr_4bit": [bsr[s]["final_val_ce_fp4"] for s in (1, 2, 3)]}
sd = {k: float(np.std(v, ddof=1)) for k, v in three.items()}
seed_sd = max(sd.values())
out["seeds"] = dict(values=three, sd=sd, seed_sd=seed_sd, one_seed_threshold=10 * seed_sd)

# P1a: ours <= float copy + 0.03 in each seed
p1a = []
for s in (1, 2, 3):
    m, se = paired(win(ours[s]), win(copy[s]))
    p1a.append(dict(seed=s, ours=ours[s]["final_val_ce"], copy=copy[s]["final_val_ce"], gap=m, se=se, ok=m <= 0.03))
out["pass"]["P1a"] = dict(rows=p1a, mean_gap=float(np.mean([r["gap"] for r in p1a])),
                          gap_se_over_seeds=float(np.std([r["gap"] for r in p1a], ddof=1) / np.sqrt(3)),
                          result=all(r["ok"] for r in p1a))

# P1b: ours at least 0.02 below naive and below ECO-Muon (seed 1), each gap above the 1-seed threshold
p1b = {}
for a in ("q4_sr_m8", "eco_muon_q4"):
    r = one[a]
    if r["diverged"]:
        p1b[a] = dict(diverged=True, last_step=r["final_step"], result="diverged (no final loss)")
        continue
    m, se = paired(win(r), win(ours[1]))
    p1b[a] = dict(loss=r["final_val_ce"], ours_below_by=m, se=se, ok=m >= 0.02 and m > 10 * seed_sd)
p1b["eco_muon_q4_first_run_fp16_ns"] = dict(diverged=eco_fp16["diverged"], note="overflowed in the first 10 steps")
out["pass"]["P1b"] = p1b

# P1c: ours below bf16-SR rounded to 4 bits, each seed
p1c = []
for s in (1, 2, 3):
    m, se = paired(win(bsr[s], fp4=True), win(ours[s]))
    p1c.append(dict(seed=s, bf16_sr_4bit=bsr[s]["final_val_ce_fp4"], ours=ours[s]["final_val_ce"], ours_below_by=m, se=se,
                    ok=m > 0))
out["pass"]["P1c"] = dict(rows=p1c, result=all(r["ok"] for r in p1c))

# P1d: ours within 0.01 of Kahan (seed 1)
m, se = paired(win(ours[1]), win(one["q4_kahan_m8"]))
out["pass"]["P1d"] = dict(ours=ours[1]["final_val_ce"], kahan=one["q4_kahan_m8"]["final_val_ce"], gap=m, se=se,
                          result=abs(m) <= 0.01)

# measured only (seed 1 unless noted)
meas = {}
for name, r, fp4 in (("float Muon", one["muon_float"], False), ("float Muon rounded to 4 bits", one["muon_float"], True),
                     ("4-bit float copy + AdamW", one["q4_float_copy_adamw"], False)):
    m, se = paired(win(ours[1]), win(r, fp4))
    meas[name] = dict(loss=r["final_val_ce_fp4" if fp4 else "final_val_ce"], ours_minus=m, se=se,
                      resolved=abs(m) > 10 * seed_sd)
for s in (1, 2, 3):
    m, se = paired(win(ours[s]), win(bsr[s]))
    meas[f"bf16-SR in bf16, seed {s}"] = dict(loss=bsr[s]["final_val_ce"], ours_minus=m, se=se)
meas["share of weights changing per step"] = {a: changed_share(a, 1) for a in ("q4_t8_m8", "q4_kahan_m8", "q4_sr_m8")}
meas["share of weights changing per step"]["eco_muon_q4"] = changed_share("eco_muon_q4", 1, tag="_nsfp32")
meas["start (pretrained GPT-2, rounded to 4 bits)"] = ours[1]["val_ce_start"]
out["measured"] = meas
out["bytes_per_weight"] = {"q4_t8_m8": 2.78, "q4_float_copy": 8, "bf16_sr_m8": 3.03, "q4_kahan_m8": 5.78,
                           "muon_float": 8, "q4_float_copy_adamw": 12, "q4_sr_m8": 1.78, "eco_muon_q4": 1.78}
out["curves"] = {os.path.basename(f)[:-5]: json.load(open(f))["evals"] for f in sorted(glob.glob(os.path.join(D, "*.json")))}

with open(os.path.join(HERE, "runs", "paper", "P1_summary.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: out[k] for k in ("seeds", "pass", "measured")}, indent=1))
