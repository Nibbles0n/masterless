"""GPTQ round (PAPER_PLAN.md, sixth and seventh change-log entries): GPTQ against round to nearest and against ours,
from runs/paper/GPTQ/*.json; and the GPTQ start of P5 (GSM8K accuracy). Writes runs/paper/GPTQ_summary.json. Run: python report/gptq_analysis.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "runs", "paper")


def paired(a, b):
    """mean and standard error over the validation windows of a - b"""
    d = np.array(a) - np.array(b)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


out = {}
ours = {"P1": json.load(open(os.path.join(R, "P1", "q4_t8_m8_lr0.001_s1.json"))),
        "P2": json.load(open(os.path.join(R, "P2", "q4_t8_m8_lr0.04_s1.json")))}
for name, study in (("pretrained", None), ("p1_muon_float_s1", "P1"), ("p1_bf16_sr_s1", "P1"),
                    ("p2_bf16_sr_s1", "P2")):
    path = os.path.join(R, "GPTQ", name + ".json")
    if not os.path.exists(path):
        continue
    g = json.load(open(path))
    if study is None:  # the pretrained model's logged losses: its float start, and ours' step-0 eval (rounded)
        g["logged_float"] = ours["P1"]["val_ce_start"]
        g["logged_rtn"] = ours["P1"]["evals"][0][1]
    row = dict(float=g["float_ce"], rtn=g["rtn_ce"], gptq=g["gptq_ce"], on_grid=g["on_grid"],
               logged_float=g["logged_float"], logged_rtn=g["logged_rtn"])
    # the reproduction check written in the plan: round to nearest (and the float model) within 0.0005 of the log
    row["reproduced"] = all(abs(a - b) <= 5e-4 for a, b in ((g["float_ce"], g["logged_float"]),
                                                             (g["rtn_ce"], g["logged_rtn"])) if b is not None)
    m, se = paired(g["rtn_per_window"], g["gptq_per_window"])
    row["gptq_below_rtn"] = dict(mean=m, se=se)
    row["share_of_rounding_loss_recovered"] = (g["rtn_ce"] - g["gptq_ce"]) / (g["rtn_ce"] - g["float_ce"])
    if study:
        m, se = paired(g["gptq_per_window"], ours[study]["final_per_window"])
        row["gptq_minus_ours"] = dict(mean=m, se=se, ours=ours[study]["final_val_ce"],
                                      gptq_above_ours_by_2se=m > 2 * se)
    out[name] = row

q = os.path.join(R, "GPTQ", "qwen_start.json")  # P5's starting model (seventh change-log entry): accuracy, no RL
if os.path.exists(q):
    g = json.load(open(q))
    out["qwen_start"] = dict(float=g["float"]["acc"], rtn=g["rtn"]["acc"], gptq=g["gptq"]["acc"], on_grid=g["on_grid"],
                             reproduced=g["reproduced"], gptq_minus_rtn=g["gptq_minus_rtn"],
                             float_minus_gptq=g["float_minus_gptq"],
                             share_of_rounding_loss_recovered=g["share_of_rounding_loss_recovered"])

if any("gptq_minus_ours" in r for r in out.values()):
    rows = [r for r in out.values() if "gptq_minus_ours" in r]
    out["claim_stands_against_gptq"] = all(r["gptq_minus_ours"]["gptq_above_ours_by_2se"] for r in rows) \
        if len(rows) == 3 else None
with open(os.path.join(R, "GPTQ_summary.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
