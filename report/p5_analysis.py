"""P5: pass lines P5a, P5b and P5d, plus the measured-only numbers, for one design.

usage: python report/p5_analysis.py short          (30 steps x 64 questions, runs/paper/P5short + the tune run as (a) s1)
       python report/p5_analysis.py main 4e-5      (200 steps x 32 questions, runs/paper/P5/*_lr4e-5.json)

Each of the 1,319 GSM8K test questions is scored 0/1. Seeds are pooled per question (the mean over seeds), the arms
are paired question by question, and the 95% interval is 1.96 x the paired standard error over the questions.
Writes runs/paper/P5<design>_summary.json.
"""
import json, math, os, sys

R = "runs/paper"
design = sys.argv[1]
if design == "short":
    files = {("a", 1): f"{R}/P5tune/p5_float_muon_vllm_s1_tune1e-4.json",
             ("a", 2): f"{R}/P5short/p5_float_muon_vllm_s2.json",
             ("c", 1): f"{R}/P5short/p5_ticker_muon_c8_vllm_s1.json",
             ("c", 2): f"{R}/P5short/p5_ticker_muon_c8_vllm_s2.json",
             ("b", 1): f"{R}/P5short/p5_lora4_adamw_vllm_s1.json"}
    out = f"{R}/P5short_summary.json"
else:
    lr = sys.argv[2]
    stem = {"a": "float_muon", "c": "ticker_muon_c8", "b": "lora4_adamw", "d": "copy4_muon"}  # (d): eighth entry
    files = {(a, s): f"{R}/P5/p5_{st}_vllm_s{s}_lr{lr}.json" for a, st in stem.items() for s in (1, 2, 3)}
    out = f"{R}/P5main_summary.json"
runs = {k: json.load(open(f)) for k, f in files.items() if os.path.exists(f)}
missing = [f for k, f in files.items() if k not in runs]


def correct(run, i):
    return [int(c) for c in run["evals"][i]["correct"]]


def pooled(arm, i=-1, seeds=(1, 2, 3)):
    rs = [correct(r, i) for (a, s), r in sorted(runs.items()) if a == arm and s in seeds]
    return [sum(q) / len(rs) for q in zip(*rs)] if rs else None


def paired(x, y):
    d = [p - q for p, q in zip(x, y)]
    n = len(d); m = sum(d) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1))
    return 100 * m, 100 * 1.96 * sd / math.sqrt(n)


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


res = dict(design=design, files={f"{a}{s}": f for (a, s), f in files.items()}, missing=missing, runs=[])
for (a, s), r in sorted(runs.items()):
    t = r["train"]
    res["runs"].append(dict(arm=a, seed=s, name=f"{a} s{s}", acc_start=r["evals"][0]["acc"], acc_end=r["evals"][-1]["acc"],
                            eval_len_end=r["evals"][-1]["mean_len"], eval_trunc_end=r["evals"][-1]["truncated"],
                            peak_mem_gb=r.get("peak_mem_gb"), state_bytes_per_weight=r.get("state_bytes_per_weight"),
                            trainable_params=r.get("trainable_params"), step_s=r.get("step_s"), wall_s=r.get("wall_s"),
                            prob_gap_mean=mean([x.get("prob_gap_mean") for x in t]),
                            gen_minus_train_logp=mean([x.get("gen_minus_train_logp") for x in t]),
                            logp_gap_max=max([x.get("logp_gap_max") or 0 for x in t]),
                            lora_rounded={k: v for k, v in (r.get("lora_rounded") or {}).items() if k != "correct"} or None,
                            changed=mean([x.get("changed") for x in t]),
                            reward=[x["reward"] for x in t], train_len=[x["mean_len"] for x in t]))
A, B, C, D = pooled("a"), pooled("b"), pooled("c"), pooled("d")
c_start = pooled("c", 0)
lines = {}
if A and C:
    m, h = paired(C, A)
    lines["P5a"] = dict(diff=m, half=h, lo=m - h, hi=m + h, pass_=m - h > -4, third_seeds=(m - h) <= -4 <= (m + h))
if A and C and any(s == 3 for a, s in runs if a in "ac"):  # the seeds 1-2 reading, for reference
    m, h = paired(pooled("c", seeds=(1, 2)), pooled("a", seeds=(1, 2)))
    lines["P5a_seeds12"] = dict(diff=m, half=h, lo=m - h, hi=m + h)
if C:
    m, h = paired(C, c_start)
    lines["P5b"] = dict(gain=m, half=h, lo=m - h, pass_=m >= 5 and m - h > 0)
if A:
    lines["a_gain"] = dict(zip(("diff", "half"), paired(A, pooled("a", 0))))
    lines["start_gap_4bit"] = dict(zip(("diff", "half"), paired(c_start, pooled("a", 0))))
if B and C:
    lines["c_minus_b"] = dict(zip(("diff", "half"), paired(C, B)))
if B:
    lines["b_gain"] = dict(zip(("diff", "half"), paired(B, pooled("b", 0))))
# LoRA merged into the 4-bit weights and rounded again (runs that have it), pooled over those seeds
rb = [(s, r) for (a, s), r in sorted(runs.items()) if a == "b" and r.get("lora_rounded")]
if rb:
    BR = [sum(q) / len(rb) for q in zip(*[[int(c) for c in r["lora_rounded"]["correct"]] for _, r in rb])]
    seeds = tuple(s for s, _ in rb)
    lines["b_rounded_seeds"] = list(seeds)
    lines["b_rounded_acc"] = 100 * sum(BR) / len(BR)
    lines["b_rounded_minus_b"] = dict(zip(("diff", "half"), paired(BR, pooled("b", seeds=seeds))))
    lines["b_rounded_minus_start"] = dict(zip(("diff", "half"), paired(BR, pooled("b", 0, seeds=seeds))))
    if C:
        lines["c_minus_b_rounded"] = dict(zip(("diff", "half"), paired(C, BR)))
if C and D:  # P5d (eighth change-log entry): the ticker against the fp32 master copy, DeepSeek-V4 style
    m, h = paired(C, D)
    lines["P5d"] = dict(diff=m, half=h, lo=m - h, hi=m + h, pass_=m - h > -3)
if D:
    lines["d_gain"] = dict(zip(("diff", "half"), paired(D, pooled("d", 0))))
    if A:
        lines["d_minus_a"] = dict(zip(("diff", "half"), paired(D, A)))
res["lines"] = lines
json.dump(res, open(out, "w"), indent=1)

print(f"design {design}; missing: {missing or 'none'}")
for r in res["runs"]:
    print(f"  {r['name']:5s} start {100*r['acc_start']:5.1f}%  end {100*r['acc_end']:5.1f}%  mem {r['peak_mem_gb']} GB  "
          f"len {r['eval_len_end']:.0f}  cut {100*r['eval_trunc_end']:.1f}%  prob gap {r['prob_gap_mean']}")
for k, v in lines.items():
    if not isinstance(v, dict):
        print(f"  {k}: {v}")
        continue
    print(f"  {k}: " + ", ".join(f"{a}={b:+.2f}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items()))
