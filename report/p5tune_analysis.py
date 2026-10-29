"""P5 pilot, Muon re-tune: each Muon learning rate against AdamW (1e-6), question by question on GSM8K test.

Reads runs/paper/p5_float_{adamw,muon}_vllm_s1_pilot.json and runs/paper/P5tune/*.json, prints a table and writes
runs/paper/P5tune_summary.json. The 95% interval is 1.96 x the paired standard error over the 1,319 questions.
"""
import glob, json, math, os

R = "runs/paper"


def correct(run, step=None):
    ev = run["evals"][-1] if step is None else next(e for e in run["evals"] if e["step"] == step)
    return [int(c) for c in ev["correct"]], ev


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d); m = sum(d) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1))
    return 100 * m, 100 * 1.96 * sd / math.sqrt(n)


def last(run, key, k=10):
    v = [t[key] for t in run["train"][-k:] if t.get(key) is not None]
    return sum(v) / len(v) if v else None


adamw = json.load(open(f"{R}/p5_float_adamw_vllm_s1_pilot.json"))
runs = [("AdamW 1e-6", adamw), ("Muon 1.37e-5 (pilot)", json.load(open(f"{R}/p5_float_muon_vllm_s1_pilot.json")))]
for f in sorted(glob.glob(f"{R}/P5tune/*.json"), key=lambda f: json.load(open(f))["muon_lr"]):
    r = json.load(open(f))
    if r.get("steps_done") == r["steps"]:
        runs.append((f"Muon {r['muon_lr']:g}", r))

a_end, _ = correct(adamw)
start, _ = correct(adamw, 0)
rows = []
print(f"{'run':24s} {'acc end':>8s} {'vs start':>16s} {'vs AdamW':>16s} {'len':>6s} {'cut':>5s} {'reward':>7s}")
for name, r in runs:
    c, ev = correct(r)
    ds, hs = paired(c, start)
    da, ha = paired(c, a_end)
    row = dict(run=name, muon_lr=r.get("muon_lr") if r["opt"] == "muon" else None, acc_end=ev["acc"],
               vs_start=[round(ds, 2), round(hs, 2)], vs_adamw=[round(da, 2), round(ha, 2)],
               eval_mean_len=ev["mean_len"], eval_truncated=ev["truncated"],
               train_reward_last10=last(r, "reward"), train_len_last10=last(r, "mean_len"),
               update_rms_last10=last(r, "update_rms"), step_s=r.get("step_s"), wall_s=r.get("wall_s"))
    rows.append(row)
    print(f"{name:24s} {100*ev['acc']:7.1f}% {ds:+6.1f} ± {hs:4.1f}   {da:+6.1f} ± {ha:4.1f}   {ev['mean_len']:6.0f} "
          f"{100*ev['truncated']:4.1f}% {row['train_reward_last10']:7.3f}")
os.makedirs(f"{R}/P5tune", exist_ok=True)
json.dump(dict(study="P5 pilot, Muon re-tune", start_acc=adamw["evals"][0]["acc"], rows=rows,
               curves={name: [dict(step=t["step"], reward=t["reward"], mean_len=t["mean_len"]) for t in r["train"]]
                       for name, r in runs}),
          open(f"{R}/P5tune_summary.json", "w"), indent=1)
