"""P2: GPT-2 6 x 512 (45M) from scratch, 0.9B tokens. Pass lines P2a and P2c, plus the tuning table.

Reads runs/paper/P2/*.json and runs/paper/P2tune/*.json, prints a table and writes runs/paper/P2_summary.json.
Paired differences are over the 256 validation windows (the same windows for every run); the 95% interval is
1.96 x the paired standard error.
"""
import glob, json, math, statistics

R = "runs/paper"


def load(p):
    return json.load(open(p))


def paired(x, y):
    d = [p - q for p, q in zip(x, y)]
    n = len(d); m = sum(d) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1))
    return m, 1.96 * sd / math.sqrt(n)


ours = {s: load(f"{R}/P2/q4_t8_m8_lr0.04_s{s}.json") for s in (1, 2, 3)}
copy = {s: load(f"{R}/P2/q4_float_copy_lr0.04_s{s}.json") for s in (1, 2, 3)}
sr = load(f"{R}/P2/bf16_sr_m8_lr0.04_s1.json")
tune = sorted((load(f) for f in glob.glob(f"{R}/P2tune/*.json")), key=lambda r: r["muon_lr"])

seeds = []
for s in (1, 2, 3):
    m, h = paired(ours[s]["final_per_window"], copy[s]["final_per_window"])
    seeds.append(dict(seed=s, ours=ours[s]["final_val_ce"], copy=copy[s]["final_val_ce"], diff=m, half=h, pass_=m <= 0.03,
                      carry_violations=ours[s]["carry_violations"], skipped_ours=ours[s]["skipped_steps"],
                      skipped_copy=copy[s]["skipped_steps"], wall_s=ours[s]["wall_s"]))
o = [ours[s]["final_val_ce"] for s in (1, 2, 3)]
c = [copy[s]["final_val_ce"] for s in (1, 2, 3)]
o_mean, o_sd = statistics.mean(o), statistics.stdev(o)
c_mean, c_sd = statistics.mean(c), statistics.stdev(c)
sr4 = sr["final_val_ce_fp4"]
res = dict(
    seeds=seeds,
    P2a=dict(pass_=all(x["pass_"] for x in seeds), worst=max(x["diff"] for x in seeds)),
    P2c=dict(ours_mean=o_mean, ours_sd=o_sd, sr_fp4=sr4, sr_float=sr["final_val_ce"], margin=sr4 - o_mean,
             need=2 * o_sd, pass_=sr4 - o_mean > 2 * o_sd),
    copy_mean=c_mean, copy_sd=c_sd, mean_diff=o_mean - c_mean,
    sr_vs_copy_float=sr["final_val_ce"] - c_mean,
    bytes=dict(ours=ours[1]["state_bytes_per_weight"], copy=copy[1]["state_bytes_per_weight"],
               sr=sr["state_bytes_per_weight"]),
    params=ours[1]["params"], steps=ours[1]["steps"], tokens=ours[1]["train_tokens"],
    tune=[dict(muon_lr=t["muon_lr"], adamw_lr=t["adamw_lr"], steps=t["steps"], val_ce=t["final_val_ce"],
               val_se=t["final_val_se"]) for t in tune],
    curves={name: [dict(step=e[0], val_ce=e[1]) for e in r["evals"]]
            for name, r in [("ours s1", ours[1]), ("copy s1", copy[1]), ("bf16-SR s1", sr)]},
)
json.dump(res, open(f"{R}/P2_summary.json", "w"), indent=1)

print(f"{'seed':>4s} {'ours':>8s} {'copy':>8s} {'ours-copy':>18s}  P2a")
for x in seeds:
    print(f"{x['seed']:4d} {x['ours']:8.4f} {x['copy']:8.4f} {x['diff']:+8.4f} ± {x['half']:.4f}  {'pass' if x['pass_'] else 'FAIL'}"
          f"  (carry overflows {x['carry_violations']}, skipped {x['skipped_ours']}/{x['skipped_copy']})")
print(f"P2a: {'PASS' if res['P2a']['pass_'] else 'FAIL'} (worst {res['P2a']['worst']:+.4f}, line +0.03)")
print(f"ours mean {o_mean:.4f} sd {o_sd:.4f}; copy mean {c_mean:.4f} sd {c_sd:.4f}")
print(f"P2c: bf16-SR float {sr['final_val_ce']:.4f}, rounded to 4 bits {sr4:.4f}; margin {sr4 - o_mean:.4f} vs need "
      f"{2 * o_sd:.4f}: {'PASS' if res['P2c']['pass_'] else 'FAIL'}")
print("tune:", [(t["muon_lr"], t["final_val_ce"]) for t in tune])
