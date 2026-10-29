"""Side study L: answers the five questions written in PLAN.md before the long runs, from runs/long/.

Run:  python report/long_summary.py   ->  prints the answers, writes report/data/long_runs.json (curves for the report)
"""
import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
LONG = os.path.join(os.path.dirname(HERE), "runs", "long")
ARMS = {"adamw": "adamw_lr0.002", "muon": "muon_lr0.005", "m3": "m3_lr0.005", "eco_adam": "eco_adam_lr0.002"}
GAP_2K = {"adamw-muon": 0.29, "m3-muon": 0.136}  # the CPU-preset numbers the questions compare with


def curves(folder):
    out = {}
    for f in sorted(glob.glob(os.path.join(LONG, folder, "*_s[0-9].jsonl"))):
        seed = int(f[-7])
        rows = [json.loads(l) for l in open(f) if '"step"' in l]
        out[seed] = {r["step"]: r["val_ce"] for r in rows if r["val_ce"] == r["val_ce"]}  # drop NaN (a diverged run)
    return out


def se(v):
    return st.stdev(v) / len(v) ** 0.5 if len(v) > 1 else float("nan")


def main():
    data = {arm: curves(folder) for arm, folder in ARMS.items()}
    final = {arm: {s: c[max(c)] for s, c in d.items()} for arm, d in data.items()}
    for arm, f in final.items():
        v = list(f.values())
        print(f"{arm:9s} seeds {sorted(f)}  final {st.mean(v):.4f} ± {se(v):.4f}  ({', '.join(f'{x:.4f}' for x in v)})")

    def gap(a, b):
        d = [final[a][s] - final[b][s] for s in sorted(set(final[a]) & set(final[b]))]
        return st.mean(d), se(d), d
    g, e, d = gap("adamw", "muon")
    print(f"Q1 AdamW - Muon at 10k: {g:+.3f} ± {e:.3f} (per seed {', '.join(f'{x:+.3f}' for x in d)}); at 2k: +0.29")
    d = [final["m3"][s] - final["adamw"][s] for s in sorted(set(final["m3"]) & set(final["adamw"]))]
    print(f"Q2 m3 below AdamW in {sum(x < 0 for x in d)} of {len(d)} seeds (m3 - AdamW: {', '.join(f'{x:+.3f}' for x in d)})")
    g, e, d = gap("m3", "muon")
    print(f"Q3 m3 - Muon at 10k: {g:+.3f} ± {e:.3f}; at 2k: +0.136")
    print("Q4 first step where the 3-seed mean reaches:")
    mean_curve = {}
    for arm, d in data.items():
        steps = sorted(set.intersection(*(set(c) for c in d.values())))
        mean_curve[arm] = {k: st.mean(c[k] for c in d.values()) for k in steps}
        hits = {lvl: next((k for k in steps if mean_curve[arm][k] <= lvl), None) for lvl in (2.0, 1.8, 1.7)}
        print(f"   {arm:9s} " + "  ".join(f"{lvl}: {hits[lvl]}" for lvl in hits))
    g, e, d = gap("eco_adam", "adamw")
    print(f"Q5 ECO-Adam - AdamW at 10k: {g:+.3f} ± {e:.3f} (per seed {', '.join(f'{x:+.3f}' for x in d)}); "
          f"within 0.03: {all(abs(x) <= 0.03 for x in d)}")
    extra = {os.path.basename(p): curves(os.path.basename(p)) for p in glob.glob(os.path.join(LONG, "*_lr*"))}
    json.dump(dict(arms=ARMS, curves=data, mean=mean_curve, all_runs=extra),
              open(os.path.join(HERE, "data", "long_runs.json"), "w"))


if __name__ == "__main__":
    main()
