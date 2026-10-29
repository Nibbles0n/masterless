"""Side study B: answers the two questions written in PLAN.md before the batch-size runs, from runs/batch/.

Run:  python report/batch_summary.py   ->  prints the answers, writes report/data/batch_runs.json
"""
import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
BATCH = os.path.join(os.path.dirname(HERE), "runs", "batch")


def runs(arm, k):
    out = {}
    for f in sorted(glob.glob(os.path.join(BATCH, f"{arm}_x{k}", "*_s[0-9].json"))):
        d = json.load(open(f))
        curve = [json.loads(l) for l in open(f[:-5] + ".jsonl") if '"step"' in l]
        out[d["seed"]] = dict(final=d["final_val_ce"], wall_s=d["wall_s"], steps=d["config"]["max_iters"],
                              curve=[[r["step"] * d["config"]["batch_size"] * d["config"]["block_size"], r["val_ce"]] for r in curve])
    return out


def ms(v):
    return st.mean(v), (st.stdev(v) / len(v) ** 0.5 if len(v) > 1 else float("nan"))


def main():
    data = {arm: {k: runs(arm, k) for k in (1, 2, 4, 8)} for arm in ("m1", "m3")}
    print("k  steps  m1 final        m3 final        m3 - m1 (seed-matched)   wall s (m1, m3)")
    for k in (1, 2, 4, 8):
        a, b = data["m1"][k], data["m3"][k]
        seeds = sorted(set(a) & set(b))
        if not seeds:
            continue
        g = [b[s]["final"] - a[s]["final"] for s in seeds]
        print(f"{k}  {a[seeds[0]]['steps']:5d}  {ms([a[s]['final'] for s in seeds])[0]:.4f} ± {ms([a[s]['final'] for s in seeds])[1]:.4f}"
              f"  {ms([b[s]['final'] for s in seeds])[0]:.4f} ± {ms([b[s]['final'] for s in seeds])[1]:.4f}"
              f"  {ms(g)[0]:+.4f} ± {ms(g)[1]:.4f} ({', '.join(f'{x:+.3f}' for x in g)})"
              f"  {st.mean(a[s]['wall_s'] for s in seeds):.0f}, {st.mean(b[s]['wall_s'] for s in seeds):.0f}")
    json.dump(data, open(os.path.join(HERE, "data", "batch_runs.json"), "w"))


if __name__ == "__main__":
    main()
