"""Milestone 6: the learning-rate pick, the pass lines written in PLAN.md before the runs, and the side comparisons.
All numbers come from one Colab T4 session (runs/m6/).

Run:  python report/m6_summary.py   ->  prints the answers, writes report/data/m6.json
"""
import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
M6 = os.path.join(os.path.dirname(HERE), "runs", "m6")
TUNE = (0.01, 0.02, 0.04, 0.08, 0.16)
ARMS = {  # key: (folder, file pattern), file pattern takes the seed
    "m1": ("base_m1", "cpu_muon_lr0.01_s{}"),
    "m2": ("base_m2", "cpu_c1_s{}"),
    "m3": ("base_m3", "cpu_c3a_s{}"),
    "m4post": ("base_m4post", "cpu_muon_post_s{}"),
    "float": ("main", "cpu_float_s{}"),
    "q4": ("main", "cpu_q4_s{}"),
    "q4_m8": ("main", "cpu_q4_m8_s{}"),
}


def load(folder, pattern, seed):
    stem = os.path.join(M6, folder, pattern.format(seed))
    if not os.path.exists(stem + ".json"):
        return None
    d = json.load(open(stem + ".json"))
    rows = [json.loads(l) for l in open(stem + ".jsonl") if '"val_ce"' in l]
    return dict(final=d["final_val_ce"], bytes=d.get("state_bytes_per_weight"), wall_s=d["wall_s"],
                curve=[[r["step"], r["val_ce"]] for r in rows if r["val_ce"] == r["val_ce"]])


def ms(v):
    return st.mean(v), (st.stdev(v) / len(v) ** 0.5 if len(v) > 1 else float("nan"))


def main():
    tune = {lr: load(f"tune_lr{lr:g}", "cpu_float_s{}", 1) for lr in TUNE}
    print("learning-rate pick, float arm, seed 1:", ", ".join(f"{lr:g} -> {r['final']:.4f}" if r else f"{lr:g} -> (none)"
                                                          for lr, r in tune.items()))
    runs = {a: {s: load(f, p, s) for s in (1, 2, 3)} for a, (f, p) in ARMS.items()}
    runs = {a: {s: r for s, r in v.items() if r} for a, v in runs.items()}
    for a, v in runs.items():
        if v:
            m, se = ms([r["final"] for r in v.values()])
            print(f"{a:7s} {m:.4f} ± {se:.4f}  seeds {', '.join(f'{r['final']:.4f}' for r in v.values())}")

    def check(arm, ref, margin):
        seeds = sorted(set(runs[arm]) & set(runs[ref]))
        diffs = [runs[arm][s]["final"] - runs[ref][s]["final"] for s in seeds]
        ok = sum(d <= margin for d in diffs)
        print(f"{arm} - {ref}: {', '.join(f'{d:+.4f}' for d in diffs)}  (mean {ms(diffs)[0]:+.4f} ± {ms(diffs)[1]:.4f});"
              f" within {margin}: {ok} of {len(seeds)} -> {'PASS' if ok == 3 else 'FAIL'}" if seeds else f"{arm}: no runs")
        return dict(diffs=diffs, passed=ok == 3) if seeds else None

    print("\npass lines (PLAN.md):")
    checks = {"q4": check("q4", "m2", 0.03), "q4_m8": check("q4_m8", "m2", 0.03), "float": check("float", "m1", 0.03)}
    print("\nalso reported:")
    extra = {k: check(a, b, 0.0) for k, (a, b) in
             {"q4_vs_m3": ("q4", "m3"), "q4_vs_m4post": ("q4", "m4post"), "q4_m8_vs_q4": ("q4_m8", "q4")}.items()}
    json.dump(dict(tune={f"{k:g}": v for k, v in tune.items()}, runs=runs, checks=checks, extra=extra),
              open(os.path.join(HERE, "data", "m6.json"), "w"))


if __name__ == "__main__":
    main()
