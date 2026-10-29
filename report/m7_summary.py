"""Milestone 7 and side study T (the ticker): the pass lines and questions written in PLAN.md before the runs, and the
side comparisons. Milestone 7's runs are in runs/m7/, the ticker's in runs/ticker/; the baselines (milestones 1-3 and
milestone 4's muon_post) are milestone 6's T4 runs in runs/m6/.

Run:  python report/m7_summary.py   ->  prints the answers, writes report/data/m7.json
"""
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(HERE), "runs")
ARMS = {  # key: (folder, file pattern taking the seed, which final number)
    "m1": ("m6/base_m1", "cpu_muon_lr0.01_s{}", "val_ce"),
    "m2": ("m6/base_m2", "cpu_c1_s{}", "val_ce"),
    "m3": ("m6/base_m3", "cpu_c3a_s{}", "val_ce"),
    "m4post": ("m6/base_m4post", "cpu_muon_post_s{}", "val_ce"),
    "q4_b4_m8": ("m7", "cpu_q4_b4_m8_s{}", "val_ce"),
    "fp8_m8": ("m7", "cpu_fp8_m8_s{}", "val_ce"),
    "fp8_fp4": ("m7", "cpu_fp8_m8_s{}", "val_ce_fp4"),  # fp8_m8's weights rounded to 4 bits after training
    "q4_t4_m8": ("ticker", "cpu_q4_t4_m8_s{}", "val_ce"),
}
BYTES_FP8_SCALES = {"q4_b4_m8": 0.5 + 1 / 16 + 0.5 + 1 / 16 + 1 + 4 / 128, "fp8_m8": 1 + 1 / 16 + 1 + 4 / 128,
                    "q4_t4_m8": 0.5 + 1 / 16 + 0.5 + 1 + 4 / 128}


def load(folder, pattern, key, seed):
    stem = os.path.join(RUNS, folder, pattern.format(seed))
    if not os.path.exists(stem + ".json"):
        return None
    d = json.load(open(stem + ".json"))
    rows = [json.loads(l) for l in open(stem + ".jsonl") if f'"{key}"' in l]
    return dict(final=d["final_" + key], bytes=d.get("state_bytes_per_weight"), wall_s=d["wall_s"],
                curve=[[r["step"], r[key]] for r in rows if r[key] == r[key]])


def ms(v):
    return st.mean(v), (st.stdev(v) / len(v) ** 0.5 if len(v) > 1 else float("nan"))


def main():
    runs = {a: {s: load(f, p, k, s) for s in (1, 2, 3)} for a, (f, p, k) in ARMS.items()}
    runs = {a: {s: r for s, r in v.items() if r} for a, v in runs.items()}
    for a, v in runs.items():
        if v:
            m, se = ms([r["final"] for r in v.values()])
            print(f"{a:9s} {m:.4f} ± {se:.4f}  seeds {', '.join(f'{r['final']:.4f}' for r in v.values())}")

    def check(arm, ref, margin):
        seeds = sorted(set(runs[arm]) & set(runs[ref]))
        diffs = [runs[arm][s]["final"] - runs[ref][s]["final"] for s in seeds]
        ok = sum(d <= margin for d in diffs)
        print(f"{arm} - {ref}: {', '.join(f'{d:+.4f}' for d in diffs)}  (mean {ms(diffs)[0]:+.4f} ± {ms(diffs)[1]:.4f});"
              f" within {margin}: {ok} of {len(seeds)} -> {'PASS' if ok == 3 else 'FAIL'}" if seeds else f"{arm}: no runs")
        return dict(diffs=diffs, passed=ok == 3) if seeds else None

    print("\npass lines (PLAN.md):")
    checks = {"q4_b4_m8": check("q4_b4_m8", "m2", 0.03), "fp8_m8": check("fp8_m8", "m1", 0.03)}
    print("\nalso reported (the margin 0 is only for the printout):")
    extra = {f"{a}_vs_{b}": check(a, b, 0.0) for a, b in
             (("fp8_fp4", "q4_b4_m8"), ("fp8_fp4", "fp8_m8"), ("q4_b4_m8", "m4post"), ("q4_b4_m8", "m3"),
              ("fp8_fp4", "m2"), ("fp8_m8", "m2"),
              ("q4_t4_m8", "q4_b4_m8"), ("q4_t4_m8", "fp8_fp4"), ("q4_t4_m8", "m2"), ("q4_t4_m8", "m4post"))}
    print("\nbytes per weight (fp32 scales; FP8 scales):",
          {a: (round(runs[a][1]["bytes"], 3), round(BYTES_FP8_SCALES[a], 3)) for a in BYTES_FP8_SCALES if runs[a]})
    json.dump(dict(runs=runs, checks=checks, extra=extra, bytes_fp8_scales=BYTES_FP8_SCALES),
              open(os.path.join(HERE, "data", "m7.json"), "w"))


if __name__ == "__main__":
    main()
