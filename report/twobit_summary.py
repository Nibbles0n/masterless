"""Side study 2B (the ticker at 2 bits): the pass line and questions written in PLAN.md before the runs. The new runs
are in runs/twobit/; the references are milestone 6's T4 runs of milestones 1 and 2 (runs/m6/), milestone 0's AdamW
(runs/m0/, CPU fp32), milestone 7's FP8 arm (runs/m7/) and the 4-bit ticker (runs/ticker/).

Run:  python report/twobit_summary.py   ->  prints the answers, writes report/data/twobit.json
"""
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(HERE), "runs")
ARMS = {  # key: (folder, file pattern taking the seed, which final number)
    "m0": ("m0", "cpu_adamw_s{}", "val_ce"),
    "m1": ("m6/base_m1", "cpu_muon_lr0.01_s{}", "val_ce"),
    "m2": ("m6/base_m2", "cpu_c1_s{}", "val_ce"),
    "m7_fp8": ("m7", "cpu_fp8_m8_s{}", "val_ce"),
    "q4_t4_m8": ("ticker", "cpu_q4_t4_m8_s{}", "val_ce"),
    "q2_t4_m8": ("twobit", "cpu_q2_t4_m8_s{}", "val_ce"),
    "q2_t8_m8": ("twobit", "cpu_q2_t8_m8_s{}", "val_ce"),
    "q2_ste_muon": ("twobit", "cpu_q2_ste_muon_s{}", "val_ce"),
    "q2_ste_adam": ("twobit", "cpu_q2_ste_adam_s{}", "val_ce"),
    "fp8_m8": ("twobit", "cpu_fp8_m8_s{}", "val_ce"),
    "fp8_q2": ("twobit", "cpu_fp8_m8_s{}", "val_ce_q2"),  # fp8_m8's weights rounded to 2 bits after training
}
BYTES_FP8_SCALES = {"q2_t4_m8": 0.25 + 1 / 16 + 0.5 + 1 + 4 / 128, "q2_t8_m8": 0.25 + 1 / 16 + 1 + 1 + 4 / 128,
                    "fp8_m8": 1 + 1 / 16 + 1 + 4 / 128}


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
            print(f"{a:12s} {m:.4f} ± {se:.4f}  seeds {', '.join(f'{r['final']:.4f}' for r in v.values())}")

    def check(arm, ref, margin):
        seeds = sorted(set(runs[arm]) & set(runs[ref]))
        diffs = [runs[arm][s]["final"] - runs[ref][s]["final"] for s in seeds]
        ok = sum(d <= margin for d in diffs)
        print(f"{arm} - {ref}: {', '.join(f'{d:+.4f}' for d in diffs)}  (mean {ms(diffs)[0]:+.4f} ± {ms(diffs)[1]:.4f});"
              f" within {margin}: {ok} of {len(seeds)} -> {'PASS' if ok == 3 else 'FAIL'}" if seeds else f"{arm}: no runs")
        return dict(diffs=diffs, passed=ok == 3) if seeds else None

    print("\npass line (PLAN.md): within 0.05 of q2_ste_muon in each seed")
    checks = {a: check(a, "q2_ste_muon", 0.05) for a in ("q2_t4_m8", "q2_t8_m8")}
    print("\nmeasure only (the margin 0 is only for the printout):")
    extra = {f"{a}_vs_{b}": check(a, b, 0.0) for a, b in
             (("q2_t8_m8", "q2_t4_m8"), ("q2_ste_muon", "m2"), ("q2_ste_adam", "m2"), ("q2_ste_adam", "q2_ste_muon"),
              ("q2_ste_adam", "m0"), ("fp8_q2", "q2_t4_m8"), ("fp8_q2", "q2_t8_m8"), ("fp8_q2", "q2_ste_muon"),
              ("q2_t4_m8", "q4_t4_m8"), ("q2_t8_m8", "q4_t4_m8"), ("q2_t4_m8", "m2"), ("q2_t8_m8", "m2"))}
    print("\ncode check, fp8_m8 against milestone 7 (within 0.002 in each seed):")
    extra["fp8_m8_vs_m7_fp8"] = check("fp8_m8", "m7_fp8", 0.002)
    print("\nbytes per weight (fp32 scales; FP8 scales):",
          {a: (round(runs[a][1]["bytes"], 3), round(BYTES_FP8_SCALES[a], 3)) for a in BYTES_FP8_SCALES if runs[a]})
    print("wall seconds per run:", {a: round(st.mean(r["wall_s"] for r in v.values())) for a, v in runs.items() if v and a.startswith(("q2", "fp8"))})
    json.dump(dict(runs=runs, checks=checks, extra=extra, bytes_fp8_scales=BYTES_FP8_SCALES),
              open(os.path.join(HERE, "data", "twobit.json"), "w"))


if __name__ == "__main__":
    main()
