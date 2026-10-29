"""Side study B2: answers the four questions written in PLAN.md before the runs, from runs/batch (old learning
rates, study B) and runs/batch_lr (learning rates times m).

Run:  python report/batch_lr_summary.py   ->  prints the answers, writes report/data/batch_lr.json
"""
import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(HERE), "runs")
KS = (1, 2, 4, 8)
MULTS = {1: (1, 2, 4), 2: (1, 2, 4, 8), 4: (1, 2, 4, 8), 8: (1, 2, 4, 8)}


def cell(arm, k, m):
    """seed -> final CE (None when the run blew up) for one setting."""
    folder = os.path.join(RUNS, "batch", f"{arm}_x{k}") if m == 1 else os.path.join(RUNS, "batch_lr", f"{arm}_x{k}_lr{m}")
    out = {}
    for jl in sorted(glob.glob(os.path.join(folder, "*_s[0-9].jsonl"))):
        seed = int(jl[-7])
        rows = [json.loads(l) for l in open(jl) if '"val_ce"' in l]
        final = json.load(open(jl[:-1]))["final_val_ce"] if os.path.exists(jl[:-1]) else None
        if final is not None and not (final == final and final <= 3):
            final = None
        out[seed] = dict(final=final, curve=[[r["step"] * 12 * k * 64, r["val_ce"]] for r in rows if r["val_ce"] == r["val_ce"]])
    return out


def ms(v):
    return st.mean(v), (st.stdev(v) / len(v) ** 0.5 if len(v) > 1 else float("nan"))


def main():
    data = {arm: {k: {m: cell(arm, k, m) for m in MULTS[k]} for k in KS} for arm in ("m1", "m3")}
    best = {}
    print("final val CE, mean ± standard error over seeds (x = runs that blew up)")
    for arm in ("m1", "m3"):
        for k in KS:
            line, cands = [], []
            for m in MULTS[k]:
                c = data[arm][k][m]
                ok = [r["final"] for r in c.values() if r["final"] is not None]
                bad = len(c) - len(ok)
                if len(ok) == 3:
                    mu, se = ms(ok)
                    cands.append((mu, m))
                    line.append(f"x{m}: {mu:.4f} ± {se:.4f}")
                else:
                    line.append(f"x{m}: {len(ok)} ok, {bad} blew up" if c else f"x{m}: no runs")
            best[(arm, k)] = min(cands) if cands else None
            b = best[(arm, k)]
            print(f"{arm} batch {k}x  " + "  ".join(line) + (f"   best rate x{b[1]}: {b[0]:.4f}" if b else ""))
    print("\nQ1/Q2: each arm at its own best rate per batch size; gap seed-matched")
    for k in KS:
        b1, b3 = best[("m1", k)], best[("m3", k)]
        if b1 and b3:
            g = [data["m3"][k][b3[1]][s]["final"] - data["m1"][k][b1[1]][s]["final"] for s in (1, 2, 3)]
            print(f"batch {k}x  m1 x{b1[1]} {b1[0]:.4f}   m3 x{b3[1]} {b3[0]:.4f}   gap {ms(g)[0]:+.4f} ± {ms(g)[1]:.4f}")
    print("\nQ3: m3 - m1 gap at a fixed batch size, by rate multiplier")
    for k in KS:
        parts = []
        for m in MULTS[k]:
            a, b = data["m1"][k][m], data["m3"][k][m]
            s = [x for x in (1, 2, 3) if x in a and x in b and a[x]["final"] is not None and b[x]["final"] is not None]
            if len(s) == 3:
                g = [b[x]["final"] - a[x]["final"] for x in s]
                parts.append(f"x{m}: {ms(g)[0]:+.4f} ± {ms(g)[1]:.4f}")
        print(f"batch {k}x  " + "  ".join(parts))
    print("\nreproduction check, seed 1 at 1x and the old rates (study B vs new session):")
    for arm in ("m1", "m3"):
        f = glob.glob(os.path.join(RUNS, "batch_lr", f"{arm}_x1_lr1", "*_s1.json"))
        if f:
            print(f"  {arm}: B {data[arm][1][1][1]['final']:.4f}  new {json.load(open(f[0]))['final_val_ce']:.4f}")
    json.dump({arm: {str(k): {str(m): {str(s): r for s, r in c.items()} for m, c in v.items()} for k, v in d.items()}
               for arm, d in data.items()}, open(os.path.join(HERE, "data", "batch_lr.json"), "w"))


if __name__ == "__main__":
    main()
