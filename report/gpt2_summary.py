"""Side study G: read runs/gpt2/<stage>/*.json, check the pass lines fixed in PLAN.md, print a table, and write
report/data/gpt2.json for the report. Differences between arms carry a paired standard error over the 256 fixed
validation windows (every arm is scored on the same windows), which is the only error bar available with one seed.

Run:  python report/gpt2_summary.py
"""
import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(HERE), "runs", "gpt2")
ARMS = ["adamw_float", "muon_float", "q4_float_copy", "fp8_m8", "q4_t4_m8"]
REF = {"fp8_m8": "muon_float", "q4_t4_m8": "q4_float_copy"}  # line 3's reference for each 2-byte arm


def load(stage):
    out = {}
    for f in glob.glob(os.path.join(RUNS, stage, "*.json")):
        d = json.load(open(f))
        rows = [json.loads(l) for l in open(f + "l") if l.startswith("{")]
        logs = [r for r in rows if "train_loss" in r]
        d["changed"] = [[r["step"], r["changed"]] for r in logs if r.get("changed") is not None]
        for c in ("changed4", "changed8"):
            d[c] = [[r["step"], r[c]] for r in logs if r.get(c) is not None]
        d["tok_per_s"] = sorted(r["tok_per_s"] for r in logs)[len(logs) // 2]  # median (the first rows include compiling)
        out[d["arm"]] = d
    return out


def paired(a, b):
    """Mean and standard error of (a - b) over the shared validation windows."""
    d = [x - y for x, y in zip(a["final_per_window"], b["final_per_window"])]
    m = sum(d) / len(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
    return m, sd / math.sqrt(len(d))


def curves(d, step_every=10):
    """Validation curve (step, loss) and a smoothed training curve (mean of each log interval)."""
    tl = d["train_loss"]
    train = [[i + step_every, sum(tl[i:i + step_every]) / len(tl[i:i + step_every])] for i in range(0, len(tl), step_every)]
    return dict(val=[[s, round(c, 5)] for s, c in d["evals"]], train=[[s, round(v, 4)] for s, v in train])


def stage_summary(stage, runs, base=None):
    rows, checks = {}, []
    for arm, d in runs.items():
        rows[arm] = dict(final=d["final_val_ce"], se=d["final_val_se"], step0=d["val_ce_step0"],
                         pretrained=d["val_ce_pretrained"], rise=d["max_val_rise"], diverged=d["diverged"],
                         skipped=d["skipped_steps"], bytes=d["state_bytes_per_weight"], wall_s=d["wall_s"],
                         tokens=d["train_tokens"], fp4=d.get("final_val_ce_fp4"),
                         max_grad=max(g for g in d["grad_norm"] if g is not None), tok_per_s=d["tok_per_s"],
                         changed=d["changed"], grad=[[i + 1, g] for i, g in enumerate(d["grad_norm"])][::5], **curves(d))
    mf = runs.get("muon_float")
    for arm, ref in REF.items():
        if arm not in runs:
            continue
        d = runs[arm]
        if mf:
            ok = (not d["diverged"]) and d["max_val_rise"] <= mf["max_val_rise"] + 0.05
            checks.append(dict(arm=arm, line=1, what="stable: finite, largest rise above own best <= muon_float's + 0.05",
                               value=f"rise {d['max_val_rise']:+.4f} vs muon_float {mf['max_val_rise']:+.4f}", ok=ok))
        checks.append(dict(arm=arm, line=2, what="improves: final < own step 0",
                           value=f"{d['final_val_ce']:.4f} vs {d['val_ce_step0']:.4f}",
                           ok=d["final_val_ce"] < d["val_ce_step0"]))
        if ref in runs:
            m, se = paired(d, runs[ref])
            checks.append(dict(arm=arm, line=3, what=f"close: <= {ref} + 0.03", value=f"{m:+.4f} ± {se:.4f}",
                               diff=m, diff_se=se, ok=m <= 0.03))
            if base and arm in base and ref in base:
                m1, _ = paired(base[arm], base[ref])
                checks.append(dict(arm=arm, line=4, what="gap to reference grows by at most 0.02 from stage 1",
                                   value=f"{m:+.4f} now vs {m1:+.4f} in stage 1 ({m - m1:+.4f})", ok=m - m1 <= 0.02))
    extras = []
    for a, b in [("q4_t4_m8", "muon_float"), ("fp8_m8", "q4_float_copy"), ("q4_t4_m8", "fp8_m8"),
                 ("muon_float", "adamw_float"), ("q4_float_copy", "muon_float")]:
        if a in runs and b in runs:
            m, se = paired(runs[a], runs[b])
            extras.append(dict(a=a, b=b, diff=round(m, 5), se=round(se, 5)))
    return dict(rows=rows, checks=checks, extras=extras)


def pw_diff(a, b):
    """Mean and standard error of a - b, both lists of per-window losses on the same windows."""
    d = [x - y for x, y in zip(a, b)]
    m = sum(d) / len(d)
    return m, math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1)) / math.sqrt(len(d))


def stage3(s3, s2):
    """Rounding after training against training in the format: the stage-3 re-runs of muon_float and fp8_m8, scored
    rounded to FP8 and to 4 bits, against the stage-2 runs trained in each format (same 256 windows)."""
    if not s3:
        return None
    mf, f8 = s3.get("muon_float"), s3.get("fp8_m8")
    fin = lambda d: d["rounded"][-1]
    pts = {}  # (trained as, stored as) -> per-window losses
    if mf:
        pts[("muon_float", "float")] = mf["final_per_window"]
        pts[("muon_float", "fp8")] = fin(mf)["fp8_per_window"]
        pts[("muon_float", "fp4")] = fin(mf)["fp4_per_window"]
    if f8:
        pts[("fp8_m8", "fp8")] = f8["final_per_window"]
        pts[("fp8_m8", "fp4")] = fin(f8)["fp4_per_window"]
    for arm in ("q4_float_copy", "q4_t4_m8"):
        if arm in s2:
            pts[(arm, "fp4")] = s2[arm]["final_per_window"]
    cells = [dict(trained=a, stored=b, loss=round(sum(v) / len(v), 5)) for (a, b), v in pts.items()]
    comps = []
    for (a, b), (c, d), what in [((("fp8_m8", "fp8")), ("muon_float", "fp8"), "FP8: trained in FP8 - float Muon rounded to FP8"),
                                 (("muon_float", "fp8"), ("muon_float", "float"), "float Muon: rounded to FP8 - float"),
                                 (("q4_t4_m8", "fp4"), ("muon_float", "fp4"), "4-bit: ticker - float Muon rounded to 4 bits"),
                                 (("q4_float_copy", "fp4"), ("muon_float", "fp4"), "4-bit: float copy - float Muon rounded to 4 bits"),
                                 (("fp8_m8", "fp4"), ("muon_float", "fp4"), "4-bit: FP8 rounded - float Muon rounded"),
                                 (("q4_t4_m8", "fp4"), ("fp8_m8", "fp4"), "4-bit: ticker - FP8 rounded to 4 bits"),
                                 (("muon_float", "fp4"), ("muon_float", "float"), "float Muon: rounded to 4 bits - float")]:
        if (a, b) in pts and (c, d) in pts:
            m, se = pw_diff(pts[(a, b)], pts[(c, d)])
            comps.append(dict(what=what, diff=round(m, 5), se=round(se, 5)))
    curves = {}
    for arm, d in s3.items():
        curves[arm] = dict(val=[[st, c] for st, c in d["evals"]],
                           **{r: [[x["step"], x[r]] for x in d["rounded"] if r in x] for r in ("fp8", "fp4")},
                           changed=d["changed"], changed4=d["changed4"], changed8=d["changed8"])
    check = {arm: dict(s3=s3[arm]["final_val_ce"], s2=s2[arm]["final_val_ce"]) for arm in s3 if arm in s2}
    rows = {a: dict(tokens=d["train_tokens"], tok_per_s=d["tok_per_s"], wall_s=d["wall_s"]) for a, d in s3.items()}
    return dict(cells=cells, comps=comps, curves=curves, check=check, rows=rows)


def probe():
    """report/ticker_probe.py's printout (runs/gpt2/ticker_probe.txt) as rows."""
    f = os.path.join(RUNS, "ticker_probe.txt")
    if not os.path.exists(f):
        return None
    rows = []
    for line in open(f):
        parts = line.split()
        if parts and parts[0][0].isdigit() and "%" in line:
            lr = float(parts[0])
            pct = [float(x.rstrip("%")) for x in parts if x.endswith("%")]
            name = " ".join(x for x in parts[1:] if not x.endswith("%"))
            rows.append(dict(lr=lr, variant=name, changed=pct[0], undone=pct[1], parked=pct[2] if len(pct) > 2 else None))
    return rows


def main():
    out = {}
    s1, s2, s3 = load("s1"), load("s2"), load("s3")
    if s1:
        out["s1"] = stage_summary("s1", s1)
    if s2:
        out["s2"] = stage_summary("s2", s2, base=s1)
    for stage, res in out.items():
        print(f"== stage {stage}")
        print(f"{'arm':14s} {'bytes':>5s} {'step0':>7s} {'final':>7s} {'±':>6s} {'rise':>7s} {'fp4':>7s} {'skip':>4s} {'max|g|':>6s} {'min':>5s}")
        for arm in ARMS:
            r = res["rows"].get(arm)
            if r:
                print(f"{arm:14s} {r['bytes']:5.2f} {r['step0']:7.4f} {r['final']:7.4f} {r['se']:6.4f} {r['rise']:+7.4f} "
                      f"{r['fp4'] if r['fp4'] is not None else float('nan'):7.4f} {r['skipped']:4d} {r['max_grad']:6.2f} {r['wall_s'] / 60:5.1f}")
        print(f"pretrained GPT-2 (float, before rounding): {next(iter(res['rows'].values()))['pretrained']:.4f}")
        for c in res["checks"]:
            print(f"  {c['arm']:9s} line {c['line']}: {'PASS' if c['ok'] else 'FAIL'}  {c['what']}: {c['value']}")
        for e in res["extras"]:
            print(f"  {e['a']} - {e['b']}: {e['diff']:+.4f} ± {e['se']:.4f}")
    q = stage3(s3, s2)
    if q:
        out["s3"] = q
        print("== stage s3 (rounding after training; final losses)")
        for c in q["cells"]:
            print(f"  trained {c['trained']:14s} stored {c['stored']:6s} {c['loss']:.4f}")
        for c in q["comps"]:
            print(f"  {c['what']:52s} {c['diff']:+.4f} ± {c['se']:.4f}")
        for arm, c in q["check"].items():
            print(f"  code check {arm}: stage 3 {c['s3']:.4f} vs stage 2 {c['s2']:.4f}")
    pr = probe()
    if pr:
        out["probe"] = pr
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "data", "gpt2.json"), "w"))


if __name__ == "__main__":
    main()
