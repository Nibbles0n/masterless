"""Build report/results.html (GPTQ, P1e, P2 and P5 results) from results_body.html, results_charts.js, the earlier report's styles
and chart helpers (template.html), and the summaries.

Run:  python report/gptq_analysis.py && python report/p1_analysis.py && python report/p2_analysis.py && python report/p5_analysis.py short && python report/p5_analysis.py main 4e-5 \
      && python report/build_results.py
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "runs", "paper")


def rounded(x):
    if isinstance(x, float):
        return float(f"{x:.6g}")
    if isinstance(x, list):
        return [rounded(v) for v in x]
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    return x


t = open(os.path.join(HERE, "template.html")).read()
css = t[t.index('<meta name="claude:viewport-fit-sides"'):t.index("</style>")]
js = t[t.index('(() => {\n"use strict";'):t.index("/* ---------- data helpers ---------- */")]
js = re.sub(r"const ARM = \{.*?\n\};\n", "", js, flags=re.S)
js = re.sub(r"const MODE2ARM = .*?\n", "", js)
js = re.sub(r"const fmt = .*?\n", "", js)
js = re.sub(r"const sgn = .*?\n", "", js)

p1 = json.load(open(os.path.join(R, "P1_summary.json")))
P1E_CURVES = {"ours": "q4_t8_m8_lr0.001_s1", "copy_adamw": "q4_float_copy_adamw_lr0.0001_s1",
              "copy_muon": "q4_float_copy_lr0.001_s1", "eco_adamw": "eco_adamw_q4_lr0.0001_s1",
              "eco_muon": "eco_muon_q4_lr0.001_s1_nsfp32"}
p1e = dict(rows=p1["pass"]["P1e"]["rows"], curves={k: [e[:2] for e in p1["curves"][v]] for k, v in P1E_CURVES.items()},
           eco_vs_copy=p1["measured"]["ECO-AdamW minus 4-bit float copy + AdamW (3 seeds)"],
           ours_vs_copy=p1["measured"]["ours minus 4-bit float copy + AdamW (3 seeds)"])
lr1e4 = json.load(open(os.path.join(R, "P5", "p5_float_muon_vllm_s1_lr1e-4.json")))
data = rounded(dict(p5main=json.load(open(os.path.join(R, "P5main_summary.json"))),
                    p5short=json.load(open(os.path.join(R, "P5short_summary.json"))),
                    p2=json.load(open(os.path.join(R, "P2_summary.json"))),
                    lr1e4=dict(reward=[x["reward"] for x in lr1e4["train"]], acc_end=lr1e4["evals"][-1]["acc"]),
                    p1e=p1e, gptq=json.load(open(os.path.join(R, "GPTQ_summary.json")))))
body = open(os.path.join(HERE, "results_body.html")).read()
charts = open(os.path.join(HERE, "results_charts.js")).read()
css += ".note { background: var(--wash); padding: 10px 14px; border-radius: 6px; }\n"
css += ".verdict { display: grid; grid-template-columns: repeat(auto-fit, minmax(12rem, 1fr)); gap: 10px; margin-block: 1.2rem; }\n"
css += ".verdict div { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; min-width: 0; }\n"
css += ".verdict b { display: block; font-family: var(--f-display); font-size: 0.95rem; }\n"
css += ".verdict span { color: var(--ink-2); font-size: 0.9rem; }\n"
css += ".two { display: grid; grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr)); gap: 10px; margin-block: 1rem; }\n"
css += ".two > div { min-width: 0; }\n.two h3 { margin: 0.4rem 0; }\n"
css += ".tbl { overflow-x: auto; }\n.pass { color: var(--c-c); font-weight: 600; }\n.fail { color: var(--c-b); font-weight: 600; }\n"
html = ("<title>P2 and P5 Results</title>\n" + css + "</style>\n" + body
        + "\n<script>\nconst DATA = " + json.dumps(data, separators=(",", ":")) + ";\n</script>\n<script>\n"
        + js + charts + "\n})();\n</script>\n")
open(os.path.join(HERE, "results.html"), "w").write(html)
print(f"results.html: {len(html) / 1e3:.0f} kB")
