"""Build report/p1.html (the P1 results page) from p1_body.html, p1_charts.js, the earlier report's styles and chart
helpers (template.html) and runs/paper/P1_summary.json.

Run:  python report/p1_analysis.py && python report/build_p1.py
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


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

s = json.load(open(os.path.join(HERE, "..", "runs", "paper", "P1_summary.json")))
m = s["measured"]
p1 = {"seeds": s["seeds"], "pass": s["pass"], "curves": {k: [[e[0], e[1]] for e in v] for k, v in s["curves"].items()},
      "start": m["start (pretrained GPT-2, rounded to 4 bits)"],
      "measured_loss": dict(kahan=s["pass"]["P1d"]["kahan"], adamw_copy=m["4-bit float copy + AdamW"]["loss"],
                             muon_float_4bit=m["float Muon rounded to 4 bits"]["loss"],
                             naive=s["pass"]["P1b"]["q4_sr_m8"]["loss"],
                             eco=s["pass"]["P1b"]["eco_muon_q4"]["loss"])}
data = rounded(dict(p1=p1))
body = open(os.path.join(HERE, "p1_body.html")).read()
charts = open(os.path.join(HERE, "p1_charts.js")).read()
css += ".note { background: var(--wash); padding: 10px 14px; border-radius: 6px; }\n"
css += ".verdict { display: grid; grid-template-columns: repeat(auto-fit, minmax(12rem, 1fr)); gap: 10px; margin: 1.2rem 0; }\n"
css += ".verdict div { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; min-width: 0; }\n"
css += ".verdict b { display: block; font-family: var(--f-display); font-size: 0.95rem; }\n"
css += ".verdict span { color: var(--ink-2); font-size: 0.9rem; }\n"
html = ("<title>P1: 4-bit GPT-2 Results</title>\n" + css + "</style>\n" + body
        + "\n<script>\nconst DATA = " + json.dumps(data, separators=(",", ":")) + ";\n</script>\n<script>\n"
        + js + charts + "\n})();\n</script>\n")
open(os.path.join(HERE, "p1.html"), "w").write(html)
print(f"p1.html: {len(html) / 1e3:.0f} kB")
