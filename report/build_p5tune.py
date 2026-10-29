"""Build report/p5tune.html (the P5 RL task and Muon re-tune page) from p5tune_body.html, p5tune_charts.js, the earlier report's styles and chart
helpers (template.html) and runs/paper/P5tune_summary.json.

Run:  python report/p5tune_analysis.py && python report/build_p5tune.py
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

data = rounded(dict(tune=json.load(open(os.path.join(HERE, "..", "runs", "paper", "P5tune_summary.json")))))
body = open(os.path.join(HERE, "p5tune_body.html")).read()
charts = open(os.path.join(HERE, "p5tune_charts.js")).read()
css += ".note { background: var(--wash); padding: 10px 14px; border-radius: 6px; }\n"
css += ".verdict { display: grid; grid-template-columns: repeat(auto-fit, minmax(12rem, 1fr)); gap: 10px; margin-block: 1.2rem; }\n"
css += ".verdict div { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; min-width: 0; }\n"
css += ".verdict b { display: block; font-family: var(--f-display); font-size: 0.95rem; }\n"
css += ".verdict span { color: var(--ink-2); font-size: 0.9rem; }\n"
css += ".arms, .two { display: grid; grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr)); gap: 10px; margin-block: 1rem; }\n"
css += ".arms div { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; min-width: 0; }\n"
css += ".arms b { display: block; font-family: var(--f-display); }\n.arms span { color: var(--ink-2); font-size: 0.92rem; }\n.two > div { min-width: 0; }\n.two h3 { margin: 0.4rem 0; }\nblockquote.note { margin-block: 1rem; }\n"
html = ("<title>P5 RL Pilot: Muon Re-tune</title>\n" + css + "</style>\n" + body
        + "\n<script>\nconst DATA = " + json.dumps(data, separators=(",", ":")) + ";\n</script>\n<script>\n"
        + js + charts + "\n})();\n</script>\n")
open(os.path.join(HERE, "p5tune.html"), "w").write(html)
print(f"p5tune.html: {len(html) / 1e3:.0f} kB")
