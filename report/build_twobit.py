"""Build report/twobit.html (the published page for side study 2B) from twobit_body.html, twobit_charts.js, the
earlier report's styles and chart helpers (template.html), and report/data/twobit.json.

Run:  python report/twobit_summary.py && python report/build_twobit.py
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def rounded(x):
    if isinstance(x, float):
        return float(f"{x:.5g}")
    if isinstance(x, list):
        return [rounded(v) for v in x]
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    return x


t = open(os.path.join(HERE, "template.html")).read()
css = t[t.index('<meta name="claude:viewport-fit-sides"'):t.index("</style>")]
css = css.replace("--r1: #86b6ef; --r2: #3987e5; --r3: #184f95;", "--r1: #a9cbf3; --r2: #5f9fe8; --r3: #2a6fc4; --r4: #143f78;")
css = css.replace("--r1: #184f95; --r2: #3987e5; --r3: #9ec5f4;", "--r1: #1d4f8f; --r2: #3277cc; --r3: #6ea6ea; --r4: #b5d3f7;")
js = t[t.index('(() => {\n"use strict";'):t.index("/* ---------- data helpers ---------- */")]
js = re.sub(r"const ARM = \{.*?\n\};\n", "", js, flags=re.S)
js = re.sub(r"const MODE2ARM = .*?\n", "", js)
js = re.sub(r"const fmt = .*?\n", "", js)
js = re.sub(r"const sgn = .*?\n", "", js)
js = js.replace('"stroke-linecap": "round", opacity: s.opacity || 1 }', '"stroke-linecap": "round", opacity: s.opacity || 1, "stroke-dasharray": s.dash || "none" }')
assert "stroke-dasharray" in js
data = rounded(dict(tb=json.load(open(os.path.join(HERE, "data", "twobit.json")))))
body = open(os.path.join(HERE, "twobit_body.html")).read()
charts = open(os.path.join(HERE, "twobit_charts.js")).read()
css += ".note { background: var(--wash); padding: 10px 14px; border-radius: 6px; }\n"
html = ("<title>The Ticker at Two Bits</title>\n" + css + "</style>\n" + body
        + "\n<script>\nconst DATA = " + json.dumps(data, separators=(",", ":")) + ";\n</script>\n<script>\n"
        + js + charts + "\n})();\n</script>\n")
open(os.path.join(HERE, "twobit.html"), "w").write(html)
print(f"twobit.html: {len(html) / 1e3:.0f} kB")
