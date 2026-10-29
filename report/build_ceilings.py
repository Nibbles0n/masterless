"""Build report/ceilings.html (rotated storage and aimed rounding error on GPT-2) from ceilings_body.html,
ceilings_charts.js, the earlier report's styles and chart helpers (template.html), and report/data/gpt2_ceilings.json.

Run:  python report/gpt2_ceilings.py && python report/build_ceilings.py
"""
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = {"c_attn": 3, "attn_c_proj": 1, "c_fc": 4, "mlp_c_proj": 4}  # matrix sizes in units of 768 x 768


def rounded(x):
    if isinstance(x, float):
        return float(f"{x:.5g}")
    if isinstance(x, list):
        return [rounded(v) for v in x]
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    return x


def keep_out(r, frac):
    """Damage cut from removing the error along the strongest frac of input directions; the same error energy is then
    spread over the remaining (1 - frac) of directions."""
    return (1 - frac) / (1 - r[f"top{round(frac * 100)}pct_share"])


c = json.load(open(os.path.join(HERE, "data", "gpt2_ceilings.json")))
rows = c["inputs"]["rows"]
wsum = sum(SIZE[r["kind"]] for r in rows)
geo = lambda f: math.exp(sum(SIZE[r["kind"]] * math.log(f(r)) for r in rows) / wsum)
gains = dict(ceiling=geo(lambda r: r["am_over_gm"]), seq=geo(lambda r: r["am_over_seq"]),
             keep1=geo(lambda r: keep_out(r, 0.01)), keep10=geo(lambda r: keep_out(r, 0.1)))
print({k: round(v, 2) for k, v in gains.items()})
for r in rows:
    r.pop("spectrum")

t = open(os.path.join(HERE, "template.html")).read()
css = t[t.index('<meta name="claude:viewport-fit-sides"'):t.index("</style>")]
js = t[t.index('(() => {\n"use strict";'):t.index("/* ---------- data helpers ---------- */")]
js = re.sub(r"const ARM = \{.*?\n\};\n", "", js, flags=re.S)
js = re.sub(r"const MODE2ARM = .*?\n", "", js)
js = re.sub(r"const fmt = .*?\n", "", js)
js = re.sub(r"const sgn = .*?\n", "", js)
data = rounded(dict(c=c, gains=gains))
body = open(os.path.join(HERE, "ceilings_body.html")).read()
charts = open(os.path.join(HERE, "ceilings_charts.js")).read()
css += ".note { background: var(--wash); padding: 10px 14px; border-radius: 6px; }\n"
html = ("<title>Aiming the Rounding Noise</title>\n" + css + "</style>\n" + body
        + "\n<script>\nconst DATA = " + json.dumps(data, separators=(",", ":")) + ";\n</script>\n<script>\n"
        + js + charts + "\n})();\n</script>\n")
open(os.path.join(HERE, "ceilings.html"), "w").write(html)
print(f"ceilings.html: {len(html) / 1e3:.0f} kB")
