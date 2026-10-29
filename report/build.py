"""Inline report/data/report_data.json into report/template.html, giving report/report.html (the published page)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def rounded(x):
    if isinstance(x, float):
        return float(f"{x:.5g}")
    if isinstance(x, list):
        return [rounded(v) for v in x]
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    return x


data = rounded(json.load(open(os.path.join(HERE, "data", "report_data.json"))))
html = open(os.path.join(HERE, "template.html")).read().replace("__DATA__", json.dumps(data, separators=(",", ":")))
open(os.path.join(HERE, "report.html"), "w").write(html)
print(f"report.html: {len(html) / 1e3:.0f} kB")
