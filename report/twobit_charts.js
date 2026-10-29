const ARM = {
  m0: { name: "m0 AdamW, float weights (CPU)", short: "m0 AdamW float", color: "--c-m0" },
  m1: { name: "m1 Muon, float weights", short: "m1 float", color: "--c-m1" },
  m2: { name: "m2 4-bit + float copy", short: "m2 4-bit + copy", color: "--c-m2" },
  q4_t4_m8: { name: "4-bit ticker (side study T)", short: "4-bit ticker", color: "--c-m3" },
  fp8_m8: { name: "FP8 + 8-bit momentum", short: "FP8", color: "--c-b" },
  fp8_q2: { name: "FP8, rounded to 2-bit after", short: "FP8 → 2-bit", color: "--c-m5", dash: "6 4" },
  q2_ste_muon: { name: "2-bit + float copy, Muon", short: "2-bit + copy, Muon", color: "--c-m2", dash: "6 4" },
  q2_ste_adam: { name: "2-bit + float copy, AdamW", short: "2-bit + copy, AdamW", color: "--c-m0", dash: "6 4" },
  q2_t4_m8: { name: "2-bit ticker, 4-bit carry", short: "2-bit, 4-bit carry", color: "--c-a" },
  q2_t8_m8: { name: "2-bit ticker, FP8 carry", short: "2-bit, FP8 carry", color: "--c-c" },
};
const NEW = k => k.startsWith("q2") || k.startsWith("fp8");
const fmt = (v, d = 3) => (v == null || !isFinite(v)) ? "–" : (+v).toFixed(d);
const sgn = (v, d = 3) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);
const R = DATA.tb.runs;
const seedsOf = k => Object.keys(R[k] || {}).sort();
const has = k => seedsOf(k).length > 0;
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const se = xs => Math.sqrt(xs.reduce((t, x) => t + (x - mean(xs)) ** 2, 0) / (xs.length - 1) / xs.length);
const finals = k => seedsOf(k).map(s => R[k][s].final);
const diffs = (a, b) => [1, 2, 3].filter(s => R[a] && R[a][s] && R[b] && R[b][s]).map(s => R[a][s].final - R[b][s].final);
function meanCurve(k) {
  const ss = seedsOf(k).map(s => R[k][s].curve), n = Math.min(...ss.map(c => c.length));
  return ss[0].slice(0, n).map((p, i) => [p[0], mean(ss.map(c => c[i][1]))]);
}
function hover(host, svg, W, H, hitAttrs, rowsFn, title, cx, cy) {
  const tip = host._tip || (host._tip = tipBox(host));
  const hit = el("svg:" + (hitAttrs.width ? "rect" : "circle"), Object.assign({ fill: "transparent", tabindex: "0" }, hitAttrs), svg);
  const on = () => { tipRows(tip, title, rowsFn()); const r = svg.getBoundingClientRect(), hr = host.getBoundingClientRect(); placeTip(host, tip, r.left - hr.left + cx * r.width / W, r.top - hr.top + cy * r.height / H); };
  hit.addEventListener("pointerenter", on); hit.addEventListener("focus", on);
  hit.addEventListener("pointerleave", () => tip.hidden = true); hit.addEventListener("blur", () => tip.hidden = true);
}

/* ---------- memory: what each byte holds ---------- */
(function () {
  const host = document.getElementById("ch-bytes");
  const PART = [
    { k: "w", label: "weights", color: "--c-m2" },
    { k: "s", label: "block scales", color: "--c-m1" },
    { k: "b", label: "carry", color: "--c-c" },
    { k: "m", label: "optimizer state", color: "--c-m3" },
  ];
  const m8 = 1 + 4 / 128;
  const rows = [
    { label: "2-bit + float copy, AdamW", parts: { w: 4, m: 8 } },
    { label: "m2 / 2-bit + float copy, Muon", parts: { w: 4, m: 4 } },
    { label: "4-bit ticker (side study T)", parts: { w: 0.5, s: 0.25, b: 0.5, m: m8 } },
    { label: "FP8 + 8-bit momentum", parts: { w: 1, s: 0.25, m: m8 } },
    { label: "2-bit ticker, FP8 carry", parts: { w: 0.25, s: 0.25, b: 1, m: m8 }, strong: true },
    { label: "2-bit ticker, 4-bit carry", parts: { w: 0.25, s: 0.25, b: 0.5, m: m8 }, strong: true },
  ];
  const draw = () => {
    header(host, { title: "Training memory per weight of the four big matrices", sub: "bytes; scales stored in 32 bits (8-bit scales would save 0.19)" });
    legend(host, PART.map(p => ({ label: p.label, color: tok(p.color) })), true);
    const W = Math.max(280, host.clientWidth - 24), narrow = W < 560, rh = 26, gap = narrow ? 30 : 14, m = { l: narrow ? 10 : 250, r: 46, t: 6, b: 30 };
    const H = m.t + rows.length * (rh + gap) + m.b;
    const svg = el("svg:svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "bytes per weight by part" }, host);
    const X = scale(0, 12, m.l, W - m.r), g = el("svg:g", {}, svg);
    for (const v of [0, 2, 4, 6, 8, 10, 12]) { el("svg:line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: tok(v ? "--grid" : "--axis") }, g); el("svg:text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, g, v); }
    rows.forEach((r, i) => {
      const yTop = m.t + i * (rh + gap) + (narrow ? 18 : 0);
      el("svg:text", { x: narrow ? m.l : m.l - 12, y: narrow ? yTop - 6 : yTop + rh / 2 + 4, "text-anchor": narrow ? "start" : "end", class: r.strong ? "lab-strong" : "lab" }, g, r.label);
      let x = 0;
      for (const p of PART) {
        const v = r.parts[p.k]; if (!v) continue;
        const x0 = X(x), x1 = X(x + v);
        el("svg:rect", { x: x0, y: yTop, width: Math.max(1, x1 - x0 - 2), height: rh, rx: 3, fill: tok(p.color) }, g);
        hover(host, svg, W, H, { x: x0, y: yTop, width: Math.max(4, x1 - x0), height: rh }, () => [{ value: v.toFixed(3) + " bytes", label: p.label, color: tok(p.color) }], r.label, (x0 + x1) / 2, yTop);
        x += v;
      }
      el("svg:text", { x: X(x) + 6, y: yTop + rh / 2 + 4, class: "lab-strong" }, g, x.toFixed(2));
    });
    numbers(host, ["setup", ...PART.map(p => p.label), "total"], rows.map(r => [r.label, ...PART.map(p => r.parts[p.k] ? r.parts[p.k].toFixed(3) : "–"), Object.values(r.parts).reduce((a, b) => a + b, 0).toFixed(2)]));
  };
  register(host, draw);
})();

/* ---------- final loss ---------- */
(function () {
  const host = document.getElementById("ch-final");
  const keys = ["m1", "m2", "q4_t4_m8", "fp8_m8", "q2_ste_muon", "q2_t8_m8", "q2_t4_m8", "m0", "q2_ste_adam", "fp8_q2"].filter(has);
  const draw = () => {
    header(host, { title: "Final validation loss", sub: "mean of seeds 1–3 (big dot) with ± one standard error; small dots are single seeds; lower is better" });
    const vals = keys.flatMap(finals), lo = Math.min(...vals) - 0.02, hi = Math.max(...vals) + 0.02;
    const W = Math.max(280, host.clientWidth - 24), narrow = W < 560, rh = 34, m = { l: narrow ? 10 : 260, r: 70, t: 6, b: 34 };
    const H = m.t + keys.length * (rh + (narrow ? 16 : 0)) + m.b;
    const svg = el("svg:svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "final loss by arm" }, host);
    const X = scale(lo, hi, m.l, W - m.r), g = el("svg:g", {}, svg);
    for (const v of niceTicks(lo, hi, narrow ? 4 : 6)) { el("svg:line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: tok("--grid") }, g); el("svg:text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, g, v.toFixed(2)); }
    keys.forEach((k, i) => {
      const yTop = m.t + i * (rh + (narrow ? 16 : 0)), y = yTop + (narrow ? 16 : 0) + rh / 2, f = finals(k), mu = mean(f), e = se(f), col = tok(ARM[k].color);
      el("svg:text", { x: narrow ? m.l : m.l - 12, y: narrow ? yTop + 12 : y + 4, "text-anchor": narrow ? "start" : "end", class: NEW(k) ? "lab-strong" : "lab" }, g, ARM[k].name);
      el("svg:line", { x1: X(mu - e), x2: X(mu + e), y1: y, y2: y, stroke: col, "stroke-width": 3, "stroke-linecap": "round" }, g);
      for (const v of f) el("svg:circle", { cx: X(v), cy: y + 9, r: 2.5, fill: col, opacity: 0.6 }, g);
      el("svg:circle", { cx: X(mu), cy: y, r: 6, fill: ARM[k].dash ? tok("--surface") : col, stroke: col, "stroke-width": 2.5 }, g);
      el("svg:text", { x: X(hi) + 6, y: y + 4, class: "lab" }, g, mu.toFixed(3));
      hover(host, svg, W, H, { cx: X(mu), cy: y, r: 14 }, () => [{ value: fmt(mu, 4) + " ± " + fmt(e, 4), label: "mean ± SE", color: col }, ...f.map((v, s) => ({ value: fmt(v, 4), label: "seed " + (s + 1) }))], ARM[k].name, X(mu), y);
    });
    numbers(host, ["arm", "mean", "SE", "seed 1", "seed 2", "seed 3"], keys.map(k => [ARM[k].name, fmt(mean(finals(k)), 4), fmt(se(finals(k)), 4), ...finals(k).map(v => fmt(v, 4))]));
  };
  register(host, draw);
})();

/* ---------- training curves ---------- */
lineChart(document.getElementById("ch-curves"), {
  title: "Validation loss during training", sub: "mean of seeds 1–3; dashed lines keep a float copy (or, for FP8 → 2-bit, are rounded at each check)",
  series: () => ["m1", "m2", "q4_t4_m8", "fp8_m8", "q2_ste_muon", "q2_ste_adam", "q2_t4_m8", "q2_t8_m8", "fp8_q2"].filter(has).map(k => ({ label: ARM[k].short, color: ARM[k].color, dash: ARM[k].dash, strong: NEW(k), width: NEW(k) ? 2.5 : 1.5, points: meanCurve(k).filter(p => p[0] > 0) })),
  x: { domain: [250, 2000], ticks: [250, 500, 1000, 1500, 2000], label: "training step" }, y: { tip: v => v.toFixed(4), tipAscending: true },
});

/* ---------- seed-by-seed differences ---------- */
function diffChart(id, title, sub, rowsAll) {
  const host = document.getElementById(id);
  const rows = rowsAll.filter(r => r.v.length);
  const draw = () => {
    header(host, { title, sub });
    const vals = rows.flatMap(r => r.v.concat(r.line == null ? [] : [r.line])), lo = Math.min(-0.05, ...vals) - 0.02, hi = Math.max(0.08, ...vals) + 0.02;
    const W = Math.max(280, host.clientWidth - 24), narrow = W < 560, rh = 40, m = { l: narrow ? 10 : 250, r: 16, t: 8, b: 34 };
    const H = m.t + rows.length * (rh + (narrow ? 16 : 0)) + m.b;
    const svg = el("svg:svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "per-seed differences" }, host);
    const X = scale(lo, hi, m.l, W - m.r), g = el("svg:g", {}, svg);
    for (const v of niceTicks(lo, hi, narrow ? 4 : 6)) { el("svg:line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: tok(Math.abs(v) < 1e-9 ? "--axis" : "--grid") }, g); el("svg:text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, g, Math.abs(v) < 1e-9 ? "0" : sgn(v, 2)); }
    rows.forEach((r, i) => {
      const yTop = m.t + i * (rh + (narrow ? 16 : 0)), y = yTop + (narrow ? 16 : 0) + rh / 2;
      el("svg:text", { x: narrow ? m.l : m.l - 12, y: narrow ? yTop + 12 : y + 4, "text-anchor": narrow ? "start" : "end", class: r.line == null ? "lab" : "lab-strong" }, g, r.label);
      if (r.line != null) { el("svg:line", { x1: X(r.line), x2: X(r.line), y1: y - 13, y2: y + 13, stroke: tok("--ink"), "stroke-width": 1.5 }, g); el("svg:text", { x: X(r.line) + 4, y: y - 15 }, g, r.lineLabel || "pass line"); }
      const col = tok(ARM[r.k].color);
      el("svg:line", { x1: X(Math.min(...r.v)), x2: X(Math.max(...r.v)), y1: y, y2: y, stroke: col, "stroke-width": 2, opacity: 0.5 }, g);
      r.v.forEach((v, s) => {
        el("svg:circle", { cx: X(v), cy: y, r: 5, fill: col, stroke: tok("--surface"), "stroke-width": 2 }, g);
        hover(host, svg, W, H, { cx: X(v), cy: y, r: 12 }, () => [{ value: sgn(v, 4), label: "seed " + (s + 1), color: col }], r.label, X(v), y);
      });
    });
    numbers(host, ["comparison", "seed 1", "seed 2", "seed 3", "mean ± SE", "line"], rows.map(r => [r.label, ...r.v.map(v => sgn(v, 4)), sgn(mean(r.v), 4) + " ± " + fmt(se(r.v), 4), r.line == null ? "measure only" : "≤ " + sgn(r.line, 2)]));
  };
  register(host, draw);
}
diffChart("ch-diffs", "Seed-by-seed differences in final loss", "each dot is one seed (same starting weights, same data order); the top two rows are the pass line: all three dots must sit left of the black tick", [
  { label: "2-bit, 4-bit carry − 2-bit + copy", k: "q2_t4_m8", v: diffs("q2_t4_m8", "q2_ste_muon"), line: 0.05 },
  { label: "2-bit, FP8 carry − 2-bit + copy", k: "q2_t8_m8", v: diffs("q2_t8_m8", "q2_ste_muon"), line: 0.05 },
  { label: "FP8 carry − 4-bit carry", k: "q2_t8_m8", v: diffs("q2_t8_m8", "q2_t4_m8"), line: null },
  { label: "FP8 → 2-bit − 2-bit, 4-bit carry", k: "fp8_q2", v: diffs("fp8_q2", "q2_t4_m8"), line: null },
  { label: "FP8 → 2-bit − 2-bit, FP8 carry", k: "fp8_q2", v: diffs("fp8_q2", "q2_t8_m8"), line: null },
  { label: "2-bit + copy, Muon − m2 (4-bit)", k: "q2_ste_muon", v: diffs("q2_ste_muon", "m2"), line: null },
  { label: "2-bit + copy: AdamW − Muon", k: "q2_ste_adam", v: diffs("q2_ste_adam", "q2_ste_muon"), line: null },
  { label: "2-bit, FP8 carry − 4-bit ticker", k: "q2_t8_m8", v: diffs("q2_t8_m8", "q4_t4_m8"), line: null },
]);

/* ---------- pass table ---------- */
(function () {
  const tb = document.querySelector("#tbl-pass tbody"), C = DATA.tb.checks;
  const rows = [["2-bit ticker, 4-bit carry ≤ 2-bit + float copy (Muon) + 0.05, each seed", C.q2_t4_m8], ["2-bit ticker, FP8 carry ≤ 2-bit + float copy (Muon) + 0.05, each seed", C.q2_t8_m8]];
  for (const [lab, c] of rows) {
    if (!c) continue;
    const tr = el("tr", {}, tb), n = c.diffs.filter(d => d <= 0.05).length;
    el("td", {}, tr, lab); el("td", { class: "n" }, tr, n + " of 3");
    el("span", { class: "pill " + (c.passed ? "pass" : "fail") }, el("td", {}, tr), c.passed ? "PASS" : "FAIL");
  }
})();

/* ---------- loss against memory ---------- */
(function () {
  const host = document.getElementById("ch-mem");
  const keys = ["m1", "m2", "q4_t4_m8", "fp8_m8", "q2_ste_muon", "q2_ste_adam", "q2_t4_m8", "q2_t8_m8"].filter(has);
  const draw = () => {
    header(host, { title: "Final loss against training memory", sub: "bytes per weight of the four big matrices (weights + optimizer state, 32-bit scales); mean of 3 seeds, small dots are single seeds" });
    const W = Math.max(280, host.clientWidth - 24), narrow = W < 560, H = narrow ? 300 : 340, m = { l: 50, r: 16, t: 14, b: 40 };
    const ys = keys.flatMap(finals), y0 = Math.min(...ys) - 0.03, y1 = Math.max(...ys) + 0.03;
    const svg = el("svg:svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "loss against memory" }, host);
    const X = scale(0, 13, m.l, W - m.r), Y = scale(y0, y1, H - m.b, m.t), g = el("svg:g", {}, svg);
    for (const v of niceTicks(y0, y1, 5)) { el("svg:line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: tok("--grid") }, g); el("svg:text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" }, g, v.toFixed(2)); }
    for (const v of [0, 2, 4, 6, 8, 10, 12]) el("svg:text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, g, v);
    el("svg:line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: tok("--axis") }, g);
    el("svg:text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle" }, g, "bytes per weight");
    const bytes = k => seedsOf(k).map(s => R[k][s].bytes)[0];
    const placed = [];
    for (const k of keys) {
      const col = tok(ARM[k].color), f = finals(k), cx = X(bytes(k)), cy = Y(mean(f));
      for (const v of f) el("svg:circle", { cx, cy: Y(v), r: 2.5, fill: col, opacity: 0.55 }, g);
      el("svg:circle", { cx, cy, r: 6, fill: ARM[k].dash ? tok("--surface") : col, stroke: col, "stroke-width": 2.5 }, g);
      let ly = cy + 4; while (placed.some(p => Math.abs(p.x - cx) < 80 && Math.abs(p.y - ly) < 13)) ly += 13; placed.push({ x: cx, y: ly });
      const right = cx < W * 0.6;
      if (!narrow) el("svg:text", { x: cx + (right ? 10 : -10), y: ly, "text-anchor": right ? "start" : "end", class: "lab-strong" }, g, ARM[k].short);
      hover(host, svg, W, H, { cx, cy, r: 14 }, () => [{ value: fmt(mean(f), 4) + " ± " + fmt(se(f), 4), label: "loss, mean ± SE", color: col }, { value: (+bytes(k)).toFixed(2), label: "bytes per weight" }], ARM[k].name, cx, cy);
    }
    if (narrow) legend(host, keys.map(k => ({ label: ARM[k].name, color: tok(ARM[k].color) })), true);
    numbers(host, ["arm", "bytes per weight", "mean loss", "SE", "seed 1", "seed 2", "seed 3"], keys.map(k => [ARM[k].name, (+bytes(k)).toFixed(2), fmt(mean(finals(k)), 4), fmt(se(finals(k)), 4), ...finals(k).map(v => fmt(v, 4))]));
  };
  register(host, draw);
})();
