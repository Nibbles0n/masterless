const ARM = {
  m1: { name: "m1 Muon, float weights", short: "m1 float", color: "--c-m1" },
  m2: { name: "m2 4-bit + float copy", short: "m2 4-bit + copy", color: "--c-m2" },
  m3: { name: "m3 4-bit, no copy", short: "m3 4-bit", color: "--c-m0" },
  m4post: { name: "m4c 4-bit + 16-bit buffer", short: "m4c", color: "--c-c" },
  q4_b4_m8: { name: "m7 4-bit + 4-bit buffer + 8-bit momentum", short: "m7 4+4+8", color: "--c-a" },
  fp8_m8: { name: "m7 FP8 + 8-bit momentum", short: "m7 FP8", color: "--c-b" },
  fp8_fp4: { name: "m7 FP8, rounded to 4-bit after", short: "FP8 → 4-bit", color: "--c-m5", dash: "6 4" },
  q4_t4_m8: { name: "ticker: 4-bit + 4-bit carry + 8-bit momentum", short: "ticker", color: "--c-m3" },
};
const fmt = (v, d = 3) => (v == null || !isFinite(v)) ? "–" : (+v).toFixed(d);
const sgn = (v, d = 3) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);
const R = DATA.m7.runs;
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
    { k: "b", label: "leftover buffer", color: "--c-c" },
    { k: "m", label: "momentum", color: "--c-m3" },
  ];
  const rows = [
    { label: "m1 / m2 (float weights, 32-bit momentum)", parts: { w: 4, m: 4 } },
    { label: "m4c (4-bit, 16-bit buffer, 32-bit momentum)", parts: { w: 0.5, s: 0.25, b: 2, m: 4 } },
    { label: "m3 (4-bit, 32-bit momentum)", parts: { w: 0.5, s: 0.25, m: 4 } },
    { label: "m7 4-bit + 4-bit buffer + 8-bit momentum", parts: { w: 0.5, s: 0.25, b: 0.5 + 0.25, m: 1 + 4 / 128 }, strong: true },
    { label: "m7 FP8 + 8-bit momentum", parts: { w: 1, s: 0.25, m: 1 + 4 / 128 }, strong: true },
    { label: "ticker: 4-bit + 4-bit carry + 8-bit momentum", parts: { w: 0.5, s: 0.25, b: 0.5, m: 1 + 4 / 128 }, strong: true },
  ];
  const draw = () => {
    header(host, { title: "Training memory per weight of the four big matrices", sub: "bytes; scales stored in 32 bits (8-bit scales would save another 0.19 bytes per scale set)" });
    legend(host, PART.map(p => ({ label: p.label, color: tok(p.color) })), true);
    const W = Math.max(280, host.clientWidth - 24), narrow = W < 560, rh = 26, gap = narrow ? 30 : 14, m = { l: narrow ? 10 : 300, r: 46, t: 6, b: 30 };
    const H = m.t + rows.length * (rh + gap) + m.b;
    const svg = el("svg:svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "bytes per weight by part" }, host);
    const X = scale(0, 8, m.l, W - m.r), g = el("svg:g", {}, svg);
    for (const v of [0, 2, 4, 6, 8]) { el("svg:line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: tok(v ? "--grid" : "--axis") }, g); el("svg:text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, g, v); }
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

/* ---------- the three evaluations and the references ---------- */
(function () {
  const host = document.getElementById("ch-three");
  const keys = ["m1", "m2", "fp8_m8", "fp8_fp4", "m4post", "q4_t4_m8", "q4_b4_m8", "m3"].filter(has);
  const draw = () => {
    header(host, { title: "Final validation loss", sub: "mean of seeds 1–3 (big dot) with ± one standard error; small dots are single seeds; lower is better" });
    const vals = keys.flatMap(finals), lo = Math.min(...vals) - 0.02, hi = Math.max(...vals) + 0.02;
    const W = Math.max(280, host.clientWidth - 24), narrow = W < 560, rh = 34, m = { l: narrow ? 10 : 300, r: 70, t: 6, b: 34 };
    const H = m.t + keys.length * (rh + (narrow ? 16 : 0)) + m.b;
    const svg = el("svg:svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "final loss by arm" }, host);
    const X = scale(lo, hi, m.l, W - m.r), g = el("svg:g", {}, svg);
    for (const v of niceTicks(lo, hi, narrow ? 4 : 6)) { el("svg:line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: tok("--grid") }, g); el("svg:text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, g, v.toFixed(2)); }
    keys.forEach((k, i) => {
      const yTop = m.t + i * (rh + (narrow ? 16 : 0)), y = yTop + (narrow ? 16 : 0) + rh / 2, f = finals(k), mu = mean(f), e = se(f), col = tok(ARM[k].color);
      const strong = !k.startsWith("m");
      el("svg:text", { x: narrow ? m.l : m.l - 12, y: narrow ? yTop + 12 : y + 4, "text-anchor": narrow ? "start" : "end", class: strong ? "lab-strong" : "lab" }, g, ARM[k].name);
      el("svg:line", { x1: X(mu - e), x2: X(mu + e), y1: y, y2: y, stroke: col, "stroke-width": 3, "stroke-linecap": "round" }, g);
      for (const v of f) el("svg:circle", { cx: X(v), cy: y + 9, r: 2.5, fill: col, opacity: 0.6 }, g);
      el("svg:circle", { cx: X(mu), cy: y, r: 6, fill: col, stroke: tok("--surface"), "stroke-width": 2 }, g);
      el("svg:text", { x: X(hi) + 6, y: y + 4, class: "lab" }, g, mu.toFixed(3));
      hover(host, svg, W, H, { cx: X(mu), cy: y, r: 14 }, () => [{ value: fmt(mu, 4) + " ± " + fmt(e, 4), label: "mean ± SE", color: col }, ...f.map((v, s) => ({ value: fmt(v, 4), label: "seed " + (s + 1) }))], ARM[k].name, X(mu), y);
    });
    numbers(host, ["arm", "mean", "SE", "seed 1", "seed 2", "seed 3"], keys.map(k => [ARM[k].name, fmt(mean(finals(k)), 4), fmt(se(finals(k)), 4), ...finals(k).map(v => fmt(v, 4))]));
  };
  register(host, draw);
})();

/* ---------- training curves ---------- */
lineChart(document.getElementById("ch-curves"), {
  title: "Validation loss during training", sub: "mean of seeds 1–3; the dashed line is the FP8 model rounded to 4 bits at each check",
  series: () => ["m1", "m2", "m4post", "m3", "fp8_m8", "q4_b4_m8", "q4_t4_m8", "fp8_fp4"].filter(has).map(k => ({ label: ARM[k].short, color: ARM[k].color, dash: ARM[k].dash, strong: k.startsWith("q4") || k.startsWith("fp8"), width: k.startsWith("m") ? 1.5 : 2.5, points: meanCurve(k).filter(p => p[0] > 0) })),
  x: { domain: [250, 2000], ticks: [250, 500, 1000, 1500, 2000], label: "training step" }, y: { domain: [1.58, 2.1], tip: v => v.toFixed(4), tipAscending: true },
});

/* ---------- seed-by-seed differences ---------- */
function diffChart(id, title, sub, rowsAll) {
  const host = document.getElementById(id);
  const rows = rowsAll.filter(r => r.v.length);
  const draw = () => {
    header(host, { title, sub });
    const vals = rows.flatMap(r => r.v), lo = Math.min(-0.05, ...vals) - 0.02, hi = Math.max(0.08, ...vals) + 0.02;
    const W = Math.max(280, host.clientWidth - 24), narrow = W < 560, rh = 40, m = { l: narrow ? 10 : 210, r: 16, t: 8, b: 34 };
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
diffChart("ch-diffs", "Seed-by-seed differences in final loss", "each dot is one seed (same seed, same data order); the top two rows are pass lines: all three dots must sit left of the black tick", [
    { label: "m7 4+4+8 − m2", k: "q4_b4_m8", v: diffs("q4_b4_m8", "m2"), line: 0.03 },
    { label: "m7 FP8 − m1", k: "fp8_m8", v: diffs("fp8_m8", "m1"), line: 0.03 },
    { label: "FP8 → 4-bit − m7 4+4+8", k: "fp8_fp4", v: diffs("fp8_fp4", "q4_b4_m8"), line: null },
    { label: "FP8 → 4-bit − m7 FP8", k: "fp8_fp4", v: diffs("fp8_fp4", "fp8_m8"), line: null },
    { label: "m7 4+4+8 − m4c", k: "q4_b4_m8", v: diffs("q4_b4_m8", "m4post"), line: null },
    { label: "m7 4+4+8 − m3", k: "q4_b4_m8", v: diffs("q4_b4_m8", "m3"), line: null },
]);
diffChart("ch-tick", "The ticker, seed by seed", "ticker minus each other run, same seed; left of zero means the ticker is better", [
  { label: "ticker − m7 4+4+8", k: "q4_t4_m8", v: diffs("q4_t4_m8", "q4_b4_m8"), line: null },
  { label: "ticker − FP8 → 4-bit", k: "q4_t4_m8", v: diffs("q4_t4_m8", "fp8_fp4"), line: null },
  { label: "ticker − m4c", k: "q4_t4_m8", v: diffs("q4_t4_m8", "m4post"), line: null },
  { label: "ticker − m7 FP8", k: "q4_t4_m8", v: diffs("q4_t4_m8", "fp8_m8"), line: null },
  { label: "ticker − m2 (float copy)", k: "q4_t4_m8", v: diffs("q4_t4_m8", "m2"), line: 0.03, lineLabel: "m4 line (reference)" },
]);

/* ---------- pass table ---------- */
(function () {
  const tb = document.querySelector("#tbl-pass tbody"), C = DATA.m7.checks;
  const rows = [["m7 4-bit + 4-bit buffer + 8-bit momentum ≤ m2 + 0.03, each seed", C.q4_b4_m8], ["m7 FP8 + 8-bit momentum ≤ m1 + 0.03, each seed", C.fp8_m8]];
  for (const [lab, c] of rows) {
    if (!c) continue;
    const tr = el("tr", {}, tb), n = c.diffs.filter(d => d <= 0.03).length;
    el("td", {}, tr, lab); el("td", { class: "n" }, tr, n + " of 3");
    el("span", { class: "pill " + (c.passed ? "pass" : "fail") }, el("td", {}, tr), c.passed ? "PASS" : "FAIL");
  }
})();

/* ---------- loss against memory ---------- */
(function () {
  const host = document.getElementById("ch-mem");
  const keys = ["m1", "m2", "m3", "m4post", "q4_b4_m8", "fp8_m8", "q4_t4_m8"].filter(has);
  const draw = () => {
    header(host, { title: "Final loss against training memory", sub: "bytes per weight of the four big matrices (weights + optimizer state, 32-bit scales); mean of 3 seeds, small dots are single seeds" });
    const W = Math.max(280, host.clientWidth - 24), narrow = W < 560, H = narrow ? 300 : 340, m = { l: 50, r: 16, t: 14, b: 40 };
    const ys = keys.flatMap(finals), y0 = Math.min(...ys) - 0.03, y1 = Math.max(...ys) + 0.03;
    const svg = el("svg:svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "loss against memory" }, host);
    const X = scale(0, 9, m.l, W - m.r), Y = scale(y0, y1, H - m.b, m.t), g = el("svg:g", {}, svg);
    if (has("m2")) { const t = mean(finals("m2")) + 0.03; el("svg:rect", { x: X(0), y: Y(Math.min(t, y1)), width: X(2.6) - X(0), height: Math.max(0, Y(y0) - Y(Math.min(t, y1))), fill: tok("--wash") }, g); el("svg:text", { x: X(0.15), y: Y(y0) - 8, class: "lab" }, g, narrow ? "goal" : "goal: about 2 bytes, ≤ m2 + 0.03"); }
    for (const v of niceTicks(y0, y1, 5)) { el("svg:line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: tok("--grid") }, g); el("svg:text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" }, g, v.toFixed(2)); }
    for (const v of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) el("svg:text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, g, v);
    el("svg:line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: tok("--axis") }, g);
    el("svg:text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle" }, g, "bytes per weight");
    const bytes = k => seedsOf(k).map(s => R[k][s].bytes)[0];
    const placed = [];
    for (const k of keys) {
      const col = tok(ARM[k].color), f = finals(k), cx = X(bytes(k)), cy = Y(mean(f));
      for (const v of f) el("svg:circle", { cx, cy: Y(v), r: 2.5, fill: col, opacity: 0.55 }, g);
      el("svg:circle", { cx, cy, r: 6, fill: col, stroke: tok("--surface"), "stroke-width": 2 }, g);
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

/* ---------- the cost of rounding after training, over training ---------- */
lineChart(document.getElementById("ch-ptq"), {
  title: "How much worse than the FP8 model, step by step", sub: "mean of seeds 1–3; zero is the FP8 model itself",
  series: () => {
    if (!has("fp8_m8")) return [];
    const base = meanCurve("fp8_m8"), at = new Map(base.map(p => [p[0], p[1]]));
    return [{ label: "m7 FP8 (zero)", color: ARM.fp8_m8.color, width: 1.5, points: base.filter(p => p[0] > 0).map(p => [p[0], 0]) },
      ...["fp8_fp4", "q4_b4_m8", "q4_t4_m8"].filter(has).map(k => ({ label: ARM[k].short, color: ARM[k].color, dash: ARM[k].dash, strong: true, points: meanCurve(k).filter(p => p[0] > 0 && at.has(p[0])).map(p => [p[0], p[1] - at.get(p[0])]) }))];
  },
  x: { domain: [250, 2000], ticks: [250, 500, 1000, 1500, 2000], label: "training step" }, y: { tip: v => sgn(v, 4), tipAscending: true },
});

/* ---------- simulation: distance from the float weight ---------- */
(function () {
  const N = DATA.noise, NAME = { random: "random rounding, no memory", fine8: "random rounding, 8× finer grid", tick4: "ticker, 4-bit carry", tick: "ticker, exact carry (= float copy)" };
  const COL = { random: "--c-m0", fine8: "--c-b", tick4: "--c-m3", tick: "--c-m2" };
  lineChart(document.getElementById("ch-noise"), {
    title: "How far a stored weight drifts from where the optimizer wanted it", sub: "simulation of 20,000 weights, typical distance in gaps of the 4-bit grid (log scale); the push follows the real learning-rate schedule",
    series: () => Object.keys(NAME).map(k => ({ label: NAME[k], color: COL[k], strong: k.startsWith("tick"), points: N.dist[k] })),
    x: { domain: [0, 2000], ticks: [0, 500, 1000, 1500, 2000], label: "training step" }, y: { log: true, domain: [0.2, 10], ticks: [0.25, 0.5, 1, 2, 4, 8], fmt: v => String(v), tip: v => v.toFixed(2) + " gaps", tipAscending: true },
  });
})();
