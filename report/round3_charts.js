const ARM = {
  m1: { name: "m1 Muon, float", color: "--c-m1" },
  m2: { name: "m2 4-bit + float copy", color: "--c-m2" },
  m3: { name: "m3 4-bit, random rounding", color: "--c-m3" },
  m4post: { name: "m4c leftover buffer after NS", color: "--c-c" },
  float: { name: "m6 reordered, float", color: "--c-m0" },
  q4: { name: "m6 reordered, 4-bit", color: "--c-a" },
  q4_m8: { name: "m6 4-bit + 8-bit momentum", color: "--c-m5" },
};
const fmt = (v, d = 3) => (v == null || !isFinite(v)) ? "–" : (+v).toFixed(d);
const sgn = (v, d = 3) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);
const R = DATA.m6.runs;
const seedsOf = k => Object.keys(R[k] || {}).sort();
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const se = xs => Math.sqrt(xs.reduce((t, x) => t + (x - mean(xs)) ** 2, 0) / (xs.length - 1) / xs.length);
const finals = k => seedsOf(k).map(s => R[k][s].final);
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

/* ---------- m6: learning-rate pick ---------- */
lineChart(document.getElementById("ch-tune"), {
  title: "Reordered Muon (float weights), seed 1, by learning rate", sub: "validation loss; normal Muon at its usual rate for reference",
  series: () => [
    ...Object.entries(DATA.m6.tune).filter(([, r]) => r).map(([lr, r], i, a) => ({ label: "rate " + lr, color: ["--r1", "--r1", "--r2", "--r3", "--r3"][i], width: [1.5, 2, 2, 2, 1.5][i], opacity: [0.6, 1, 1, 1, 0.6][i], points: r.curve.filter(p => p[0] > 0) })),
    { label: "m1 Muon, rate 0.01", color: "--c-m1", strong: true, points: (R.m1 && R.m1[1] ? R.m1[1].curve : []).filter(p => p[0] > 0) },
  ],
  x: { domain: [250, 2000], ticks: [250, 500, 1000, 1500, 2000], label: "training step" }, y: { domain: [1.55, 2.4], tip: v => v.toFixed(4), tipAscending: true },
});

/* ---------- m6: training curves ---------- */
(function () {
  const host = document.getElementById("ch-m6curves");
  let show = "quant";
  const sets = { quant: ["m2", "m3", "m4post", "q4", "q4_m8"], float: ["m1", "float", "m2", "q4"] };
  lineChart(host, {
    title: "Validation loss during training", sub: "mean of seeds 1–3, all on one T4",
    controls: h => {
      const c = el("div", { class: "controls" }, h), g = el("div", { class: "grp", role: "group", "aria-label": "which arms" }, c);
      for (const [k, lab] of [["quant", "4-bit arms"], ["float", "float arms"]]) {
        const b = el("button", { type: "button", "aria-pressed": String(show === k) }, g, lab);
        b.addEventListener("click", () => { show = k; REG.find(r => r.host === host).draw(); });
      }
    },
    series: () => sets[show].filter(k => R[k] && seedsOf(k).length).map(k => ({ label: ARM[k].name, color: ARM[k].color, strong: k === "q4", points: meanCurve(k).filter(p => p[0] > 0) })),
    x: { domain: [250, 2000], ticks: [250, 500, 1000, 1500, 2000], label: "training step" }, y: { domain: [1.58, 2.3], tip: v => v.toFixed(4), tipAscending: true },
  });
})();

/* ---------- m6: seed-by-seed differences ---------- */
(function () {
  const host = document.getElementById("ch-m6diffs");
  const d = (a, b) => [1, 2, 3].filter(s => R[a] && R[a][s] && R[b] && R[b][s]).map(s => R[a][s].final - R[b][s].final);
  const rows = [
    { label: "m6 4-bit − m2", k: "q4", v: d("q4", "m2"), line: 0.03, side: "pass ←" },
    { label: "m6 4-bit + 8-bit mom. − m2", k: "q4_m8", v: d("q4_m8", "m2"), line: 0.03, side: "pass ←" },
    { label: "m6 float − m1", k: "float", v: d("float", "m1"), line: 0.03, side: "pass ←" },
    { label: "m6 4-bit − m3", k: "q4", v: d("q4", "m3"), line: null },
    { label: "m6 4-bit − m4c", k: "q4", v: d("q4", "m4post"), line: null },
    { label: "m3 − m2 (for scale)", k: "m3", v: d("m3", "m2"), line: null },
  ].filter(r => r.v.length);
  const draw = () => {
    header(host, { title: "Seed-by-seed differences in final loss", sub: "each dot is one seed; left of the black tick passes" });
    const vals = rows.flatMap(r => r.v), lo = Math.min(-0.05, ...vals) - 0.02, hi = Math.max(0.15, ...vals) + 0.02;
    const W = Math.max(280, host.clientWidth - 24), narrow = W < 560, rh = 40, m = { l: narrow ? 10 : 210, r: 16, t: 8, b: 34 };
    const H = m.t + rows.length * (rh + (narrow ? 16 : 0)) + m.b;
    const svg = el("svg:svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "per-seed differences" }, host);
    const X = scale(lo, hi, m.l, W - m.r), g = el("svg:g", {}, svg);
    for (const v of niceTicks(lo, hi, narrow ? 4 : 6)) { el("svg:line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: tok(Math.abs(v) < 1e-9 ? "--axis" : "--grid") }, g); el("svg:text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, g, Math.abs(v) < 1e-9 ? "0" : sgn(v, 2)); }
    rows.forEach((r, i) => {
      const yTop = m.t + i * (rh + (narrow ? 16 : 0)), y = yTop + (narrow ? 16 : 0) + rh / 2;
      el("svg:text", { x: narrow ? m.l : m.l - 12, y: narrow ? yTop + 12 : y + 4, "text-anchor": narrow ? "start" : "end", class: "lab" }, g, r.label);
      if (r.line != null) { el("svg:line", { x1: X(r.line), x2: X(r.line), y1: y - 13, y2: y + 13, stroke: tok("--ink"), "stroke-width": 1.5 }, g); el("svg:text", { x: X(r.line) - 4, y: y - 15, "text-anchor": "end" }, g, r.side); }
      const col = tok(ARM[r.k].color);
      el("svg:line", { x1: X(Math.min(...r.v)), x2: X(Math.max(...r.v)), y1: y, y2: y, stroke: col, "stroke-width": 2, opacity: 0.5 }, g);
      r.v.forEach((v, s) => {
        el("svg:circle", { cx: X(v), cy: y, r: 5, fill: col, stroke: tok("--surface"), "stroke-width": 2 }, g);
        hover(host, svg, W, H, { cx: X(v), cy: y, r: 12 }, () => [{ value: sgn(v, 4), label: "seed " + (s + 1), color: col }], r.label, X(v), y);
      });
    });
    numbers(host, ["comparison", "seed 1", "seed 2", "seed 3", "pass line"], rows.map(r => [r.label, ...r.v.map(v => sgn(v, 4)), r.line == null ? "measure only" : "≤ " + sgn(r.line, 2)]));
  };
  register(host, draw);
})();

/* ---------- m6: loss against memory ---------- */
(function () {
  const host = document.getElementById("ch-m6mem");
  const keys = Object.keys(ARM).filter(k => R[k] && seedsOf(k).length);
  const draw = () => {
    header(host, { title: "Final loss against training memory", sub: "bytes per weight of the four big matrices (weights + optimizer state); mean of 3 seeds, small dots are single seeds" });
    const W = Math.max(280, host.clientWidth - 24), narrow = W < 560, H = narrow ? 300 : 340, m = { l: 50, r: 16, t: 14, b: 40 };
    const ys = keys.flatMap(finals), y0 = Math.min(...ys) - 0.03, y1 = Math.max(...ys) + 0.03;
    const svg = el("svg:svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "loss against memory" }, host);
    const X = scale(0, 9, m.l, W - m.r), Y = scale(y0, y1, H - m.b, m.t), g = el("svg:g", {}, svg);
    if (R.m2) { const t = mean(finals("m2")) + 0.03; el("svg:rect", { x: X(0), y: Y(Math.min(t, y1)), width: X(2) - X(0), height: Math.max(0, Y(y0) - Y(Math.min(t, y1))), fill: tok("--wash") }, g); el("svg:text", { x: X(0.15), y: Y(y0) - 8, class: "lab" }, g, narrow ? "goal" : "goal: ≤ 2 bytes, ≤ m2 + 0.03"); }
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
      let ly = cy + 4; while (placed.some(p => Math.abs(p.x - cx) < 60 && Math.abs(p.y - ly) < 13)) ly += 13; placed.push({ x: cx, y: ly });
      const right = cx < W * 0.6;
      if (!narrow) el("svg:text", { x: cx + (right ? 10 : -10), y: ly, "text-anchor": right ? "start" : "end", class: "lab-strong" }, g, ARM[k].name);
      hover(host, svg, W, H, { cx, cy, r: 14 }, () => [{ value: fmt(mean(f), 4) + " ± " + fmt(se(f), 4), label: "loss, mean ± SE", color: col }, { value: String(bytes(k)), label: "bytes per weight" }], ARM[k].name, cx, cy);
    }
    if (narrow) legend(host, keys.map(k => ({ label: ARM[k].name, color: tok(ARM[k].color) })), true);
    numbers(host, ["arm", "bytes per weight", "mean loss", "SE", "seed 1", "seed 2", "seed 3"], keys.map(k => [ARM[k].name, bytes(k), fmt(mean(finals(k)), 4), fmt(se(finals(k)), 4), ...finals(k).map(v => fmt(v, 4))]));
  };
  register(host, draw);
})();

/* ---------- batch x learning rate ---------- */
const B = DATA.blr;
const KCOL = { 1: "--r1", 2: "--r2", 4: "--r3", 8: "--r4" };
function cellStats(arm, k, m) {
  const c = (B[arm][k] || {})[m]; if (!c) return null;
  const ok = Object.values(c).map(r => r.final).filter(v => v != null);
  return { n: Object.keys(c).length, ok: ok.length, mean: ok.length === 3 ? mean(ok) : null, se: ok.length === 3 ? se(ok) : null };
}
function mults(k) { return Object.keys(B.m3[k]).map(Number).sort((a, b) => a - b); }
function rateChart(id, title, sub, valueFn, yfmt, hl) {
  lineChart(document.getElementById(id), {
    title, sub, noEndLabels: false,
    series: () => ["1", "2", "4", "8"].map(k => ({ label: "batch " + k + "×", color: KCOL[k], points: mults(k).map(m => [m, valueFn(k, m)]).filter(p => p[1] != null) })),
    x: { log: true, domain: [0.85, 9.5], ticks: [1, 2, 4, 8], fmt: v => v + "×", label: "learning-rate multiplier (log scale)", tipLabel: "rate", tipFmt: v => v + "×" },
    y: { tip: v => yfmt(v), tipAscending: true, ...(hl || {}) },
  });
}
rateChart("ch-blr-m3", "m3 (4-bit, no copy): final loss", "mean of 3 seeds; a missing point means a run blew up", (k, m) => { const c = cellStats("m3", k, m); return c && c.mean; }, v => v.toFixed(4));
rateChart("ch-blr-m1", "m1 (Muon, float): final loss", "mean of 3 seeds", (k, m) => { const c = cellStats("m1", k, m); return c && c.mean; }, v => v.toFixed(4));
rateChart("ch-blr-gap", "Damage from 4-bit rounding: m3 − m1", "seed-matched, mean of 3 seeds", (k, m) => {
  const a = (B.m1[k] || {})[m], b = (B.m3[k] || {})[m]; if (!a || !b) return null;
  const g = [1, 2, 3].filter(s => a[s] && b[s] && a[s].final != null && b[s].final != null).map(s => b[s].final - a[s].final);
  return g.length === 3 ? mean(g) : null;
}, v => sgn(v, 4));

/* ---------- scales: how much one ruler per matrix costs (round test, trained m1 model) ---------- */
(function () {
  const host = document.getElementById("ch-scales"), T = DATA.round.trained, base = T.full.val_ce;
  const rows = [
    { k: "nearest / tensor", label: "one per matrix", bytes: "≈ 0" },
    { k: "nearest / row", label: "one per row", bytes: "0.008–0.03" },
    { k: "nearest / block_64", label: "one per 64 weights", bytes: "FP8 0.016 · fp32 0.06" },
    { k: "nearest / block_16", label: "one per 16 (what we use)", bytes: "FP8 0.06 · fp32 0.25" },
    { k: "gptq / block_16", label: "one per 16, GPTQ rounding", bytes: "FP8 0.06 · fp32 0.25" },
  ];
  const draw = () => {
    header(host, { title: "Loss added by rounding a trained model to 4 bits once", sub: "milestone 1's model (seed 1), validation loss increase over full precision; nearest rounding unless named" });
    const W = Math.max(280, host.clientWidth - 24), narrow = W < 560, rh = 30, m = { l: narrow ? 8 : 190, r: narrow ? 56 : 190, t: 6, b: 30 };
    const H = m.t + rows.length * (rh + (narrow ? 16 : 0)) + m.b;
    const svg = el("svg:svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "loss increase by scale granularity" }, host);
    const X = scale(0, 0.03, m.l, W - m.r), g = el("svg:g", {}, svg);
    for (const v of [0, 0.01, 0.02, 0.03]) { el("svg:line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: tok(v ? "--grid" : "--axis") }, g); el("svg:text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, g, v ? "+" + v.toFixed(2) : "0"); }
    rows.forEach((r, i) => {
      const yTop = m.t + i * (rh + (narrow ? 16 : 0)), y = yTop + (narrow ? 16 : 0) + rh / 2, v = T[r.k].val_ce - base;
      el("svg:text", { x: narrow ? m.l : m.l - 10, y: narrow ? yTop + 12 : y + 4, "text-anchor": narrow ? "start" : "end", class: "lab" }, g, r.label);
      const col = tok(r.k.startsWith("gptq") ? "--c-c" : "--c-m2");
      el("svg:rect", { x: X(0), y: y - 8, width: Math.max(1, X(v) - X(0)), height: 16, rx: 3, fill: col }, g);
      el("svg:text", { x: X(v) + 6, y: y + 4, class: "lab-strong" }, g, "+" + v.toFixed(3));
      if (!narrow) el("svg:text", { x: W - m.r + 58, y: y + 4 }, g, r.bytes);
      hover(host, svg, W, H, { x: m.l, y: y - rh / 2, width: W - m.r - m.l, height: rh }, () => [{ value: "+" + v.toFixed(4), label: "loss added", color: col }, { value: r.bytes, label: "bytes per weight for the scales" }], r.label, X(v), y);
    });
    numbers(host, ["scales", "loss added", "scale bytes per weight"], rows.map(r => [r.label, "+" + (T[r.k].val_ce - base).toFixed(4), r.bytes]));
  };
  register(host, draw);
})();

/* ---------- pass-line table ---------- */
(function () {
  const tb = document.querySelector("#tbl-pass tbody"), C = DATA.m6.checks;
  const rows = [["q4", "m6 reordered 4-bit: each seed ≤ m2 + 0.03"], ["q4_m8", "m6 4-bit with 8-bit momentum: each seed ≤ m2 + 0.03"], ["float", "m6 reordered float: each seed ≤ m1 + 0.03"]];
  for (const [k, label] of rows) {
    const c = C[k], tr = el("tr", {}, tb);
    el("td", {}, tr, label);
    el("td", { class: "n" }, tr, c ? c.diffs.filter(d => d <= 0.03).length + " of " + c.diffs.length : "–");
    el("span", { class: "pill " + (c && c.passed ? "pass" : "fail") }, el("td", {}, tr), c ? (c.passed ? "PASS" : "FAIL") : "NOT RUN");
  }
})();

/* ---------- why m6 failed: one-weight simulation ---------- */
(function () {
  const host = document.getElementById("ch-sim");
  let which = "steady";
  lineChart(host, {
    title: "One weight on a grid with gap 1, pushed 0.05 gaps per step", sub: "where the stored weight is, in grid gaps",
    controls: h => {
      const c = el("div", { class: "controls" }, h), g = el("div", { class: "grp", role: "group", "aria-label": "push" }, c);
      for (const [k, lab] of [["steady", "steady push"], ["noisy", "noisy push"]]) {
        const b = el("button", { type: "button", "aria-pressed": String(which === k) }, g, lab);
        b.addEventListener("click", () => { which = k; REG.find(r => r.host === host).draw(); });
      }
    },
    series: () => {
      const d = DATA.sim[which], pts = a => a.map((v, i) => [i + 1, v]);
      return [
        { label: "float weight (the target)", color: "--c-m0", points: pts(d.float) },
        { label: "separate leftover buffer (m4c)", color: "--c-c", points: pts(d.buffer) },
        { label: "leftover in the momentum (m6)", color: "--c-a", strong: true, points: pts(d.m6) },
      ];
    },
    x: { domain: [1, 400], ticks: [1, 100, 200, 300, 400], label: "step" }, y: { tip: v => v.toFixed(2) + " gaps" },
  });
})();

/* ---------- B2: gap against how far float training got ---------- */
(function () {
  const host = document.getElementById("ch-blr-prog");
  const pts = [];
  for (const k of ["1", "2", "4", "8"]) for (const m of mults(k)) {
    const a = B.m1[k][m], b = B.m3[k][m];
    if (!a || !b || Object.keys(a).length < 3) continue;
    const ss = ["1", "2", "3"].filter(s => a[s] && b[s] && a[s].final != null && b[s].final != null);
    if (ss.length < 3) continue;
    pts.push({ k, m, x: mean(ss.map(s => a[s].final)), y: mean(ss.map(s => b[s].final - a[s].final)) });
  }
  const a1 = B.m1["1"]["1"], b1 = B.m3["1"]["1"], n = a1["1"].curve.length;
  const ref = [...Array(n).keys()].map(i => [mean(["1", "2", "3"].map(s => a1[s].curve[i][1])), mean(["1", "2", "3"].map(s => b1[s].curve[i][1] - a1[s].curve[i][1]))]).filter(p => p[0] < 2.1);
  const draw = () => {
    header(host, { title: "Rounding damage against how far training got", sub: "x: float Muon's final loss in that setting; y: m3 − m1; line: batch 1×, old rate, read along training" });
    legend(host, ["1", "2", "4", "8"].map(k => ({ label: "batch " + k + "×", color: tok(KCOL[k]) })), true);
    const W = Math.max(280, host.clientWidth - 24), narrow = W < 560, H = narrow ? 300 : 340, m = { l: 50, r: 16, t: 14, b: 40 };
    const svg = el("svg:svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "gap against progress" }, host);
    const X = scale(2.05, 1.6, m.l, W - m.r), Y = scale(0, 0.24, H - m.b, m.t), g = el("svg:g", {}, svg);
    for (const v of [0, 0.05, 0.1, 0.15, 0.2]) { el("svg:line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: tok("--grid") }, g); el("svg:text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" }, g, v ? "+" + v.toFixed(2) : "0"); }
    for (const v of [2.0, 1.9, 1.8, 1.7, 1.6]) el("svg:text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, g, v.toFixed(1));
    el("svg:line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: tok("--axis") }, g);
    el("svg:text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle" }, g, "float Muon's loss (further right = further along)");
    el("svg:path", { d: ref.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1)).join(""), fill: "none", stroke: tok("--ink-2"), "stroke-width": 2, "stroke-dasharray": "5 4" }, g);
    if (!narrow) el("svg:text", { x: X(ref[1][0]) + 6, y: Y(ref[1][1]) - 8, class: "lab" }, g, "batch 1×, old rate, during training");
    for (const p of pts) {
      const col = tok(KCOL[p.k]), cx = X(p.x), cy = Y(Math.min(p.y, 0.24));
      el("svg:circle", { cx, cy, r: 6, fill: col, stroke: tok("--surface"), "stroke-width": 2 }, g);
      if (!narrow) el("svg:text", { x: cx + 9, y: cy + 4 }, g, "×" + p.m);
      hover(host, svg, W, H, { cx, cy, r: 12 }, () => [{ value: sgn(p.y, 4), label: "m3 − m1", color: col }, { value: fmt(p.x, 4), label: "m1 final loss" }], `batch ${p.k}×, rate ×${p.m}`, cx, cy);
    }
    numbers(host, ["setting", "m1 final", "m3 − m1"], pts.map(p => [`batch ${p.k}×, rate ×${p.m}`, fmt(p.x, 4), sgn(p.y, 4)]));
  };
  register(host, draw);
})();
