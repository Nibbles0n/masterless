const ARM = {
  adamw_float: { name: "AdamW, float weights", short: "AdamW float", color: "--c-m0" },
  muon_float: { name: "Muon, float weights", short: "Muon float", color: "--c-m1" },
  q4_float_copy: { name: "4-bit + float copy (Muon)", short: "4-bit + copy", color: "--c-m2", dash: "6 4" },
  fp8_m8: { name: "FP8 + 8-bit momentum", short: "FP8", color: "--c-b" },
  q4_t4_m8: { name: "4-bit ticker", short: "4-bit ticker", color: "--c-m3" },
};
const ORDER = ["adamw_float", "muon_float", "q4_float_copy", "fp8_m8", "q4_t4_m8"];
const TWO = k => k === "fp8_m8" || k === "q4_t4_m8";
const fmt = (v, d = 3) => (v == null || !isFinite(v)) ? "–" : (+v).toFixed(d);
const sgn = (v, d = 3) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);
const S1 = DATA.g.s1, S2 = DATA.g.s2;
const LAST = S2 || S1, LASTN = S2 ? "stage 2 (100M tokens)" : "stage 1 (25M tokens)";
const arms = st => ORDER.filter(k => st && st.rows[k]);
const PRE = S1.rows.muon_float.pretrained;
const CU_PER_H = 1.07;
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
    { k: "c", label: "float copy", color: "--c-m0" },
    { k: "s", label: "block scales", color: "--c-m1" },
    { k: "b", label: "carry", color: "--c-c" },
    { k: "m", label: "optimizer state", color: "--c-m3" },
  ];
  const m8 = 1 + 4 / 128;
  const rows = [
    { label: "AdamW, float weights", parts: { w: 4, m: 8 } },
    { label: "Muon, float weights", parts: { w: 4, m: 4 } },
    { label: "4-bit + float copy (Muon)", parts: { c: 4, m: 4 } },
    { label: "FP8 + 8-bit momentum", parts: { w: 1, s: 0.25, m: m8 }, strong: true },
    { label: "4-bit ticker", parts: { w: 0.5, s: 0.25, b: 0.5, m: m8 }, strong: true },
  ];
  const draw = () => {
    header(host, { title: "Training memory per weight of the 48 block matrices", sub: "bytes per weight: weights + optimizer state; scales stored in 32 bits; the 4-bit weights the float-copy run multiplies with are made fresh from the copy each step" });
    legend(host, PART.map(p => ({ label: p.label, color: tok(p.color) })), true);
    const W = Math.max(280, host.clientWidth - 24), narrow = W < 560, rh = 26, gap = narrow ? 30 : 14, m = { l: narrow ? 10 : 210, r: 46, t: 6, b: 30 };
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

/* ---------- start, end, and rounding after training ---------- */
(function () {
  const host = document.getElementById("ch-start");
  const keys = arms(LAST);
  const draw = () => {
    header(host, { title: "Where each run started and where it ended, " + LASTN, sub: "hollow dot: step 0, after rounding the pretrained weights onto the run's grid; filled dot: end of training; diamond: the float runs rounded to 4 bits after training; grey line: pretrained GPT-2 in float" });
    const rows = keys.map(k => LAST.rows[k]);
    const vals = rows.flatMap(r => [r.step0, r.final, r.fp4].filter(v => v != null)).concat([PRE]);
    const lo = Math.min(...vals) - 0.01, hi = Math.max(...vals) + 0.01;
    const W = Math.max(280, host.clientWidth - 24), narrow = W < 560, rh = 34, m = { l: narrow ? 10 : 210, r: 16, t: 22, b: 34 };
    const H = m.t + keys.length * (rh + (narrow ? 16 : 0)) + m.b;
    const svg = el("svg:svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "start and end loss by arm" }, host);
    const X = scale(lo, hi, m.l, W - m.r), g = el("svg:g", {}, svg);
    for (const v of niceTicks(lo, hi, narrow ? 4 : 6)) { el("svg:line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: tok("--grid") }, g); el("svg:text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, g, v.toFixed(2)); }
    el("svg:line", { x1: X(PRE), x2: X(PRE), y1: m.t - 6, y2: H - m.b, stroke: tok("--ink-2"), "stroke-width": 1.5, "stroke-dasharray": "3 3" }, g);
    el("svg:text", { x: X(PRE), y: m.t - 10, "text-anchor": "middle", class: "lab" }, g, "pretrained, float " + PRE.toFixed(3));
    el("svg:text", { x: (m.l + W - m.r) / 2, y: H - 2, "text-anchor": "middle" }, g, "validation loss (nats per token), lower is better");
    keys.forEach((k, i) => {
      const r = LAST.rows[k], yTop = m.t + i * (rh + (narrow ? 16 : 0)), y = yTop + (narrow ? 16 : 0) + rh / 2, col = tok(ARM[k].color);
      el("svg:text", { x: narrow ? m.l : m.l - 12, y: narrow ? yTop + 12 : y + 4, "text-anchor": narrow ? "start" : "end", class: TWO(k) ? "lab-strong" : "lab" }, g, ARM[k].name);
      el("svg:line", { x1: X(r.step0), x2: X(r.final), y1: y, y2: y, stroke: col, "stroke-width": 2, opacity: 0.5 }, g);
      el("svg:circle", { cx: X(r.step0), cy: y, r: 5, fill: tok("--surface"), stroke: col, "stroke-width": 2 }, g);
      el("svg:circle", { cx: X(r.final), cy: y, r: 6, fill: col, stroke: tok("--surface"), "stroke-width": 2 }, g);
      if (r.fp4 != null) { const x = X(r.fp4); el("svg:path", { d: `M${x},${y - 6}L${x + 6},${y}L${x},${y + 6}L${x - 6},${y}Z`, fill: tok("--surface"), stroke: col, "stroke-width": 2 }, g); }
      const tipRowsFn = () => [{ value: fmt(r.step0, 4), label: "step 0 (after rounding)", color: col }, { value: fmt(r.final, 4), label: "end of training", color: col }, ...(r.fp4 != null ? [{ value: fmt(r.fp4, 4), label: "rounded to 4 bits after training" }] : []), { value: sgn(r.final - PRE, 4), label: "end − pretrained float" }];
      hover(host, svg, W, H, { x: m.l, y: y - rh / 2, width: W - m.r - m.l, height: rh }, tipRowsFn, ARM[k].name, X(r.final), y);
    });
    numbers(host, ["arm", "step 0", "end", "4-bit after training", "end − pretrained float"], keys.map(k => { const r = LAST.rows[k]; return [ARM[k].name, fmt(r.step0, 4), fmt(r.final, 4), fmt(r.fp4, 4), sgn(r.final - PRE, 4)]; }));
  };
  register(host, draw);
})();

/* ---------- validation curves ---------- */
for (const [id, st, name] of [["ch-curves-s1", S1, "stage 1, 760 steps (25M tokens)"], ["ch-curves-s2", S2, "stage 2, 3,050 steps (100M tokens)"]]) {
  const host = document.getElementById(id);
  if (!st) { host.closest("figure").hidden = true; continue; }
  lineChart(host, {
    title: "Validation loss during training, " + name, sub: "256 fixed windows of 1,024 tokens; step 0 is the pretrained model rounded onto each run's grid; grey line: pretrained GPT-2 in float",
    series: () => arms(st).map(k => ({ label: ARM[k].short, color: ARM[k].color, dash: ARM[k].dash, strong: TWO(k), width: TWO(k) ? 2.5 : 1.5, points: st.rows[k].val })),
    hlines: [{ y: PRE, label: "pretrained, float" }],
    x: { label: "training step" }, y: { tip: v => v.toFixed(4), tipAscending: true },
  });
}

/* ---------- paired differences, both stages ---------- */
(function () {
  const host = document.getElementById("ch-diffs");
  const ex = (st, a, b) => st && st.extras.find(e => e.a === a && e.b === b);
  const ck = (st, a) => st && st.checks.find(c => c.arm === a && c.line === 3);
  const pick = (st, a, b) => { const c = (a === "fp8_m8" && b === "muon_float") || (a === "q4_t4_m8" && b === "q4_float_copy") ? ck(st, a) : null; if (c) return { d: c.diff, se: c.diff_se }; const e = ex(st, a, b); return e && { d: e.diff, se: e.se }; };
  const ROWS = [
    { a: "fp8_m8", b: "muon_float", label: "FP8 − Muon float", line: 0.03 },
    { a: "q4_t4_m8", b: "q4_float_copy", label: "4-bit ticker − 4-bit + copy", line: 0.03 },
    { a: "q4_t4_m8", b: "muon_float", label: "4-bit ticker − Muon float" },
    { a: "q4_t4_m8", b: "fp8_m8", label: "4-bit ticker − FP8" },
    { a: "fp8_m8", b: "q4_float_copy", label: "FP8 − 4-bit + copy" },
    { a: "q4_float_copy", b: "muon_float", label: "4-bit + copy − Muon float" },
    { a: "muon_float", b: "adamw_float", label: "Muon float − AdamW float" },
  ].map(r => Object.assign(r, { s1: pick(S1, r.a, r.b), s2: pick(S2, r.a, r.b) })).filter(r => r.s1 || r.s2);
  const draw = () => {
    header(host, { title: "Differences in final loss between runs", sub: "same batches, same 256 windows; bar: ± one standard error over the windows; hollow: stage 1 (25M tokens), filled: stage 2 (100M tokens); black tick: pass line" });
    const vals = ROWS.flatMap(r => [r.s1, r.s2].filter(Boolean).flatMap(p => [p.d - p.se, p.d + p.se])).concat([0, 0.03]);
    const lo = Math.min(...vals) - 0.01, hi = Math.max(...vals) + 0.01;
    const W = Math.max(280, host.clientWidth - 24), narrow = W < 560, rh = 40, m = { l: narrow ? 10 : 230, r: 16, t: 8, b: 34 };
    const H = m.t + ROWS.length * (rh + (narrow ? 16 : 0)) + m.b;
    const svg = el("svg:svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "paired differences" }, host);
    const X = scale(lo, hi, m.l, W - m.r), g = el("svg:g", {}, svg);
    for (const v of niceTicks(lo, hi, narrow ? 4 : 6)) { el("svg:line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: tok(Math.abs(v) < 1e-9 ? "--axis" : "--grid") }, g); el("svg:text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, g, Math.abs(v) < 1e-9 ? "0" : sgn(v, 2)); }
    ROWS.forEach((r, i) => {
      const yTop = m.t + i * (rh + (narrow ? 16 : 0)), y = yTop + (narrow ? 16 : 0) + rh / 2, col = tok(ARM[r.a].color);
      el("svg:text", { x: narrow ? m.l : m.l - 12, y: narrow ? yTop + 12 : y + 4, "text-anchor": narrow ? "start" : "end", class: r.line ? "lab-strong" : "lab" }, g, r.label);
      if (r.line) { el("svg:line", { x1: X(r.line), x2: X(r.line), y1: y - 14, y2: y + 14, stroke: tok("--ink"), "stroke-width": 1.5 }, g); }
      [["s1", -6], ["s2", 6]].forEach(([s, dy]) => {
        const p = r[s]; if (!p) return;
        el("svg:line", { x1: X(p.d - p.se), x2: X(p.d + p.se), y1: y + dy, y2: y + dy, stroke: col, "stroke-width": 2.5, "stroke-linecap": "round" }, g);
        el("svg:circle", { cx: X(p.d), cy: y + dy, r: 5, fill: s === "s1" ? tok("--surface") : col, stroke: col, "stroke-width": 2 }, g);
      });
      hover(host, svg, W, H, { x: m.l, y: y - rh / 2, width: W - m.r - m.l, height: rh }, () => [r.s1 && { value: sgn(r.s1.d, 4) + " ± " + fmt(r.s1.se, 4), label: "stage 1", color: col }, r.s2 && { value: sgn(r.s2.d, 4) + " ± " + fmt(r.s2.se, 4), label: "stage 2", color: col }, r.line && { value: "≤ " + sgn(r.line, 2), label: "pass line" }].filter(Boolean), r.label, X((r.s2 || r.s1).d), y);
    });
    numbers(host, ["comparison", "stage 1", "stage 2", "line"], ROWS.map(r => [r.label, r.s1 ? sgn(r.s1.d, 4) + " ± " + fmt(r.s1.se, 4) : "–", r.s2 ? sgn(r.s2.d, 4) + " ± " + fmt(r.s2.se, 4) : "–", r.line ? "≤ " + sgn(r.line, 2) : "measure only"]));
  };
  register(host, draw);
})();

/* ---------- pass table ---------- */
(function () {
  const tb = document.querySelector("#tbl-pass tbody");
  for (const [st, name] of [[S1, "1"], [S2, "2"]]) {
    if (!st) continue;
    for (const c of st.checks) {
      const tr = el("tr", {}, tb);
      el("td", {}, tr, name); el("td", {}, tr, ARM[c.arm].short); el("td", {}, tr, c.line + ". " + c.what); el("td", { class: "n" }, tr, c.value);
      el("span", { class: "pill " + (c.ok ? "pass" : "fail") }, el("td", {}, tr), c.ok ? "PASS" : "FAIL");
    }
  }
})();

/* ---------- stability: training loss and gradient size ---------- */
lineChart(document.getElementById("ch-train"), {
  title: "Training loss, " + LASTN, sub: "mean over each 10 steps; the batches are identical across runs, so the lines can be compared step by step",
  series: () => arms(LAST).map(k => ({ label: ARM[k].short, color: ARM[k].color, dash: ARM[k].dash, strong: TWO(k), width: TWO(k) ? 2 : 1.25, noDot: true, points: LAST.rows[k].train })),
  x: { label: "training step" }, y: { tip: v => v.toFixed(3), tipAscending: true }, table: false,
});
lineChart(document.getElementById("ch-grad"), {
  title: "Gradient size, " + LASTN, sub: "length of the full gradient vector, every 5th step; a run going unstable shows rising or spiking gradients",
  series: () => arms(LAST).map(k => ({ label: ARM[k].short, color: ARM[k].color, strong: TWO(k), width: 1.25, noDot: true, points: LAST.rows[k].grad.filter(p => p[1] != null) })),
  x: { label: "training step" }, y: { tip: v => v.toFixed(2) }, table: false,
});

/* ---------- share of weights that move per step ---------- */
lineChart(document.getElementById("ch-changed"), {
  title: "Share of stored weights that change value each step, " + LASTN, sub: "fraction of the 84.9M block-matrix weights whose code differs from the step before; dashed: the codes that float Muon (and FP8) weights would have if rounded to nearest, from the stage-3 re-runs",
  series: () => ["fp8_m8", "q4_t4_m8"].filter(k => LAST.rows[k]).map(k => ({ label: ARM[k].short + (k === "fp8_m8" ? ", FP8 codes" : ", 4-bit codes"), color: ARM[k].color, strong: true, width: 2, points: LAST.rows[k].changed.map(p => [p[0], 100 * p[1]]) })).concat(
    DATA.g.s3 && DATA.g.s3.curves.muon_float ? [
      { label: "Muon float, FP8 codes", color: "--c-m1", dash: "6 4", width: 1.5, points: DATA.g.s3.curves.muon_float.changed8.map(p => [p[0], 100 * p[1]]) },
      { label: "Muon float, 4-bit codes", color: "--c-m0", dash: "6 4", width: 1.5, points: DATA.g.s3.curves.muon_float.changed4.map(p => [p[0], 100 * p[1]]) }] : [],
    DATA.g.s3 && DATA.g.s3.curves.fp8_m8 ? [{ label: "FP8, 4-bit codes", color: "--c-b", dash: "2 3", width: 1.5, points: DATA.g.s3.curves.fp8_m8.changed4.map(p => [p[0], 100 * p[1]]) }] : []),
  x: { label: "training step" }, y: { tip: v => v.toFixed(2) + "%", fmt: v => v + "%" }, table: false,
});

/* ---------- time and cost ---------- */
(function () {
  const tb = document.querySelector("#tbl-time tbody");
  let tot = 0;
  for (const [st, name] of [[S1, "1"], [S2, "2"], [DATA.g.s3, "3 (re-runs)"]]) {
    if (!st) continue;
    for (const k of arms(st)) {
      const r = st.rows[k], h = r.wall_s / 3600; tot += h;
      const tr = el("tr", {}, tb);
      el("td", {}, tr, name); el("td", {}, tr, ARM[k].name); el("td", { class: "n" }, tr, (r.tokens / 1e6).toFixed(1) + "M");
      el("td", { class: "n" }, tr, (r.tok_per_s / 1e3).toFixed(1) + "k"); el("td", { class: "n" }, tr, (60 * h).toFixed(0) + " min"); el("td", { class: "n" }, tr, (h * CU_PER_H).toFixed(1));
    }
  }
  const tr = el("tr", { class: "total" }, tb);
  el("td", {}, tr, ""); el("td", {}, tr, "all runs above"); el("td", {}, tr, ""); el("td", {}, tr, "");
  el("td", { class: "n" }, tr, tot.toFixed(1) + " h"); el("td", { class: "n" }, tr, (tot * CU_PER_H).toFixed(0));
})();

/* ---------- stage 3: rounding after training against training in the format ---------- */
const Q = DATA.g.s3, PROBE = DATA.g.probe;
const CELL = (a, b) => Q && (Q.cells.find(c => c.trained === a && c.stored === b) || {}).loss;
(function () {
  const host = document.getElementById("ch-quant");
  if (!Q) { host.closest("figure").hidden = true; return; }
  const ROWS = [
    { fmt: "FP8 storage (1 byte per weight + scales)" },
    { label: "float Muon, rounded to FP8 after training", k: "muon_float", v: CELL("muon_float", "fp8"), hollow: true },
    { label: "trained in FP8 (FP8 + 8-bit momentum)", k: "fp8_m8", v: CELL("fp8_m8", "fp8") },
    { fmt: "4-bit storage (half a byte per weight + scales)" },
    { label: "float Muon, rounded to 4 bits after training", k: "muon_float", v: CELL("muon_float", "fp4"), hollow: true },
    { label: "FP8-trained, rounded to 4 bits after training", k: "fp8_m8", v: CELL("fp8_m8", "fp4"), hollow: true },
    { label: "trained in 4 bits with a float copy", k: "q4_float_copy", v: CELL("q4_float_copy", "fp4") },
    { label: "trained in 4 bits, ticker", k: "q4_t4_m8", v: CELL("q4_t4_m8", "fp4") },
  ].filter(r => r.fmt || r.v != null);
  const FL = CELL("muon_float", "float");
  const draw = () => {
    header(host, { title: "Storing in a format against training in it, after 100M tokens", sub: "final validation loss; hollow: trained in float (or FP8) and rounded to nearest at the end; filled: trained in the format; grey line: float Muon, not rounded" });
    const vals = ROWS.filter(r => r.v != null).map(r => r.v).concat([FL]), lo = Math.min(...vals) - 0.01, hi = Math.max(...vals) + 0.01;
    const W = Math.max(280, host.clientWidth - 24), narrow = W < 560, rh = 30, m = { l: narrow ? 10 : 300, r: 60, t: 22, b: 34 };
    const H = m.t + ROWS.reduce((t, r) => t + (r.fmt ? 26 : rh + (narrow ? 14 : 0)), 0) + m.b;
    const svg = el("svg:svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "loss by training and storage format" }, host);
    const X = scale(lo, hi, m.l, W - m.r), g = el("svg:g", {}, svg);
    for (const v of niceTicks(lo, hi, narrow ? 4 : 6)) { el("svg:line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: tok("--grid") }, g); el("svg:text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, g, v.toFixed(2)); }
    el("svg:line", { x1: X(FL), x2: X(FL), y1: m.t - 6, y2: H - m.b, stroke: tok("--ink-2"), "stroke-width": 1.5, "stroke-dasharray": "3 3" }, g);
    el("svg:text", { x: X(FL), y: m.t - 10, "text-anchor": "middle", class: "lab" }, g, "float Muon " + FL.toFixed(3));
    let y = m.t;
    for (const r of ROWS) {
      if (r.fmt) { el("svg:text", { x: narrow ? 10 : 10, y: y + 18, class: "lab-strong" }, g, r.fmt); y += 26; continue; }
      const yTop = y, cy = yTop + (narrow ? 14 : 0) + rh / 2, col = tok(ARM[r.k].color);
      el("svg:text", { x: narrow ? m.l : m.l - 12, y: narrow ? yTop + 11 : cy + 4, "text-anchor": narrow ? "start" : "end", class: "lab" }, g, r.label);
      el("svg:line", { x1: X(FL), x2: X(r.v), y1: cy, y2: cy, stroke: col, "stroke-width": 2, opacity: 0.35 }, g);
      el("svg:circle", { cx: X(r.v), cy, r: 6, fill: r.hollow ? tok("--surface") : col, stroke: col, "stroke-width": 2.5 }, g);
      el("svg:text", { x: X(hi) + 6, y: cy + 4, class: "lab" }, g, r.v.toFixed(3));
      hover(host, svg, W, H, { x: m.l, y: yTop, width: W - m.r - m.l, height: rh + (narrow ? 14 : 0) }, () => [{ value: fmt(r.v, 4), label: "final loss", color: col }, { value: sgn(r.v - FL, 4), label: "− float Muon" }], r.label, X(r.v), cy);
      y += rh + (narrow ? 14 : 0);
    }
    numbers(host, ["trained", "stored", "final loss", "− float Muon"], ROWS.filter(r => !r.fmt).map(r => [r.label, r.hollow ? "rounded after" : "trained in it", fmt(r.v, 4), sgn(r.v - FL, 4)]));
  };
  register(host, draw);
})();
(function () {
  const tb = document.querySelector("#tbl-quant tbody");
  if (!Q) { tb.closest(".tablewrap").hidden = true; return; }
  for (const c of Q.comps) { const tr = el("tr", {}, tb); el("td", {}, tr, c.what); el("td", { class: "n" }, tr, sgn(c.diff, 4) + " ± " + fmt(c.se, 4)); }
})();
if (Q) lineChart(document.getElementById("ch-qcurves"), {
  title: "Rounded after training, at every 500 steps", sub: "validation loss of the weights as trained (solid) and rounded to nearest (dashed); the ticker and 4-bit + copy are their stage-2 runs",
  series: () => [
    { label: "Muon float", color: "--c-m1", points: Q.curves.muon_float ? Q.curves.muon_float.val.filter(p => p[0] % 500 === 0 || p[0] === 3050) : [] },
    { label: "Muon → FP8", color: "--c-m1", dash: "6 4", points: Q.curves.muon_float ? Q.curves.muon_float.fp8 : [] },
    { label: "Muon → 4-bit", color: "--c-m0", dash: "6 4", points: Q.curves.muon_float ? Q.curves.muon_float.fp4 : [] },
    { label: "FP8", color: "--c-b", strong: true, width: 2.5, points: Q.curves.fp8_m8 ? Q.curves.fp8_m8.val.filter(p => p[0] % 500 === 0 || p[0] === 3050) : [] },
    { label: "FP8 → 4-bit", color: "--c-b", dash: "6 4", points: Q.curves.fp8_m8 ? Q.curves.fp8_m8.fp4 : [] },
    { label: "4-bit + copy", color: "--c-m2", points: S2.rows.q4_float_copy.val.filter(p => p[0] % 500 === 0 || p[0] === 3050) },
    { label: "4-bit ticker", color: "--c-m3", strong: true, width: 2.5, points: S2.rows.q4_t4_m8.val.filter(p => p[0] % 500 === 0 || p[0] === 3050) },
  ].filter(s => s.points.length),
  x: { label: "training step" }, y: { tip: v => v.toFixed(4), tipAscending: true },
});
else document.getElementById("ch-qcurves").closest("figure").hidden = true;

/* ---------- the ticker probe: share changed per step against learning rate ---------- */
if (PROBE) lineChart(document.getElementById("ch-probe"), {
  title: "Share of weights whose 4-bit value changes per step, against learning rate", sub: "four GPT-2 matrices (layer 5), 35 steps after a 15-step settle, Muon-like steps on random gradients; both axes on a log scale",
  series: () => [
    { v: "float + RTN", label: "exact (float, rounded)", color: "--c-m1" },
    { v: "ticker 4-bit", label: "ticker, carry k/14", color: "--c-m3", strong: true, width: 2.5 },
    { v: "ticker 8-bit", label: "ticker, 8-bit carry", color: "--c-c" },
    { v: "ticker 4-bit /15", label: "ticker, carry k/15", color: "--c-a", dash: "6 4" },
  ].map(s => Object.assign(s, { points: PROBE.filter(r => r.variant === s.v && r.changed > 0).map(r => [r.lr, r.changed]) })),
  x: { log: true, label: "learning rate", tipLabel: "learning rate", fmt: v => v.toExponential(0), tipFmt: v => v.toExponential(0) }, y: { log: true, fmt: v => v + "%", tip: v => v.toFixed(3) + "%" },
});
