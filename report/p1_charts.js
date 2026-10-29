const fmt = (v, d = 3) => (v == null || !isFinite(v)) ? "–" : (+v).toFixed(d);
const sgn = (v, d = 4) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);
const P = DATA.p1, C = P.curves;
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;

function hover(host, svg, W, H, hitAttrs, rowsFn, title, cx, cy) {
  const tip = host._tip || (host._tip = tipBox(host));
  const hit = el("svg:" + (hitAttrs.width ? "rect" : "circle"), Object.assign({ fill: "transparent", tabindex: "0" }, hitAttrs), svg);
  const on = () => { tipRows(tip, title, rowsFn()); const r = svg.getBoundingClientRect(), hr = host.getBoundingClientRect(); placeTip(host, tip, r.left - hr.left + cx * r.width / W, r.top - hr.top + cy * r.height / H); };
  hit.addEventListener("pointerenter", on); hit.addEventListener("focus", on);
  hit.addEventListener("pointerleave", () => tip.hidden = true); hit.addEventListener("blur", () => tip.hidden = true);
}

/* ---------- memory against loss ---------- */
(function () {
  const host = document.getElementById("ch-mem");
  const S = P.seeds.values;
  const pts = [
    { label: "ours (ticker)", bytes: 2.78, loss: mean(S.ours), color: "--c-m2", strong: true, dx: 10, dy: 16, seeds: S.ours },
    { label: "4-bit + float copy, Muon", bytes: 8, loss: mean(S.float_copy), color: "--c-m1", dx: -10, dy: -10, anchor: "end", seeds: S.float_copy },
    { label: "Kahan (exact carry)", bytes: 5.78, loss: P.measured_loss.kahan, color: "--c-c", dx: 0, dy: 22, anchor: "middle" },
    { label: "4-bit + float copy, AdamW", bytes: 12, loss: P.measured_loss.adamw_copy, color: "--c-a", dx: -10, dy: 18, anchor: "end" },
    { label: "bf16-SR, rounded to 4 bits", bytes: 3.03, loss: mean(S.bf16_sr_4bit), color: "--c-m3", dx: 10, dy: 4, seeds: S.bf16_sr_4bit },
    { label: "float Muon, rounded to 4 bits", bytes: 8, loss: P.measured_loss.muon_float_4bit, color: "--c-m0", dx: -10, dy: 4, anchor: "end" },
    { label: "naive (random rounding)", bytes: 1.78, loss: P.measured_loss.naive, color: "--c-b", dx: 10, dy: 4 },
    { label: "ECO-Muon", bytes: 1.78, loss: P.measured_loss.eco, color: "--c-m5", dx: -10, dy: 4, anchor: "end" },
  ];
  const draw = () => {
    header(host, { title: "Final loss against optimizer memory, GPT-2 small after 100M tokens", sub: "validation loss (lower is better) against bytes per weight for the weights and optimizer state; mean of 3 seeds where there are 3" });
    const W = Math.max(280, host.clientWidth - 24), narrow = W < 560, H = narrow ? 300 : 340;
    const m = { l: 52, r: 16, t: 14, b: 40 };
    const svg = el("svg:svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "final loss against bytes per weight" }, host);
    const X = scale(0, 13, m.l, W - m.r), Y = scale(3.26, 3.46, H - m.b, m.t), g = el("svg:g", {}, svg);
    for (const v of [3.28, 3.32, 3.36, 3.40, 3.44]) { el("svg:line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: tok("--grid") }, g); el("svg:text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" }, g, v.toFixed(2)); }
    el("svg:line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: tok("--axis") }, g);
    for (const v of [0, 2, 4, 6, 8, 10, 12]) el("svg:text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, g, v);
    el("svg:text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle" }, g, "bytes per weight");
    const start = P.start;
    el("svg:line", { x1: m.l, x2: W - m.r, y1: Y(start), y2: Y(start), stroke: tok("--ink-2"), "stroke-dasharray": "4 4" }, g);
    el("svg:text", { x: W - m.r - 4, y: Y(start) - 5, "text-anchor": "end", class: "lab" }, g, "start: GPT-2 rounded to 4 bits, " + fmt(start, 3));
    for (const p of pts) {
      const cx = X(p.bytes), cy = Y(p.loss);
      el("svg:circle", { cx, cy, r: p.strong ? 7 : 5.5, fill: tok(p.color), stroke: tok("--surface"), "stroke-width": 2 }, g);
      if (!narrow) el("svg:text", { x: cx + p.dx, y: cy + p.dy, "text-anchor": p.anchor || "start", class: p.strong ? "lab-strong" : "lab" }, g, p.label);
      hover(host, svg, W, H, { cx, cy, r: 12 }, () => [{ value: fmt(p.loss, 4), label: "loss" + (p.seeds ? " (seeds " + p.seeds.map(v => fmt(v, 4)).join(" · ") + ")" : ""), color: tok(p.color) }, { value: String(p.bytes), label: "bytes per weight" }], p.label, cx, cy);
    }
    if (narrow) legend(host, pts.map(p => ({ label: p.label, color: tok(p.color) })), false);
    numbers(host, ["arm", "bytes per weight", "final loss"], pts.map(p => [p.label, String(p.bytes), fmt(p.loss, 4)]));
  };
  register(host, draw);
})();

/* ---------- the headline gap, seed by seed ---------- */
(function () {
  const host = document.getElementById("ch-gap");
  const rows = P.pass.P1a.rows;
  const draw = () => {
    header(host, { title: "Ours minus the 4-bit float copy, each seed", sub: "final validation loss; bars show ± 2 paired standard errors over the 256 validation windows; the pass line is +0.03" });
    const W = Math.max(280, host.clientWidth - 24), narrow = W < 560, rh = 34, m = { l: 64, r: 20, t: 8, b: 34 };
    const H = m.t + rows.length * rh + m.b;
    const svg = el("svg:svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "gap per seed" }, host);
    const X = scale(-0.005, 0.035, m.l, W - m.r), g = el("svg:g", {}, svg);
    for (const v of [0, 0.01, 0.02, 0.03]) { el("svg:line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: tok(v ? "--grid" : "--axis") }, g); el("svg:text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, g, v ? "+" + v.toFixed(2) : "0"); }
    el("svg:line", { x1: X(0.03), x2: X(0.03), y1: m.t, y2: H - m.b, stroke: tok("--c-b"), "stroke-width": 2 }, g);
    el("svg:text", { x: X(0.03) - 6, y: H - m.b - 6, "text-anchor": "end", class: "lab" }, g, narrow ? "pass line" : "pass line (P1a)");
    rows.forEach((r, i) => {
      const y = m.t + i * rh + rh / 2;
      el("svg:text", { x: m.l - 10, y: y + 4, "text-anchor": "end" }, g, "seed " + r.seed);
      el("svg:line", { x1: X(r.gap - 2 * r.se), x2: X(r.gap + 2 * r.se), y1: y, y2: y, stroke: tok("--c-m2"), "stroke-width": 2, "stroke-linecap": "round" }, g);
      el("svg:circle", { cx: X(r.gap), cy: y, r: 5.5, fill: tok("--c-m2"), stroke: tok("--surface"), "stroke-width": 2 }, g);
      el("svg:text", { x: X(r.gap + 2 * r.se) + 8, y: y + 4, class: "lab" }, g, sgn(r.gap));
      hover(host, svg, W, H, { cx: X(r.gap), cy: y, r: 12 }, () => [{ value: sgn(r.gap) + " ± " + fmt(r.se, 4), label: "ours − float copy", color: tok("--c-m2") }, { value: fmt(r.ours, 4) + " vs " + fmt(r.copy, 4), label: "losses" }], "seed " + r.seed, X(r.gap), y);
    });
    numbers(host, ["seed", "ours", "float copy", "gap", "paired SE"], rows.map(r => [String(r.seed), fmt(r.ours, 5), fmt(r.copy, 5), sgn(r.gap), fmt(r.se, 4)]));
  };
  register(host, draw);
})();

/* ---------- training curves, seed 1 ---------- */
const CURVE = [
  ["q4_t8_m8_lr0.001_s1", "ours (ticker)", "--c-m2", true],
  ["q4_float_copy_lr0.001_s1", "4-bit + float copy", "--c-m1"],
  ["q4_kahan_m8_lr0.001_s1", "Kahan", "--c-c"],
  ["bf16_sr_m8_lr0.001_s1", "bf16-SR (in bf16)", "--c-m3"],
  ["muon_float_lr0.001_s1", "float Muon (fp32)", "--c-m0"],
  ["eco_muon_q4_lr0.001_s1_nsfp32", "ECO-Muon", "--c-m5"],
  ["q4_sr_m8_lr0.001_s1", "naive", "--c-b"],
];
lineChart(document.getElementById("ch-curves"), {
  title: "Validation loss during training, seed 1",
  sub: "every 100 steps (3.3M tokens); bf16-SR and float Muon are shown before rounding to 4 bits, so they are 16- and 32-bit models here",
  height: 320,
  x: { ticks: [0, 500, 1000, 1500, 2000, 2500, 3000], domain: [0, 3050], label: "step" },
  y: { domain: [3.25, 3.47], ticks: [3.26, 3.30, 3.34, 3.38, 3.42, 3.46], fmt: v => v.toFixed(2), tip: v => fmt(v, 4), tipAscending: true },
  series: () => CURVE.map(([k, label, color, strong]) => ({ label, color, strong, points: (C[k] || []) })),
});
