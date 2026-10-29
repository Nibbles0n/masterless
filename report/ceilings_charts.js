const fmt = (v, d = 2) => (v == null || !isFinite(v)) ? "–" : (+v).toFixed(d);
const KIND = {
  c_attn: { name: "attention input (c_attn)", color: "--c-m2" },
  attn_c_proj: { name: "attention output (attn_c_proj)", color: "--c-a" },
  c_fc: { name: "MLP input (c_fc)", color: "--c-m3" },
  mlp_c_proj: { name: "MLP output (mlp_c_proj)", color: "--c-c" },
};
const ROWS = DATA.c.inputs.rows;

function hover(host, svg, W, H, hitAttrs, rowsFn, title, cx, cy) {
  const tip = host._tip || (host._tip = tipBox(host));
  const hit = el("svg:rect", Object.assign({ fill: "transparent", tabindex: "0" }, hitAttrs), svg);
  const on = () => { tipRows(tip, title, rowsFn()); const r = svg.getBoundingClientRect(), hr = host.getBoundingClientRect(); placeTip(host, tip, r.left - hr.left + cx * r.width / W, r.top - hr.top + cy * r.height / H); };
  hit.addEventListener("pointerenter", on); hit.addEventListener("focus", on);
  hit.addEventListener("pointerleave", () => tip.hidden = true); hit.addEventListener("blur", () => tip.hidden = true);
}

/* one horizontal bar per row: {label, v, color, tip} */
function barChart(host, o) {
  const draw = () => {
    header(host, { title: o.title, sub: o.sub });
    const W = Math.max(280, host.clientWidth - 24), narrow = W < 560, bh = 16, gap = narrow ? 30 : 12, m = { l: narrow ? 10 : 230, r: 50, t: 6, b: 30 };
    const H = m.t + o.rows.length * (bh + gap) + m.b;
    const svg = el("svg:svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.title }, host);
    const X = scale(o.domain[0], o.domain[1], m.l, W - m.r), g = el("svg:g", {}, svg);
    for (const v of o.ticks) { el("svg:line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: tok(v === o.domain[0] ? "--axis" : "--grid") }, g); el("svg:text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, g, o.tickFmt ? o.tickFmt(v) : v); }
    o.rows.forEach((r, i) => {
      const y0 = m.t + i * (bh + gap) + (narrow ? 16 : 0), x0 = X(o.domain[0]);
      el("svg:text", { x: narrow ? m.l : m.l - 12, y: narrow ? y0 - 5 : y0 + bh - 4, "text-anchor": narrow ? "start" : "end", class: "lab" }, g, r.label);
      el("svg:rect", { x: x0, y: y0, width: Math.max(1, X(r.v) - x0), height: bh, rx: 3, fill: tok(r.color) }, g);
      el("svg:text", { x: X(r.v) + 6, y: y0 + bh - 4, class: "lab" }, g, o.valFmt(r.v));
      hover(host, svg, W, H, { x: x0, y: y0, width: Math.max(8, X(r.v) - x0), height: bh }, () => r.tip, r.label, X(r.v), y0);
    });
    if (o.table) numbers(host, o.table.head, o.table.rows);
  };
  register(host, draw);
}

/* ---------- rotation ---------- */
(function () {
  const R = DATA.c.rotation.pretrained.all, T = DATA.c.rotation.float_muon_s2.all;
  const names = { "none": "stored as is", "hadamard16 (in)": "Hadamard within each block of 16", "hadamard256 (in)": "Hadamard over 256 inputs",
    "hadamard256 (in and out)": "Hadamard over 256, both sides", "random full (in)": "random rotation, all inputs" };
  const keys = Object.keys(names);
  barChart(document.getElementById("ch-rot"), {
    title: "Average 4-bit gap ÷ typical weight", sub: "smaller is better; the grey bar is no rotation",
    domain: [0.28, 0.32], ticks: [0.28, 0.29, 0.30, 0.31, 0.32], valFmt: v => fmt(v, 4),
    rows: keys.map((k, i) => ({ label: names[k], v: R[k].gap, color: i ? "--c-m2" : "--c-m0",
      tip: [{ label: "gap ÷ weight", value: fmt(R[k].gap, 4) }, { label: "block max ÷ weight", value: fmt(R[k].scale, 3) }, { label: "round-to-nearest error ÷ weight", value: fmt(R[k].rtn, 4) }] })),
    table: { head: ["storage", "gap", "block max", "rounding error", "gap, trained weights"],
      rows: keys.map(k => [names[k], fmt(R[k].gap, 4), fmt(R[k].scale, 3), fmt(R[k].rtn, 4), fmt(T[k].gap, 4)]) },
  });
})();

/* ---------- whole-model gains ---------- */
(function () {
  const G = DATA.gains;
  const rows = [
    { label: "keep error out of the top 1%", v: G.keep1, color: "--c-m3" },
    { label: "sequential feedback (GPTQ rule)", v: G.seq, color: "--c-a" },
    { label: "keep error out of the top 10%", v: G.keep10, color: "--c-c" },
    { label: "ceiling (perfect aiming)", v: G.ceiling, color: "--c-m1" },
  ];
  rows.forEach(r => r.tip = [{ label: "less disturbance", value: fmt(r.v) + "×" }, { label: "like dividing the noise by", value: fmt(Math.sqrt(r.v)) }]);
  barChart(document.getElementById("ch-ways"), {
    title: "Less output disturbance for the same rounding error", sub: "1 = rounding error spread evenly, as now",
    domain: [1, 4], ticks: [1, 2, 3, 4], tickFmt: v => v + "×", valFmt: v => fmt(v) + "×", rows,
    table: { head: ["method", "gain", "noise ÷ by"], rows: rows.map(r => [r.label, fmt(r.v) + "×", fmt(Math.sqrt(r.v))]) },
  });
})();

/* ---------- by layer ---------- */
function byLayer(id, key, o) {
  lineChart(document.getElementById(id), Object.assign({
    x: { label: "layer", ticks: [0, 2, 4, 6, 8, 10, 11], domain: [0, 11] },
    series: () => Object.entries(KIND).map(([k, c]) => ({ label: c.name, color: c.color,
      points: ROWS.filter(r => r.kind === k).map(r => [r.layer, r[key]]) })),
  }, o));
}
byLayer("ch-layers", "am_over_gm", { title: "Ceiling: average ÷ geometric-mean input energy", sub: "how many times less rounding error could disturb the outputs, if aimed perfectly",
  y: { domain: [0, 30], ticks: [0, 5, 10, 15, 20, 25, 30], fmt: v => v + "×", tip: v => fmt(v, 1) + "×" } });
byLayer("ch-share", "top10pct_share", { title: "Input energy in the strongest 10% of directions", sub: "10% would mean evenly spread",
  y: { domain: [0, 1], ticks: [0, 0.25, 0.5, 0.75, 1], fmt: v => Math.round(v * 100) + "%", tip: v => Math.round(v * 100) + "%" },
  hlines: [{ y: 0.1, label: "even (10%)" }] });

/* ---------- budget table ---------- */
(function () {
  const tb = document.querySelector("#tbl-budget tbody"), f = Math.sqrt(DATA.gains.keep10);
  [["plain 4-bit, random rounding", 4.0], ["FP8 (milestone 7)", 1.8], ["ticker, 4-bit carry (k/14)", 1.1], ["ticker, FP8 carry", 0.6], ["ticker, 8-bit integer carry (k/254)", 0.25]]
    .forEach(([n, v]) => { const tr = document.createElement("tr"); tr.innerHTML = `<td>${n}</td><td class="n">${fmt(v)}</td><td class="n">${fmt(v / f)}</td>`; tb.appendChild(tr); });
})();
