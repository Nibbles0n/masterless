const fmt = (v, d = 3) => (v == null || !isFinite(v)) ? "–" : (+v).toFixed(d);
const Z = DATA.zz, B = ["0.9", "0.95", "0.99"].filter(b => Z[b]);
const BETA = { "0.9": "--c-m0", "0.95": "--c-m2", "0.99": "--c-m3" };
const CAND = {
  muon_cautious: { name: "Muon, only where this batch agrees", color: "--c-c" },
  rank8: { name: "Muon on the 8 strongest directions", color: "--c-a" },
  damped: { name: "Muon, weak directions shrunk", color: "--c-b" },
  muon_top25: { name: "Muon, largest quarter only", color: "--c-m5" },
  muon_shrink: { name: "Muon, soft threshold (half stay)", color: "--c-m3" },
  sign: { name: "sign of momentum", color: "--c-m0" },
};

function hover(host, svg, W, H, hitAttrs, rowsFn, title, cx, cy) {
  const tip = host._tip || (host._tip = tipBox(host));
  const hit = el("svg:" + (hitAttrs.width ? "rect" : "circle"), Object.assign({ fill: "transparent", tabindex: "0" }, hitAttrs), svg);
  const on = () => { tipRows(tip, title, rowsFn()); const r = svg.getBoundingClientRect(), hr = host.getBoundingClientRect(); placeTip(host, tip, r.left - hr.left + cx * r.width / W, r.top - hr.top + cy * r.height / H); };
  hit.addEventListener("pointerenter", on); hit.addEventListener("focus", on);
  hit.addEventListener("pointerleave", () => tip.hidden = true); hit.addEventListener("blur", () => tip.hidden = true);
}

/* ---------- the rounding law: prediction vs simulation ---------- */
(function () {
  const host = document.getElementById("ch-law");
  const sim = k => { const d = DATA.noise.dist[k]; return d[d.length - 1][1]; };
  const rows = [
    { label: "plain 4-bit, random rounding (m3)", pred: Math.sqrt(183), sim: sim("random") },
    { label: "a grid 8 times finer (about FP8)", pred: 2.4, sim: sim("fine8") },
    { label: "ticker, 4-bit carry", pred: Math.sqrt(2000 / 196 / 5), sim: sim("tick4") },
  ];
  const draw = () => {
    header(host, { title: "How far the stored weight drifts from where float training would put it", sub: "after 2000 steps, in gaps of the 4-bit grid; formula from section 2 against the simulation" });
    legend(host, [{ label: "formula", color: tok("--c-m1") }, { label: "simulation", color: tok("--c-m2") }], true);
    const W = Math.max(280, host.clientWidth - 24), narrow = W < 560, bh = 12, gap = narrow ? 34 : 20, m = { l: narrow ? 10 : 250, r: 46, t: 6, b: 30 };
    const H = m.t + rows.length * (2 * bh + 2 + gap) + m.b;
    const svg = el("svg:svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "drift by storage type" }, host);
    const X = scale(0, 14, m.l, W - m.r), g = el("svg:g", {}, svg);
    for (const v of [0, 2, 4, 6, 8, 10, 12, 14]) { el("svg:line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: tok(v ? "--grid" : "--axis") }, g); el("svg:text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, g, v); }
    rows.forEach((r, i) => {
      const y0 = m.t + i * (2 * bh + 2 + gap) + (narrow ? 18 : 0);
      el("svg:text", { x: narrow ? m.l : m.l - 12, y: narrow ? y0 - 6 : y0 + bh + 4, "text-anchor": narrow ? "start" : "end", class: "lab" }, g, r.label);
      [["formula", r.pred, "--c-m1", 0], ["simulation", r.sim, "--c-m2", bh + 2]].forEach(([lab, v, c, dy]) => {
        el("svg:rect", { x: X(0), y: y0 + dy, width: Math.max(1, X(v) - X(0)), height: bh, rx: 3, fill: tok(c) }, g);
        el("svg:text", { x: X(v) + 6, y: y0 + dy + bh - 2, class: "lab" }, g, fmt(v, 1));
        hover(host, svg, W, H, { x: X(0), y: y0 + dy, width: Math.max(8, X(v) - X(0)), height: bh }, () => [{ value: fmt(v, 2) + " gaps", label: lab, color: tok(c) }], r.label, X(v), y0 + dy);
      });
    });
    numbers(host, ["storage", "formula", "simulation"], rows.map(r => [r.label, fmt(r.pred, 2), fmt(r.sim, 2)]));
  };
  register(host, draw);
})();

/* ---------- zig-zag by phase and momentum ---------- */
lineChart(document.getElementById("ch-zz"), {
  title: "Zig-zag factor Z = distance travelled ÷ net movement",
  sub: "real Muon run, float weights; each point is one stretch of training (steps 0–500, 500–1500, 1500–2000)",
  x: { label: "middle of the stretch (step)", ticks: [0, 500, 1000, 1500, 2000], domain: [0, 2000] },
  y: { log: true, ticks: [1, 2, 5, 10, 20, 50], fmt: v => v, tip: v => fmt(v, 1) },
  series: () => B.map(b => ({ label: "momentum " + b, color: BETA[b], strong: b === "0.95",
    points: Z[b].phases.map(p => [(p.steps[0] + p.steps[1]) / 2, p.zigzag]) })),
});

/* ---------- progress per unit of path, Muon variants ---------- */
function effChart(id, key, title, sub) {
  lineChart(document.getElementById(id), {
    title, sub, height: 300,
    x: { ticks: [100, 500, 1000, 1500, 2000], domain: [0, 2000], label: "step" },
    y: { log: true, ticks: [0.5, 1, 2, 4, 8], fmt: v => v + "×", tip: v => fmt(v, 2) + "×" },
    hlines: [{ y: 1, label: "Muon = 1" }],
    series: () => Object.entries(CAND).map(([k, c]) => ({ label: c.name, color: c.color,
      points: Z["0.95"].checks.map(r => [r.step, Math.max(0.3, r[key][k])]) })),
  });
}
effChart("ch-l1", "per_l1", "Downhill progress per unit of path (what low-bit storage pays for)", "⟨step, true gradient⟩ ÷ sum of |moves|, as a multiple of Muon's; same momentum, same moment, median over the 16 matrices");
effChart("ch-l2", "per_l2", "Downhill progress per unit of step length (the usual measure)", "⟨step, true gradient⟩ ÷ length of the step, as a multiple of Muon's");

/* ---------- where Muon's progress comes from ---------- */
lineChart(document.getElementById("ch-top8"), {
  title: "Share of Muon's downhill progress from the momentum's 8 strongest directions",
  sub: "of 128 directions per matrix (median over the 16 matrices); Muon moves all 128 equally far",
  x: { ticks: [100, 500, 1000, 1500, 2000], domain: [0, 2000], label: "step" },
  y: { domain: [0, 1], ticks: [0, 0.25, 0.5, 0.75, 1], fmt: v => Math.round(v * 100) + "%", tip: v => Math.round(v * 100) + "%" },
  hlines: [{ y: 8 / 128, label: "if spread evenly (6%)" }],
  series: () => [{ label: "top 8 of 128", color: "--c-a", points: Z["0.95"].checks.map(r => [r.step, r.top8_share]) }],
});
