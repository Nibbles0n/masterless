const fmt = (v, d = 3) => (v == null || !isFinite(v)) ? "–" : (+v).toFixed(d);
const sgn = (v, d = 1) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);
const T = DATA.tune;
const COL = { "AdamW 1e-6": "--c-a", "Muon 1.37e-5 (pilot)": "--c-m0", "Muon 4e-05": "--c-c", "Muon 0.0001": "--c-m2", "Muon 0.00025": "--c-m3" };
const NAME = { "AdamW 1e-6": "AdamW 1e-6", "Muon 1.37e-5 (pilot)": "Muon 1.37e-5 (pilot)", "Muon 4e-05": "Muon 4e-5", "Muon 0.0001": "Muon 1e-4", "Muon 0.00025": "Muon 2.5e-4" };

function hover(host, svg, W, H, hitAttrs, rowsFn, title, cx, cy) {
  const tip = host._tip || (host._tip = tipBox(host));
  const hit = el("svg:rect", Object.assign({ fill: "transparent", tabindex: "0" }, hitAttrs), svg);
  const on = () => { tipRows(tip, title, rowsFn()); const r = svg.getBoundingClientRect(), hr = host.getBoundingClientRect(); placeTip(host, tip, r.left - hr.left + cx * r.width / W, r.top - hr.top + cy * r.height / H); };
  hit.addEventListener("pointerenter", on); hit.addEventListener("focus", on);
  hit.addEventListener("pointerleave", () => tip.hidden = true); hit.addEventListener("blur", () => tip.hidden = true);
}

/* ---------- accuracy after 30 steps ---------- */
(function () {
  const host = document.getElementById("ch-acc");
  const rows = T.rows;
  const draw = () => {
    header(host, { title: "GSM8K exam accuracy after 30 steps of RL, float model", sub: "1,319 questions, greedy; bars are 95% intervals of the difference from AdamW, question by question; dashed: the start, 54.7%" });
    const W = Math.max(280, host.clientWidth - 24), narrow = W < 560, rh = 36, m = { l: narrow ? 118 : 150, r: narrow ? 46 : 60, t: 8, b: 34 };
    const H = m.t + rows.length * rh + m.b;
    const svg = el("svg:svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "accuracy per run" }, host);
    const X = scale(50, 75, m.l, W - m.r), g = el("svg:g", {}, svg);
    for (const v of [50, 55, 60, 65, 70, 75]) { el("svg:line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: tok("--grid") }, g); el("svg:text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, g, v + "%"); }
    const s = 100 * T.start_acc;
    el("svg:line", { x1: X(s), x2: X(s), y1: m.t, y2: H - m.b, stroke: tok("--ink-2"), "stroke-dasharray": "4 4" }, g);
    const a = 100 * rows[0].acc_end;
    rows.forEach((r, i) => {
      const y = m.t + i * rh + rh / 2, v = 100 * r.acc_end, c = tok(COL[r.run]), h = r.vs_adamw[1];
      el("svg:text", { x: m.l - 10, y: y + 4, "text-anchor": "end" }, g, NAME[r.run]);
      if (i > 0) el("svg:line", { x1: X(v - h), x2: X(v + h), y1: y, y2: y, stroke: c, "stroke-width": 2, "stroke-linecap": "round" }, g);
      el("svg:circle", { cx: X(v), cy: y, r: r.run === "Muon 0.0001" ? 7 : 5.5, fill: c, stroke: tok("--surface"), "stroke-width": 2 }, g);
      el("svg:text", { x: X(Math.max(v + h, v)) + 9, y: y + 4, class: "lab" }, g, v.toFixed(1) + "%");
      hover(host, svg, W, H, { x: 0, y: y - rh / 2, width: W, height: rh }, () => [
        { value: v.toFixed(1) + "%", label: "accuracy", color: c },
        { value: sgn(r.vs_start[0]) + " ± " + fmt(r.vs_start[1], 1), label: "points vs start" },
        ...(i ? [{ value: sgn(r.vs_adamw[0]) + " ± " + fmt(r.vs_adamw[1], 1), label: "points vs AdamW" }] : []),
        { value: fmt(r.eval_mean_len, 0), label: "answer length, tokens" }], NAME[r.run], X(v), y);
    });
    el("svg:line", { x1: X(a), x2: X(a), y1: m.t, y2: H - m.b, stroke: tok("--c-a"), "stroke-opacity": 0.35 }, g);
    numbers(host, ["run", "accuracy", "vs start (points)", "vs AdamW (points)", "answer length"], rows.map(r => [NAME[r.run], fmt(100 * r.acc_end, 1) + "%", sgn(r.vs_start[0]) + " ± " + fmt(r.vs_start[1], 1), r.run.startsWith("AdamW") ? "–" : sgn(r.vs_adamw[0]) + " ± " + fmt(r.vs_adamw[1], 1), fmt(r.eval_mean_len, 0)]));
  };
  register(host, draw);
})();

/* ---------- training reward ---------- */
const smooth = (pts, k = 5) => pts.map((p, i) => { const w = pts.slice(Math.max(0, i - k + 1), i + 1); return [p[0], w.reduce((s, q) => s + q[1], 0) / w.length]; });
lineChart(document.getElementById("ch-reward"), {
  title: "Share of practice answers marked right, during training",
  sub: "mean over the last 5 steps; 64 questions × 8 answers per step",
  height: 300,
  x: { ticks: [0, 5, 10, 15, 20, 25, 30], domain: [1, 30], label: "step" },
  y: { domain: [0.38, 0.72], ticks: [0.4, 0.5, 0.6, 0.7], fmt: v => v.toFixed(1), tip: v => fmt(v, 3) },
  series: () => Object.keys(COL).map(k => ({ label: NAME[k], color: COL[k], strong: k === "Muon 0.0001", points: smooth(T.curves[k].map(c => [c.step, c.reward])) })),
});
